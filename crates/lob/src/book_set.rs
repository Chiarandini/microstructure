//! Routing a whole ITCH session into per-symbol books.
//!
//! Every ITCH message carries `stock_locate` in its header, including the
//! order messages that identify their order only by id. That makes routing a
//! direct index by `u16` rather than a lookup through an order-id map, which
//! is both simpler and avoids maintaining a table with an entry per live
//! order across the session.

use crate::{Book, Event};
use itch::{Body, Header, Symbol, symbol_str};

/// `stock_locate` is a `u16`, so a flat table covers the whole space for
/// 256 KiB and removes hashing from the hot path entirely.
const LOCATE_SPACE: usize = u16::MAX as usize + 1;

/// The result of routing and applying one message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Applied {
    /// Index of the book the message was routed to.
    pub book: usize,
    /// The book event it produced, if any.
    pub event: Option<Event>,
}

/// A set of order books, one per tracked symbol, driven by an ITCH stream.
pub struct BookSet {
    books: Vec<Book>,
    symbols: Vec<String>,
    /// `stock_locate` to book index. `u32::MAX` means untracked.
    by_locate: Vec<u32>,
    /// Symbols to track. `None` tracks everything the directory announces.
    filter: Option<Vec<String>>,
    /// Symbols named in the filter that the directory never announced.
    announced: usize,
}

const UNTRACKED: u32 = u32::MAX;

impl BookSet {
    /// Track every symbol the session announces.
    pub fn all() -> Self {
        Self::new(None)
    }

    /// Track only the named symbols. Names are matched case-insensitively
    /// against the trimmed ITCH symbol.
    pub fn only(symbols: impl IntoIterator<Item = String>) -> Self {
        let f: Vec<String> = symbols
            .into_iter()
            .map(|s| s.trim().to_uppercase())
            .collect();
        Self::new(Some(f))
    }

    fn new(filter: Option<Vec<String>>) -> Self {
        BookSet {
            books: Vec::new(),
            symbols: Vec::new(),
            by_locate: vec![UNTRACKED; LOCATE_SPACE],
            filter,
            announced: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.books.len()
    }

    pub fn is_empty(&self) -> bool {
        self.books.is_empty()
    }

    /// Number of symbols the session's directory announced, tracked or not.
    pub fn announced(&self) -> usize {
        self.announced
    }

    pub fn book(&self, idx: usize) -> &Book {
        &self.books[idx]
    }

    pub fn book_mut(&mut self, idx: usize) -> &mut Book {
        &mut self.books[idx]
    }

    pub fn symbol(&self, idx: usize) -> &str {
        &self.symbols[idx]
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Book)> {
        self.symbols
            .iter()
            .map(String::as_str)
            .zip(self.books.iter())
    }

    fn wants(&self, symbol: &str) -> bool {
        self.filter
            .as_ref()
            .is_none_or(|f| f.iter().any(|s| s == symbol))
    }

    /// Register a symbol from a directory message.
    ///
    /// The directory block precedes trading, so every locate a later order
    /// message can reference has been registered by the time it arrives.
    fn register(&mut self, locate: u16, stock: &Symbol) {
        self.announced += 1;
        let name = symbol_str(stock);
        if !self.wants(name) {
            return;
        }
        // A duplicate directory entry must not create a second book, which
        // would silently split one symbol's flow across two states.
        if self.by_locate[locate as usize] != UNTRACKED {
            return;
        }
        self.by_locate[locate as usize] = self.books.len() as u32;
        self.books.push(Book::new());
        self.symbols.push(name.to_string());
    }

    /// Route a message to its book and apply it.
    ///
    /// Returns `None` when the message belongs to an untracked symbol or is
    /// one the book does not model.
    pub fn apply(&mut self, header: &Header, body: &Body) -> Option<Applied> {
        if let Body::StockDirectory { stock, .. } = body {
            self.register(header.stock_locate, stock);
            return None;
        }
        let idx = self.by_locate[header.stock_locate as usize];
        if idx == UNTRACKED {
            return None;
        }
        let idx = idx as usize;
        let event = self.books[idx].apply(body);
        Some(Applied { book: idx, event })
    }

    /// Total anomalies across every book.
    pub fn anomalies(&self) -> crate::Anomalies {
        let mut total = crate::Anomalies::default();
        for b in &self.books {
            total.unknown_order_ref += b.anomalies.unknown_order_ref;
            total.oversized_removal += b.anomalies.oversized_removal;
            total.duplicate_order_id += b.anomalies.duplicate_order_id;
            total.level_inconsistent += b.anomalies.level_inconsistent;
            total.crossed_book += b.anomalies.crossed_book;
        }
        total
    }

    pub fn live_orders(&self) -> usize {
        self.books.iter().map(Book::live_orders).sum()
    }

    /// Symbols whose level aggregates disagree with their order map.
    pub fn reconcile(&self) -> Vec<&str> {
        self.iter()
            .filter(|(_, b)| !b.depth_matches_orders())
            .map(|(s, _)| s)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use itch::Side;

    fn header(locate: u16) -> Header {
        Header {
            stock_locate: locate,
            tracking_number: 0,
            timestamp: 1,
        }
    }

    fn directory(stock: &[u8; 8]) -> Body {
        Body::StockDirectory {
            stock: *stock,
            round_lot_size: 100,
        }
    }

    fn add(order_ref: u64, side: Side, shares: u32, price: u32) -> Body {
        Body::AddOrder {
            order_ref,
            side,
            shares,
            price,
            stock: *b"XXXX    ",
            attributed: false,
        }
    }

    #[test]
    fn directory_creates_one_book_per_symbol() {
        let mut s = BookSet::all();
        s.apply(&header(1), &directory(b"AAPL    "));
        s.apply(&header(2), &directory(b"MSFT    "));
        assert_eq!(s.len(), 2);
        assert_eq!(s.announced(), 2);
        assert_eq!(s.symbol(0), "AAPL");
        assert_eq!(s.symbol(1), "MSFT");
    }

    #[test]
    fn filter_tracks_only_the_named_symbols() {
        let mut s = BookSet::only(["msft".to_string()]);
        s.apply(&header(1), &directory(b"AAPL    "));
        s.apply(&header(2), &directory(b"MSFT    "));
        assert_eq!(s.len(), 1);
        assert_eq!(s.announced(), 2);
        assert_eq!(s.symbol(0), "MSFT");

        // An untracked symbol's messages route nowhere rather than erroring.
        assert!(s.apply(&header(1), &add(1, Side::Buy, 100, 1000)).is_none());
        assert!(s.apply(&header(2), &add(2, Side::Buy, 100, 1000)).is_some());
    }

    /// The routing property that the old order-id map existed to provide:
    /// messages naming only an order id still reach the right book, because
    /// the header carries the locate.
    #[test]
    fn order_messages_route_by_locate_not_by_order_id() {
        let mut s = BookSet::all();
        s.apply(&header(7), &directory(b"AAPL    "));
        s.apply(&header(9), &directory(b"MSFT    "));

        s.apply(&header(7), &add(100, Side::Buy, 500, 1000));
        s.apply(&header(9), &add(200, Side::Sell, 300, 2000));

        let a = s.apply(
            &header(7),
            &Body::OrderCancel {
                order_ref: 100,
                cancelled_shares: 200,
            },
        );
        assert_eq!(a.unwrap().book, 0);
        assert_eq!(s.book(0).best_bid().unwrap().1.shares, 300);
        assert_eq!(s.book(1).best_ask().unwrap().1.shares, 300);
    }

    /// Replace rebinds an order to a new id. Routing must not depend on that,
    /// which is the simplification locate-based routing buys.
    #[test]
    fn replace_needs_no_id_rebinding() {
        let mut s = BookSet::all();
        s.apply(&header(3), &directory(b"AAPL    "));
        s.apply(&header(3), &add(1, Side::Buy, 100, 1000));
        s.apply(
            &header(3),
            &Body::OrderReplace {
                original_order_ref: 1,
                new_order_ref: 2,
                shares: 250,
                price: 1005,
            },
        );
        let a = s.apply(&header(3), &Body::OrderDelete { order_ref: 2 });
        assert_eq!(a.unwrap().book, 0);
        assert!(s.book(0).best_bid().is_none());
        assert_eq!(s.anomalies().unknown_order_ref, 0);
    }

    #[test]
    fn messages_for_unannounced_locates_are_ignored() {
        let mut s = BookSet::all();
        s.apply(&header(1), &directory(b"AAPL    "));
        assert!(
            s.apply(&header(999), &add(1, Side::Buy, 100, 1000))
                .is_none()
        );
        assert_eq!(s.len(), 1);
    }

    /// A repeated directory entry must not split one symbol across two books.
    #[test]
    fn duplicate_directory_entries_do_not_create_a_second_book() {
        let mut s = BookSet::all();
        s.apply(&header(1), &directory(b"AAPL    "));
        s.apply(&header(1), &directory(b"AAPL    "));
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn aggregates_roll_up_across_books() {
        let mut s = BookSet::all();
        s.apply(&header(1), &directory(b"AAPL    "));
        s.apply(&header(2), &directory(b"MSFT    "));
        s.apply(&header(1), &add(1, Side::Buy, 100, 1000));
        s.apply(&header(2), &add(2, Side::Sell, 100, 2000));
        assert_eq!(s.live_orders(), 2);
        assert!(s.reconcile().is_empty());

        s.apply(&header(1), &Body::OrderDelete { order_ref: 42 });
        assert_eq!(s.anomalies().unknown_order_ref, 1);
    }
}

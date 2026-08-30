//! Limit order book reconstruction from an ITCH message stream.
//!
//! One [`Book`] per symbol, driven by [`Book::apply`]. The book tracks
//! aggregate displayed depth per price level, plus the order map needed to
//! interpret messages that reference an order by id and carry neither price
//! nor side.
//!
//! # What is deliberately not modelled
//!
//! Queue position within a price level. ITCH gives enough information to
//! maintain it, but nothing in the current study depends on it, and carrying
//! it would roughly triple the per-message cost. [`Order`] keeps the fields
//! that would be needed to add it later.
//!
//! # Correctness
//!
//! A book that is subtly wrong yields features that are subtly wrong and a
//! result that is confidently false, so the invariants in [`Book::check`] are
//! run over full days rather than trusted.

pub mod book_set;

pub use book_set::{Applied, BookSet};

use itch::{Body, Side};
use std::collections::BTreeMap;
use std::collections::HashMap;

/// Price in ITCH fixed-point ticks of $0.0001.
pub type Price = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Order {
    pub side: Side,
    pub price: Price,
    /// Remaining displayed quantity.
    pub shares: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Level {
    pub shares: u64,
    /// Number of live orders resting at this price.
    pub orders: u32,
}

/// Something that happened at a point in time, emitted as the book is driven.
///
/// The distinction between a trade that consumed displayed depth and one that
/// did not is the reason this exists: both are "trades" to a tape reader, but
/// only the former is a book event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Event {
    /// Displayed depth was consumed by an execution.
    Trade {
        side: Side,
        price: Price,
        shares: u32,
        /// False for the non-printable leg of an execute-with-price.
        printable: bool,
    },
    /// A trade against non-displayed liquidity. Reported by the venue but not
    /// a change to the visible book.
    HiddenTrade { price: Price, shares: u32 },
    /// Opening, closing, or halt cross.
    Cross { price: Price, shares: u64 },
    /// Displayed depth was added.
    Add {
        side: Side,
        price: Price,
        shares: u32,
    },
    /// Displayed depth was withdrawn, by cancel or delete.
    Cancel {
        side: Side,
        price: Price,
        shares: u32,
    },
}

/// Counters describing anything the stream did that the book could not
/// explain. All of these should be zero or negligible on a clean run; a
/// non-trivial count means the reconstruction is wrong somewhere.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Anomalies {
    /// A message referenced an order id the book has never seen. Expected in
    /// small numbers only when starting mid-stream.
    pub unknown_order_ref: u64,
    /// An execution or cancel claimed more shares than the order held.
    pub oversized_removal: u64,
    /// An add reused an order id that was already live. Distinct from an
    /// oversized removal: this means a removal was missed, not that a removal
    /// was too large.
    pub duplicate_order_id: u64,
    /// A price level's aggregate share count and resting-order count
    /// disagreed about whether the level was empty.
    pub level_inconsistent: u64,
    /// Best bid met or crossed best ask outside an auction.
    pub crossed_book: u64,
}

#[derive(Debug, Default)]
pub struct Book {
    bids: BTreeMap<Price, Level>,
    asks: BTreeMap<Price, Level>,
    orders: HashMap<u64, Order>,
    pub anomalies: Anomalies,
    /// Set while the symbol is halted or in an auction, when a crossed book
    /// is legitimate and the invariant must not fire.
    pub auction: bool,
}

impl Book {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn best_bid(&self) -> Option<(Price, Level)> {
        self.bids.iter().next_back().map(|(&p, &l)| (p, l))
    }

    pub fn best_ask(&self) -> Option<(Price, Level)> {
        self.asks.iter().next().map(|(&p, &l)| (p, l))
    }

    /// Mid-price in ticks, `None` when either side is empty.
    ///
    /// Returned as `f64` because a mid can land on a half-tick; rounding it
    /// to an integer would quantise exactly the small moves the study is
    /// trying to measure.
    pub fn mid(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some((b, _)), Some((a, _))) => Some((b as f64 + a as f64) / 2.0),
            _ => None,
        }
    }

    pub fn spread(&self) -> Option<u32> {
        match (self.best_bid(), self.best_ask()) {
            (Some((b, _)), Some((a, _))) => Some(a.saturating_sub(b)),
            _ => None,
        }
    }

    /// Depth at the touch as `(bid_shares, ask_shares)`.
    pub fn touch_depth(&self) -> (u64, u64) {
        (
            self.best_bid().map_or(0, |(_, l)| l.shares),
            self.best_ask().map_or(0, |(_, l)| l.shares),
        )
    }

    /// Aggregate shares over the best `n` levels of each side.
    pub fn depth_n(&self, n: usize) -> (u64, u64) {
        let b = self.bids.values().rev().take(n).map(|l| l.shares).sum();
        let a = self.asks.values().take(n).map(|l| l.shares).sum();
        (b, a)
    }

    pub fn live_orders(&self) -> usize {
        self.orders.len()
    }

    fn side_map(&mut self, side: Side) -> &mut BTreeMap<Price, Level> {
        match side {
            Side::Buy => &mut self.bids,
            Side::Sell => &mut self.asks,
        }
    }

    fn add_depth(&mut self, side: Side, price: Price, shares: u32) {
        let level = self.side_map(side).entry(price).or_default();
        level.shares += shares as u64;
        level.orders += 1;
    }

    /// Remove `shares` from a level, dropping the level when it empties.
    ///
    /// `closing` marks the removal of a whole order, which also decrements
    /// the resting-order count; a partial cancel leaves the order in place.
    fn remove_depth(&mut self, side: Side, price: Price, shares: u32, closing: bool) {
        let map = match side {
            Side::Buy => &mut self.bids,
            Side::Sell => &mut self.asks,
        };
        let Some(level) = map.get_mut(&price) else {
            return;
        };
        level.shares = level.shares.saturating_sub(shares as u64);
        if closing {
            level.orders = level.orders.saturating_sub(1);
        }
        // The two counters are maintained together and should empty together.
        // Treating disagreement as "close enough" would hide exactly the
        // bookkeeping error that `depth_matches_orders` exists to detect, so
        // record it before cleaning up.
        let empty = level.shares == 0;
        if empty != (level.orders == 0) {
            self.anomalies.level_inconsistent += 1;
        }
        if empty || level.orders == 0 {
            map.remove(&price);
        }
    }

    /// Take `shares` off an existing order, returning what it actually held
    /// and whether that emptied it.
    fn take_from_order(&mut self, order_ref: u64, shares: u32) -> Option<(Order, u32, bool)> {
        let order = match self.orders.get_mut(&order_ref) {
            Some(o) => o,
            None => {
                self.anomalies.unknown_order_ref += 1;
                return None;
            }
        };
        let snapshot = *order;
        let taken = shares.min(order.shares);
        if taken < shares {
            self.anomalies.oversized_removal += 1;
        }
        order.shares -= taken;
        let emptied = order.shares == 0;
        if emptied {
            self.orders.remove(&order_ref);
        }
        Some((snapshot, taken, emptied))
    }

    /// Drive the book with one message, returning any event it produced.
    pub fn apply(&mut self, body: &Body) -> Option<Event> {
        match *body {
            Body::AddOrder {
                order_ref,
                side,
                shares,
                price,
                ..
            } => {
                // A duplicate id would silently corrupt depth. Overwriting is
                // wrong; the venue guarantees uniqueness among live orders, so
                // this only fires if we have lost track of a removal.
                if self
                    .orders
                    .insert(
                        order_ref,
                        Order {
                            side,
                            price,
                            shares,
                        },
                    )
                    .is_some()
                {
                    self.anomalies.duplicate_order_id += 1;
                }
                self.add_depth(side, price, shares);
                Some(Event::Add {
                    side,
                    price,
                    shares,
                })
            }

            Body::OrderExecuted {
                order_ref,
                executed_shares,
                ..
            } => {
                let (order, taken, emptied) = self.take_from_order(order_ref, executed_shares)?;
                self.remove_depth(order.side, order.price, taken, emptied);
                Some(Event::Trade {
                    side: order.side,
                    price: order.price,
                    shares: taken,
                    printable: true,
                })
            }

            Body::OrderExecutedWithPrice {
                order_ref,
                executed_shares,
                printable,
                execution_price,
                ..
            } => {
                let (order, taken, emptied) = self.take_from_order(order_ref, executed_shares)?;
                // Depth leaves at the order's resting price, not the price it
                // printed at. Removing at `execution_price` would corrupt a
                // level the order was never on.
                self.remove_depth(order.side, order.price, taken, emptied);
                Some(Event::Trade {
                    side: order.side,
                    price: execution_price,
                    shares: taken,
                    printable,
                })
            }

            Body::OrderCancel {
                order_ref,
                cancelled_shares,
            } => {
                let (order, taken, emptied) = self.take_from_order(order_ref, cancelled_shares)?;
                self.remove_depth(order.side, order.price, taken, emptied);
                Some(Event::Cancel {
                    side: order.side,
                    price: order.price,
                    shares: taken,
                })
            }

            Body::OrderDelete { order_ref } => {
                let order = match self.orders.remove(&order_ref) {
                    Some(o) => o,
                    None => {
                        self.anomalies.unknown_order_ref += 1;
                        return None;
                    }
                };
                self.remove_depth(order.side, order.price, order.shares, true);
                Some(Event::Cancel {
                    side: order.side,
                    price: order.price,
                    shares: order.shares,
                })
            }

            Body::OrderReplace {
                original_order_ref,
                new_order_ref,
                shares,
                price,
            } => {
                // Replace is delete-then-add under a new id, and loses queue
                // priority. Side is inherited from the replaced order, since
                // the message does not carry it.
                let old = match self.orders.remove(&original_order_ref) {
                    Some(o) => o,
                    None => {
                        self.anomalies.unknown_order_ref += 1;
                        return None;
                    }
                };
                self.remove_depth(old.side, old.price, old.shares, true);
                self.orders.insert(
                    new_order_ref,
                    Order {
                        side: old.side,
                        price,
                        shares,
                    },
                );
                self.add_depth(old.side, price, shares);
                Some(Event::Add {
                    side: old.side,
                    price,
                    shares,
                })
            }

            // Hidden liquidity was never in the displayed book, so applying
            // this would double-count depth removal.
            Body::TradeNonCross { price, shares, .. } => Some(Event::HiddenTrade { price, shares }),

            Body::CrossTrade {
                cross_price,
                shares,
                ..
            } => Some(Event::Cross {
                price: cross_price,
                shares,
            }),

            Body::TradingAction { trading_state, .. } => {
                // 'T' is trading; anything else (halted, quotation-only,
                // paused) permits a crossed or empty book.
                self.auction = trading_state != b'T';
                None
            }

            _ => None,
        }
    }

    /// Whether best bid meets or crosses best ask while trading normally.
    ///
    /// A crossed book during a halt or auction is legitimate, so that case
    /// reports false. Pure, so it can be called on a shared book; use
    /// [`Book::record_cross_check`] when the violation should also be tallied.
    pub fn is_crossed(&self) -> bool {
        if self.auction {
            return false;
        }
        match (self.best_bid(), self.best_ask()) {
            (Some((bid, _)), Some((ask, _))) => bid >= ask,
            _ => false,
        }
    }

    /// [`Book::is_crossed`], recording a violation in [`Book::anomalies`].
    /// Returns true when the book is well formed.
    pub fn record_cross_check(&mut self) -> bool {
        if self.is_crossed() {
            self.anomalies.crossed_book += 1;
            return false;
        }
        true
    }

    /// Total displayed shares on each side, used to check conservation.
    pub fn total_depth(&self) -> (u64, u64) {
        (
            self.bids.values().map(|l| l.shares).sum(),
            self.asks.values().map(|l| l.shares).sum(),
        )
    }

    /// Recompute level aggregates from the order map and compare.
    ///
    /// This is the strong consistency check: the incremental depth
    /// bookkeeping and the order map are maintained independently, so if they
    /// agree after a full day, both are almost certainly right. Too expensive
    /// for the hot path, so callers run it at checkpoints.
    pub fn depth_matches_orders(&self) -> bool {
        let mut bids: BTreeMap<Price, u64> = BTreeMap::new();
        let mut asks: BTreeMap<Price, u64> = BTreeMap::new();
        for o in self.orders.values() {
            let m = match o.side {
                Side::Buy => &mut bids,
                Side::Sell => &mut asks,
            };
            *m.entry(o.price).or_default() += o.shares as u64;
        }
        let lhs_b: BTreeMap<Price, u64> = self.bids.iter().map(|(&p, l)| (p, l.shares)).collect();
        let lhs_a: BTreeMap<Price, u64> = self.asks.iter().map(|(&p, l)| (p, l.shares)).collect();
        lhs_b == bids && lhs_a == asks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn add(order_ref: u64, side: Side, shares: u32, price: u32) -> Body {
        Body::AddOrder {
            order_ref,
            side,
            shares,
            price,
            stock: *b"TEST    ",
            attributed: false,
        }
    }

    #[test]
    fn adds_build_both_sides() {
        let mut b = Book::new();
        b.apply(&add(1, Side::Buy, 100, 1000));
        b.apply(&add(2, Side::Buy, 200, 999));
        b.apply(&add(3, Side::Sell, 300, 1001));

        assert_eq!(b.best_bid().unwrap().0, 1000);
        assert_eq!(b.best_ask().unwrap().0, 1001);
        assert_eq!(b.mid().unwrap(), 1000.5);
        assert_eq!(b.spread().unwrap(), 1);
        assert_eq!(b.touch_depth(), (100, 300));
        assert_eq!(b.depth_n(2), (300, 300));
        assert!(b.depth_matches_orders());
    }

    #[test]
    fn orders_at_one_price_aggregate() {
        let mut b = Book::new();
        b.apply(&add(1, Side::Buy, 100, 1000));
        b.apply(&add(2, Side::Buy, 150, 1000));
        let (_, level) = b.best_bid().unwrap();
        assert_eq!(level.shares, 250);
        assert_eq!(level.orders, 2);
        assert!(b.depth_matches_orders());
    }

    #[test]
    fn partial_cancel_leaves_the_order_resting() {
        let mut b = Book::new();
        b.apply(&add(1, Side::Buy, 100, 1000));
        let ev = b.apply(&Body::OrderCancel {
            order_ref: 1,
            cancelled_shares: 40,
        });
        assert_eq!(
            ev,
            Some(Event::Cancel {
                side: Side::Buy,
                price: 1000,
                shares: 40
            })
        );
        assert_eq!(b.best_bid().unwrap().1.shares, 60);
        assert_eq!(b.live_orders(), 1);
        assert!(b.depth_matches_orders());
    }

    #[test]
    fn delete_removes_the_level_when_it_empties() {
        let mut b = Book::new();
        b.apply(&add(1, Side::Buy, 100, 1000));
        b.apply(&Body::OrderDelete { order_ref: 1 });
        assert!(b.best_bid().is_none());
        assert_eq!(b.live_orders(), 0);
        assert!(b.depth_matches_orders());
    }

    #[test]
    fn execution_consumes_depth_and_reports_a_trade() {
        let mut b = Book::new();
        b.apply(&add(1, Side::Sell, 500, 2000));
        let ev = b.apply(&Body::OrderExecuted {
            order_ref: 1,
            executed_shares: 200,
            match_number: 7,
        });
        assert_eq!(
            ev,
            Some(Event::Trade {
                side: Side::Sell,
                price: 2000,
                shares: 200,
                printable: true
            })
        );
        assert_eq!(b.best_ask().unwrap().1.shares, 300);
        assert!(b.depth_matches_orders());
    }

    /// The execution price is what printed; the depth must still come off the
    /// level the order was actually resting on.
    #[test]
    fn execute_with_price_removes_depth_at_the_resting_price() {
        let mut b = Book::new();
        b.apply(&add(1, Side::Sell, 500, 2000));
        let ev = b.apply(&Body::OrderExecutedWithPrice {
            order_ref: 1,
            executed_shares: 200,
            match_number: 7,
            printable: false,
            execution_price: 1950,
        });
        assert_eq!(
            ev,
            Some(Event::Trade {
                side: Side::Sell,
                price: 1950,
                shares: 200,
                printable: false
            })
        );
        assert_eq!(b.best_ask().unwrap().0, 2000);
        assert_eq!(b.best_ask().unwrap().1.shares, 300);
        assert!(b.depth_matches_orders());
    }

    #[test]
    fn replace_moves_depth_and_rebinds_the_id() {
        let mut b = Book::new();
        b.apply(&add(1, Side::Buy, 100, 1000));
        b.apply(&Body::OrderReplace {
            original_order_ref: 1,
            new_order_ref: 2,
            shares: 250,
            price: 1005,
        });
        assert_eq!(
            b.best_bid().unwrap(),
            (
                1005,
                Level {
                    shares: 250,
                    orders: 1
                }
            )
        );
        assert_eq!(b.live_orders(), 1);
        // The old id must be gone: a later message referencing it is an error.
        assert!(b.apply(&Body::OrderDelete { order_ref: 1 }).is_none());
        assert_eq!(b.anomalies.unknown_order_ref, 1);
        assert!(b.depth_matches_orders());
    }

    /// The single most important negative test: hidden trades must not touch
    /// the displayed book.
    #[test]
    fn hidden_trades_do_not_change_depth() {
        let mut b = Book::new();
        b.apply(&add(1, Side::Buy, 100, 1000));
        let before = b.total_depth();
        let ev = b.apply(&Body::TradeNonCross {
            order_ref: 0,
            side: Side::Buy,
            shares: 999,
            stock: *b"TEST    ",
            price: 1000,
            match_number: 1,
        });
        assert_eq!(
            ev,
            Some(Event::HiddenTrade {
                price: 1000,
                shares: 999
            })
        );
        assert_eq!(b.total_depth(), before);
        assert!(b.depth_matches_orders());
    }

    #[test]
    fn cross_trades_do_not_change_depth() {
        let mut b = Book::new();
        b.apply(&add(1, Side::Buy, 100, 1000));
        let before = b.total_depth();
        b.apply(&Body::CrossTrade {
            shares: 50_000,
            stock: *b"TEST    ",
            cross_price: 1000,
            match_number: 1,
            cross_type: b'O',
        });
        assert_eq!(b.total_depth(), before);
    }

    #[test]
    fn unknown_order_refs_are_counted_not_panicked_on() {
        let mut b = Book::new();
        assert!(b.apply(&Body::OrderDelete { order_ref: 99 }).is_none());
        assert!(
            b.apply(&Body::OrderCancel {
                order_ref: 99,
                cancelled_shares: 1
            })
            .is_none()
        );
        assert_eq!(b.anomalies.unknown_order_ref, 2);
    }

    #[test]
    fn oversized_removal_is_clamped_and_counted() {
        let mut b = Book::new();
        b.apply(&add(1, Side::Buy, 100, 1000));
        let ev = b.apply(&Body::OrderCancel {
            order_ref: 1,
            cancelled_shares: 500,
        });
        assert_eq!(
            ev,
            Some(Event::Cancel {
                side: Side::Buy,
                price: 1000,
                shares: 100
            })
        );
        assert_eq!(b.anomalies.oversized_removal, 1);
        assert!(b.best_bid().is_none());
        assert!(b.depth_matches_orders());
    }

    #[test]
    fn crossed_book_is_detected_while_trading() {
        let mut b = Book::new();
        b.apply(&add(1, Side::Buy, 100, 1010));
        b.apply(&add(2, Side::Sell, 100, 1000));
        assert!(b.is_crossed());
        assert!(!b.record_cross_check());
        assert_eq!(b.anomalies.crossed_book, 1);
    }

    /// During a halt or auction a crossed book is legitimate and must not be
    /// reported as a reconstruction failure.
    #[test]
    fn crossed_book_is_allowed_during_an_auction() {
        let mut b = Book::new();
        b.apply(&Body::TradingAction {
            stock: *b"TEST    ",
            trading_state: b'H',
        });
        b.apply(&add(1, Side::Buy, 100, 1010));
        b.apply(&add(2, Side::Sell, 100, 1000));
        assert!(!b.is_crossed());
        assert!(b.record_cross_check());
        assert_eq!(b.anomalies.crossed_book, 0);

        b.apply(&Body::TradingAction {
            stock: *b"TEST    ",
            trading_state: b'T',
        });
        assert!(b.is_crossed());
    }

    /// A one-sided book cannot be crossed, and must not be reported as such.
    #[test]
    fn one_sided_book_is_not_crossed() {
        let mut b = Book::new();
        b.apply(&add(1, Side::Buy, 100, 1000));
        assert!(!b.is_crossed());
        assert!(b.record_cross_check());
    }

    /// A reused order id means a removal was missed, which is a different
    /// diagnosis from a removal that was too large.
    #[test]
    fn duplicate_order_id_is_counted_separately() {
        let mut b = Book::new();
        b.apply(&add(1, Side::Buy, 100, 1000));
        b.apply(&add(1, Side::Buy, 100, 1000));
        assert_eq!(b.anomalies.duplicate_order_id, 1);
        assert_eq!(b.anomalies.oversized_removal, 0);
    }

    /// Depth bookkeeping and the order map are maintained separately; after a
    /// long mixed sequence they must still agree.
    #[test]
    fn depth_and_orders_agree_after_a_mixed_sequence() {
        let mut b = Book::new();
        for i in 0..200u64 {
            let side = if i % 2 == 0 { Side::Buy } else { Side::Sell };
            let price = if i % 2 == 0 {
                1000 - (i as u32 % 5)
            } else {
                1010 + (i as u32 % 5)
            };
            b.apply(&add(i, side, 100 + (i as u32 % 7), price));
        }
        for i in (0..200u64).step_by(3) {
            b.apply(&Body::OrderCancel {
                order_ref: i,
                cancelled_shares: 30,
            });
        }
        for i in (0..200u64).step_by(5) {
            b.apply(&Body::OrderExecuted {
                order_ref: i,
                executed_shares: 20,
                match_number: i,
            });
        }
        for i in (0..200u64).step_by(7) {
            b.apply(&Body::OrderReplace {
                original_order_ref: i,
                new_order_ref: 10_000 + i,
                shares: 50,
                price: 1000,
            });
        }
        for i in (0..200u64).step_by(11) {
            b.apply(&Body::OrderDelete { order_ref: i });
        }
        assert!(b.depth_matches_orders());
    }
}

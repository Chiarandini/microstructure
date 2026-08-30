//! Per-event export: one gzipped CSV per symbol.
//!
//! Rust writes events; Python computes features. Nothing here aggregates,
//! smooths, or samples, because feature definitions will change many times
//! during the study and each change should cost a re-read of these files
//! rather than a re-parse of a multi-gigabyte session.
//!
//! Each row carries the top of book both *before* and *after* the event.
//! Order flow imbalance is then computable downstream without replaying the
//! book, and a queue-reactive fit gets its (queue state, transition) pairs
//! directly.
//!
//! # Look-ahead
//!
//! The `*_before` columns describe the book strictly prior to the event on
//! that row. Any predictor built from them uses only information available
//! before the event occurred. The `*_after` columns are the outcome and must
//! never be used as a feature for the same row. This is the one invariant a
//! downstream analysis can violate silently, so it is stated here and tested
//! in [`tests::before_excludes_the_event_itself`].

use flate2::Compression;
use flate2::write::GzEncoder;
use lob::{Event, TopOfBook};
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

pub const HEADER: &str = "ts_ns,event,side,price,shares,old_price,old_shares,\
bid_px_before,ask_px_before,bid_sz_before,ask_sz_before,\
bid_px_after,ask_px_after,bid_sz_after,ask_sz_after\n";

/// Classification written to the `event` column.
///
/// A replace gets its own label rather than being folded into `add`. It is a
/// cancellation plus a resubmission that loses queue priority, so its net
/// depth effect is `new - old`, and a model treating it as a submission would
/// systematically overstate incoming liquidity.
fn event_label(event: &Event) -> &'static str {
    match event {
        Event::Add { .. } => "add",
        Event::Cancel { .. } => "cancel",
        Event::Replace { .. } => "replace",
        Event::Trade { .. } => "trade",
        Event::HiddenTrade { .. } => "hidden",
        Event::Cross { .. } => "cross",
    }
}

/// `B`, `S`, or empty for events that have no displayed side.
fn event_side(event: &Event) -> &'static str {
    let side = match event {
        Event::Add { side, .. }
        | Event::Cancel { side, .. }
        | Event::Replace { side, .. }
        | Event::Trade { side, .. } => *side,
        // Hidden trades and crosses do not consume a displayed side.
        Event::HiddenTrade { .. } | Event::Cross { .. } => return "",
    };
    match side {
        itch::Side::Buy => "B",
        itch::Side::Sell => "S",
    }
}

/// The event's own price and size. For a replace this is the *new* leg; the
/// withdrawn leg goes to the `old_*` columns.
fn event_price_shares(event: &Event) -> (u32, u64) {
    match *event {
        Event::Add { price, shares, .. }
        | Event::Cancel { price, shares, .. }
        | Event::Trade { price, shares, .. }
        | Event::HiddenTrade { price, shares } => (price, shares as u64),
        Event::Replace {
            new_price,
            new_shares,
            ..
        } => (new_price, new_shares as u64),
        Event::Cross { price, shares } => (price, shares),
    }
}

/// The withdrawn leg of a replace, empty for every other event.
///
/// Carrying it means a consumer can decompose a replace into its cancel and
/// add legs without replaying the book.
fn event_old_leg(event: &Event) -> Option<(u32, u32)> {
    match *event {
        Event::Replace {
            old_price,
            old_shares,
            ..
        } => Some((old_price, old_shares)),
        _ => None,
    }
}

/// One gzipped CSV per symbol, created lazily on that symbol's first event.
pub struct EventWriter {
    dir: PathBuf,
    stem: String,
    /// Parallel to the [`lob::BookSet`] book indices.
    sinks: Vec<Option<GzEncoder<BufWriter<File>>>>,
    rows: Vec<u64>,
    /// Reused per row so writing does not allocate.
    scratch: String,
}

impl EventWriter {
    /// `stem` prefixes every file, and should identify the session, e.g.
    /// `20190730.NASDAQ_ITCH50`.
    ///
    /// Book indices are assigned as the session's directory block is read, so
    /// the writer grows to fit rather than being sized up front.
    pub fn new(dir: impl AsRef<Path>, stem: &str) -> io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        Ok(EventWriter {
            dir,
            stem: stem.to_string(),
            sinks: Vec::new(),
            rows: Vec::new(),
            scratch: String::with_capacity(256),
        })
    }

    pub fn path_for(&self, symbol: &str) -> PathBuf {
        self.dir.join(format!("{}_{symbol}.csv.gz", self.stem))
    }

    pub fn rows_written(&self, book: usize) -> u64 {
        self.rows.get(book).copied().unwrap_or(0)
    }

    pub fn total_rows(&self) -> u64 {
        self.rows.iter().sum()
    }

    fn sink(&mut self, book: usize, symbol: &str) -> io::Result<&mut GzEncoder<BufWriter<File>>> {
        if book >= self.sinks.len() {
            self.sinks.resize_with(book + 1, || None);
            self.rows.resize(book + 1, 0);
        }
        if self.sinks[book].is_none() {
            let file = File::create(self.path_for(symbol))?;
            // Level 1: these files are written once and read many times by
            // pandas, so decompression speed and write throughput matter more
            // than the last few percent of size.
            let mut enc = GzEncoder::new(BufWriter::new(file), Compression::new(1));
            enc.write_all(HEADER.as_bytes())?;
            self.sinks[book] = Some(enc);
        }
        Ok(self.sinks[book].as_mut().expect("just populated"))
    }

    /// Write one event row.
    ///
    /// `before` must be the top of book prior to applying the message that
    /// produced `event`, and `after` the state once applied.
    pub fn write(
        &mut self,
        book: usize,
        symbol: &str,
        ts_ns: u64,
        event: &Event,
        before: &TopOfBook,
        after: &TopOfBook,
    ) -> io::Result<()> {
        let (price, shares) = event_price_shares(event);
        let label = event_label(event);
        let side = event_side(event);

        // Formatted into a buffer owned by `self` and reused across rows, so
        // a session's worth of events costs no per-row allocation. Taken out
        // of `self` for the duration because `sink()` needs `&mut self`.
        let mut row = std::mem::take(&mut self.scratch);
        row.clear();
        use std::fmt::Write as _;
        let _ = write!(row, "{ts_ns},{label},{side},{price},{shares},");
        if let Some((old_price, old_shares)) = event_old_leg(event) {
            let _ = write!(row, "{old_price},{old_shares},");
        } else {
            row.push_str(",,");
        }
        push_px(&mut row, before.bid_px);
        row.push(',');
        push_px(&mut row, before.ask_px);
        let _ = write!(row, ",{},{},", before.bid_sz, before.ask_sz);
        push_px(&mut row, after.bid_px);
        row.push(',');
        push_px(&mut row, after.ask_px);
        let _ = writeln!(row, ",{},{}", after.bid_sz, after.ask_sz);

        let result = self
            .sink(book, symbol)
            .and_then(|s| s.write_all(row.as_bytes()));
        self.scratch = row;
        result?;
        self.rows[book] += 1;
        Ok(())
    }

    /// Finish every open file. Must be called; dropping without it leaves
    /// the gzip streams unterminated and the files unreadable.
    pub fn finish(&mut self) -> io::Result<()> {
        for sink in self.sinks.iter_mut() {
            if let Some(enc) = sink.take() {
                enc.finish()?.flush()?;
            }
        }
        Ok(())
    }
}

/// An absent price is written as an empty field, not as zero: an empty side
/// and a side priced at zero are different states.
fn push_px(out: &mut String, px: Option<u32>) {
    if let Some(p) = px {
        use std::fmt::Write as _;
        let _ = write!(out, "{p}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use itch::{Body, Header, Side};
    use lob::BookSet;

    fn header(locate: u16, ts: u64) -> Header {
        Header {
            stock_locate: locate,
            tracking_number: 0,
            timestamp: ts,
        }
    }

    fn directory() -> Body {
        Body::StockDirectory {
            stock: *b"TEST    ",
            round_lot_size: 100,
        }
    }

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

    fn read_back(path: &Path) -> Vec<String> {
        use std::io::Read;
        let f = File::open(path).unwrap();
        let mut s = String::new();
        flate2::read::GzDecoder::new(f)
            .read_to_string(&mut s)
            .unwrap();
        s.lines().map(str::to_string).collect()
    }

    /// Look a field up by column name, so tests do not break when the schema
    /// grows a column.
    fn field<'a>(lines: &'a [String], row: usize, column: &str) -> &'a str {
        let idx = lines[0]
            .split(',')
            .position(|c| c == column)
            .unwrap_or_else(|| panic!("no column {column:?} in header"));
        lines[row]
            .split(',')
            .nth(idx)
            .expect("row shorter than header")
    }

    /// Drive a BookSet the way replay does, capturing before and after.
    fn run(dir: &Path, messages: &[(u64, Body)]) -> Vec<String> {
        let mut books = BookSet::all();
        let mut w = EventWriter::new(dir, "test").unwrap();
        books.apply(&header(1, 0), &directory());

        for (ts, body) in messages {
            let h = header(1, *ts);
            let Some(idx) = books.route(h.stock_locate) else {
                continue;
            };
            let before = books.book(idx).top_of_book();
            if let Some(applied) = books.apply(&h, body)
                && let Some(event) = applied.event
            {
                let after = books.book(idx).top_of_book();
                w.write(idx, "TEST", *ts, &event, &before, &after).unwrap();
            }
        }
        w.finish().unwrap();
        read_back(&w.path_for("TEST"))
    }

    #[test]
    fn writes_a_header_and_one_row_per_event() {
        let dir = std::env::temp_dir().join("export_test_basic");
        let _ = fs::remove_dir_all(&dir);
        let lines = run(
            &dir,
            &[
                (100, add(1, Side::Buy, 500, 1000)),
                (200, add(2, Side::Sell, 300, 1010)),
            ],
        );
        assert_eq!(lines[0], HEADER.trim_end());
        assert_eq!(lines.len(), 3);
        assert!(lines[1].starts_with("100,add,B,1000,500,"));
        assert!(lines[2].starts_with("200,add,S,1010,300,"));
    }

    /// The look-ahead invariant. The `before` columns must describe the book
    /// without the row's own event; the `after` columns must include it.
    #[test]
    fn before_excludes_the_event_itself() {
        let dir = std::env::temp_dir().join("export_test_lookahead");
        let _ = fs::remove_dir_all(&dir);
        let lines = run(&dir, &[(100, add(1, Side::Buy, 500, 1000))]);

        // The book was empty before this add.
        assert_eq!(
            field(&lines, 1, "bid_px_before"),
            "",
            "before must not contain the event's own add"
        );
        assert_eq!(field(&lines, 1, "bid_sz_before"), "0");
        // The add is now resting.
        assert_eq!(field(&lines, 1, "bid_px_after"), "1000");
        assert_eq!(field(&lines, 1, "bid_sz_after"), "500");
    }

    /// An empty side is an empty field, distinguishable from a zero price.
    #[test]
    fn absent_prices_are_empty_not_zero() {
        let dir = std::env::temp_dir().join("export_test_empty");
        let _ = fs::remove_dir_all(&dir);
        let lines = run(&dir, &[(100, add(1, Side::Buy, 500, 1000))]);
        assert_eq!(
            field(&lines, 1, "ask_px_before"),
            "",
            "ask side is empty, not zero"
        );
        assert_eq!(field(&lines, 1, "ask_px_after"), "", "ask side still empty");
    }

    /// A replace at the same price nets out to `new - old`. Labelling it as a
    /// plain add would misreport that as `+new`, so it gets its own label and
    /// carries the withdrawn leg.
    #[test]
    fn replace_is_labelled_and_carries_the_withdrawn_leg() {
        let dir = std::env::temp_dir().join("export_test_replace");
        let _ = fs::remove_dir_all(&dir);
        let lines = run(
            &dir,
            &[
                (1, add(1, Side::Buy, 100, 1000)),
                (
                    2,
                    Body::OrderReplace {
                        original_order_ref: 1,
                        new_order_ref: 2,
                        shares: 250,
                        price: 1000,
                    },
                ),
            ],
        );
        assert_eq!(field(&lines, 2, "event"), "replace");
        assert_eq!(field(&lines, 2, "side"), "B");
        assert_eq!(field(&lines, 2, "price"), "1000");
        assert_eq!(field(&lines, 2, "shares"), "250");
        assert_eq!(field(&lines, 2, "old_price"), "1000");
        assert_eq!(field(&lines, 2, "old_shares"), "100");

        // Net depth change is +150, which only decomposes correctly if both
        // legs are present.
        assert_eq!(field(&lines, 2, "bid_sz_before"), "100");
        assert_eq!(field(&lines, 2, "bid_sz_after"), "250");
    }

    /// Non-replace events leave the old-leg columns empty.
    #[test]
    fn non_replace_events_have_no_old_leg() {
        let dir = std::env::temp_dir().join("export_test_no_old_leg");
        let _ = fs::remove_dir_all(&dir);
        let lines = run(&dir, &[(1, add(1, Side::Buy, 100, 1000))]);
        assert_eq!(field(&lines, 1, "old_price"), "");
        assert_eq!(field(&lines, 1, "old_shares"), "");
    }

    #[test]
    fn trades_and_cancels_are_labelled_and_sided() {
        let dir = std::env::temp_dir().join("export_test_labels");
        let _ = fs::remove_dir_all(&dir);
        let lines = run(
            &dir,
            &[
                (1, add(1, Side::Sell, 500, 2000)),
                (
                    2,
                    Body::OrderExecuted {
                        order_ref: 1,
                        executed_shares: 200,
                        match_number: 9,
                    },
                ),
                (
                    3,
                    Body::OrderCancel {
                        order_ref: 1,
                        cancelled_shares: 100,
                    },
                ),
            ],
        );
        assert!(lines[2].starts_with("2,trade,S,2000,200,"));
        assert!(lines[3].starts_with("3,cancel,S,2000,100,"));
    }

    /// Hidden trades carry no displayed side and must not perturb the book,
    /// so before and after have to match.
    #[test]
    fn hidden_trades_have_no_side_and_do_not_move_the_book() {
        let dir = std::env::temp_dir().join("export_test_hidden");
        let _ = fs::remove_dir_all(&dir);
        let lines = run(
            &dir,
            &[
                (1, add(1, Side::Buy, 500, 1000)),
                (
                    2,
                    Body::TradeNonCross {
                        order_ref: 0,
                        side: Side::Buy,
                        shares: 99,
                        stock: *b"TEST    ",
                        price: 1000,
                        match_number: 1,
                    },
                ),
            ],
        );
        assert_eq!(field(&lines, 2, "event"), "hidden");
        assert_eq!(
            field(&lines, 2, "side"),
            "",
            "hidden trades have no displayed side"
        );
        for col in ["bid_px", "ask_px", "bid_sz", "ask_sz"] {
            assert_eq!(
                field(&lines, 2, &format!("{col}_before")),
                field(&lines, 2, &format!("{col}_after")),
                "hidden trade must not move {col}"
            );
        }
    }

    /// A file left unfinished is an unterminated gzip stream and unreadable,
    /// so finish() is not optional.
    #[test]
    fn finish_produces_a_readable_file() {
        let dir = std::env::temp_dir().join("export_test_finish");
        let _ = fs::remove_dir_all(&dir);
        let mut books = BookSet::all();
        let mut w = EventWriter::new(&dir, "test").unwrap();
        books.apply(&header(1, 0), &directory());
        let h = header(1, 5);
        let before = lob::TopOfBook::default();
        let applied = books.apply(&h, &add(1, Side::Buy, 100, 1000)).unwrap();
        let after = books.book(0).top_of_book();
        w.write(0, "TEST", 5, &applied.event.unwrap(), &before, &after)
            .unwrap();
        assert_eq!(w.total_rows(), 1);
        w.finish().unwrap();

        assert_eq!(read_back(&w.path_for("TEST")).len(), 2);
    }
}

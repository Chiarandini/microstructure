//! Replay an ITCH session: reconstruct books, verify invariants, report.

use clap::Parser;
use export::EventWriter;
use itch::{Body, MessageKind};
use lob::BookSet;
use std::path::Path;
use std::time::Instant;

#[derive(Parser)]
#[command(about = "Reconstruct limit order books from a Nasdaq ITCH 5.0 file and verify them")]
struct Args {
    /// Path to an ITCH file, `.gz` or plain.
    path: String,

    /// Symbols to reconstruct, comma-separated. Omit to track every symbol.
    #[arg(long, value_delimiter = ',')]
    symbols: Option<Vec<String>>,

    /// Stop after this many messages. Useful for a fast smoke test.
    #[arg(long)]
    limit: Option<u64>,

    /// Run the full depth-versus-order-map reconciliation every N messages.
    /// Expensive; 0 checks only at the end.
    #[arg(long, default_value_t = 0)]
    reconcile_every: u64,

    /// Write a per-event CSV per tracked symbol into this directory.
    ///
    /// Requires `--symbols`: exporting every symbol of a session would write
    /// hundreds of millions of rows, which is not a thing anyone wants by
    /// accident.
    #[arg(long)]
    out_dir: Option<String>,

    /// Restrict exported events to the continuous session.
    #[arg(long, default_value_t = false)]
    session_only: bool,
}

/// Continuous session, 09:30 to 16:00 ET, as nanoseconds since midnight.
///
/// The crossed-book invariant only holds here. Outside it, auction interest
/// accumulates on both sides and a crossed book is expected.
const SESSION_OPEN_NS: u64 = 9 * 3_600_000_000_000 + 30 * 60_000_000_000;
const SESSION_CLOSE_NS: u64 = 16 * 3_600_000_000_000;

/// Everything the run observed, for reporting.
struct Summary {
    counts: [u64; MessageKind::COUNT],
    total: u64,
    first_ts: u64,
    last_ts: u64,
    cross_checks: u64,
    cross_failures: u64,
    exported: u64,
    elapsed: std::time::Duration,
}

impl Summary {
    fn messages_of(&self, kind: MessageKind) -> u64 {
        self.counts[kind as usize]
    }
}

fn main() {
    let args = Args::parse();

    let mut reader = match itch::reader::open(&args.path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("cannot open {}: {e}", args.path);
            std::process::exit(1);
        }
    };

    let mut books = match &args.symbols {
        Some(s) => BookSet::only(s.iter().cloned()),
        None => BookSet::all(),
    };

    if args.out_dir.is_some() && args.symbols.is_none() {
        eprintln!("--out-dir requires --symbols");
        std::process::exit(1);
    }

    let summary = match replay(&mut reader, &mut books, &args) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    report(&args, &summary, &books);
}

/// Session identifier used to prefix exported files.
///
/// Only `.gz` is stripped: the rest of the name carries the date and venue
/// (`20190730.NASDAQ_ITCH50`), which is exactly what distinguishes one
/// exported session from another.
fn session_stem(path: &str) -> String {
    let name = Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "session".to_string());
    name.strip_suffix(".gz").unwrap_or(&name).to_string()
}

fn replay<R: std::io::Read>(
    reader: &mut itch::reader::Reader<R>,
    books: &mut BookSet,
    args: &Args,
) -> Result<Summary, String> {
    let mut counts = [0u64; MessageKind::COUNT];
    let mut total = 0u64;
    let mut first_ts = u64::MAX;
    let mut last_ts = 0u64;
    let mut cross_checks = 0u64;
    let mut cross_failures = 0u64;

    let mut writer = match &args.out_dir {
        Some(dir) => Some(
            EventWriter::new(dir, &session_stem(&args.path))
                .map_err(|e| format!("cannot open {dir}: {e}"))?,
        ),
        None => None,
    };

    let start = Instant::now();

    loop {
        let (header, body) = match reader.next_message() {
            Ok(Some(m)) => m,
            Ok(None) => break,
            Err(e) => return Err(format!("fatal at message {total}: {e}")),
        };
        total += 1;
        counts[body.kind() as usize] += 1;

        if header.timestamp > 0 {
            first_ts = first_ts.min(header.timestamp);
            last_ts = last_ts.max(header.timestamp);
        }

        let in_session = (SESSION_OPEN_NS..=SESSION_CLOSE_NS).contains(&header.timestamp);

        // Snapshot before applying, so the exported row's `before` columns
        // cannot contain the event on that row. Routing is an array index, so
        // this costs nothing on runs that are not exporting.
        let before = writer.as_ref().and_then(|_| {
            books
                .route(header.stock_locate)
                .map(|idx| (idx, books.book(idx).top_of_book()))
        });

        if let Some(applied) = books.apply(&header, &body) {
            // Only quote-moving messages can newly cross the book, so
            // checking on the others would cost time without finding
            // anything the next add or replace would not.
            let quote_moving = matches!(body, Body::AddOrder { .. } | Body::OrderReplace { .. });
            if quote_moving && in_session {
                cross_checks += 1;
                if !books.book_mut(applied.book).record_cross_check() {
                    cross_failures += 1;
                }
            }

            if let (Some(w), Some((idx, before)), Some(event)) =
                (writer.as_mut(), before, applied.event)
                && (in_session || !args.session_only)
            {
                let after = books.book(idx).top_of_book();
                let symbol = books.symbol(idx);
                w.write(idx, symbol, header.timestamp, &event, &before, &after)
                    .map_err(|e| format!("writing {symbol}: {e}"))?;
            }
        }

        if args.reconcile_every > 0 && total.is_multiple_of(args.reconcile_every) {
            let diverged = books.reconcile();
            if !diverged.is_empty() {
                return Err(format!(
                    "depth and order map diverged at message {total} for: {}",
                    diverged.join(", ")
                ));
            }
        }

        if args.limit.is_some_and(|l| total >= l) {
            break;
        }
    }

    let mut exported = 0;
    if let Some(w) = writer.as_mut() {
        // Unterminated gzip streams are unreadable, so this is not optional
        // and must happen before the elapsed time is taken.
        w.finish()
            .map_err(|e| format!("closing export files: {e}"))?;
        exported = w.total_rows();
    }

    Ok(Summary {
        counts,
        total,
        first_ts,
        last_ts,
        cross_checks,
        cross_failures,
        exported,
        elapsed: start.elapsed(),
    })
}

fn report(args: &Args, s: &Summary, books: &BookSet) {
    let secs = s.elapsed.as_secs_f64();
    println!("file                {}", args.path);
    println!("symbols announced   {}", books.announced());
    println!("books tracked       {}", books.len());
    println!(
        "session             {} to {}",
        fmt_ns(s.first_ts),
        fmt_ns(s.last_ts)
    );
    println!();

    println!("messages            {:>13}", s.total);
    for kind in MessageKind::ALL {
        println!("  {:<16}{:>13}", kind.label(), s.messages_of(kind));
    }
    println!();
    println!(
        "elapsed             {secs:.2} s  ({:.2} M msg/s)",
        s.total as f64 / secs / 1e6
    );
    println!();

    let a = books.anomalies();
    println!("reconstruction");
    println!("  crossed-book checks {:>11}", s.cross_checks);
    println!("  crossed-book fails  {:>11}", s.cross_failures);
    println!("  unknown order refs  {:>11}", a.unknown_order_ref);
    println!("  oversized removals  {:>11}", a.oversized_removal);
    println!("  duplicate order ids {:>11}", a.duplicate_order_id);
    println!("  level inconsistent  {:>11}", a.level_inconsistent);
    println!("  orders still live   {:>11}", books.live_orders());

    let diverged = books.reconcile();
    println!(
        "  depth vs order map  {:>11}",
        if diverged.is_empty() {
            "consistent"
        } else {
            "DIVERGED"
        }
    );
    if let Some(dir) = &args.out_dir {
        println!();
        println!("exported            {:>13} rows to {dir}", s.exported);
    }
    for symbol in diverged.iter().take(10) {
        println!("    diverged: {symbol}");
    }

    let mut sample: Vec<(&str, &lob::Book)> = books.iter().collect();
    sample.sort_by_key(|(s, _)| *s);
    if !sample.is_empty() {
        println!();
        println!(
            "{:<10} {:>12} {:>12} {:>10} {:>10}",
            "symbol", "bid", "ask", "bid sz", "ask sz"
        );
        for (symbol, book) in sample.iter().take(8) {
            let (bid_sz, ask_sz) = book.touch_depth();
            println!(
                "{:<10} {:>12} {:>12} {:>10} {:>10}",
                symbol,
                book.best_bid().map_or("-".to_string(), |(p, _)| fmt_px(p)),
                book.best_ask().map_or("-".to_string(), |(p, _)| fmt_px(p)),
                bid_sz,
                ask_sz
            );
        }
    }
}

/// Nanoseconds since midnight as HH:MM:SS.
fn fmt_ns(ns: u64) -> String {
    if ns == u64::MAX {
        return "-".into();
    }
    let s = ns / 1_000_000_000;
    format!("{:02}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60)
}

/// ITCH fixed-point ticks as dollars.
fn fmt_px(p: u32) -> String {
    format!("{:.4}", p as f64 / 10_000.0)
}

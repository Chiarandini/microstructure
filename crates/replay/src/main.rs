//! Replay an ITCH day: reconstruct books, verify invariants, report.
//!
//! Two modes. With no `--symbols`, it validates the whole file and prints a
//! reconstruction report, which is how the correctness claims in the README
//! are checked. With `--symbols`, it additionally tracks those symbols'
//! books in full so features can be derived from them.

use clap::Parser;
use itch::{Body, symbol_str};
use lob::Book;
use std::collections::HashMap;
use std::time::Instant;

#[derive(Parser)]
#[command(
    about = "Reconstruct limit order books from a Nasdaq ITCH 5.0 file and verify them"
)]
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
    /// Expensive; 0 disables it and only the end-of-run check happens.
    #[arg(long, default_value_t = 0)]
    reconcile_every: u64,
}

/// Nanoseconds since midnight for the continuous session, 09:30 to 16:00 ET.
const MARKET_OPEN_NS: u64 = 9 * 3_600_000_000_000 + 30 * 60_000_000_000;
const MARKET_CLOSE_NS: u64 = 16 * 3_600_000_000_000;

#[derive(Default)]
struct Counts {
    total: u64,
    add: u64,
    executed: u64,
    canceled: u64,
    deleted: u64,
    replaced: u64,
    hidden_trades: u64,
    crosses: u64,
    other: u64,
}

fn main() {
    let args = Args::parse();

    let tracked: Option<Vec<String>> = args
        .symbols
        .as_ref()
        .map(|v| v.iter().map(|s| s.trim().to_uppercase()).collect());

    let mut reader = match itch::reader::open(&args.path) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("cannot open {}: {e}", args.path);
            std::process::exit(1);
        }
    };

    // stock_locate -> symbol, populated from the directory messages the file
    // opens with. Order messages carry only the locate id.
    let mut locate_symbol: HashMap<u16, String> = HashMap::new();

    // One book per tracked symbol, plus a router from order id to book, so
    // that messages for untracked symbols can be skipped without decoding
    // their state.
    let mut books: Vec<Book> = Vec::new();
    let mut book_of_symbol: HashMap<String, usize> = HashMap::new();
    let mut route: HashMap<u64, usize> = HashMap::new();

    let mut counts = Counts::default();
    let mut first_ts = u64::MAX;
    let mut last_ts = 0u64;
    let mut checked = 0u64;
    let mut cross_violations = 0u64;

    let start = Instant::now();

    loop {
        let msg = match reader.next_message() {
            Ok(Some(m)) => m,
            Ok(None) => break,
            Err(e) => {
                eprintln!("\nfatal at message {}: {e}", counts.total);
                std::process::exit(1);
            }
        };
        let (header, body) = msg;
        counts.total += 1;

        if header.timestamp > 0 {
            first_ts = first_ts.min(header.timestamp);
            last_ts = last_ts.max(header.timestamp);
        }

        // The directory block precedes trading, so this map is complete
        // before any order message arrives.
        if let Body::StockDirectory { stock, .. } = body {
            let name = symbol_str(&stock).to_string();
            let wanted = tracked.as_ref().is_none_or(|t| t.contains(&name));
            if wanted {
                let idx = books.len();
                books.push(Book::new());
                book_of_symbol.insert(name.clone(), idx);
            }
            locate_symbol.insert(header.stock_locate, name);
            continue;
        }

        // Route the message to a book. Adds carry the symbol and establish
        // the routing; everything else looks the order id up.
        let book_idx = match body {
            Body::AddOrder { order_ref, stock, .. } => {
                let name = symbol_str(&stock);
                match book_of_symbol.get(name) {
                    Some(&i) => {
                        route.insert(order_ref, i);
                        Some(i)
                    }
                    None => None,
                }
            }
            Body::OrderExecuted { order_ref, .. }
            | Body::OrderExecutedWithPrice { order_ref, .. }
            | Body::OrderCancel { order_ref, .. } => route.get(&order_ref).copied(),
            Body::OrderDelete { order_ref } => {
                let i = route.remove(&order_ref);
                i
            }
            Body::OrderReplace { original_order_ref, new_order_ref, .. } => {
                match route.remove(&original_order_ref) {
                    Some(i) => {
                        route.insert(new_order_ref, i);
                        Some(i)
                    }
                    None => None,
                }
            }
            Body::TradeNonCross { stock, .. } | Body::CrossTrade { stock, .. } => {
                book_of_symbol.get(symbol_str(&stock)).copied()
            }
            Body::TradingAction { stock, .. } => book_of_symbol.get(symbol_str(&stock)).copied(),
            _ => None,
        };

        match body {
            Body::AddOrder { .. } => counts.add += 1,
            Body::OrderExecuted { .. } | Body::OrderExecutedWithPrice { .. } => {
                counts.executed += 1
            }
            Body::OrderCancel { .. } => counts.canceled += 1,
            Body::OrderDelete { .. } => counts.deleted += 1,
            Body::OrderReplace { .. } => counts.replaced += 1,
            Body::TradeNonCross { .. } => counts.hidden_trades += 1,
            Body::CrossTrade { .. } => counts.crosses += 1,
            _ => counts.other += 1,
        }

        if let Some(i) = book_idx {
            books[i].apply(&body);

            // Only assert the crossed-book invariant during the continuous
            // session. Pre-open and post-close books legitimately cross while
            // auction interest accumulates.
            let in_session =
                header.timestamp >= MARKET_OPEN_NS && header.timestamp <= MARKET_CLOSE_NS;
            if in_session && matches!(body, Body::AddOrder { .. } | Body::OrderReplace { .. }) {
                checked += 1;
                if !books[i].check() {
                    cross_violations += 1;
                }
            }
        }

        if args.reconcile_every > 0 && counts.total % args.reconcile_every == 0 {
            for (name, &i) in &book_of_symbol {
                assert!(
                    books[i].depth_matches_orders(),
                    "depth and order map diverged for {name} at message {}",
                    counts.total
                );
            }
        }

        if args.limit.is_some_and(|l| counts.total >= l) {
            break;
        }
    }

    let elapsed = start.elapsed();
    report(
        &args, &counts, &books, &book_of_symbol, &locate_symbol, first_ts, last_ts, checked,
        cross_violations, elapsed,
    );
}

#[allow(clippy::too_many_arguments)]
fn report(
    args: &Args,
    counts: &Counts,
    books: &[Book],
    book_of_symbol: &HashMap<String, usize>,
    locate_symbol: &HashMap<u16, String>,
    first_ts: u64,
    last_ts: u64,
    checked: u64,
    cross_violations: u64,
    elapsed: std::time::Duration,
) {
    let secs = elapsed.as_secs_f64();
    println!("file                {}", args.path);
    println!("symbols in file     {}", locate_symbol.len());
    println!("books tracked       {}", books.len());
    println!(
        "session             {} to {}",
        fmt_ns(first_ts),
        fmt_ns(last_ts)
    );
    println!();
    println!("messages            {:>13}", counts.total);
    println!("  add               {:>13}", counts.add);
    println!("  executed          {:>13}", counts.executed);
    println!("  cancel            {:>13}", counts.canceled);
    println!("  delete            {:>13}", counts.deleted);
    println!("  replace           {:>13}", counts.replaced);
    println!("  hidden trade      {:>13}", counts.hidden_trades);
    println!("  cross             {:>13}", counts.crosses);
    println!("  other             {:>13}", counts.other);
    println!();
    println!(
        "elapsed             {:.2} s  ({:.2} M msg/s)",
        secs,
        counts.total as f64 / secs / 1e6
    );
    println!();

    // Aggregate anomalies across every reconstructed book.
    let mut unknown = 0u64;
    let mut oversized = 0u64;
    let mut crossed = 0u64;
    let mut live = 0usize;
    for b in books {
        unknown += b.anomalies.unknown_order_ref;
        oversized += b.anomalies.oversized_removal;
        crossed += b.anomalies.crossed_book;
        live += b.live_orders();
    }

    println!("reconstruction");
    println!("  crossed-book checks {:>11}", checked);
    println!("  crossed-book fails  {:>11}", cross_violations);
    println!("  unknown order refs  {:>11}", unknown);
    println!("  oversized removals  {:>11}", oversized);
    println!("  crossed (recorded)  {:>11}", crossed);
    println!("  orders still live   {:>11}", live);

    let mut reconciled = true;
    for (name, &i) in book_of_symbol {
        if !books[i].depth_matches_orders() {
            println!("  RECONCILE FAILED for {name}");
            reconciled = false;
        }
    }
    println!(
        "  depth vs order map  {:>11}",
        if reconciled { "consistent" } else { "DIVERGED" }
    );

    // A sample of end-of-day books, as a smell test that the numbers are
    // plausible rather than structurally empty.
    let mut named: Vec<(&String, usize)> =
        book_of_symbol.iter().map(|(n, &i)| (n, i)).collect();
    named.sort_by(|a, b| a.0.cmp(b.0));
    let shown: Vec<(&String, usize)> = named.into_iter().take(8).collect();
    if !shown.is_empty() {
        println!();
        println!("{:<10} {:>12} {:>12} {:>10} {:>10}", "symbol", "bid", "ask", "bid sz", "ask sz");
        for (name, i) in shown {
            let b = &books[i];
            let (bs, as_) = b.touch_depth();
            println!(
                "{:<10} {:>12} {:>12} {:>10} {:>10}",
                name,
                b.best_bid().map_or("-".to_string(), |(p, _)| fmt_px(p)),
                b.best_ask().map_or("-".to_string(), |(p, _)| fmt_px(p)),
                bs,
                as_
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

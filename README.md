# microstructure

Limit order book reconstruction from raw Nasdaq ITCH 5.0, and a statistical
study of short-horizon predictability built on top of it.

Rust does the data path, Python does the statistics. The research plan, the
evaluation protocol, and the things this project deliberately refuses to do
are in [`DESIGN.md`](DESIGN.md).

## Status

| Phase | State |
|---|---|
| 0. Data fetch and skeleton | done |
| 1. ITCH 5.0 decoder | done |
| 2. Book reconstruction, validated | done |
| 3. Feature extraction and export | next |
| 4. OFI predictability study | not started |
| 5. Extension: cross-impact or a fitted queue-reactive model | not started |
| 6. Writeup | not started |

Nothing below is a research finding yet. What follows is an engineering
result: the book the study will rest on has been checked rather than assumed.

## Reconstruction result

Full Nasdaq BX session, 2019-07-30, all 8,849 symbols reconstructed
simultaneously.

```
messages                 28,734,686
  add                    10,629,593
  executed                  681,693
  cancel                      299,199
  delete                 10,164,658
  replace                 2,046,443
  hidden trade              243,778

elapsed                        4.82 s   (5.96 M msg/s)

crossed-book checks      12,435,862
crossed-book fails                0
unknown order refs                0
oversized removals                0
orders still live                 0
depth vs order map       consistent
```

The four zeros are the point.

- **No crossed books.** Across 12.4 million checks during the continuous
  session, best bid never met or crossed best ask. Checked only between 09:30
  and 16:00, because a crossed book while halted or accumulating auction
  interest is legitimate and flagging it would be a false positive.
- **No unknown order references.** Every execution, cancel, delete, and
  replace referred to an order the book already knew about. A non-zero count
  here means messages are being dropped or misrouted.
- **No oversized removals.** No message ever tried to take more shares off an
  order than it held.
- **No orders left live at end of day.** Every one of the 10.6 million orders
  opened during the session was accounted for and removed. This is the
  strongest of the four: it is a conservation law over the whole day, and
  almost any bookkeeping error would break it.

Separately, the level aggregates and the order map are maintained
independently and reconciled at the end: recomputing per-price depth from the
order map reproduces the incrementally maintained levels exactly.

Two reconstruction traps that these checks exist to catch, both handled:

- **Hidden trades (`P`) must not touch the visible book.** Those shares were
  never displayed. Applying them double-counts depth consumption and silently
  inflates every order-flow feature.
- **Execute-with-price (`C`) removes depth at the order's resting price**, not
  at the price that printed. Removing at the execution price corrupts a level
  the order was never on.

## Throughput

5.96 M messages/second, single-threaded, including gzip decompression and
maintaining 8,849 live books. A full BX day replays in under five seconds,
which is the property that matters: re-running the entire study after a bug
fix is a coffee-free operation rather than an overnight job.

The decoder borrows from the read buffer and allocates nothing per message.
Prices stay as integer ticks of $0.0001 end to end; converting to floating
point happens once, in the analysis layer.

## Failing loudly

The reader treats an unknown message type or a length prefix disagreeing with
the specification as fatal. Both mean the stream position is wrong, so
everything after would be plausible-looking garbage.

This earned its keep on the first real run: the file stopped at message 58,160
on message type `N`, which the decoder did not yet know. A parser that skipped
unknown types would have silently desynchronised and produced a book that
looked fine and was wrong. `N` is the Retail Price Improvement Indicator; it
is now handled.

## Layout

```
crates/itch      ITCH 5.0 decoder and streaming reader
crates/lob       order book state machine
crates/replay    binary: drives a day, verifies, reports
scripts/fetch.sh downloads a session from Nasdaq
py/              analysis (phase 4 onward)
data/            gitignored; fetched, never committed
```

## Running it

```sh
cargo build --release
cargo test --release          # 27 tests

./scripts/fetch.sh list                 # what Nasdaq publishes
./scripts/fetch.sh bx 20190730          # 391 MB, good for development
./scripts/fetch.sh nasdaq 20190730      # 3.7 GB, the real thing

./target/release/replay data/raw/20190730.BX_ITCH_50.gz
./target/release/replay data/raw/20190730.BX_ITCH_50.gz --symbols AAPL,MSFT
```

`--reconcile-every N` runs the full depth-versus-order-map reconciliation
every N messages instead of only at the end. Slow, but it turns a silent
divergence into an immediate assertion at the message that caused it.

## Data

Nasdaq publishes historical ITCH 5.0 sample days openly at
<https://emi.nasdaq.com/ITCH/>, no account required: seven dates across 2019
and January 2020 for both Nasdaq and Nasdaq BX.

Seven distinct days matters for the study. Out-of-sample will mean a
different day, not a later slice of the same one; intraday autocorrelation
and a shared regime make same-day holdout optimistic by construction.

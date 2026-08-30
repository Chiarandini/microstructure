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

Full Nasdaq session, 2019-07-30, all 8,849 symbols reconstructed
simultaneously. The BX session for the same date is shown alongside it.

```
                          Nasdaq          BX
messages             282,229,684   28,734,686
  add                125,460,750   10,629,593
  executed             7,717,995      681,693
  cancel               2,358,032      299,199
  delete             119,999,061   10,164,658
  replace             21,253,951    2,046,443
  hidden trade         1,461,010      243,778
  cross                   17,700            0

elapsed                  50.46 s       3.73 s
throughput          5.59 M msg/s  7.70 M msg/s

crossed-book checks  141,170,621   12,435,862
crossed-book fails             0            0
unknown order refs             0            0
oversized removals             0            0
duplicate order ids            0            0
level inconsistent             0            0
orders still live              0            0
depth vs order map    consistent   consistent
```

The four zeros are the point.

- **No crossed books.** Across 141 million checks during the continuous
  session, best bid never met or crossed best ask. Checked only between 09:30
  and 16:00, because a crossed book while halted or accumulating auction
  interest is legitimate and flagging it would be a false positive.
- **No unknown order references.** Every one of the 151 million executions,
  cancels, deletes, and replaces referred to an order the book already knew
  about. A non-zero count here means messages are being dropped or misrouted.
- **No oversized removals.** No message ever tried to take more shares off an
  order than it held.
- **No orders left live at end of day.** Every one of the 125.5 million orders
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

5.6 M messages/second single-threaded on the full Nasdaq feed, including gzip
decompression and maintaining 8,849 live books. A whole trading day replays in
50 seconds, which is the property that matters: re-running the entire study
after a bug fix is a coffee-length operation rather than an overnight job.

The decoder borrows from the read buffer and allocates nothing per message.
Prices stay as integer ticks of $0.0001 end to end; converting to floating
point happens once, in the analysis layer.

Routing is a direct index by the header's `stock_locate`. Every ITCH message
carries one, including the order messages that identify their order only by
id, so no order-id to book map is needed and no hashing happens per message.

Decompression is about a quarter of wall time (`gunzip` alone on the BX
session is 0.98 s against a 3.73 s replay), so a faster inflate backend is
worth perhaps 15%. Not currently a priority.

## Failing loudly

The reader treats an unknown message type, or a length prefix disagreeing with
the specification, as fatal. Both mean the stream position is wrong, so
everything decoded after that point would be plausible-looking garbage.

The alternative, skipping unrecognised messages, is worse than useless here: a
book built from a desynchronised stream still looks like a book. It has a bid,
an ask, and a spread; it is simply wrong, and nothing downstream would notice.
Refusing to guess is what makes the invariant results above mean anything.

## Layout

```
crates/itch      ITCH 5.0 decoder and streaming reader
crates/lob       order book state machine, and BookSet session routing
crates/replay    binary: drives a session, verifies, reports
scripts/fetch.sh downloads a session from Nasdaq
data/            gitignored; fetched, never committed
```

## Running it

```sh
cargo build --release
cargo test --release          # 36 tests

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

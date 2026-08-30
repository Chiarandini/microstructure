# Design and research plan

## What this is

A limit order book research stack: a Rust engine that reconstructs the full
book from raw Nasdaq ITCH 5.0 message data, and a statistical study built on
top of it asking a narrow, honest question about short-horizon predictability.

The point is not to produce a trading strategy. The point is to do a piece of
empirical microstructure research correctly: real data, a reconstruction that
is validated rather than assumed, features with an economic rationale, and an
evaluation protocol chosen before the results are seen.

## Why this shape

Two halves, deliberately split by language.

- **Rust for the data path.** A single Nasdaq trading day is roughly 4 to 5 GB
  gzipped and on the order of 300 million messages. Book reconstruction is a
  sequential state machine over that stream: it cannot be vectorised away and
  it has to be fast enough that re-running the whole study after a bug fix is
  cheap rather than an overnight job. This is the same class of problem as
  `collatz-chains` and it reuses the same instincts.
- **Python for the statistics.** Regression, cross-validation, bootstrap, and
  plotting belong where the ecosystem is. The Rust side's job is to emit a
  clean tabular artifact; the Python side never parses a binary protocol.

The interface between them is a columnar file per (symbol, day), so the
expensive parse happens once and the statistical work iterates freely.

## Data

Nasdaq publishes historical ITCH 5.0 sample days publicly at
`https://emi.nasdaq.com/ITCH/`. No account, no credentials.

| Venue | Days available | Size (gz) |
|---|---|---|
| Nasdaq | 2019-01-30, 03-27, 07-30, 08-30, 10-30, 12-30, 2020-01-30 | 3.5 to 5.6 GB |
| Nasdaq BX | same seven dates plus 2018-12-28, 2020-01-30 | 0.4 to 1.7 GB |

Two things this buys us:

1. **Seven distinct days** spread across a year, so out-of-sample can mean
   "a different day" rather than "a later slice of the same day". Same-day
   holdout in microstructure is nearly worthless: intraday autocorrelation
   and a shared regime make it optimistic by construction.
2. **Two venues on the same dates**, which later permits a cross-venue check:
   a signal that only exists on one venue is more likely an artifact of that
   venue's participants than a property of the asset.

Development runs on the smallest BX day; the study runs on full Nasdaq days.
`data/` is gitignored. Nothing in the repo depends on data that a reader
cannot fetch themselves, and the fetch is scripted.

## Architecture

Built:

```
crates/itch      ITCH 5.0 decoder: bytes -> typed messages, no allocation
                 per message; streaming reader over gzipped BinaryFILE
crates/lob       order book state machine, plus BookSet, which routes a
                 whole session into per-symbol books
crates/replay    the binary: drives a session, verifies invariants, reports
scripts/fetch.sh session download
```

Planned, and not yet written:

```
crates/export    book + event stream -> per-event rows on disk (phase 3)
py/              analysis: regressions, evaluation protocol, figures (phase 4)
```

### `itch`

ITCH 5.0 is a length-prefixed binary protocol with about twenty message types.
Fields are big-endian, prices are fixed-point with four implied decimals, and
timestamps are nanoseconds since midnight packed into six bytes.

Decoding borrows from the input buffer rather than allocating per message. The
decoder is a pure function from a byte slice to a typed enum, which makes it
trivially testable and keeps I/O out of the hot path.

Only a subset of message types affects the book: add order (with and without
MPID), executed, executed with price, cancel, delete, replace. Trades against
hidden liquidity are reported separately and must not be applied to the
visible book, which is a standard source of reconstruction error.

### `lob`

Maintains order-id to (side, price, size) plus per-price-level aggregate
depth. The operations that matter are add, partial cancel, delete, and
replace, where replace is a delete followed by an add with a new order id and
loses queue priority.

`BookSet` routes a session into per-symbol books. Routing is a direct index by
the header's `stock_locate`, which every message carries, including the order
messages that name their order only by id. An order-id to book map is
therefore unnecessary.

**Correctness is the crux of the whole project.** A book reconstruction that
is subtly wrong produces features that are subtly wrong and a result that is
confidently false.

Implemented, and run over full sessions:

1. **Crossed-book invariant.** Best bid must stay below best ask during the
   continuous session. Checked on every quote-moving message; suppressed
   during halts and auctions, where a crossed book is legitimate.
2. **Conservation.** Every executed and cancelled quantity must be traceable
   to a live order of at least that size, and the session must end with no
   unexplained live orders.
3. **Independent reconciliation.** Level aggregates and the order map are
   maintained by separate code paths; recomputing depth from the order map
   must reproduce the incrementally maintained levels exactly.

Not implemented: cross-venue sanity, meaning reconstructed trade prints
aggregated to the minute compared against the venue's published volume for
that symbol and day. This would catch a class of error the three checks above
cannot, since all three are internal consistency checks and would all pass on
a book that is self-consistent but systematically missing flow. Worth adding
before any result is published.

There is no CI. The checks run when `replay` is run.

### `export` (planned, phase 3)

**Rust emits a per-event log; Python computes every feature.** The Rust side
writes one row per book event, carrying the pre-event and post-event top of
book. Nothing is aggregated, smoothed, or sampled on the way out.

This split is deliberate. Feature definitions will change many times during
the study, and each change should cost a Python re-run over an existing file
rather than a full re-parse of a multi-gigabyte session. It also means the
same artifact serves both the regression and the point-process fit, which want
very different views of the same events.

Row schema, one per event:

```
ts_ns, event, side, price, shares,
old_price, old_shares,                            withdrawn leg of a replace
bid_px_before, ask_px_before, bid_sz_before, ask_sz_before,
bid_px_after,  ask_px_after,  bid_sz_after,  ask_sz_after
```

A replace is labelled `replace`, not `add`. It is a cancellation plus a
resubmission that loses queue priority, so where both legs rest at the same
price its net depth effect is `new - old`. The withdrawn leg is carried so a
consumer can decompose it without replaying the book.

Carrying both sides of the event is what makes order flow imbalance
computable without replaying the book in Python, and what gives the
queue-reactive fit its (queue state -> transition) pairs directly.

Retaining nanosecond timestamps and event types unaggregated is what the
phase-5 model needs: a Hawkes fit is estimation on event times, and
discretising them at export would destroy the object being estimated.

Features to be computed downstream, all functions of information strictly
before the timestamp they are stamped with:

- **Order flow imbalance (OFI)**, in the Cont-Kukanov-Stoikov sense: the
  signed change in depth at the best quotes, which is the quantity that maps
  linearly to price change, rather than raw trade imbalance.
- **Queue imbalance**: `(bid_size - ask_size) / (bid_size + ask_size)`.
- **Trade sign imbalance** over a trailing event window.
- **Realised and effective spread** at several horizons.

Sampling is on an event clock, not a wall clock: calendar time oversamples
quiet periods and undersamples exactly the moments where anything happens.

Format is gzipped CSV rather than Parquet. Per symbol-day the row count is in
the millions, not the hundreds of millions, so CSV costs disk and some read
time but no dependency and no schema tooling. Revisit if the study grows to
many symbols across all seven days at once.

## The research question

> Over the next `k` events, does order flow imbalance predict the change in
> mid-price, and does that predictability survive an honest accounting of
> transaction costs and multiple testing?

This is a known result: OFI is one of the most robust short-horizon
predictors in the literature. Reproducing a known result carefully is the
right first target, because it means a null result indicates a bug in my
pipeline rather than an interesting discovery. Once the reproduction holds,
the extensions below are where the actual research sits.

### Protocol, fixed in advance

- **Split by day, never within a day.** Train on the earlier days, test on the
  later ones, in chronological order. No shuffling.
- **Report the decay curve**, not a single horizon. The interesting object is
  how R-squared falls off as `k` grows, and where it crosses zero net of the
  spread. A single number at a single horizon invites cherry-picking.
- **Cost-aware from the start.** A signal that predicts a mid-price move
  smaller than half the spread is not tradeable. Every result is reported both
  gross and net of half-spread.
- **Count the hypotheses.** Every feature, horizon, and symbol combination
  tried gets logged to `py/experiments.jsonl` as it is run. The final report
  applies a deflated Sharpe / Benjamini-Hochberg style correction against that
  logged count, not against a count reconstructed after the fact.
- **Stationary bootstrap for error bars**, since observations are serially
  dependent and i.i.d. bootstrap would understate the standard errors badly.

### Extensions, in priority order

1. **Cross-impact.** Does OFI in one symbol predict returns in a correlated
   symbol? Requires handling asynchronous event clocks across symbols, which
   is the interesting technical difficulty.
2. **A fitted model rather than a regression.** Either a queue-reactive model
   (Huang-Lehalle-Rosenbaum), where order arrival intensities depend on the
   current queue state, or a multivariate Hawkes process for the self- and
   cross-excitation of order flow. Then simulate from the fitted model and ask
   whether it reproduces the empirical stylised facts it was not fit to. This
   is the part that demonstrates something beyond running a regression, and it
   is the natural home for measure-theoretic probability.
3. **Regime dependence.** Seven days across a year, including one just before
   the 2020 volatility spike, is enough to ask whether the relationship is
   stable across regimes.

## Phases

| Phase | Deliverable | Done when |
|---|---|---|
| 0 | Data fetch script, project skeleton | `just fetch` retrieves a day and verifies its checksum |
| 1 | `itch` decoder | Round-trips every message type; parses a full day without error |
| 2 | `lob` reconstruction | All three invariants hold across a full day |
| 3 | `features` + Parquet export | One file per symbol-day, look-ahead test passes |
| 4 | OFI reproduction | Decay curve, out-of-sample by day, with bootstrap error bars |
| 5 | Extension 1 or 2 | A result that is not just a reproduction |
| 6 | Writeup | README carries the finding, its error bars, and its limits |

## Non-goals

- Not a backtester. No fills, no queue simulation, no PnL curve. Those invite
  exactly the overfitting the protocol above is designed to prevent, and a
  fake PnL curve is worse than no PnL curve.
- Not a trading strategy, and the writeup will say so explicitly.
- Not multi-venue consolidated. Single venue at a time, by choice: a
  consolidated book from public data is its own large project and the errors
  it introduces would contaminate everything downstream.

"""Compare the same symbol and day reconstructed from two venues.

Every other check in this project is internal: it asks whether the book agrees
with itself. All of them would pass on a book that is self-consistent but
systematically missing flow, or that misreads prices in a consistent way.

Nasdaq and Nasdaq BX publish independent message streams for the same session,
so reconstructing a symbol from each gives two independent views of one
underlying price.

The comparison is on **trade prints**, not on midpoints. BX is a thin venue:
its median AAPL spread on 2019-07-30 is $0.07 against Nasdaq's $0.01, reaching
$0.39 at the 95th percentile. A midpoint inside a spread that wide is a noisy
estimate of the true price, so comparing mids measures how the two venues
quote rather than whether either book is right. Executions do not have that
problem: Reg NMS trade-through protection pins prints on both venues to the
same national best bid and offer, whatever each venue's own quote looks like.

Its limit, stated plainly: both sides run the same decoder and book, so a bug
affecting both venues identically would not show up here. This catches
data-dependent errors, not shared logic errors.

    python3 py/cross_venue_check.py AAPL MSFT INTC
"""

import sys
from pathlib import Path

import pandas as pd

TICK = 10_000
EVENTS = Path("data/events")
EVENTS_BX = Path("data/events_bx")
SESSION = "20190730"
GRID_NS = 1_000_000_000


def load(path):
    return pd.read_csv(
        path,
        usecols=["ts_ns", "event", "price", "bid_px_before", "ask_px_before"],
        dtype={"ts_ns": "int64", "price": "int64"},
    )


def mid_grid(df):
    """Last-known mid per second, in dollars."""
    d = df[df.bid_px_before.notna() & df.ask_px_before.notna()]
    mid = (d.bid_px_before + d.ask_px_before) / 2 / TICK
    return pd.Series(mid.to_numpy(), index=(d.ts_ns // GRID_NS).to_numpy()).groupby(level=0).last()


def quote_grid(df, column):
    d = df[df[column].notna()]
    return (
        pd.Series((d[column] / TICK).to_numpy(), index=(d.ts_ns // GRID_NS).to_numpy())
        .groupby(level=0)
        .last()
    )


def check(symbol):
    a = EVENTS / f"{SESSION}.NASDAQ_ITCH50_{symbol}.csv.gz"
    b = EVENTS_BX / f"{SESSION}.BX_ITCH_50_{symbol}.csv.gz"
    for p in (a, b):
        if not p.exists():
            print(f"missing {p}")
            return None

    nasdaq, bx = load(a), load(b)
    nas_mid = mid_grid(nasdaq)

    # Every BX print, priced against the Nasdaq mid for that second. If BX
    # prices were being misparsed, or its book were built from the wrong
    # messages, these would not line up.
    bx_trades = bx[bx.event.isin(["trade", "hidden"])].copy()
    bx_trades["sec"] = bx_trades.ts_ns // GRID_NS
    bx_trades["ref"] = bx_trades.sec.map(nas_mid)
    matched = bx_trades.dropna(subset=["ref"])
    err = (matched.price / TICK - matched.ref).abs() * 100  # cents

    # The same statistic for Nasdaq's own prints against its own mid, as a
    # scale reference: a print is normally about half a spread from the mid,
    # so this is the floor the cross-venue number should be compared against.
    nas_trades = nasdaq[nasdaq.event.isin(["trade", "hidden"])].copy()
    nas_trades["sec"] = nas_trades.ts_ns // GRID_NS
    nas_trades["ref"] = nas_trades.sec.map(nas_mid)
    nas_matched = nas_trades.dropna(subset=["ref"])
    nas_err = (nas_matched.price / TICK - nas_matched.ref).abs() * 100

    # Two venues cannot stay crossed against each other: a BX bid above the
    # Nasdaq ask is a standing arbitrage. Brief crossings are real, a
    # persistent one means a book is wrong.
    bx_bid, bx_ask = quote_grid(bx, "bid_px_before"), quote_grid(bx, "ask_px_before")
    nas_bid, nas_ask = quote_grid(nasdaq, "bid_px_before"), quote_grid(nasdaq, "ask_px_before")
    q = pd.concat(
        [
            bx_bid.rename("bx_bid"),
            bx_ask.rename("bx_ask"),
            nas_bid.rename("nas_bid"),
            nas_ask.rename("nas_ask"),
        ],
        axis=1,
    ).dropna()
    crossed = ((q.bx_bid > q.nas_ask) | (q.nas_bid > q.bx_ask)).mean() * 100

    print(f"{symbol}")
    print(f"  BX prints matched to a Nasdaq quote   {len(matched):,}")
    print(f"  |BX print - Nasdaq mid|, cents")
    print(f"    median {err.median():.2f}   p95 {err.quantile(0.95):.2f}   "
          f"<=5c {100 * (err <= 5).mean():.2f}%")
    print(f"  |Nasdaq print - Nasdaq mid|, cents  (same-venue reference)")
    print(f"    median {nas_err.median():.2f}   p95 {nas_err.quantile(0.95):.2f}   "
          f"<=5c {100 * (nas_err <= 5).mean():.2f}%")
    print(f"  seconds with the venues crossed       {crossed:.3f}%")

    # A print landing within a few cents of the other venue's mid, essentially
    # as often as it lands near its own venue's mid, is the signal that both
    # reconstructions describe the same security at the same price.
    ok = err.median() <= 5 and (err <= 5).mean() > 0.95 and crossed < 1.0
    print(f"  verdict: {'consistent' if ok else 'DIVERGENT, investigate'}")
    return ok


def main(argv):
    symbols = argv[1:] or ["AAPL", "MSFT", "INTC"]
    results = []
    for s in symbols:
        r = check(s)
        print()
        results.append(r)
    if all(r for r in results if r is not None):
        print("cross-venue: consistent")
        return 0
    print("cross-venue: DIVERGENT")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))

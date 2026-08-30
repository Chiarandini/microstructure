"""Sanity-check an exported event file before any analysis is built on it.

The Rust side proves internal consistency of the book. This proves the
*exported* view is faithful and usable: that the file says what the book
meant, and that the look-ahead invariant survived the trip to disk.

Run before trusting a session:

    python3 py/validate_export.py data/events/20190730.NASDAQ_ITCH50_AAPL.csv.gz
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ITCH prices are fixed-point with four implied decimals.
TICK = 10_000
SESSION_OPEN_NS = int(9.5 * 3600 * 1e9)
SESSION_CLOSE_NS = int(16 * 3600 * 1e9)


def load(path):
    df = pd.read_csv(
        path,
        dtype={
            "ts_ns": "int64",
            "event": "category",
            "side": "category",
            "price": "int64",
            "shares": "int64",
            "old_price": "float64",
            "old_shares": "float64",
            "bid_px_before": "float64",
            "ask_px_before": "float64",
            "bid_sz_before": "int64",
            "ask_sz_before": "int64",
            "bid_px_after": "float64",
            "ask_px_after": "float64",
            "bid_sz_after": "int64",
            "ask_sz_after": "int64",
        },
    )
    return df


class Checks:
    def __init__(self):
        self.failures = 0

    def check(self, name, ok, detail=""):
        status = "ok  " if ok else "FAIL"
        if not ok:
            self.failures += 1
        print(f"  [{status}] {name}{'  ' + detail if detail else ''}")


def validate(df, path):
    print(f"{path}")
    print(f"  {len(df):,} rows")
    c = Checks()

    c.check("timestamps non-decreasing", bool(df.ts_ns.is_monotonic_increasing))

    in_session = (df.ts_ns >= SESSION_OPEN_NS) & (df.ts_ns <= SESSION_CLOSE_NS)
    c.check(
        "all rows within the continuous session",
        bool(in_session.all()),
        f"{(~in_session).sum():,} outside",
    )

    # A two-sided book must never be crossed. This is the same invariant the
    # Rust asserts, re-checked on the artifact rather than on the live book,
    # so a bug in the writer cannot hide behind a correct reconstruction.
    both = df.bid_px_before.notna() & df.ask_px_before.notna()
    crossed = both & (df.bid_px_before >= df.ask_px_before)
    c.check("no crossed book (before)", not bool(crossed.any()), f"{crossed.sum():,} crossed")

    both_a = df.bid_px_after.notna() & df.ask_px_after.notna()
    crossed_a = both_a & (df.bid_px_after >= df.ask_px_after)
    c.check("no crossed book (after)", not bool(crossed_a.any()), f"{crossed_a.sum():,} crossed")

    # Crosses are excluded: the venue emits a cross message per symbol even
    # when no cross occurred, and those legitimately carry price 0 shares 0.
    real = df.event != "cross"
    c.check("shares positive", bool((df.loc[real, "shares"] > 0).all()))
    c.check("prices positive", bool((df.loc[real, "price"] > 0).all()))

    # A price with no size, or size with no price, means the two halves of
    # the snapshot disagree.
    bad_bid = (df.bid_px_before.notna() & (df.bid_sz_before == 0)) | (
        df.bid_px_before.isna() & (df.bid_sz_before != 0)
    )
    bad_ask = (df.ask_px_before.notna() & (df.ask_sz_before == 0)) | (
        df.ask_px_before.isna() & (df.ask_sz_before != 0)
    )
    c.check("price and size agree on emptiness", not bool((bad_bid | bad_ask).any()))

    # The look-ahead invariant, restated on the artifact: an add that lands
    # exactly at the prevailing best bid must increase bid size by its own
    # shares, and must not already be included in the `before` snapshot.
    at_bid = (
        (df.event == "add")
        & (df.side == "B")
        & df.bid_px_before.notna()
        & (df.price == df.bid_px_before)
    )
    if at_bid.any():
        sub = df[at_bid]
        delta = sub.bid_sz_after - sub.bid_sz_before
        c.check(
            "adds at the touch move size by exactly their own shares",
            bool((delta == sub.shares).all()),
            f"{(delta != sub.shares).sum():,} of {at_bid.sum():,} mismatched",
        )

    # A replace is a cancel plus a resubmission. Its net effect on bid size is
    # the new leg (if it rests at the touch) minus the old leg (if it was
    # resting there). Checking that decomposition is what caught the earlier
    # version of this exporter labelling replaces as plain adds.
    #
    # Restricted to replaces that left the touch price unchanged. When the
    # withdrawn leg was the only order at the best bid, removing it moves the
    # best bid, and `bid_sz_after` then describes a different price level, so
    # the arithmetic below does not apply. That is correct behaviour, not an
    # error, and is reported separately.
    rep = (
        (df.event == "replace")
        & (df.side == "B")
        & df.bid_px_before.notna()
        & (df.bid_px_after == df.bid_px_before)
    )
    if rep.any():
        sub = df[rep]
        added = np.where(sub.price == sub.bid_px_before, sub.shares, 0)
        removed = np.where(sub.old_price == sub.bid_px_before, sub.old_shares, 0)
        delta = (sub.bid_sz_after - sub.bid_sz_before).to_numpy()
        c.check(
            "replaces decompose into their new and old legs",
            bool((delta == added - removed).all()),
            f"{(delta != added - removed).sum():,} of {rep.sum():,} at a fixed touch",
        )

    all_rep = df.event == "replace"
    if all_rep.any():
        sub = df[all_rep]
        c.check(
            "replaces carry a withdrawn leg",
            bool(sub.old_price.notna().all() and sub.old_shares.notna().all()),
        )
        moved = (sub.bid_px_after != sub.bid_px_before) & sub.bid_px_before.notna()
        print(f"    ({moved.sum():,} replaces moved the bid touch)")

    non_replace = df.event != "replace"
    c.check(
        "only replaces carry a withdrawn leg",
        not bool(df.loc[non_replace, "old_price"].notna().any()),
    )

    # Hidden trades are reported by the venue but were never displayed, so
    # they must leave the visible book untouched.
    hidden = df.event == "hidden"
    if hidden.any():
        h = df[hidden]
        unchanged = (
            (h.bid_px_before.fillna(-1) == h.bid_px_after.fillna(-1))
            & (h.ask_px_before.fillna(-1) == h.ask_px_after.fillna(-1))
            & (h.bid_sz_before == h.bid_sz_after)
            & (h.ask_sz_before == h.ask_sz_after)
        )
        c.check(
            "hidden trades leave the book unchanged",
            bool(unchanged.all()),
            f"{(~unchanged).sum():,} of {hidden.sum():,} moved the book",
        )

    return c.failures


def describe(df):
    mid = (df.bid_px_before + df.ask_px_before) / 2 / TICK
    spread = (df.ask_px_before - df.bid_px_before) / TICK
    valid = mid.notna()

    print("  event mix")
    for name, n in df.event.value_counts().items():
        print(f"    {name:<8} {n:>12,}  ({100 * n / len(df):5.2f}%)")

    print("  quotes")
    print(f"    mid       ${mid[valid].min():.2f} to ${mid[valid].max():.2f}")
    print(f"    spread    median ${spread[valid].median():.4f}, "
          f"p99 ${spread[valid].quantile(0.99):.4f}")

    # Event-time spacing, which is what an event-clock study samples on.
    dt = np.diff(df.ts_ns.to_numpy())
    dt = dt[dt > 0]
    if len(dt):
        print(f"    inter-event median {np.median(dt) / 1e3:.1f} us, "
              f"p99 {np.percentile(dt, 99) / 1e3:.1f} us")


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    total_failures = 0
    for path in argv[1:]:
        df = load(path)
        total_failures += validate(df, Path(path).name)
        describe(df)
        print()
    if total_failures:
        print(f"{total_failures} check(s) FAILED")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

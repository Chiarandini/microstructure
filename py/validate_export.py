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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from schema import DTYPES, SESSION_CLOSE_NS, SESSION_OPEN_NS, TICK


def load(path):
    return pd.read_csv(path, dtype=DTYPES)


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

    # The session filter should clip at the boundaries, not merely near them.
    # A file starting minutes late would mean events are being dropped.
    span_start = (df.ts_ns.min() - SESSION_OPEN_NS) / 1e9
    span_end = (SESSION_CLOSE_NS - df.ts_ns.max()) / 1e9
    c.check(
        "coverage starts at the opening bell",
        span_start < 1.0,
        f"first event {span_start:.3f}s after 09:30",
    )
    c.check(
        "coverage runs to the closing bell",
        span_end < 60.0,
        f"last event {span_end:.3f}s before 16:00",
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

    # Every touch-level check below runs on both sides. Checking only the bid
    # would leave a sell-side-only defect invisible, and the two sides are
    # maintained by separate map entries in the book.
    for side, label in (("B", "bid"), ("S", "ask")):
        px_b, px_a = f"{label}_px_before", f"{label}_px_after"
        sz_b, sz_a = f"{label}_sz_before", f"{label}_sz_after"

        # The look-ahead invariant, restated on the artifact: an order that
        # lands exactly at the prevailing touch must move size by its own
        # shares, and must not already be included in the `before` snapshot.
        at_touch = (
            (df.event == "add") & (df.side == side) & df[px_b].notna() & (df.price == df[px_b])
        )
        if at_touch.any():
            sub = df[at_touch]
            delta = sub[sz_a] - sub[sz_b]
            c.check(
                f"adds at the {label} move size by exactly their own shares",
                bool((delta == sub.shares).all()),
                f"{(delta != sub.shares).sum():,} of {at_touch.sum():,} mismatched",
            )

        # A cancel at the touch removes its own shares, unless it emptied the
        # level and moved the price, in which case the two sizes describe
        # different levels.
        cx = (
            (df.event == "cancel")
            & (df.side == side)
            & df[px_b].notna()
            & (df.price == df[px_b])
            & (df[px_a] == df[px_b])
        )
        if cx.any():
            sub = df[cx]
            delta = sub[sz_b] - sub[sz_a]
            c.check(
                f"cancels at the {label} remove exactly their own shares",
                bool((delta == sub.shares).all()),
                f"{(delta != sub.shares).sum():,} of {cx.sum():,} at a fixed touch",
            )

        # A trade consumes resting depth on the side it executed against.
        # Execute-with-price prints away from the resting price, so restrict
        # to trades whose price is the touch.
        tr = (
            (df.event == "trade")
            & (df.side == side)
            & df[px_b].notna()
            & (df.price == df[px_b])
            & (df[px_a] == df[px_b])
        )
        if tr.any():
            sub = df[tr]
            delta = sub[sz_b] - sub[sz_a]
            c.check(
                f"trades at the {label} consume exactly their own shares",
                bool((delta == sub.shares).all()),
                f"{(delta != sub.shares).sum():,} of {tr.sum():,} at a fixed touch",
            )

        # A replace is a cancel plus a resubmission, so its net effect is the
        # new leg (if it rests at the touch) minus the old leg (if it was
        # resting there). Checking the decomposition is what distinguishes a
        # replace from a plain add, whose net effect would be the new leg
        # alone.
        #
        # Restricted to replaces that left the touch price unchanged: when the
        # withdrawn leg was the only order there, removing it moves the touch
        # and the two sizes describe different levels. That is correct
        # behaviour, not an error, and is reported separately below.
        rep = (
            (df.event == "replace")
            & (df.side == side)
            & df[px_b].notna()
            & (df[px_a] == df[px_b])
        )
        if rep.any():
            sub = df[rep]
            added = np.where(sub.price == sub[px_b], sub.shares, 0)
            removed = np.where(sub.old_price == sub[px_b], sub.old_shares, 0)
            delta = (sub[sz_a] - sub[sz_b]).to_numpy()
            c.check(
                f"replaces at the {label} decompose into new and old legs",
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

    non_replace = df.event != "replace"
    c.check(
        "only replaces carry a withdrawn leg",
        not bool(df.loc[non_replace, "old_price"].notna().any()),
    )

    # Execution detail belongs to trades and nothing else, and both columns
    # must appear together: a printable flag without a resting price would
    # leave the aggressor unidentifiable.
    trades = df.event == "trade"
    c.check(
        "only trades carry execution detail",
        not bool(
            df.loc[~trades, "printable"].notna().any()
            or df.loc[~trades, "resting_price"].notna().any()
        ),
    )
    if trades.any():
        t = df[trades]
        c.check(
            "trades carry execution detail",
            bool(t.printable.notna().all() and t.resting_price.notna().all()),
        )
        c.check("printable is 0 or 1", bool(t.printable.isin([0, 1]).all()))

        # Depth leaves at the resting price, so that price must be a real
        # level: at or inside the prevailing quote on the side that was
        # consumed. This is what makes the resting price usable, rather than
        # merely present.
        rest_ok = (
            ((t.side == "B") & (t.resting_price <= t.bid_px_before))
            | ((t.side == "S") & (t.resting_price >= t.ask_px_before))
            | t.bid_px_before.isna()
            | t.ask_px_before.isna()
        )
        c.check(
            "resting price sits on its own side of the book",
            bool(rest_ok.all()),
            f"{int((~rest_ok).sum()):,} of {int(trades.sum()):,} off-side",
        )

        away = t.price != t.resting_price
        print(f"    ({int(away.sum()):,} of {int(trades.sum()):,} executions "
              f"printed away from the resting price; "
              f"{int((t.printable == 0).sum()):,} non-printable)")

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

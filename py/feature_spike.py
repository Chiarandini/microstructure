"""Validate feature construction without evaluating any prediction.

This deliberately stops short of forward returns. The point is to get sign
conventions and distributions right while it is still impossible to tune them
against a result, because once you have seen how a feature predicts, every
later choice about that feature is made by someone who has already looked.
The regression belongs in phase 4, after this is settled.

So the checks below are all internal properties of the features themselves:
identities that must hold arithmetically, ranges that must be respected, and
one sign convention that is testable because getting it backwards makes a
non-negative quantity go negative.

    python3 py/feature_spike.py data/events/*.csv.gz
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F


class Checks:
    def __init__(self):
        self.failures = 0
        self.lines = []

    def check(self, name, ok, detail=""):
        if not ok:
            self.failures += 1
        self.lines.append((name, bool(ok), detail))

    def report(self, indent="  "):
        for name, ok, detail in self.lines:
            status = "ok  " if ok else "FAIL"
            print(f"{indent}[{status}] {name}{'  ' + detail if detail else ''}")


def check_file(df, c):
    """Checks that must hold for any correct construction."""
    ts = F.two_sided(df)

    # The row-local OFI is only valid because each row's `before` is the
    # previous row's `after`. If that chain ever broke, every feature built on
    # it would be quietly wrong.
    breaks = 0
    for col in ("bid_px", "ask_px", "bid_sz", "ask_sz"):
        prev_after = df[f"{col}_after"].shift(1)
        cur_before = df[f"{col}_before"]
        mism = ~((prev_after == cur_before) | (prev_after.isna() & cur_before.isna()))
        mism.iloc[0] = False
        breaks += int(mism.sum())
    c.check("book state chains across rows", breaks == 0, f"{breaks:,} breaks")

    e = F.ofi(df)

    # When neither touch price moves, OFI must reduce exactly to the change in
    # bid depth minus the change in ask depth. This is the case that dominates
    # the data, and it pins the indicator logic.
    flat = ts & (df.bid_px_after == df.bid_px_before) & (df.ask_px_after == df.ask_px_before)
    if flat.any():
        expected = (
            (df.bid_sz_after - df.bid_sz_before) - (df.ask_sz_after - df.ask_sz_before)
        )[flat]
        c.check(
            "OFI reduces to depth change when the touch is fixed",
            bool((e[flat] == expected).all()),
            f"{int((e[flat] != expected).sum()):,} of {int(flat.sum()):,} mismatched",
        )

    # An add resting at the bid, with the touch unchanged, is unambiguously
    # buying pressure of exactly its own size.
    add_bid = (
        ts
        & (df.event == "add")
        & (df.side == "B")
        & (df.price == df.bid_px_before)
        & (df.bid_px_after == df.bid_px_before)
        & (df.ask_px_after == df.ask_px_before)
        & (df.ask_sz_after == df.ask_sz_before)
    )
    if add_bid.any():
        c.check(
            "an add at the bid contributes +shares",
            bool((e[add_bid] == df.shares[add_bid]).all()),
            f"{int((e[add_bid] != df.shares[add_bid]).sum()):,} of {int(add_bid.sum()):,}",
        )

    # And an add at the ask is selling pressure of the same magnitude. If the
    # ask terms carried the wrong sign, this is where it would show.
    add_ask = (
        ts
        & (df.event == "add")
        & (df.side == "S")
        & (df.price == df.ask_px_before)
        & (df.ask_px_after == df.ask_px_before)
        & (df.bid_px_after == df.bid_px_before)
        & (df.bid_sz_after == df.bid_sz_before)
    )
    if add_ask.any():
        c.check(
            "an add at the ask contributes -shares",
            bool((e[add_ask] == -df.shares[add_ask]).all()),
            f"{int((e[add_ask] != -df.shares[add_ask]).sum()):,} of {int(add_ask.sum()):,}",
        )

    # Rows where a side is empty have no defined imbalance and must not leak a
    # nonzero contribution.
    c.check("OFI is zero where the book is one-sided", bool((e[~ts] == 0).all()))

    qi = F.queue_imbalance(df)
    c.check("queue imbalance within [-1, 1]", bool(qi.between(-1, 1).all()))

    # The decisive test of the aggressor convention. ITCH reports the resting
    # side, so the aggressor is its opposite; if that were inverted, a buyer
    # would appear to pay below the mid and this would go negative.
    es = F.effective_spread(df).dropna()
    if len(es):
        neg = int((es < 0).sum())
        c.check(
            "effective spread non-negative",
            neg == 0,
            f"{neg:,} of {len(es):,} negative",
        )
        # It should also not exceed the quoted spread by much: an aggressor
        # cannot do worse than the far touch on a normal execution.
        quoted = F.spread_before(df)[es.index]
        worse = int((es > quoted + 1e-9).sum())
        c.check(
            "effective spread does not exceed quoted",
            worse / len(es) < 0.02,
            f"{worse:,} of {len(es):,} ({100 * worse / len(es):.3f}%)",
        )

    # Trades execute against resting depth, so a trade priced at the touch
    # must be on the side whose touch it matches. Restricted to tape trades:
    # an execute-with-price prints away from the resting order, so its price
    # carries no information about which side it consumed.
    tr = F.tape_trades(df)
    at_bid = tr & (df.price == df.bid_px_before)
    at_ask = tr & (df.price == df.ask_px_before)
    if at_bid.any():
        c.check(
            "trades at the bid are resting-buy executions",
            bool((df.side[at_bid & ~at_ask] == "B").all()),
        )
    if at_ask.any():
        c.check(
            "trades at the ask are resting-sell executions",
            bool((df.side[at_ask & ~at_bid] == "S").all()),
        )

    return e, qi, es


def describe(df, e, qi, es, k):
    agg = F.aggregate(df, k)
    tape = F.tape_trades(df)
    sv = F.signed_volume(df)
    excluded = int((df.event == "trade").sum() - tape.sum())

    print(f"  per-event OFI      mean {e.mean():+.2f}  sd {e.std():.1f}  "
          f"p1 {e.quantile(0.01):+.0f}  p99 {e.quantile(0.99):+.0f}")
    print(f"  queue imbalance    mean {qi.mean():+.4f}  sd {qi.std():.4f}")
    if len(es):
        print(f"  effective spread   median ${es.median():.4f}  "
              f"mean ${es.mean():.4f}  (quoted median ${F.spread_before(df).median():.4f})")
    print(f"  signed volume      net {sv.sum():+,.0f} of {df.shares[tape].sum():,} traded")
    print(f"  executions excluded from signing  {excluded:,} of "
          f"{int((df.event == 'trade').sum()):,} "
          f"({100 * excluded / max(1, (df.event == 'trade').sum()):.2f}%)")
    print(f"  {k}-event buckets  n {len(agg):,}  "
          f"OFI sd {agg.ofi.std():,.0f}  trades/bucket {agg.trades.mean():.1f}")

    # Autocorrelation of bucketed OFI, a property of the predictor alone. It
    # matters for the study because strong persistence means effective sample
    # size is far below the row count, which is what the stationary bootstrap
    # in phase 4 exists to handle.
    if len(agg) > 10:
        a = agg.ofi.to_numpy()
        ac1 = np.corrcoef(a[:-1], a[1:])[0, 1]
        print(f"  bucketed OFI ac(1) {ac1:+.4f}")


def main(argv):
    paths = argv[1:]
    if not paths:
        print(__doc__)
        return 2
    k = 200

    total_failures = 0
    detail_shown = False
    per_file = []

    for path in paths:
        df = F.load_events(path)
        c = Checks()
        e, qi, es = check_file(df, c)
        total_failures += c.failures
        name = Path(path).name.replace(".csv.gz", "")

        if not detail_shown:
            print(f"{name}   {len(df):,} rows")
            c.report()
            describe(df, e, qi, es, k)
            print()
            detail_shown = True

        per_file.append(
            {
                "file": name,
                "rows": len(df),
                "failures": c.failures,
                "ofi_sd": e.std(),
                "qi_mean": qi.mean(),
                "eff_spread": es.median() if len(es) else np.nan,
                "quoted": F.spread_before(df).median(),
            }
        )

    summary = pd.DataFrame(per_file)
    print(f"{len(summary)} files, {summary.rows.sum():,} rows, "
          f"{int(summary.failures.sum())} check failures")
    print()
    print("  effective vs quoted spread, by file (cents)")
    s = summary.dropna(subset=["eff_spread"])
    print(f"    effective  median {100 * s.eff_spread.median():.2f}  "
          f"min {100 * s.eff_spread.min():.2f}  max {100 * s.eff_spread.max():.2f}")
    print(f"    quoted     median {100 * s.quoted.median():.2f}  "
          f"min {100 * s.quoted.min():.2f}  max {100 * s.quoted.max():.2f}")
    print(f"  queue imbalance mean across files  "
          f"{summary.qi_mean.min():+.4f} to {summary.qi_mean.max():+.4f}")

    if total_failures:
        print(f"\n{total_failures} check(s) FAILED")
        return 1
    print("\nall checks passed; no forward returns computed")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

"""Feature construction from an exported event log.

Every function here is a pure transform of the columns written by
`crates/export`, and every one uses only the `*_before` state of its own row
plus the row's own event. Nothing reads forward in time.

Sign conventions, which are the part that is easy to get silently wrong:

- **OFI > 0 is buying pressure.** Depth arriving at the bid, or leaving the
  ask, is positive.
- **Aggressor sign +1 is a buyer-initiated trade.** ITCH reports the side of
  the *resting* order that was executed, so the aggressor is the opposite: an
  execution against a resting bid means someone sold into it.
- **Effective spread is non-negative** by construction when the aggressor sign
  is right, which is what makes it a test of the convention rather than just
  another statistic.
"""

import numpy as np
import pandas as pd

# ITCH prices are fixed-point with four implied decimals.
TICK = 10_000

EVENT_COLUMNS = [
    "ts_ns",
    "event",
    "side",
    "price",
    "shares",
    "printable",
    "resting_price",
    "bid_px_before",
    "ask_px_before",
    "bid_sz_before",
    "ask_sz_before",
    "bid_px_after",
    "ask_px_after",
    "bid_sz_after",
    "ask_sz_after",
]


def load_events(path, columns=None):
    return pd.read_csv(path, usecols=columns or EVENT_COLUMNS)


def two_sided(df):
    """Rows where both sides of the book are populated before and after.

    Order flow imbalance is undefined when a side is empty: there is no queue
    whose change could be measured. Rare intraday for liquid names, and
    counted rather than silently dropped by callers.
    """
    return (
        df.bid_px_before.notna()
        & df.ask_px_before.notna()
        & df.bid_px_after.notna()
        & df.ask_px_after.notna()
    )


def ofi(df):
    """Per-event order flow imbalance, Cont-Kukanov-Stoikov.

        e = 1{Pb1 >= Pb0} qb1 - 1{Pb1 <= Pb0} qb0
          - 1{Pa1 <= Pa0} qa1 + 1{Pa1 >= Pa0} qa0

    Three cases make the intuition concrete. If the bid price is unchanged,
    both bid indicators fire and the contribution is the change in bid depth.
    If the bid improves, the whole new queue counts as fresh demand. If the
    bid falls, the whole old queue counts as demand withdrawn. The ask terms
    mirror it with the sign flipped, since depth arriving at the ask is
    selling pressure.

    Computed from one row's own before-to-after transition rather than by
    comparing consecutive rows. The two are identical because each row's
    `before` is the previous row's `after`, and the single-row form is
    unaffected by any filtering the caller has already applied.
    """
    pb0, pb1 = df.bid_px_before, df.bid_px_after
    qb0, qb1 = df.bid_sz_before, df.bid_sz_after
    pa0, pa1 = df.ask_px_before, df.ask_px_after
    qa0, qa1 = df.ask_sz_before, df.ask_sz_after

    e = (
        (pb1 >= pb0) * qb1
        - (pb1 <= pb0) * qb0
        - (pa1 <= pa0) * qa1
        + (pa1 >= pa0) * qa0
    )
    return e.where(two_sided(df), 0.0).astype("float64")


def queue_imbalance(df):
    """(bid - ask) / (bid + ask) at the touch, before the event.

    In [-1, 1]; positive means the bid queue is deeper, which is the standard
    static measure of directional pressure.
    """
    total = df.bid_sz_before + df.ask_sz_before
    qi = (df.bid_sz_before - df.ask_sz_before) / total
    return qi.where(total > 0, 0.0)


def tape_trades(df):
    """Executions that are both a tape print and priced at the resting order.

    Two exclusions, and each removes a case where a trade-based feature would
    otherwise be wrong rather than merely noisy.

    A non-printable execution removes displayed depth without reaching the
    tape, so counting it inflates volume relative to any published figure.

    An execution priced away from the resting order (ITCH's
    `OrderExecutedWithPrice`) breaks the rule that identifies the aggressor.
    Normally the resting side tells you who initiated: an execution against a
    resting bid means someone sold into it. When the print lands at a
    different price from the order's own quote, that inference no longer
    holds, and on AAPL those trades produce a negative effective spread, which
    is the arithmetic signature of a mislabelled aggressor.

    About 1% of AAPL executions are excluded here. They remain in the file and
    still consume depth through the OFI path, which does not depend on any
    signing rule.
    """
    return (
        (df.event == "trade")
        & (df.printable == 1)
        & (df.price == df.resting_price)
    )


def aggressor_sign(df):
    """+1 buyer-initiated, -1 seller-initiated, 0 where undefined.

    ITCH reports the side of the resting order that was executed, so the
    aggressor is the opposite: an execution against a resting buy order means
    an incoming sell. Getting this backwards inverts every trade-flow feature
    and drives the effective spread negative, which is why
    `effective_spread` doubles as the test for it.

    Defined only on `tape_trades`; zero elsewhere, including on executions
    whose aggressor cannot be identified.
    """
    sign = np.where(df.side == "B", -1.0, 1.0)
    return pd.Series(np.where(tape_trades(df), sign, 0.0), index=df.index)


def mid_before(df):
    """Mid price in dollars, before the event. NaN when one side is empty."""
    return (df.bid_px_before + df.ask_px_before) / 2 / TICK


def spread_before(df):
    """Quoted spread in dollars, before the event."""
    return (df.ask_px_before - df.bid_px_before) / TICK


def effective_spread(df):
    """2 * sign * (price - mid), in dollars, on tape trades.

    The standard transaction-cost measure: what the aggressor actually paid
    relative to the prevailing mid, doubled to make it comparable to the
    quoted spread. Non-negative whenever the aggressor sign is correct, since
    a buyer pays at or above the mid.
    """
    s = aggressor_sign(df)
    px = df.price / TICK
    es = 2.0 * s * (px - mid_before(df))
    return es.where(tape_trades(df))


def signed_volume(df):
    """Aggressor-signed traded shares; zero on non-trade rows."""
    return aggressor_sign(df) * df.shares


def event_buckets(df, k):
    """Bucket index for a fixed number of events per bucket.

    An event clock, not a wall clock. Calendar-time sampling oversamples quiet
    periods and undersamples exactly the moments when anything happens, and
    most microstructure relationships are far more stable in event time.
    """
    return np.arange(len(df)) // k


def aggregate(df, k):
    """Per-bucket features on a `k`-event clock.

    Returns one row per bucket with the summed order flow imbalance, summed
    signed volume, trade count, and the mid at the bucket's start and end.

    `mid_end` is included because the study needs it, but nothing in this
    module relates it to any feature: constructing a predictor and evaluating
    it are deliberately separate steps.
    """
    b = event_buckets(df, k)
    work = pd.DataFrame(
        {
            "bucket": b,
            "ts_ns": df.ts_ns.to_numpy(),
            "ofi": ofi(df).to_numpy(),
            "signed_volume": signed_volume(df).to_numpy(),
            "is_trade": tape_trades(df).to_numpy(),
            "mid": mid_before(df).to_numpy(),
            "qi": queue_imbalance(df).to_numpy(),
            "spread": spread_before(df).to_numpy(),
        }
    )
    g = work.groupby("bucket", sort=True)
    out = pd.DataFrame(
        {
            "ts_start": g.ts_ns.first(),
            "ts_end": g.ts_ns.last(),
            "ofi": g.ofi.sum(),
            "signed_volume": g.signed_volume.sum(),
            "trades": g.is_trade.sum(),
            "qi_mean": g.qi.mean(),
            "spread_mean": g.spread.mean(),
            "mid_start": g.mid.first(),
            "mid_end": g.mid.last(),
        }
    )
    return out.reset_index(drop=True)

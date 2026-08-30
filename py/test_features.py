"""Unit tests for feature construction.

The spike in `feature_spike.py` checks these features against real sessions,
which catches whether they behave. These check what they mean, on rows built
by hand so the expected answer is known independently of the data.

    python3 -m pytest py/ -q
"""

import numpy as np
import pandas as pd
import pytest

import features as F
from schema import TICK


def row(**kw):
    """One event row, with a quiet two-sided book as the default background.

    Bid 100.00 for 500, ask 100.01 for 400, unchanged before to after, so any
    test only has to state the fields it is actually exercising.
    """
    base = {
        "ts_ns": 1,
        "event": "add",
        "side": "B",
        "price": 1_000_000,
        "shares": 100,
        "old_price": np.nan,
        "old_shares": np.nan,
        "printable": np.nan,
        "resting_price": np.nan,
        "bid_px_before": 1_000_000,
        "ask_px_before": 1_000_100,
        "bid_sz_before": 500,
        "ask_sz_before": 400,
        "bid_px_after": 1_000_000,
        "ask_px_after": 1_000_100,
        "bid_sz_after": 500,
        "ask_sz_after": 400,
    }
    base.update(kw)
    return base


def frame(*rows):
    return pd.DataFrame(list(rows))


# --- order flow imbalance -------------------------------------------------


def test_ofi_is_depth_change_when_the_touch_does_not_move():
    df = frame(row(bid_sz_after=600, ask_sz_after=350))
    # +100 of bid arrived, 50 of ask left; both are buying pressure.
    assert F.ofi(df).iloc[0] == 150


def test_ofi_counts_the_whole_new_queue_when_the_bid_improves():
    # A better bid replaces the old one: all of it is fresh demand, and the
    # old queue is not subtracted because it was at a worse price.
    df = frame(row(bid_px_after=1_000_050, bid_sz_after=80))
    assert F.ofi(df).iloc[0] == 80


def test_ofi_counts_the_whole_old_queue_when_the_bid_falls():
    # The bid backed away: the entire previous queue is demand withdrawn.
    df = frame(row(bid_px_after=999_900, bid_sz_after=70))
    assert F.ofi(df).iloc[0] == -500


def test_ofi_is_negative_when_the_ask_improves():
    # A better ask is fresh supply, so it pushes OFI down by its full size.
    df = frame(row(ask_px_after=1_000_050, ask_sz_after=90))
    assert F.ofi(df).iloc[0] == -90


def test_ofi_is_positive_when_the_ask_retreats():
    # Supply withdrawn at the touch is buying pressure.
    df = frame(row(ask_px_after=1_000_200, ask_sz_after=90))
    assert F.ofi(df).iloc[0] == 400


def test_ofi_is_antisymmetric_between_the_sides():
    bid = frame(row(bid_sz_after=600))
    ask = frame(row(ask_sz_after=500))
    assert F.ofi(bid).iloc[0] == 100
    assert F.ofi(ask).iloc[0] == -100


def test_ofi_is_zero_on_a_one_sided_book():
    df = frame(row(ask_px_before=np.nan, ask_sz_before=0, ask_px_after=np.nan, ask_sz_after=0))
    assert F.ofi(df).iloc[0] == 0.0


def test_ofi_is_zero_when_nothing_at_the_touch_changes():
    # A deep-book event leaves the touch untouched and must not register.
    assert F.ofi(frame(row())).iloc[0] == 0.0


# --- queue imbalance ------------------------------------------------------


def test_queue_imbalance_sign_and_value():
    df = frame(row(bid_sz_before=600, ask_sz_before=400))
    assert F.queue_imbalance(df).iloc[0] == pytest.approx(0.2)


def test_queue_imbalance_is_bounded():
    df = frame(
        row(bid_sz_before=100, ask_sz_before=0, ask_px_before=np.nan),
        row(bid_sz_before=0, ask_sz_before=100, bid_px_before=np.nan),
    )
    qi = F.queue_imbalance(df)
    assert qi.iloc[0] == 1.0
    assert qi.iloc[1] == -1.0


def test_queue_imbalance_is_zero_on_an_empty_book():
    df = frame(row(bid_sz_before=0, ask_sz_before=0))
    assert F.queue_imbalance(df).iloc[0] == 0.0


# --- trade classification -------------------------------------------------


def trade(**kw):
    d = {"event": "trade", "printable": 1.0, "resting_price": kw.get("price", 1_000_100)}
    d.update(kw)
    return row(**d)


def test_tape_trades_excludes_non_printable_executions():
    df = frame(trade(side="S", price=1_000_100, printable=0.0))
    assert not F.tape_trades(df).iloc[0]


def test_tape_trades_excludes_prints_away_from_the_resting_price():
    # An OrderExecutedWithPrice: printed at the bid while resting at the ask.
    df = frame(trade(side="S", price=1_000_000, resting_price=1_000_100))
    assert not F.tape_trades(df).iloc[0]


def test_tape_trades_accepts_an_ordinary_execution():
    df = frame(trade(side="S", price=1_000_100, resting_price=1_000_100))
    assert F.tape_trades(df).iloc[0]


def test_aggressor_is_the_opposite_of_the_resting_side():
    # Resting sell consumed means a buyer lifted it, and vice versa.
    lifted = frame(trade(side="S", price=1_000_100))
    hit = frame(trade(side="B", price=1_000_000))
    assert F.aggressor_sign(lifted).iloc[0] == +1.0
    assert F.aggressor_sign(hit).iloc[0] == -1.0


def test_aggressor_sign_is_zero_off_the_tape():
    assert F.aggressor_sign(frame(row())).iloc[0] == 0.0
    df = frame(trade(side="S", price=1_000_100, printable=0.0))
    assert F.aggressor_sign(df).iloc[0] == 0.0


# --- spreads --------------------------------------------------------------


def test_effective_spread_equals_quoted_for_a_touch_execution():
    # Buyer lifts the one-cent ask: pays half a cent over the mid, doubled.
    df = frame(trade(side="S", price=1_000_100))
    assert F.effective_spread(df).iloc[0] == pytest.approx(0.01)
    assert F.spread_before(df).iloc[0] == pytest.approx(0.01)


def test_effective_spread_is_non_negative_on_both_sides():
    both = frame(
        trade(side="S", price=1_000_100),
        trade(side="B", price=1_000_000),
    )
    es = F.effective_spread(both)
    assert (es >= 0).all()
    assert es.iloc[0] == pytest.approx(es.iloc[1])


def test_effective_spread_rewards_price_improvement():
    # A buyer filled at the mid pays nothing relative to it.
    df = frame(trade(side="S", price=1_000_050, resting_price=1_000_050))
    assert F.effective_spread(df).iloc[0] == pytest.approx(0.0)


def test_effective_spread_would_go_negative_if_the_sign_were_flipped():
    """The guard that makes the convention testable rather than assumed."""
    df = frame(trade(side="S", price=1_000_100))
    flipped = -F.aggressor_sign(df)
    es = 2.0 * flipped * (df.price / TICK - F.mid_before(df))
    assert es.iloc[0] < 0


def test_signed_volume_carries_the_aggressor_sign():
    df = frame(
        trade(side="S", price=1_000_100, shares=300),
        trade(side="B", price=1_000_000, shares=100),
    )
    assert F.signed_volume(df).tolist() == [300.0, -100.0]


# --- bucketing ------------------------------------------------------------


def test_aggregate_sums_within_buckets():
    rows = [row(bid_sz_after=500 + 10) for _ in range(4)]  # +10 OFI each
    agg = F.aggregate(frame(*rows), k=2)
    assert len(agg) == 2
    assert agg.ofi.tolist() == [20.0, 20.0]


def test_aggregate_drops_a_ragged_final_bucket():
    # Five events at k=2 gives two full buckets; the fifth is discarded
    # because its summed flow would cover fewer events than the others.
    agg = F.aggregate(frame(*[row() for _ in range(5)]), k=2)
    assert len(agg) == 2


def test_aggregate_returns_empty_when_short_of_one_bucket():
    agg = F.aggregate(frame(row()), k=10)
    assert len(agg) == 0
    assert "mid_start" in agg.columns


def test_mid_start_is_the_boundary_value_not_the_first_non_null():
    """A groupby `first()` would skip the NaN and report 100.005 instead.

    That would silently shift the price at a bucket boundary to a value from
    inside the bucket, which is exactly the kind of small displacement that
    turns into a look-ahead error downstream.
    """
    one_sided = row(bid_px_before=np.nan, bid_sz_before=0)
    agg = F.aggregate(frame(one_sided, row(), row(), row()), k=4)
    assert np.isnan(agg.mid_start.iloc[0])


def test_mid_start_is_taken_before_the_bucket_s_own_events():
    rows = [
        row(bid_px_before=1_000_000, ask_px_before=1_000_100),
        row(bid_px_before=1_000_200, ask_px_before=1_000_300),
    ]
    agg = F.aggregate(frame(*rows), k=1)
    assert agg.mid_start.tolist() == [100.005, 100.025]


def test_bucket_timestamps_are_the_first_in_each_bucket():
    rows = [row(ts_ns=t) for t in (10, 20, 30, 40)]
    agg = F.aggregate(frame(*rows), k=2)
    assert agg.ts_start.tolist() == [10, 30]

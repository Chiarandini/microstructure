"""Unit tests for queue-state extraction and the queue-reactive model."""

import numpy as np
import pandas as pd
import pytest

import queue_reactive as QR
import queue_states as Q


def row(**kw):
    """One event row with a quiet two-sided book, mirroring test_features."""
    base = {
        "ts_ns": 0,
        "event": "add",
        "price": 1_000_000,
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


def anchor(ts=0):
    """A leading state change, so the row after it has a predecessor.

    The first state-changing event of a series is always dropped: there is no
    earlier change to measure a holding time from. Real sessions lose their
    first transition the same way.
    """
    return row(ts_ns=ts, bid_sz_before=400, bid_sz_after=500,
               ask_sz_before=300, ask_sz_after=400)


# --- extraction -----------------------------------------------------------


def test_channels_are_classified_from_the_transition():
    df = frame(
        anchor(),
        row(ts_ns=1_000_000_000, bid_sz_before=500, bid_sz_after=600),          # L
        row(ts_ns=2_000_000_000, bid_sz_before=600, bid_sz_after=550, event="cancel"),  # C
        row(ts_ns=3_000_000_000, bid_sz_before=550, bid_sz_after=500, event="trade"),   # M
        row(ts_ns=4_000_000_000, bid_sz_before=500, bid_sz_after=300,
            bid_px_after=999_900),                                              # P
    )
    t = Q.touch_transitions(df, "B")
    assert t.channel.tolist() == ["L", "C", "M", "P"]


def test_a_price_move_is_a_price_move_even_when_the_queue_grows():
    """Channel P takes precedence: the queue was replaced, not added to."""
    df = frame(
        anchor(),
        row(ts_ns=1_000_000_000, bid_sz_before=100, bid_sz_after=900, bid_px_after=1_000_050),
    )
    assert Q.touch_transitions(df, "B").channel.tolist() == ["P"]


def test_holding_time_is_the_gap_to_the_previous_state_change():
    df = frame(
        anchor(),
        row(ts_ns=5_000_000_000, bid_sz_before=500, bid_sz_after=600),
        row(ts_ns=8_000_000_000, bid_sz_before=600, bid_sz_after=700),
    )
    t = Q.touch_transitions(df, "B")
    assert t.dt_ns.tolist() == [5_000_000_000, 3_000_000_000]


def test_events_that_do_not_move_the_touch_are_excluded():
    # A deep-book event leaves both touch price and size alone.
    df = frame(anchor(), row(ts_ns=1), row(ts_ns=2),
               row(ts_ns=3, bid_sz_before=500, bid_sz_after=600))
    t = Q.touch_transitions(df, "B")
    assert len(t) == 1
    assert t.channel.tolist() == ["L"]


def test_the_ask_side_is_extracted_independently():
    df = frame(
        anchor(),
        row(ts_ns=1_000_000_000, ask_sz_before=400, ask_sz_after=500),
    )
    assert Q.touch_transitions(df, "S").channel.tolist() == ["L"]
    assert Q.touch_transitions(df, "B").empty


def test_sufficient_statistics_partition_time_and_count_departures():
    df = frame(
        anchor(),
        row(ts_ns=1_000_000_000, bid_sz_before=500, bid_sz_after=600),
        row(ts_ns=3_000_000_000, bid_sz_before=600, bid_sz_after=500, event="cancel"),
    )
    t = Q.touch_transitions(df, "B")
    s = Q.sufficient_statistics(t)
    # Total holding time must equal the span between first and last change.
    assert s.T_ns.sum() == 3_000_000_000
    assert s.loc[s.q == 500, "n_L"].item() == 1
    assert s.loc[s.q == 600, "n_C"].item() == 1


# --- estimation -----------------------------------------------------------


def stats_frame(rows):
    return pd.DataFrame(rows, columns=["q", "T_ns", "n_L", "n_C", "n_M", "n_P"])


def test_intensity_is_close_to_counts_over_time():
    # 100 cancels in 10 seconds is about 10 per second; the weak prior pulls
    # it very slightly towards the pooled rate.
    s = stats_frame([[100, 10 * 10**9, 0, 100, 0, 0]])
    m = QR.fit(s, n_bins=4)
    b = QR.bin_of(m, 100)
    assert m["rates"]["C"][b] == pytest.approx(10.0, rel=0.05)


def test_the_poisson_null_is_flat_in_queue_size():
    s = stats_frame([[10, 10**9, 1, 0, 0, 0], [1000, 10**9, 100, 0, 0, 0]])
    m = QR.fit(s, n_bins=8, state_dependent=False)
    assert len(set(np.round(m["rates"]["L"], 9))) == 1


def test_the_state_dependent_fit_is_not_flat_when_the_data_is_not():
    s = stats_frame([[10, 10**9, 1, 0, 0, 0], [1000, 10**9, 100, 0, 0, 0]])
    m = QR.fit(s, n_bins=8, state_dependent=True)
    lo = m["rates"]["L"][QR.bin_of(m, 10)]
    hi = m["rates"]["L"][QR.bin_of(m, 1000)]
    assert hi > 10 * lo


def test_no_intensity_is_exactly_zero():
    """Zero would assert impossibility and make held-out likelihood -inf."""
    s = stats_frame([[100, 10**9, 5, 0, 0, 0], [200, 10**9, 5, 3, 1, 1]])
    m = QR.fit(s, n_bins=6)
    for ch in QR.CHANNELS:
        assert np.all(m["rates"][ch] > 0)


def test_held_out_likelihood_prefers_the_model_that_generated_the_data():
    """State-dependent data should be better explained by a state-dependent fit."""
    train = stats_frame([[10, 10**9, 100, 1, 1, 1], [1000, 10**9, 1, 100, 1, 1]])
    test = stats_frame([[10, 10**9, 100, 1, 1, 1], [1000, 10**9, 1, 100, 1, 1]])
    qr = QR.fit(train, n_bins=8, state_dependent=True)
    po = QR.fit(train, n_bins=8, state_dependent=False)
    ll_qr, n = QR.log_likelihood(qr, test)
    ll_po, _ = QR.log_likelihood(po, test)
    assert np.isfinite(ll_qr)
    assert ll_qr > ll_po
    assert n == 206


def test_likelihood_is_finite_on_states_absent_from_training():
    train = stats_frame([[100, 10**9, 10, 10, 1, 1]])
    test = stats_frame([[5000, 10**9, 10, 10, 1, 1]])
    m = QR.fit(train, n_bins=8)
    ll, _ = QR.log_likelihood(m, test)
    assert np.isfinite(ll)


def test_parameter_count_reflects_the_model():
    s = stats_frame([[10, 10**9, 1, 1, 1, 1], [1000, 10**9, 1, 1, 1, 1]])
    qr = QR.fit(s, n_bins=8, state_dependent=True)
    po = QR.fit(s, n_bins=8, state_dependent=False)
    assert QR.parameter_count(po) == 4
    assert QR.parameter_count(qr) > QR.parameter_count(po)


# --- simulation -----------------------------------------------------------


def test_simulation_stays_positive_and_returns_holding_times():
    s = stats_frame([[100, 10**9, 10, 10, 5, 2], [400, 10**9, 10, 20, 5, 2]])
    m = QR.fit(s, n_bins=6)
    sizes = pd.DataFrame({"channel": ["L", "C", "M"], "size": [100, 100, 100]})
    restarts = pd.DataFrame({"q0": [100, 200]})
    q, dt = QR.simulate(m, sizes, restarts, 500, np.random.default_rng(0))
    assert len(q) == len(dt) == 500
    assert (q >= 1).all()
    assert (dt >= 0).all()


def test_occupancy_is_a_time_weighted_distribution():
    edges = np.array([1, 10, 100, 1000])
    q = np.array([5, 50, 50])
    dt = np.array([1.0, 1.0, 2.0])
    occ = QR.occupancy(q, dt, edges)
    assert occ.sum() == pytest.approx(1.0)
    assert occ[0] == pytest.approx(0.25)
    assert occ[1] == pytest.approx(0.75)


def test_total_variation_bounds():
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert QR.total_variation(p, p) == pytest.approx(0.0)
    assert QR.total_variation(p, q) == pytest.approx(1.0)


def test_bin_edges_are_increasing_and_cover_the_range():
    e = QR.bin_edges(10, 5000)
    assert (np.diff(e) > 0).all()
    assert e[0] == 1
    assert e[-1] >= 5000

"""Touch-queue state transitions, extracted from the event log.

The queue-reactive model of Huang, Lehalle and Rosenbaum treats the queue at
the best quote as a continuous-time Markov chain whose transition intensities
depend on the current queue size. This module turns an event log into the
sufficient statistics that model needs.

For one side of the book, every event that changes the touch is a departure
from the state that preceded it, through one of four channels:

- **L**, a limit order joining the queue, which grows it.
- **C**, a cancellation leaving the queue.
- **M**, a market order consuming the queue.
- **P**, a price move, which ends the episode and replaces the queue entirely.

For a Markov jump process, the maximum-likelihood estimate of the intensity of
channel `k` out of state `q` is just `N_k(q) / T(q)`: the number of departures
through that channel divided by the total time spent in that state. Both are
additive over visits, so a whole session reduces to two small tables.

Statistics are accumulated on a fine grid of queue sizes in shares and re-binned
at analysis time. Choosing the analysis bins here would bake a decision into a
cache that takes minutes to rebuild.
"""

import numpy as np
import pandas as pd

CHANNELS = ["L", "C", "M", "P"]

# Queue sizes above this are pooled. Chosen well above the median touch depth
# of the deepest symbol in the universe (INTC, ~1,800 shares) so the pooled
# bin holds only a thin tail.
Q_CAP = 20_000

# Cap on how many event sizes to retain per channel for resampling during
# simulation. The distribution is heavily concentrated on round lots, so a
# large sample adds nothing.
SIZE_SAMPLE = 20_000


def touch_transitions(df, side):
    """Per-transition state, holding time and channel for one side.

    Returns a frame with one row per event that changed the touch on `side`.
    `q` is the queue size the transition departed from, `dt_ns` the time spent
    in that state, and `channel` how it was left.
    """
    px_b = f"{'bid' if side == 'B' else 'ask'}_px_before"
    px_a = f"{'bid' if side == 'B' else 'ask'}_px_after"
    sz_b = f"{'bid' if side == 'B' else 'ask'}_sz_before"
    sz_a = f"{'bid' if side == 'B' else 'ask'}_sz_after"

    changed = (df[px_b] != df[px_a]) | (df[sz_b] != df[sz_a])
    # NaN-vs-NaN compares unequal, so an empty side would otherwise register
    # as a change on every row.
    both_known = df[px_b].notna() & df[px_a].notna()
    sel = changed & both_known
    d = df[sel]
    if len(d) < 2:
        return pd.DataFrame(columns=["q", "dt_ns", "channel", "delta"])

    ts = d.ts_ns.to_numpy()
    q = d[sz_b].to_numpy()
    q_after = d[sz_a].to_numpy()
    p_before = d[px_b].to_numpy()
    p_after = d[px_a].to_numpy()
    event = d.event.to_numpy()

    # The state that ended at event i began at event i-1.
    dt = np.diff(ts)
    q = q[1:]
    q_after = q_after[1:]
    delta = q_after - q
    moved = p_after[1:] != p_before[1:]
    ev = event[1:]

    channel = np.where(
        moved,
        "P",
        np.where(delta > 0, "L", np.where(ev == "trade", "M", "C")),
    )

    return pd.DataFrame({"q": q, "dt_ns": dt, "channel": channel, "delta": delta})


def sufficient_statistics(trans):
    """Holding time and per-channel departure counts, by queue size.

    One row per occupied queue size: the total time spent there and how many
    times each channel was taken out of it.
    """
    if trans.empty:
        return pd.DataFrame(columns=["q", "T_ns"] + [f"n_{c}" for c in CHANNELS])

    q = np.minimum(trans.q.to_numpy(), Q_CAP)
    out = pd.DataFrame({"q": q, "dt_ns": trans.dt_ns.to_numpy(), "channel": trans.channel})
    time = out.groupby("q", sort=True).dt_ns.sum().rename("T_ns")
    counts = (
        out.groupby(["q", "channel"], sort=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=CHANNELS, fill_value=0)
    )
    counts.columns = [f"n_{c}" for c in CHANNELS]
    return pd.concat([time, counts], axis=1).reset_index()


def size_samples(trans, rng):
    """A sample of absolute event sizes per channel, for simulation.

    Simulation needs to draw a size once a channel is chosen. Holding this
    distribution fixed across model variants means any difference in their fit
    is attributable to the intensities, which is the thing under test.
    """
    rows = []
    for ch in ["L", "C", "M"]:
        v = np.abs(trans.delta.to_numpy()[trans.channel.to_numpy() == ch])
        v = v[v > 0]
        if len(v) == 0:
            continue
        if len(v) > SIZE_SAMPLE:
            v = rng.choice(v, SIZE_SAMPLE, replace=False)
        rows.append(pd.DataFrame({"channel": ch, "size": v}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["channel", "size"])


def post_move_queue(df, side, rng):
    """Queue sizes observed immediately after a price move.

    A price move replaces the queue, so a simulation that runs past one needs
    somewhere to restart from. Drawn from the data rather than assumed.
    """
    sz_a = f"{'bid' if side == 'B' else 'ask'}_sz_after"
    px_b = f"{'bid' if side == 'B' else 'ask'}_px_before"
    px_a = f"{'bid' if side == 'B' else 'ask'}_px_after"
    moved = (df[px_b] != df[px_a]) & df[px_b].notna() & df[px_a].notna()
    v = df.loc[moved, sz_a].to_numpy()
    v = v[v > 0]
    if len(v) > SIZE_SAMPLE:
        v = rng.choice(v, SIZE_SAMPLE, replace=False)
    return pd.DataFrame({"q0": v})

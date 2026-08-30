"""Fit, simulate and test a queue-reactive model of the touch.

The model treats the queue at the best quote as a continuous-time Markov
chain. From a state of `q` shares it leaves through one of four channels, each
with its own intensity: a limit order joining (L), a cancellation (C), a
market order consuming (M), or a price move that replaces the queue (P).

The question this phase asks is whether those intensities actually depend on
`q`. Huang, Lehalle and Rosenbaum argue they do, strongly, and that the
dependence is what makes the book mean-reverting rather than a random walk in
depth. The null is a homogeneous Poisson model whose intensities are constant.

Both models are fit on the training sessions and simulated forward with the
*same* event-size distribution, so any difference in how well they reproduce
the data is attributable to state dependence and nothing else.

They are then judged on the stationary distribution of queue size in the
held-out sessions, which neither model was fit to: the fit used holding times
and departure counts, not the occupancy distribution the simulation produces.
"""

import numpy as np
import pandas as pd

CHANNELS = ["L", "C", "M", "P"]
NS_PER_S = 1e9


def bin_edges(n_bins, q_max):
    """Log-spaced queue bins.

    Queue sizes are heavily right-skewed, so equal-width bins would put almost
    every observation in the first one and estimate the tail from nothing.
    """
    return np.unique(np.round(np.geomspace(1, max(q_max, 2), n_bins + 1)).astype(np.int64))


def fit(stats, n_bins=24, state_dependent=True):
    """Intensities per queue bin, by maximum likelihood.

    For a Markov jump process the MLE of an intensity is departures divided by
    time at risk, which is what this computes once the sufficient statistics
    are pooled into bins.

    With `state_dependent=False` every bin gets the same intensity, pooled
    over all states. That is the homogeneous Poisson null.
    """
    q = stats.q.to_numpy()
    edges = bin_edges(n_bins, int(q.max()))
    idx = np.clip(np.searchsorted(edges, q, side="right") - 1, 0, len(edges) - 2)

    time_s = np.bincount(idx, weights=stats.T_ns.to_numpy(), minlength=len(edges) - 1) / NS_PER_S
    rates = {}
    for ch in CHANNELS:
        n = np.bincount(idx, weights=stats[f"n_{ch}"].to_numpy(), minlength=len(edges) - 1)
        total_t = time_s.sum()
        pooled = n.sum() / total_t if total_t > 0 else 0.0
        if state_dependent:
            # Posterior mean under a weak Gamma prior centred on the pooled
            # rate and worth one pseudo-event: lambda = (N + 1) / (T + 1/pooled).
            #
            # The raw N/T estimate assigns exactly zero to any bin where a
            # channel happened not to fire, which asserts the channel is
            # impossible there. The data does not support that, and a single
            # such departure in held-out data makes the likelihood negatively
            # infinite. Shrinking towards the pooled rate also tames the
            # sparsely visited tail bins, where N/T is estimated from seconds
            # of occupancy.
            prior_t = 1.0 / pooled if pooled > 0 else 0.0
            rates[ch] = (n + 1.0) / (time_s + prior_t) if pooled > 0 else np.zeros_like(time_s)
        else:
            rates[ch] = np.full(len(time_s), pooled)

    return {
        "edges": edges,
        "time_s": time_s,
        "rates": rates,
        "counts": {ch: np.bincount(idx, weights=stats[f"n_{ch}"].to_numpy(),
                                   minlength=len(edges) - 1) for ch in CHANNELS},
        "state_dependent": state_dependent,
    }


def bin_of(model, q):
    edges = model["edges"]
    return np.clip(np.searchsorted(edges, q, side="right") - 1, 0, len(edges) - 2)


def simulate(model, sizes, restarts, n_events, rng):
    """Run the fitted chain forward, returning time-weighted queue occupancy.

    Returns the visited queue sizes and the time spent at each, so the
    stationary distribution can be formed the same way it is measured in the
    data: weighted by holding time, not by event count.
    """
    size_pool = {ch: sizes.loc[sizes.channel == ch, "size"].to_numpy() for ch in ["L", "C", "M"]}
    for ch in ["L", "C", "M"]:
        if len(size_pool[ch]) == 0:
            size_pool[ch] = np.array([100])
    q0_pool = restarts.q0.to_numpy()
    if len(q0_pool) == 0:
        q0_pool = np.array([100])

    qs = np.empty(n_events, dtype=np.int64)
    dts = np.empty(n_events)

    # Per-bin rate matrix and its normalised cumulative sums, built once. The
    # loop is inherently sequential, so the only thing to optimise is the work
    # done inside each step.
    edges = model["edges"]
    n_bins = len(edges) - 1
    rate_matrix = np.column_stack([model["rates"][ch] for ch in CHANNELS])
    totals = rate_matrix.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cum = np.where(totals[:, None] > 0, np.cumsum(rate_matrix, axis=1) / totals[:, None], 1.0)

    # All randomness drawn up front: per-step generator calls otherwise
    # dominate the runtime.
    u_ch = rng.random(n_events)
    exp_draws = rng.exponential(size=n_events)
    picks = np.column_stack([rng.choice(size_pool[ch], n_events) for ch in ["L", "C", "M"]])
    pick_q0 = rng.choice(q0_pool, n_events)

    q = int(pick_q0[0])
    for i in range(n_events):
        b = np.searchsorted(edges, q, side="right") - 1
        b = 0 if b < 0 else (n_bins - 1 if b >= n_bins else b)
        total = totals[b]
        if total <= 0:
            # An unvisited bin has no estimated dynamics, so restart rather
            # than invent a rate.
            q = int(pick_q0[i])
            qs[i], dts[i] = q, 0.0
            continue

        qs[i] = q
        dts[i] = exp_draws[i] / total

        c = int(np.searchsorted(cum[b], u_ch[i]))
        if c == 3:  # P, a price move replaces the queue
            q = int(pick_q0[i])
        elif c == 0:  # L
            q += int(picks[i, 0])
        else:  # C or M
            q = max(q - int(picks[i, c - 1]), 1)

    return qs, dts


def occupancy(qs, dts, edges):
    """Time-weighted distribution of queue size over bins."""
    idx = np.clip(np.searchsorted(edges, qs, side="right") - 1, 0, len(edges) - 2)
    w = np.bincount(idx, weights=dts, minlength=len(edges) - 1)
    total = w.sum()
    return w / total if total > 0 else w


def empirical_occupancy(stats, edges):
    q = stats.q.to_numpy()
    idx = np.clip(np.searchsorted(edges, q, side="right") - 1, 0, len(edges) - 2)
    w = np.bincount(idx, weights=stats.T_ns.to_numpy(), minlength=len(edges) - 1)
    total = w.sum()
    return w / total if total > 0 else w


def total_variation(p, q):
    """Half the L1 distance between two distributions, in [0, 1]."""
    return 0.5 * np.abs(p - q).sum()


def log_likelihood(model, stats):
    """Log-likelihood of held-out data under a fitted model.

    For a Markov jump process observed continuously, the log-likelihood is

        sum over states and channels of  N_k(q) log(lambda_k(q)) - lambda_k(q) T(q)

    which depends on the data only through the same sufficient statistics used
    to fit it. Evaluating it on sessions the model never saw is the sharp
    comparison between the state-dependent model and the constant-rate null:
    unlike a distance between simulated and observed distributions, it
    involves no simulation and no choice of summary statistic.
    """
    edges = model["edges"]
    q = stats.q.to_numpy()
    idx = np.clip(np.searchsorted(edges, q, side="right") - 1, 0, len(edges) - 2)
    time_s = np.bincount(idx, weights=stats.T_ns.to_numpy(), minlength=len(edges) - 1) / NS_PER_S

    ll = 0.0
    events = 0.0
    for ch in CHANNELS:
        n = np.bincount(idx, weights=stats[f"n_{ch}"].to_numpy(), minlength=len(edges) - 1)
        lam = model["rates"][ch]
        positive = lam > 0
        ll += float(np.sum(n[positive] * np.log(lam[positive])))
        ll -= float(np.sum(lam * time_s))
        # A channel with zero fitted rate but observed departures would be
        # infinitely surprising. The pooled fallback in `fit` prevents it, so
        # this only guards against a future change breaking that.
        if np.any(n[~positive] > 0):
            return -np.inf, float(n.sum())
        events += float(n.sum())
    return ll, events


def parameter_count(model):
    """Free intensities in the model, for a like-for-like comparison."""
    if not model["state_dependent"]:
        return len(CHANNELS)
    return int(np.sum(model["time_s"] > 0)) * len(CHANNELS)


def summarise_rates(model, label, top=8):
    """Intensity by queue bin, for reading the state dependence off directly."""
    edges, rates, time_s = model["edges"], model["rates"], model["time_s"]
    rows = []
    for b in range(len(edges) - 1):
        if time_s[b] <= 0:
            continue
        rows.append(
            {
                "q_lo": edges[b],
                "q_hi": edges[b + 1],
                "time_s": time_s[b],
                **{ch: rates[ch][b] for ch in CHANNELS},
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["C_per_share"] = df.C / df.q_lo.clip(lower=1)
    print(f"  {label}")
    print(f"    {'queue':>14}{'time s':>10}{'lambda_L':>10}{'lambda_C':>10}"
          f"{'lambda_M':>10}{'lambda_P':>10}")
    step = max(1, len(df) // top)
    for _, r in df.iloc[::step].iterrows():
        rng_lbl = f"{int(r.q_lo)}-{int(r.q_hi)}"
        print(f"    {rng_lbl:>14}{r.time_s:>10.0f}{r.L:>10.3f}{r.C:>10.3f}"
              f"{r.M:>10.3f}{r.P:>10.3f}")
    return df

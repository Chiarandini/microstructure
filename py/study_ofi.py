"""Does order flow imbalance predict short-horizon mid-price changes?

The protocol below was fixed in DESIGN.md before this script existed and
before any coefficient had been computed. It is restated here so a reader can
check the two against each other.

Two regressions, and the distinction between them is the whole point.

**Impact** is contemporaneous: the mid change *across* bucket i on the order
flow *during* bucket i. It is not a prediction, it cannot be traded, and it is
a known result with a high R-squared. It runs first as a reproduction: if it
fails, the pipeline is broken, so a null result in the predictive regression
would mean a bug rather than a finding.

**Prediction** is strictly forward: order flow during bucket i against the mid
change over the `h` buckets that follow it. The windows do not overlap by a
single event.

Rules, all fixed in advance:

- Train on the earlier sessions, test on the later ones, in chronological
  order. Never split within a session: intraday autocorrelation and a shared
  regime make a same-day holdout optimistic by construction.
- Report the whole decay curve over horizons, not one number at one horizon,
  which would invite picking the horizon that worked.
- Report gross and net of half the quoted spread. A signal that predicts a
  move smaller than the spread it must cross is not tradeable.
- Error bars come from a stationary bootstrap, because bucket observations are
  serially dependent and an i.i.d. bootstrap would understate them badly.
- Every symbol-horizon fit is appended to `py/experiments.jsonl` as it runs,
  so the multiple-testing correction is applied against the number of
  hypotheses actually examined rather than the number eventually reported.

    python3 py/study_ofi.py --k 200
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

BUCKETS = Path("data/buckets")
EXPERIMENTS = Path("py/experiments.jsonl")

# Fixed in advance. Horizons are in buckets, so at k=200 events these span
# roughly 200 to 4,000 events ahead.
HORIZONS = [1, 2, 3, 5, 10, 20]

# The first four sessions train, the last three test. Chronological, and
# chosen by count rather than by looking at which split flattered the result.
N_TRAIN_SESSIONS = 4

BOOTSTRAP_DRAWS = 500
# Mean block length for the stationary bootstrap, in buckets.
BOOTSTRAP_BLOCK = 50


def ols(x, y):
    """Slope and intercept of y on x, by least squares."""
    a = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(a, y, rcond=None)
    return coef[0], coef[1]


def r2_in_sample(x, y):
    beta, alpha = ols(x, y)
    pred = alpha + beta * x
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def r2_out_of_sample(beta, alpha, x, y, baseline):
    """Out-of-sample R-squared against a baseline learned on the training set.

    The baseline is the training mean, not the test mean. Using the test mean
    would let the model benefit from information it did not have, which is
    precisely the leak this whole protocol exists to avoid.
    """
    pred = alpha + beta * x
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - baseline) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def stationary_bootstrap_indices(n, block, rng):
    """Politis-Romano stationary bootstrap: geometric blocks, wrapped.

    Each step either continues the current block or jumps to a fresh random
    start, with the jump probability set so the mean block length is `block`.
    Keeping whole blocks preserves the serial dependence that makes an i.i.d.
    bootstrap far too optimistic here.
    """
    p = 1.0 / block
    t = np.arange(n)
    jump = rng.random(n) < p
    jump[0] = True

    # Vectorised rather than looped: for each position, find the most recent
    # jump, and walk forward from that block's random start. Equivalent to the
    # sequential construction, and fast enough to bootstrap hundreds of
    # thousands of buckets without dominating the run.
    reset = np.maximum.accumulate(np.where(jump, t, 0))
    starts = rng.integers(0, n, size=n)
    return (starts[reset] + (t - reset)) % n


def bootstrap_ci(x, y, beta, alpha, baseline, rng, draws, block):
    """Percentile interval for out-of-sample R-squared and for the slope."""
    n = len(x)
    r2s = np.empty(draws)
    betas = np.empty(draws)
    for d in range(draws):
        idx = stationary_bootstrap_indices(n, block, rng)
        xb, yb = x[idx], y[idx]
        r2s[d] = r2_out_of_sample(beta, alpha, xb, yb, baseline)
        betas[d] = ols(xb, yb)[0]
    return (
        np.nanpercentile(r2s, 2.5),
        np.nanpercentile(r2s, 97.5),
        np.nanpercentile(betas, 2.5),
        np.nanpercentile(betas, 97.5),
    )


def build_windows(g, h):
    """Feature and target arrays for one symbol-session at horizon `h`.

    Bucket `i` supplies the feature. Its own price change runs from
    `mid_start[i]` to `mid_start[i+1]`, so the forward window starts at
    `i+1` and ends at `i+1+h`. The two never share an event.

    Returns feature, contemporaneous change, forward change, and the mean
    spread over the feature bucket, all aligned.
    """
    mid = g.mid_start.to_numpy()
    ofi = g.ofi.to_numpy()
    spread = g.spread_mean.to_numpy()
    n = len(g)
    last = n - 1 - h
    if last <= 0:
        return None
    i = np.arange(last)
    contemporaneous = mid[i + 1] - mid[i]
    forward = mid[i + 1 + h] - mid[i + 1]
    ok = np.isfinite(contemporaneous) & np.isfinite(forward) & np.isfinite(ofi[i])
    return ofi[i][ok], contemporaneous[ok], forward[ok], spread[i][ok]


def log_experiment(record):
    EXPERIMENTS.parent.mkdir(parents=True, exist_ok=True)
    with EXPERIMENTS.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def benjamini_hochberg(pvals, alpha=0.05):
    """Return the BH-adjusted significance threshold and a rejection mask."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    thresholds = alpha * (np.arange(1, m + 1) / m)
    passed = p[order] <= thresholds
    if not passed.any():
        return 0.0, np.zeros(m, dtype=bool)
    kmax = np.max(np.where(passed)[0])
    cutoff = thresholds[kmax]
    return cutoff, p <= cutoff


def normal_two_sided_p(t):
    """Two-sided p-value from a t-statistic, normal approximation.

    Bucket counts here are in the tens of thousands, so the normal tail is
    indistinguishable from the t tail.
    """
    from math import erfc, sqrt

    return erfc(abs(t) / sqrt(2.0))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=200, help="events per bucket")
    ap.add_argument("--draws", type=int, default=BOOTSTRAP_DRAWS)
    ap.add_argument("--seed", type=int, default=20260830)
    args = ap.parse_args(argv)

    path = BUCKETS / f"k{args.k}.csv.gz"
    if not path.exists():
        print(f"missing {path}; run: python3 py/build_buckets.py {args.k}")
        return 2

    panel = pd.read_csv(path, dtype={"date": "string", "symbol": "string"})
    sessions = sorted(panel.date.unique())
    train_days, test_days = sessions[:N_TRAIN_SESSIONS], sessions[N_TRAIN_SESSIONS:]
    symbols = sorted(panel.symbol.unique())
    rng = np.random.default_rng(args.seed)
    run_id = datetime.now(timezone.utc).isoformat(timespec="seconds")

    print(f"k = {args.k} events per bucket")
    print(f"train  {', '.join(train_days)}")
    print(f"test   {', '.join(test_days)}")
    print(f"{len(panel):,} buckets, {len(symbols)} symbols")
    print()

    # --- impact, the contemporaneous reproduction ------------------------
    print("Impact (contemporaneous, in-sample; NOT a prediction)")
    print(f"  {'symbol':<8}{'R2':>8}{'lambda $/1k sh':>16}{'buckets':>10}")
    impact = {}
    for sym in symbols:
        g = panel[panel.symbol == sym]
        xs, ys = [], []
        for _, gg in g.groupby("date", sort=True):
            w = build_windows(gg, 1)
            if w is None:
                continue
            xs.append(w[0])
            ys.append(w[1])
        x, y = np.concatenate(xs), np.concatenate(ys)
        beta, _ = ols(x, y)
        r2 = r2_in_sample(x, y)
        impact[sym] = (r2, beta)
        print(f"  {sym:<8}{r2:>8.3f}{1000 * beta:>16.4f}{len(x):>10,}")
        log_experiment(
            {
                "run": run_id,
                "kind": "impact_contemporaneous",
                "k": args.k,
                "symbol": sym,
                "horizon": 0,
                "r2_in_sample": float(r2),
                "beta": float(beta),
                "n": int(len(x)),
            }
        )
    print()

    # --- prediction, strictly forward ------------------------------------
    print("Prediction (out-of-sample by session, forward windows)")
    print(f"  {'symbol':<8}{'h':>4}{'OOS R2':>10}{'95% CI':>20}"
          f"{'beta t':>9}{'pred move':>11}{'half spr':>10}")
    results = []
    for sym in symbols:
        g = panel[panel.symbol == sym]
        for h in HORIZONS:
            tr_x, tr_y, te_x, te_y, te_sp = [], [], [], [], []
            for day, gg in g.groupby("date", sort=True):
                w = build_windows(gg, h)
                if w is None:
                    continue
                if day in train_days:
                    tr_x.append(w[0])
                    tr_y.append(w[2])
                else:
                    te_x.append(w[0])
                    te_y.append(w[2])
                    te_sp.append(w[3])
            if not tr_x or not te_x:
                continue
            xtr, ytr = np.concatenate(tr_x), np.concatenate(tr_y)
            xte, yte = np.concatenate(te_x), np.concatenate(te_y)
            spte = np.concatenate(te_sp)

            beta, alpha = ols(xtr, ytr)
            baseline = ytr.mean()
            r2 = r2_out_of_sample(beta, alpha, xte, yte, baseline)

            lo, hi, blo, bhi = bootstrap_ci(
                xte, yte, beta, alpha, baseline, rng, args.draws, BOOTSTRAP_BLOCK
            )

            # A t-statistic for the slope refit on the test set, which is what
            # the p-value and the multiplicity correction are about.
            b_te, a_te = ols(xte, yte)
            resid = yte - (a_te + b_te * xte)
            sx = np.sum((xte - xte.mean()) ** 2)
            se = np.sqrt(np.sum(resid**2) / max(len(xte) - 2, 1) / sx) if sx > 0 else np.nan
            t = b_te / se if se and np.isfinite(se) and se > 0 else np.nan
            p = normal_two_sided_p(t) if np.isfinite(t) else 1.0

            # Typical predicted move against the cost of crossing. Compared at
            # the same quantile of |OFI| so the comparison is like for like.
            typical_ofi = np.nanpercentile(np.abs(xte), 90)
            pred_move = abs(beta) * typical_ofi
            half_spread = np.nanmedian(spte) / 2.0

            results.append(
                {
                    "symbol": sym,
                    "h": h,
                    "r2": r2,
                    "lo": lo,
                    "hi": hi,
                    "beta": beta,
                    "t": t,
                    "p": p,
                    "pred_move": pred_move,
                    "half_spread": half_spread,
                    "n_test": len(xte),
                }
            )
            log_experiment(
                {
                    "run": run_id,
                    "kind": "predictive",
                    "k": args.k,
                    "symbol": sym,
                    "horizon": h,
                    "r2_oos": float(r2),
                    "r2_ci": [float(lo), float(hi)],
                    "beta": float(beta),
                    "beta_ci": [float(blo), float(bhi)],
                    "t": float(t) if np.isfinite(t) else None,
                    "p": float(p),
                    "pred_move_at_p90_ofi": float(pred_move),
                    "half_spread": float(half_spread),
                    "n_train": int(len(xtr)),
                    "n_test": int(len(xte)),
                }
            )
            print(f"  {sym:<8}{h:>4}{r2:>10.5f}  [{lo:>+7.5f},{hi:>+8.5f}]"
                  f"{t:>9.1f}{pred_move:>11.4f}{half_spread:>10.4f}")
    print()

    res = pd.DataFrame(results)
    cutoff, rejected = benjamini_hochberg(res.p.to_numpy())
    res["significant"] = rejected
    res["tradeable"] = res.pred_move > res.half_spread

    # The correction applies to the hypotheses of this run. The cumulative
    # log is provenance: it records every fit ever performed, including
    # repeated runs, so the count of *distinct* specifications ever tried can
    # be recovered rather than reconstructed from memory.
    total_logged = sum(1 for _ in EXPERIMENTS.open())
    print(f"Multiplicity: {len(res)} predictive hypotheses this run; "
          f"{total_logged} fits logged cumulatively in {EXPERIMENTS}")
    print(f"  Benjamini-Hochberg at 5%: threshold p <= {cutoff:.2e}, "
          f"{int(res.significant.sum())} of {len(res)} survive")
    print(f"  positive OOS R2:          {int((res.r2 > 0).sum())} of {len(res)}")
    print(f"  CI excludes zero:         {int((res.lo > 0).sum())} of {len(res)}")
    print(f"  predicted move > half spread: {int(res.tradeable.sum())} of {len(res)}")
    print()

    print("Decay curve, median OOS R2 across symbols")
    print(f"  {'h':>4}{'median R2':>12}{'n>0':>6}{'median CI low':>16}")
    for h in HORIZONS:
        sub = res[res.h == h]
        print(f"  {h:>4}{sub.r2.median():>12.5f}{int((sub.r2 > 0).sum()):>6}"
              f"{sub.lo.median():>16.5f}")

    out = Path(f"py/results_ofi_k{args.k}.csv")
    res.to_csv(out, index=False)
    print()
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

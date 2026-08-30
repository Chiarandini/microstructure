"""Phase 5: is the touch queue state-dependent, and does that matter?

Fits two continuous-time Markov models of the queue at the best quote on the
training sessions, and judges both on the held-out ones.

- **Queue-reactive**: intensities depend on the current queue size.
- **Poisson null**: the same four channels, but constant intensities.

Both are simulated with the same event-size distribution and the same restart
distribution, so any difference in fit is attributable to state dependence
alone.

The test is the stationary distribution of queue size in the held-out
sessions, measured by time spent rather than by event count. Neither model was
fit to it: estimation used holding times and departure counts state by state,
never the occupancy the simulation has to reproduce. A model can match every
local transition rate and still get the aggregate distribution wrong, which is
what makes this a real test rather than a restatement of the fit.

    python3 py/study_queue_reactive.py
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import queue_reactive as QR

QUEUE = Path("data/queue")
EXPERIMENTS = Path("py/experiments.jsonl")

# The same chronological split as the phase-4 study, for the same reason.
N_TRAIN_SESSIONS = 4
N_SIM_EVENTS = 300_000


def log_experiment(record):
    with EXPERIMENTS.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bins", type=int, default=24)
    ap.add_argument("--events", type=int, default=N_SIM_EVENTS)
    ap.add_argument("--seed", type=int, default=20260831)
    args = ap.parse_args(argv)

    stats = pd.read_csv(QUEUE / "stats.csv.gz", dtype={"date": "string"})
    sizes = pd.read_csv(QUEUE / "sizes.csv.gz")
    restarts = pd.read_csv(QUEUE / "restarts.csv.gz", dtype={"date": "string"})

    sessions = sorted(stats.date.unique())
    train_days, test_days = sessions[:N_TRAIN_SESSIONS], sessions[N_TRAIN_SESSIONS:]
    symbols = sorted(stats.symbol.unique())
    rng = np.random.default_rng(args.seed)
    run_id = datetime.now(timezone.utc).isoformat(timespec="seconds")

    print(f"train  {', '.join(train_days)}")
    print(f"test   {', '.join(test_days)}")
    print(f"{len(stats):,} state rows, {len(symbols)} symbols, both sides")
    print()

    # One symbol's intensity profile, printed so the state dependence can be
    # read off directly rather than inferred from a distance metric.
    demo = stats[(stats.symbol == "AAPL") & (stats.side == "B") & stats.date.isin(train_days)]
    demo_pooled = demo.groupby("q", as_index=False).sum(numeric_only=True)
    print("Intensity by queue size (AAPL bid, training sessions, events/sec)")
    QR.summarise_rates(QR.fit(demo_pooled, args.bins), "")
    print()

    rows = []
    print("Held-out log-likelihood per event (nats), and simulated occupancy fit")
    print(f"  {'symbol':<8}{'side':>5}{'QR ll/ev':>11}{'Poisson':>10}{'gain':>9}"
          f"{'TV QR':>9}{'TV Pois':>9}")
    for sym in symbols:
        for side in ("B", "S"):
            tr = stats[(stats.symbol == sym) & (stats.side == side) & stats.date.isin(train_days)]
            te = stats[(stats.symbol == sym) & (stats.side == side) & stats.date.isin(test_days)]
            if tr.empty or te.empty:
                continue
            tr_pooled = tr.groupby("q", as_index=False).sum(numeric_only=True)
            te_pooled = te.groupby("q", as_index=False).sum(numeric_only=True)

            sz = sizes[(sizes.symbol == sym) & (sizes.side == side)]
            rs = restarts[
                (restarts.symbol == sym)
                & (restarts.side == side)
                & restarts.date.isin(train_days)
            ]

            qr = QR.fit(tr_pooled, args.bins, state_dependent=True)
            po = QR.fit(tr_pooled, args.bins, state_dependent=False)
            edges = qr["edges"]

            ll_qr, n_ev = QR.log_likelihood(qr, te_pooled)
            ll_po, _ = QR.log_likelihood(po, te_pooled)
            per_qr, per_po = ll_qr / n_ev, ll_po / n_ev

            empirical = QR.empirical_occupancy(te_pooled, edges)
            q1, d1 = QR.simulate(qr, sz, rs, args.events, rng)
            q2, d2 = QR.simulate(po, sz, rs, args.events, rng)
            tv_qr = QR.total_variation(QR.occupancy(q1, d1, edges), empirical)
            tv_po = QR.total_variation(QR.occupancy(q2, d2, edges), empirical)

            rows.append(
                {
                    "symbol": sym,
                    "side": side,
                    "ll_qr_per_event": per_qr,
                    "ll_poisson_per_event": per_po,
                    "ll_gain_per_event": per_qr - per_po,
                    "test_events": n_ev,
                    "params_qr": QR.parameter_count(qr),
                    "params_poisson": QR.parameter_count(po),
                    "tv_qr": tv_qr,
                    "tv_poisson": tv_po,
                }
            )
            print(f"  {sym:<8}{side:>5}{per_qr:>11.3f}{per_po:>10.3f}"
                  f"{per_qr - per_po:>+9.3f}{tv_qr:>9.3f}{tv_po:>9.3f}")
            log_experiment(
                {
                    "run": run_id,
                    "kind": "queue_reactive",
                    "symbol": sym,
                    "side": side,
                    "bins": args.bins,
                    "sim_events": args.events,
                    "ll_qr_per_event": float(per_qr),
                    "ll_poisson_per_event": float(per_po),
                    "ll_gain_per_event": float(per_qr - per_po),
                    "test_events": int(n_ev),
                    "tv_queue_reactive": float(tv_qr),
                    "tv_poisson": float(tv_po),
                }
            )

    res = pd.DataFrame(rows)
    total_gain = float((res.ll_gain_per_event * res.test_events).sum())
    print()
    print("Held-out likelihood (the sharp test)")
    print(f"  queue-reactive wins on {int((res.ll_gain_per_event > 0).sum())} "
          f"of {len(res)} symbol-sides")
    print(f"  median gain            {res.ll_gain_per_event.median():+.4f} nats/event")
    print(f"  total gain             {total_gain:+,.0f} nats over "
          f"{int(res.test_events.sum()):,} held-out events")
    print(f"  extra parameters       {int(res.params_qr.median())} vs "
          f"{int(res.params_poisson.median())} per symbol-side")
    print()
    print("Simulated stationary occupancy (the generative test)")
    print(f"  queue-reactive closer on {int((res.tv_qr < res.tv_poisson).sum())} "
          f"of {len(res)} symbol-sides")
    print(f"  median total variation   queue-reactive {res.tv_qr.median():.4f}, "
          f"Poisson {res.tv_poisson.median():.4f}")

    out = Path("py/results_queue_reactive.csv")
    res.to_csv(out, index=False)
    print()
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

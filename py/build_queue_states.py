"""Cache queue-state sufficient statistics for the whole panel.

Reduces 86 million events to per-symbol-side tables of holding time and
departure counts by queue size, plus the size samples a simulation needs.
Small enough to reload instantly, so the modelling can iterate without
re-reading the panel.

    python3 py/build_queue_states.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F
import queue_states as Q

EVENTS = Path("data/events")
OUT = Path("data/queue")

NEEDED = [
    "ts_ns",
    "event",
    "price",
    "bid_px_before",
    "ask_px_before",
    "bid_sz_before",
    "ask_sz_before",
    "bid_px_after",
    "ask_px_after",
    "bid_sz_after",
    "ask_sz_after",
]

NAME = re.compile(r"^(\d{8})\.(\w+)_(\w+)\.csv\.gz$")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260831)

    stats, sizes, restarts = [], [], []
    paths = sorted(EVENTS.glob("*.csv.gz"))
    if not paths:
        print(f"no event files in {EVENTS}")
        return 2

    for path in paths:
        m = NAME.match(path.name)
        if not m:
            continue
        date, _venue, symbol = m.groups()
        df = F.load_events(path, columns=NEEDED)

        for side in ("B", "S"):
            trans = Q.touch_transitions(df, side)
            if trans.empty:
                continue

            s = Q.sufficient_statistics(trans)
            s.insert(0, "side", side)
            s.insert(0, "symbol", symbol)
            s.insert(0, "date", date)
            stats.append(s)

            z = Q.size_samples(trans, rng)
            z.insert(0, "side", side)
            z.insert(0, "symbol", symbol)
            sizes.append(z)

            r = Q.post_move_queue(df, side, rng)
            r.insert(0, "side", side)
            r.insert(0, "symbol", symbol)
            r.insert(0, "date", date)
            restarts.append(r)

        print(f"  {date} {symbol:<6} {len(df):>10,} events")

    pd.concat(stats, ignore_index=True).to_csv(OUT / "stats.csv.gz", index=False)
    pd.concat(sizes, ignore_index=True).to_csv(OUT / "sizes.csv.gz", index=False)
    pd.concat(restarts, ignore_index=True).to_csv(OUT / "restarts.csv.gz", index=False)
    print()
    print(f"wrote {OUT}/stats.csv.gz, sizes.csv.gz, restarts.csv.gz")
    return 0


if __name__ == "__main__":
    sys.exit(main())

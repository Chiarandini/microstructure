"""Reduce the event panel to bucket aggregates, once.

The study iterates on 86 million event rows only if it has to. Aggregating to
a `k`-event clock collapses the panel to a few hundred thousand rows, which
fits in memory and re-reads in seconds, so the regression work can be rerun
freely without re-parsing the panel each time.

    python3 py/build_buckets.py 200
"""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F

EVENTS = Path("data/events")
BUCKETS = Path("data/buckets")

# Columns aggregate() needs. Reading only these roughly halves load time.
NEEDED = [
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

NAME = re.compile(r"^(\d{8})\.(\w+)_(\w+)\.csv\.gz$")


def main(argv):
    k = int(argv[1]) if len(argv) > 1 else 200
    BUCKETS.mkdir(parents=True, exist_ok=True)
    out_path = BUCKETS / f"k{k}.csv.gz"

    frames = []
    paths = sorted(EVENTS.glob("*.csv.gz"))
    if not paths:
        print(f"no event files in {EVENTS}")
        return 2

    for path in paths:
        m = NAME.match(path.name)
        if not m:
            print(f"skipping unrecognised name {path.name}")
            continue
        date, _venue, symbol = m.groups()

        df = F.load_events(path, columns=NEEDED)
        agg = F.aggregate(df, k)
        agg.insert(0, "symbol", symbol)
        agg.insert(0, "date", date)
        frames.append(agg)
        print(f"  {date} {symbol:<6} {len(df):>10,} events -> {len(agg):>8,} buckets")

    panel = pd.concat(frames, ignore_index=True)
    panel.to_csv(out_path, index=False)
    print()
    print(f"wrote {out_path}  ({len(panel):,} buckets, "
          f"{panel.date.nunique()} sessions, {panel.symbol.nunique()} symbols)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

#!/usr/bin/env bash
# Build the study panel: fetch each Nasdaq session, export the universe's
# per-event logs, validate them.
#
# Resumable. A day whose exports already exist for every symbol is skipped, so
# an interrupted run can simply be restarted.
#
#   ./scripts/build_dataset.sh              # fetch, export, validate
#   ./scripts/build_dataset.sh --drop-raw   # also delete each raw session
#                                           # once its export succeeds
#
# Raw sessions are ~4 GB each and are kept by default: the export schema
# changes during development, and re-exporting from a local file takes
# seconds where re-downloading takes minutes. Use --drop-raw only if disk is
# actually short.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# The universe, fixed in DESIGN.md before any result was computed. Changing it
# means changing that section too, and saying why.
SYMBOLS="AAPL,MSFT,AMZN,GOOGL,INTC,CSCO,SPY,QQQ"

# The seven sessions Nasdaq publishes. 2019-05-30 is deliberately absent: BX
# has it, Nasdaq does not.
DATES="20190130 20190327 20190730 20190830 20191030 20191230 20200130"

DROP_RAW=0
[ "${1:-}" = "--drop-raw" ] && DROP_RAW=1

REPLAY="./target/release/replay"
if [ ! -x "$REPLAY" ]; then
  echo "building replay"
  cargo build --release
fi

mkdir -p data/events

n_symbols=$(echo "$SYMBOLS" | tr ',' '\n' | wc -l | tr -d ' ')

for date in $DATES; do
  stem="${date}.NASDAQ_ITCH50"
  raw="data/raw/${stem}.gz"

  # find, not ls: `ls` on no matches returns non-zero, which `pipefail`
  # propagates and `set -e` turns into a silent exit.
  have=$(find data/events -maxdepth 1 -name "${stem}_*.csv.gz" | wc -l | tr -d ' ')
  if [ "$have" -eq "$n_symbols" ]; then
    echo "== $date: already exported ($have symbols), skipping"
    continue
  fi

  echo "== $date"
  ./scripts/fetch.sh nasdaq "$date"

  # --session-only keeps the continuous session, where the crossed-book
  # invariant holds and where the study is defined.
  "$REPLAY" "$raw" --symbols "$SYMBOLS" --out-dir data/events --session-only \
    | grep -E "books tracked|elapsed|fails|unknown order|oversized|duplicate|level incon|still live|depth vs|exported"

  got=$(find data/events -maxdepth 1 -name "${stem}_*.csv.gz" | wc -l | tr -d ' ')
  if [ "$got" -ne "$n_symbols" ]; then
    echo "WARNING: $date produced $got files, expected $n_symbols" >&2
    echo "  missing:" >&2
    for s in $(echo "$SYMBOLS" | tr ',' ' '); do
      [ -f "data/events/${stem}_${s}.csv.gz" ] || echo "    $s" >&2
    done
  fi

  if [ "$DROP_RAW" -eq 1 ]; then
    echo "dropping $raw"
    rm -f "$raw"
  fi
done

echo
echo "== validating"
python3 py/validate_export.py data/events/*.csv.gz

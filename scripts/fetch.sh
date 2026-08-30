#!/usr/bin/env bash
# Fetch a Nasdaq ITCH 5.0 sample day into data/raw/.
#
# Nasdaq publishes these openly at https://emi.nasdaq.com/ITCH/ with no
# account and no credentials. Files are large: a Nasdaq day is 3.5 to 5.6 GB
# gzipped, a BX day 0.4 to 1.7 GB.
#
#   ./scripts/fetch.sh bx 20190730        # small, for development
#   ./scripts/fetch.sh nasdaq 20190730    # full venue, for the study
#   ./scripts/fetch.sh list               # what is available

set -euo pipefail

BASE="https://emi.nasdaq.com/ITCH"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data/raw"

# The seven dates published for both venues, plus the BX-only extras.
DATES="20190130 20190327 20190530 20190730 20190830 20191030 20191230 20200130"

usage() {
  sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 1
}

[ $# -ge 1 ] || usage

if [ "$1" = "list" ]; then
  echo "Nasdaq:"
  curl -s "$BASE/Nasdaq%20ITCH/" | sed 's/<br>/\n/g' |
    grep -o '[0-9]\{1,\} <A HREF="[^"]*ITCH50.gz"' | sed 's/.*\///;s/"//'
  echo
  echo "BX:"
  curl -s "$BASE/Nasdaq%20BX%20ITCH/" | sed 's/<br>/\n/g' |
    grep -o '[0-9]\{1,\} <A HREF="[^"]*\.gz"' | sed 's/.*\///;s/"//'
  exit 0
fi

[ $# -eq 2 ] || usage
venue="$1"
date="$2"

case "$venue" in
  bx)
    # BX names files YYYYMMDD.
    url="$BASE/Nasdaq%20BX%20ITCH/${date}.BX_ITCH_50.gz"
    out="$DEST/${date}.BX_ITCH_50.gz"
    ;;
  nasdaq)
    # Nasdaq names files MMDDYYYY, which is a genuine trap when scripting
    # against the two venues together.
    mmddyyyy="${date:4:2}${date:6:2}${date:0:4}"
    url="$BASE/Nasdaq%20ITCH/${mmddyyyy}.NASDAQ_ITCH50.gz"
    out="$DEST/${date}.NASDAQ_ITCH50.gz"
    ;;
  *)
    echo "unknown venue '$venue' (expected 'bx' or 'nasdaq')" >&2
    exit 1
    ;;
esac

case "$DATES" in
  *"$date"*) ;;
  *) echo "warning: $date is not one of the known published dates" >&2 ;;
esac

mkdir -p "$DEST"

# Only skip when the local file is actually complete. Skipping on mere
# existence would strand a partial multi-GB download as permanently broken,
# and a truncated gzip fails deep into the replay rather than at open time.
remote_size="$(curl -fsSL --head "$url" | awk 'tolower($1) ~ /^content-length:/ {print $2}' | tr -d '\r' | tail -1)"
local_size=0
[ -f "$out" ] && local_size="$(wc -c < "$out" | tr -d ' ')"

if [ -n "$remote_size" ] && [ "$local_size" = "$remote_size" ]; then
  echo "already have $out ($local_size bytes)"
  exit 0
fi

if [ "$local_size" != "0" ]; then
  echo "resuming $out at $local_size of ${remote_size:-unknown} bytes"
fi

echo "fetching $url"
# --continue-at lets an interrupted multi-GB download resume rather than
# restart, which matters on a laptop.
# Progress meter only when attached to a terminal; in a log it emits a
# few hundred KB of carriage-returned noise per file.
progress=""
[ -t 1 ] || progress="--no-progress-meter"
curl -fSL --retry 3 --continue-at - $progress -o "$out" "$url"
echo "wrote $out"

# A truncated file is worse than a missing one: it parses happily until it
# does not. Verify the size we ended up with.
final_size="$(wc -c < "$out" | tr -d ' ')"
if [ -n "$remote_size" ] && [ "$final_size" != "$remote_size" ]; then
  echo "SIZE MISMATCH: expected $remote_size bytes, got $final_size" >&2
  exit 1
fi

# Nasdaq publishes .md5sum siblings for some days; verify when one exists.
if curl -fsS --head "${url}.md5sum" >/dev/null 2>&1; then
  expected="$(curl -fsS "${url}.md5sum" | awk '{print $1}')"
  actual="$(md5 -q "$out" 2>/dev/null || md5sum "$out" | awk '{print $1}')"
  if [ "$expected" = "$actual" ]; then
    echo "checksum ok"
  else
    echo "CHECKSUM MISMATCH: expected $expected, got $actual" >&2
    exit 1
  fi
else
  echo "no published checksum for this day; skipping verification"
fi

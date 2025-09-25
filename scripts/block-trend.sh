#!/usr/bin/env bash
# Quick CLI to print block height trend and last N timestamps.
# Usage: block-trend.sh [--url <adapter_base>] [--n <count>]
# Default adapter base: http://localhost:18099

set -euo pipefail
BASE=${1:-}
N=30

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url) BASE="$2"; shift 2 ;;
    --n) N="$2"; shift 2 ;;
    *) echo "Unknown: $1"; exit 2 ;;
  esac
 done

BASE=${BASE:-"http://localhost:18099"}

summary=$(curl -fsS "$BASE/explorer/summary")
height=$(printf '%s' "$summary" | sed -n 's/.*"height"\s*:\s*\([0-9]\+\).*/\1/p' | head -n1)

blocks=$(curl -fsS "$BASE/explorer/blocks?limit=$N")

echo "Tip: $height"
echo "Last $N blocks:"
printf '%s' "$blocks" | jq -r '.blocks[] | [.height, (.header.block_header.timestamp // .header.timestamp // 0)] | @tsv' 2>/dev/null \
  | awk '{ cmd="date -r "$2" "+%F %T""; cmd | getline d; close(cmd); printf("%8s  %s\n", $1, d) }'

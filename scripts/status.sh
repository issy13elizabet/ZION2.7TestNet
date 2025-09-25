#!/usr/bin/env bash
# Quick status: chain height, block stats, wallet balance via adapter/proxy
# Usage:
#   scripts/status.sh [--base http://HOST:8080] [--adapter http://HOST:18099]
# Defaults:
#   BASE (proxy base): http://localhost:8080  -> uses /adapter/... paths
#   If --adapter provided, queries adapter directly and ignores BASE

set -euo pipefail

BASE=${BASE:-"http://localhost:8080"}
ADAPTER=${ADAPTER:-""}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base) BASE="$2"; shift 2 ;;
    --adapter) ADAPTER="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

if [[ -n "$ADAPTER" ]]; then
  ADP="$ADAPTER"
else
  ADP="${BASE%/}/adapter"
fi

JQ=jq
if ! command -v jq >/dev/null 2>&1; then JQ=; fi

echo "[status] Using adapter: $ADP"

echo "[status] Fetching explorer summary…"
sum=$(curl -fsS --max-time 6 "$ADP/explorer/summary" || true)
echo "$sum" | sed -E 's/\s+/ /g' | cut -c1-160 | sed 's/$/.../'

echo "[status] Fetching stats (n=120)…"
stats=$(curl -fsS --max-time 6 "$ADP/explorer/stats?n=120" || true)
if [[ -n "$JQ" ]]; then
  echo "$stats" | jq '{tip, avgIntervalSec, blocksLastHour, bphApprox}'
else
  echo "$stats" | sed -n 's/.*"tip"\s*:\s*\([0-9]\+\).*/tip: \1/p'
  echo "$stats" | sed -n 's/.*"avgIntervalSec"\s*:\s*\([0-9]\+\).*/avgIntervalSec: \1s/p'
  echo "$stats" | sed -n 's/.*"blocksLastHour"\s*:\s*\([0-9]\+\).*/blocksLastHour: \1/p'
  echo "$stats" | sed -n 's/.*"bphApprox"\s*:\s*\([0-9.]\+\).*/bphApprox: \1/p'
fi

echo "[status] Fetching wallet balance…"
bal=$(curl -fsS --max-time 6 "$ADP/wallet/balance" || true)
if [[ -n "$JQ" ]]; then
  echo "$bal" | jq '{availableBalance, lockedAmount}'
else
  echo "$bal" | sed -n 's/.*"availableBalance"\s*:\s*\([0-9]\+\).*/availableBalance: \1/p'
  echo "$bal" | sed -n 's/.*"lockedAmount"\s*:\s*\([0-9]\+\).*/lockedAmount: \1/p'
fi

echo "[status] Done."

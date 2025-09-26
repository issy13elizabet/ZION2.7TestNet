#!/usr/bin/env bash
# Poll rpc-shim height until target (default 60) reached.
# Usage: ./tools/watch_height.sh [target] [interval]
set -euo pipefail
TARGET=${1:-60}
INTERVAL=${2:-5}
URL=${ZION_SHIM_URL:-http://localhost:18089/json_rpc}

echo "[watch_height] Waiting for height >= $TARGET (poll every ${INTERVAL}s) via $URL"
while true; do
  JSON=$(curl -s -m 3 -H 'Content-Type: application/json' -d '{"jsonrpc":"2.0","id":"0","method":"get_info"}' "$URL" || true)
  HEIGHT=$(echo "$JSON" | grep -o '"height"[[:space:]]*:[[:space:]]*[0-9]*' | head -1 | grep -o '[0-9]*' || echo 0)
  TS=$(date -Iseconds)
  echo "$TS height=$HEIGHT"
  if [ "${HEIGHT}" -ge "${TARGET}" ]; then
    echo "[watch_height] Target reached."; exit 0
  fi
  sleep "$INTERVAL"
done

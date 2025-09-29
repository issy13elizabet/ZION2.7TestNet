#!/usr/bin/env bash
set -euo pipefail
CORE_URL="${CORE_URL:-http://localhost:${PORT:-8602}}"

red() { printf "\033[31m%s\033[0m\n" "$*"; }
green() { printf "\033[32m%s\033[0m\n" "$*"; }

fail() { red "[FAIL] $*"; exit 1; }

echo "[smoke] Strict mode bridge status check @ $CORE_URL"
BRIDGE_JSON=$(curl -sf "$CORE_URL/api/bridge/status") || fail "bridge status endpoint unreachable"
ENABLED=$(echo "$BRIDGE_JSON" | grep -o '"enabled":true' || true)
if [ -z "$ENABLED" ]; then
  echo "$BRIDGE_JSON"
  fail "bridge not enabled"
fi
HEIGHT=$(echo "$BRIDGE_JSON" | sed -n 's/.*"height":\([0-9]\+\).*/\1/p')
if [ -z "$HEIGHT" ] || [ "$HEIGHT" -le 0 ]; then
  echo "$BRIDGE_JSON"
  fail "invalid or zero height"
fi
green "[ok] bridge enabled height=$HEIGHT"

# Try fetch block template via RPC adapter (if available)
if curl -sf -X POST "$CORE_URL/api/rpc/json_rpc" -H 'Content-Type: application/json' -d '{"jsonrpc":"2.0","id":1,"method":"get_block_template","params":{}}' > /dev/null; then
  green "[ok] get_block_template JSON-RPC reachable"
else
  red "[warn] get_block_template RPC failed (may be strict failing if daemon warming)"
fi

echo "[smoke] SUCCESS"
#!/usr/bin/env bash
set -euo pipefail

POOL_CONT=${1:-zion-uzi-pool}
PATCH_SRC=${2:-/opt/zion/Zion/docker/uzi-pool/patch-rx.js}

if ! docker ps --format '{{.Names}}' | grep -qx "$POOL_CONT"; then
  echo "[patch] Container $POOL_CONT not running" >&2
  exit 2
fi

if [ ! -f "$PATCH_SRC" ]; then
  echo "[patch] patch-rx.js not found at $PATCH_SRC" >&2
  exit 3
fi

echo "[patch] Copy patch-rx.js into container..."
docker cp "$PATCH_SRC" "$POOL_CONT:/patch-rx.js"

echo "[patch] Run patch inside container..."
docker exec "$POOL_CONT" sh -lc 'node /patch-rx.js || true; if [ -f /app/lib/pool.js ]; then grep -n "getTargetHex" /app/lib/pool.js | head -n 3 || true; fi'

echo "[patch] Restart container..."
docker restart "$POOL_CONT" >/dev/null
sleep 2

echo "[patch] Tail last logs:"
docker logs --tail=80 "$POOL_CONT" 2>&1 | tail -n 80

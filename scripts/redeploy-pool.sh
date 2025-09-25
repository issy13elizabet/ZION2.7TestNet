#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
COMPOSE_FILE="$ROOT_DIR/docker/compose.pool-seeds.yml"

echo "[1/4] Building rpc-shim (no cache)…"
docker compose -f "$COMPOSE_FILE" build --no-cache rpc-shim

echo "[2/4] Restarting rpc-shim…"
docker compose -f "$COMPOSE_FILE" up -d --no-deps --force-recreate rpc-shim

echo "[3/4] Ensuring redis & uzi-pool are up…"
docker compose -f "$COMPOSE_FILE" up -d redis uzi-pool

echo "[4/4] Probing shim health (http://localhost:18089/)…"
curl -s http://localhost:18089/ || true
echo

echo "Pool status (docker ps):"
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | (head -n1; grep -E 'zion-(rpc-shim|uzi-pool|redis)') || true

echo
echo "Follow logs (Ctrl+C to stop):"
docker logs -f zion-uzi-pool &
POOL_LOG_PID=$!
sleep 2
echo "-- shim recent logs --"
docker logs --tail=80 zion-rpc-shim || true
wait $POOL_LOG_PID || true

#!/usr/bin/env bash

# Simple Ryzen bring-up for ZION stack (seeds + rpc-shim + walletd + pool + adapter + redis)
# - No edits to existing files needed
# - Creates a small Compose override to point rpc-shim and walletd to seed1/seed2
# - Builds required images and starts services

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$ROOT_DIR"

COMPOSE_FILE="docker/compose.pool-seeds.yml"
OVERRIDE_FILE="docker/compose.override.ryzen.yml"
PROXY_COMPOSE="docker/compose.proxy.yml"
PROM_COMPOSE="docker/compose.prometheus.yml"

echo "[ryzen-up] Using repo at: $ROOT_DIR"

# Resolve compose command
if command -v docker &>/dev/null && docker compose version &>/dev/null; then
  COMPOSE_CMD=(docker compose)
elif command -v docker-compose &>/dev/null; then
  COMPOSE_CMD=(docker-compose)
else
  echo "ERROR: docker compose (or docker-compose) not found." >&2
  exit 1
fi

# Ensure network
echo "[ryzen-up] Ensuring docker network 'zion-seeds' exists"
docker network create zion-seeds >/dev/null 2>&1 || true

# Create override compose for Ryzen
echo "[ryzen-up] Writing $OVERRIDE_FILE"
cat > "$OVERRIDE_FILE" <<'YAML'
services:
  rpc-shim:
    environment:
      - ZION_RPC_URLS=http://zion-seed1:18081/json_rpc,http://zion-seed2:18081/json_rpc
  walletd:
    command: ["--bind-address", "0.0.0.0", "--bind-port", "8070", "--container-file", "/home/zion/.zion/pool.wallet", "--container-password", "testpass", "--daemon-address", "zion-seed1", "--daemon-port", "18081"]
YAML

# Build core daemon image (once)
echo "[ryzen-up] Building zion:production-fixed (core daemon)"
docker build -t zion:production-fixed -f docker/Dockerfile.zion-cryptonote.prod .

# Build Uzi Pool image (used by service 'uzi-pool')
echo "[ryzen-up] Building zion:uzi-pool (stratum pool)"
docker build -t zion:uzi-pool -f docker/uzi-pool/Dockerfile .

# Build adapter images referenced with build contexts in compose
echo "[ryzen-up] Building adapter images (rpc-shim, wallet-adapter)"
"${COMPOSE_CMD[@]}" -f "$COMPOSE_FILE" -f "$OVERRIDE_FILE" build rpc-shim wallet-adapter

# Bring up seeds first
echo "[ryzen-up] Starting seed1 and seed2"
"${COMPOSE_CMD[@]}" -f "$COMPOSE_FILE" -f "$OVERRIDE_FILE" up -d seed1 seed2

# Bring up remaining services
echo "[ryzen-up] Starting redis, rpc-shim, walletd, uzi-pool, wallet-adapter"
"${COMPOSE_CMD[@]}" -f "$COMPOSE_FILE" -f "$OVERRIDE_FILE" up -d redis rpc-shim walletd uzi-pool wallet-adapter

# Optionally bring up reverse proxy if compose file exists
if [[ -f "$PROXY_COMPOSE" ]]; then
  echo "[ryzen-up] Starting nginx reverse proxy on :8080"
  "${COMPOSE_CMD[@]}" -f "$PROXY_COMPOSE" up -d zion-proxy
fi

# Optionally bring up Prometheus
if [[ -f "$PROM_COMPOSE" ]]; then
  echo "[ryzen-up] Starting Prometheus on :9090"
  "${COMPOSE_CMD[@]}" -f "$PROM_COMPOSE" up -d prometheus
  echo "[ryzen-up] Starting Alertmanager on :9093"
  "${COMPOSE_CMD[@]}" -f "$PROM_COMPOSE" up -d alertmanager
  echo "[ryzen-up] Starting Grafana on :3000"
  "${COMPOSE_CMD[@]}" -f "$PROM_COMPOSE" up -d grafana
fi

echo "[ryzen-up] Done. Quick checks:"
echo "  docker ps | grep zion-"
echo "  curl -s http://localhost:18089/"
echo "  curl -s http://localhost:18089/getheight"
if [[ -f "$PROXY_COMPOSE" ]]; then
  echo "  curl -s http://localhost:8080/healthz"
  echo "  curl -s http://localhost:8080/shim/metrics.json | jq ."
  echo "  curl -s http://localhost:8080/shim/metrics | head -n 5"
fi
echo "  Prometheus:  http://localhost:9090/graph?g0.expr=zion_shim_last_height"
echo "  Alertmanager: http://localhost:9093"
echo "  Grafana:     http://localhost:3000 (user: admin, pass: \"${GRAFANA_ADMIN_PASSWORD:-admin}\")"

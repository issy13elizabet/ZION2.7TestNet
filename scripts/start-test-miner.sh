#!/usr/bin/env bash
set -euo pipefail

# Start a test XMRig miner connected to the uzi-pool on the same Docker network.
# Local usage:
#   ./scripts/start-test-miner.sh
# Remote usage (SSH):
#   ./scripts/start-test-miner.sh <server-ip> [user]

if [[ $# -gt 0 ]]; then
  SERVER_IP="$1"; USER="${2:-root}"
  echo "[ssh] Starting test miner on ${USER}@${SERVER_IP}…"
  ssh "${USER}@${SERVER_IP}" "docker compose -f /opt/zion/Zion/docker/compose.test-miner.yml up -d xmrig-test && docker logs -f zion-xmrig-test"
  exit $?
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
COMPOSE_FILE="$ROOT_DIR/docker/compose.test-miner.yml"

echo "[local] Ensuring network 'zion-seeds' exists…"
docker network create zion-seeds >/dev/null 2>&1 || true

echo "[local] Starting test miner container…"
docker compose -f "$COMPOSE_FILE" up -d xmrig-test

echo "[local] Following miner logs (Ctrl+C to stop)…"
docker logs -f zion-xmrig-test

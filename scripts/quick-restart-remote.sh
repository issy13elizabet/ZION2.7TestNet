#!/usr/bin/env bash
set -euo pipefail

# Quick restart of zion services remotely using docker compose
# Usage: ./scripts/quick-restart-remote.sh <server-ip> [user]

SERVER_IP="${1:-}"
SERVER_USER="${2:-root}"

if [[ -z "${SERVER_IP}" ]]; then
  echo "Usage: $0 <server-ip> [user]" >&2
  exit 1
fi

echo "[local] Performing quick restart on ${SERVER_USER}@${SERVER_IP}…"
ssh "${SERVER_USER}@${SERVER_IP}" "CLEAN=${CLEAN:-0}" bash -s << 'REMOTE'
set -euo pipefail
REPO_DIR="/opt/zion/Zion"
cd "$REPO_DIR" || { echo "[remote] Repo not found at $REPO_DIR" >&2; exit 2; }

COMPOSE="docker compose"
if command -v docker-compose >/dev/null 2>&1; then
  COMPOSE="docker-compose"
fi

echo "[remote] Pulling latest repo changes…"
git fetch --all --prune
git reset --hard origin/master

echo "[remote] CLEAN mode: ${CLEAN:-0} (1 = remove volumes)"
if [[ "${CLEAN:-0}" == "1" ]]; then
  echo "[remote] Bringing stack down and removing volumes…"
  $COMPOSE -f docker/compose.pool-seeds.yml down -v || true
fi

echo "[remote] Restarting services…"
$COMPOSE -f docker/compose.pool-seeds.yml up -d --force-recreate seed1 seed2 redis rpc-shim uzi-pool

echo "[remote] Services status:"
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | (head -n1; grep -E 'zion-(seed1|seed2|redis|rpc-shim|uzi-pool)') || true

echo "[remote] Shim health:"
curl -s http://localhost:18089/ || true; echo
REMOTE

echo "[local] Quick restart complete."

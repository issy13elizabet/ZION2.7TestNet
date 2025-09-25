#!/usr/bin/env bash
# Note: ensure executable (chmod +x) to avoid exit code 126.
set -euo pipefail

# Remote redeploy of seed nodes, rpc-shim and uzi-pool over SSH
# Usage: ./scripts/ssh-redeploy-pool.sh <server-ip> [user]

SERVER_IP="${1:-}"
SERVER_USER="${2:-root}"
PUSH_LOCAL="${PUSH_LOCAL:-1}" # set to 0 to skip copying local changes
CLEAN_REDEPLOY="${CLEAN:-0}"   # set CLEAN=1 to nuke volumes and restart fresh

if [[ -z "${SERVER_IP}" ]]; then
  echo "Usage: $0 <server-ip> [user]" >&2
  exit 1
fi

REMOTE_BASE="/opt/zion"
REMOTE_REPO_DIR="$REMOTE_BASE/Zion"
REPO_URL="https://github.com/Yose144/Zion.git"

echo "[ssh] Testing SSH connectivity to ${SERVER_USER}@${SERVER_IP}..."
if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "${SERVER_USER}@${SERVER_IP}" 'echo SSH_OK' 2>/dev/null | grep -q SSH_OK; then
  echo "[ssh] Passwordless SSH not yet configured."
  echo "[ssh] Tip: run ./scripts/ssh-key-setup.sh ${SERVER_IP} ${SERVER_USER} to install your SSH key and enable multiplexing."
  echo "[ssh] Or use password-based deployment: ./scripts/deploy-ssh-pool.sh ${SERVER_IP} ${SERVER_USER}" >&2
  exit 2
fi

echo "[ssh] Preparing remote workspace at ${REMOTE_REPO_DIR}..."
ssh "${SERVER_USER}@${SERVER_IP}" "CLEAN=${CLEAN_REDEPLOY}" bash -s << 'REMOTE_CMDS'
set -euo pipefail
REMOTE_BASE="/opt/zion"
REMOTE_REPO_DIR="$REMOTE_BASE/Zion"
REPO_URL="https://github.com/Yose144/Zion.git"

sudo mkdir -p "$REMOTE_BASE"
sudo chown -R "$USER":"$USER" "$REMOTE_BASE"

if [[ -d "$REMOTE_REPO_DIR/.git" ]]; then
  echo "[remote] Repo exists, pulling latest…"
  git -C "$REMOTE_REPO_DIR" fetch --all --prune
  git -C "$REMOTE_REPO_DIR" reset --hard origin/master
  # No submodules required (core vendored)
else
  echo "[remote] Cloning repo fresh…"
  git clone "$REPO_URL" "$REMOTE_REPO_DIR"
  # No submodules required (core vendored)
fi

cd "$REMOTE_REPO_DIR"

echo "[remote] CLEAN mode: ${CLEAN:-0} (1 = remove volumes)"
if [[ "${CLEAN:-0}" == "1" ]]; then
  echo "[remote] Stopping stack and removing volumes (seed1-data, seed2-data, pool-data)…"
  docker compose -f docker/compose.pool-seeds.yml down -v || true
  # Safety: also try volume names directly in case compose names differ
  docker volume rm seed1-data seed2-data pool-data >/dev/null 2>&1 || true
  # Remove potentially conflicting containers
  for c in zion-uzi-pool zion-rpc-shim zion-seed1 zion-seed2 zion-redis; do
    if docker ps -a --format '{{.Names}}' | grep -qx "$c"; then
      echo "[remote] Removing leftover container $c"; docker rm -f "$c" || true;
    fi
  done
fi

echo "[remote] Ensuring docker network 'zion-seeds' exists…"
docker network create zion-seeds >/dev/null 2>&1 || true

echo "[remote] Checking daemon image tag…"
if ! docker image inspect zion:production-fixed >/dev/null 2>&1; then
  if docker image inspect zion:production >/dev/null 2>&1; then
    echo "[remote] Tagging zion:production -> zion:production-fixed"
    docker image tag zion:production zion:production-fixed
  else
    echo "[remote] WARNING: Neither zion:production-fixed nor zion:production is present. Seed containers may fail."
  fi
fi

echo "[remote] Building uzi-pool image (zion:uzi-pool)…"
docker build -t zion:uzi-pool -f docker/uzi-pool/Dockerfile .

echo "[remote] Building rpc-shim (no cache)…"
docker compose -f docker/compose.pool-seeds.yml build --no-cache rpc-shim

echo "[remote] Starting core services (seed1, seed2, redis)…"
docker compose -f docker/compose.pool-seeds.yml up -d seed1 seed2 redis

echo "[remote] Restarting rpc-shim & uzi-pool…"
# Ensure no name conflicts block recreation
if docker ps -a --format '{{.Names}}' | grep -qx 'zion-uzi-pool'; then docker rm -f zion-uzi-pool || true; fi
if docker ps -a --format '{{.Names}}' | grep -qx 'zion-rpc-shim'; then docker rm -f zion-rpc-shim || true; fi
docker compose -f docker/compose.pool-seeds.yml up -d --force-recreate rpc-shim uzi-pool

echo "[remote] Current service states:"
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | (head -n1; grep -E 'zion-(seed1|seed2|redis|rpc-shim|uzi-pool)') || true

echo "[remote] Probing shim health:"
curl -s http://localhost:18089/ || true
echo
REMOTE_CMDS

# Optionally push local changes to remote (targeted files only), then rebuild seeds & shim
if [[ "${PUSH_LOCAL}" == "1" ]]; then
  echo "[ssh] Pushing local changes (targeted files) to remote..."
  scp -q \
    "${PWD}/zion-cryptonote/src/Rpc/RpcServer.cpp" \
    "${SERVER_USER}@${SERVER_IP}:${REMOTE_REPO_DIR}/zion-cryptonote/src/Rpc/RpcServer.cpp" || true
  scp -q \
    "${PWD}/docker/compose.pool-seeds.yml" \
    "${SERVER_USER}@${SERVER_IP}:${REMOTE_REPO_DIR}/docker/compose.pool-seeds.yml" || true
  scp -q \
    "${PWD}/adapters/zion-rpc-shim/server.js" \
    "${SERVER_USER}@${SERVER_IP}:${REMOTE_REPO_DIR}/adapters/zion-rpc-shim/server.js" || true
  scp -q \
    "${PWD}/adapters/uzi-pool-config/config.json" \
    "${SERVER_USER}@${SERVER_IP}:${REMOTE_REPO_DIR}/adapters/uzi-pool-config/config.json" || true

  echo "[ssh] Rebuilding seeds and rpc-shim with local changes..."
  ssh "${SERVER_USER}@${SERVER_IP}" bash -s << 'REMOTE_CMDS_2'
set -euo pipefail
cd /opt/zion/Zion
docker compose -f docker/compose.pool-seeds.yml build --no-cache seed1 seed2 rpc-shim
docker compose -f docker/compose.pool-seeds.yml up -d seed1 seed2 rpc-shim
echo "[remote] Services restarted with local changes."
REMOTE_CMDS_2
fi

echo "[ssh] Remote redeploy completed. You can follow logs with: ssh ${SERVER_USER}@${SERVER_IP} 'docker logs -f zion-uzi-pool'"

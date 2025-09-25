#!/usr/bin/env bash
set -euo pipefail

# Password-based SSH deployment of the pool stack (port 3333), seeds, redis, rpc-shim, and optional test miner.
# Usage: ./scripts/deploy-ssh-pool.sh <server-ip> [user]

SERVER_IP="${1:-}"
SERVER_USER="${2:-root}"

if [[ -z "${SERVER_IP}" ]]; then
  echo "Usage: $0 <server-ip> [user]" >&2
  exit 1
fi

echo "🚀 ZION Pool SSH Deployment to ${SERVER_USER}@${SERVER_IP}"

# Build a small tarball with only what's needed in case of fallback usage
PKG="zion-pool-ssh-deploy.tar.gz"
tar -czf "$PKG" docker/ adapters/ scripts/ docker/compose.pool-seeds.yml docker/uzi-pool/Dockerfile docker/compose.test-miner.yml || true

echo "⬆️  Uploading helper package (for fallback only)…"
scp "$PKG" "${SERVER_USER}@${SERVER_IP}:/tmp/" || true

echo "🔗 Connecting to server (you may be prompted for password)…"
ssh "${SERVER_USER}@${SERVER_IP}" << 'REMOTE'
set -euo pipefail

echo "[remote] Preparing system (Docker, Compose, Git, Curl)…"
apt-get update -y
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sh
  systemctl enable docker; systemctl start docker
fi

# Prefer docker-compose v2 standalone if not present, otherwise rely on plugin 'docker compose'
if ! command -v docker-compose >/dev/null 2>&1; then
  DC_BIN="/usr/local/bin/docker-compose"
  if ! command -v docker compose >/dev/null 2>&1; then
    echo "[remote] Installing docker-compose v2…"
    curl -L "https://github.com/docker/compose/releases/download/v2.21.0/docker-compose-$(uname -s)-$(uname -m)" -o "$DC_BIN"
    chmod +x "$DC_BIN"
  fi
fi

if ! command -v git >/dev/null 2>&1; then
  apt-get install -y git
fi
if ! command -v curl >/dev/null 2>&1; then
  apt-get install -y curl
fi

COMPOSE_CMD="docker compose"
if command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD="docker-compose"
fi

BASE="/opt/zion"
REPO_DIR="$BASE/Zion"
REPO_URL="https://github.com/Yose144/Zion.git"

mkdir -p "$BASE"
cd "$BASE"

if [[ -d "$REPO_DIR/.git" ]]; then
  echo "[remote] Repo exists, updating…"
  git -C "$REPO_DIR" fetch --all --prune
  git -C "$REPO_DIR" reset --hard origin/master
  # No submodules required (core vendored)
else
  echo "[remote] Cloning repo…"
  git clone "$REPO_URL" "$REPO_DIR"
  # No submodules required (core vendored)
fi

cd "$REPO_DIR"

echo "[remote] Ensure docker network 'zion-seeds'…"
docker network create zion-seeds >/dev/null 2>&1 || true

echo "[remote] Tagging daemon image if needed…"
if ! docker image inspect zion:production-fixed >/dev/null 2>&1; then
  if docker image inspect zion:production >/dev/null 2>&1; then
    docker image tag zion:production zion:production-fixed
  else
    echo "[remote] WARNING: Missing zion:production-fixed and zion:production images. Seeds may fail unless present."
  fi
fi

echo "[remote] Building uzi-pool image…"
docker build -t zion:uzi-pool -f docker/uzi-pool/Dockerfile .

echo "[remote] Building rpc-shim…"
$COMPOSE_CMD -f docker/compose.pool-seeds.yml build rpc-shim || true

echo "[remote] Starting seed1, seed2, redis…"
$COMPOSE_CMD -f docker/compose.pool-seeds.yml up -d seed1 seed2 redis

echo "[remote] Starting rpc-shim & uzi-pool (port 3333)…"
$COMPOSE_CMD -f docker/compose.pool-seeds.yml up -d rpc-shim uzi-pool

echo "[remote] Services status:"
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | (head -n1; grep -E 'zion-(seed1|seed2|redis|rpc-shim|uzi-pool)') || true

echo "[remote] Shim health:"
curl -s http://localhost:18089/ || true; echo

echo "[remote] Launching test miner (xmrig)…"
$COMPOSE_CMD -f docker/compose.test-miner.yml up -d xmrig-test || true
sleep 3
echo "--- xmrig last logs ---"
docker logs --tail=50 zion-xmrig-test || true

echo "--- pool last logs ---"
docker logs --tail=100 zion-uzi-pool || true

echo "[remote] Done."
REMOTE

echo "🟢 Deployment complete. Follow live logs:"
echo "  ssh ${SERVER_USER}@${SERVER_IP} 'docker logs -f zion-uzi-pool'"
echo "  ssh ${SERVER_USER}@${SERVER_IP} 'docker logs -f zion-xmrig-test'"

#!/usr/bin/env bash
set -euo pipefail

HOST=${1:-"91.98.122.165"}
USER=${2:-"root"}

echo "==> Deploying Zion pool + seeds to $USER@$HOST"

ssh "$USER@$HOST" bash -lc '
  set -euo pipefail
  echo "[server] Ensuring docker + compose present"
  if ! command -v docker >/dev/null 2>&1; then
    apt-get update && apt-get install -y docker.io
    systemctl enable --now docker
  fi
  if ! docker compose version >/dev/null 2>&1; then
    apt-get update && apt-get install -y docker-compose-plugin || true
  fi

  echo "[server] Creating /opt/zion-pool workspace"
  mkdir -p /opt/zion-pool

  echo "[server] Writing compose file"
  cat > /opt/zion-pool/compose.yml <<EOF
version: "3.8"
services:
  seed1:
    image: zion:production-fixed
    container_name: zion-seed1
    restart: unless-stopped
    environment:
      ZION_MODE: daemon
      ZION_LOG_LEVEL: info
      P2P_PORT: "18080"
      RPC_PORT: "18081"
    volumes:
      - seed1-data:/home/zion/.zion
    networks: [zion-seeds]

  seed2:
    image: zion:production-fixed
    container_name: zion-seed2
    restart: unless-stopped
    environment:
      ZION_MODE: daemon
      ZION_LOG_LEVEL: info
      P2P_PORT: "18080"
      RPC_PORT: "18081"
    volumes:
      - seed2-data:/home/zion/.zion
    depends_on: [seed1]
    networks: [zion-seeds]

  pool:
    image: zion:pool-latest
    container_name: zion-pool
    restart: unless-stopped
    environment:
      ZION_MODE: pool
      ZION_LOG_LEVEL: info
      POOL_PORT: "3333"
      POOL_BIND: 0.0.0.0
      POOL_DIFFICULTY: "1000"
      POOL_FEE: "1"
    volumes:
      - pool-data:/home/zion/.zion
    depends_on: [seed1, seed2]
    ports:
      - "3333:3333"
    networks: [zion-seeds]

volumes:
  seed1-data: { driver: local }
  seed2-data: { driver: local }
  pool-data:  { driver: local }

networks:
  zion-seeds: { name: zion-seeds, driver: bridge }
EOF

  echo "[server] Building pool image from provided Dockerfile (if present)"
  if [ -d /opt/zion-src ]; then rm -rf /opt/zion-src; fi
  mkdir -p /opt/zion-src
  # Expect caller to rsync sources into /opt/zion-src; if not, try to clone minimal context
  if [ ! -f /opt/zion-src/Dockerfile ]; then
    echo "[server] No sources synced, cloning minimal context..."
    git clone https://github.com/Yose144/Zion.git /opt/zion-src || true
  fi
  cd /opt/zion-src
  docker build -t zion:pool-latest .

  echo "[server] Starting stack"
  cd /opt/zion-pool
  docker compose up -d

  echo "[server] Status:"
  docker compose ps
'

echo "==> Done. Pool should be on tcp://$HOST:3333"

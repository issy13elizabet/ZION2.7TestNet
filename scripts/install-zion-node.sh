#!/usr/bin/env bash
set -euo pipefail

# Simple installer for running a ZION node via Docker Compose
# Requirements: Docker + Docker Compose plugin

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
DOCKER_DIR="$ROOT_DIR/docker"
COMPOSE_FILE="$DOCKER_DIR/compose.single-node.yml"
CONF_DIR="$DOCKER_DIR/node"

mkdir -p "$CONF_DIR" "$CONF_DIR/data" "$CONF_DIR/logs" "$CONF_DIR/config"
# Seed default config if missing
if [[ ! -f "$CONF_DIR/config/zion.conf" ]]; then
  cp "$DOCKER_DIR/node/zion.conf" "$CONF_DIR/config/zion.conf"
fi

# Build image (can be skipped if image prebuilt is provided/pulled)
if ! docker image inspect zion:node-latest > /dev/null 2>&1; then
  echo "Building zion:node-latest image..."
  docker compose -f "$COMPOSE_FILE" build
fi

echo "Starting ZION node..."
docker compose -f "$COMPOSE_FILE" up -d

echo "Done. Useful checks:"
echo "  - curl http://localhost:18081/getheight"
echo "  - docker logs -f zion-node"

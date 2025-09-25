#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
DOCKER_DIR="$ROOT_DIR/docker"
COMPOSE_FILE="$DOCKER_DIR/compose.single-node.yml"

echo "Stopping ZION node..."
docker compose -f "$COMPOSE_FILE" down --remove-orphans

read -r -p "Remove persistent data in docker/node/data and logs? [y/N] " ans
case "${ans:-N}" in
  [yY][eE][sS]|[yY])
    rm -rf "$DOCKER_DIR/node/data" "$DOCKER_DIR/node/logs"
    echo "Data and logs removed."
    ;;
  *)
    echo "Data preserved in $DOCKER_DIR/node/."
    ;;
 esac

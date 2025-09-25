#!/usr/bin/env bash
set -euo pipefail

# Ensure Docker is running
if ! docker info >/dev/null 2>&1; then
  echo "Docker daemon is not running. Please start Docker Desktop and rerun." >&2
  exit 1
fi

# Prune builder cache, images, containers, networks, and volumes
yes | docker builder prune -af || true
yes | docker system prune -a --volumes || true

# Remove dangling volumes just in case
docker volume ls -q | xargs -r docker volume rm -f || true

docker system df || true

echo "Docker cleanup done."
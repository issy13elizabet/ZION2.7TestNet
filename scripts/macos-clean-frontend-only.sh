#!/usr/bin/env bash

# macOS cleanup for frontend-only workflow
# - Removes Zion-specific Docker containers, images, volumes, and the zion-seeds network
# - Leaves the system otherwise intact (no global prune by default)
# Usage:
#   bash scripts/macos-clean-frontend-only.sh [--prune]
#   --prune : also prune dangling images and builder cache

set -euo pipefail

echo "[clean] Frontend-only cleanup started"

if ! command -v docker >/dev/null 2>&1; then
  echo "[clean] ERROR: docker is not installed or not in PATH" >&2
  exit 1
fi

DO_PRUNE=0
if [[ "${1-}" == "--prune" ]]; then
  DO_PRUNE=1
fi

# Remove Zion containers (names starting with zion-)
ZION_CONTAINERS=$(docker ps -a --format '{{.ID}} {{.Names}}' | awk '$2 ~ /^zion-/ {print $1}')
if [[ -n "${ZION_CONTAINERS}" ]]; then
  echo "[clean] Removing containers: ${ZION_CONTAINERS}"
  docker rm -f ${ZION_CONTAINERS}
else
  echo "[clean] No zion-* containers to remove"
fi

# Remove Zion images (repository 'zion')
ZION_IMAGES=$(docker images --format '{{.Repository}}:{{.Tag}} {{.ID}}' | awk '$1 ~ /^zion:/ {print $2}')
if [[ -n "${ZION_IMAGES}" ]]; then
  echo "[clean] Removing images: ${ZION_IMAGES}"
  docker rmi -f ${ZION_IMAGES}
else
  echo "[clean] No zion:* images to remove"
fi

# Remove known project volumes and any volumes matching zion_*
echo "[clean] Checking project volumes"
VOL_LIST=(
  docker_pool-data
  docker_seed1-data
  docker_seed2-data
  docker_wallet-data
  docker_zion_data
  zion_pool-data
  zion_seed-data
)

# Add any volumes that start with zion_
ZION_MATCH_VOLS=$(docker volume ls --format '{{.Name}}' | grep -E '^zion_' || true)
if [[ -n "${ZION_MATCH_VOLS}" ]]; then
  while IFS= read -r v; do VOL_LIST+=("$v"); done <<< "${ZION_MATCH_VOLS}"
fi

for V in "${VOL_LIST[@]}"; do
  if docker volume inspect "$V" >/dev/null 2>&1; then
    echo "[clean] Removing volume: $V"
    docker volume rm -f "$V" >/dev/null 2>&1 || true
  fi
done

# Remove network zion-seeds if present
if docker network inspect zion-seeds >/dev/null 2>&1; then
  echo "[clean] Removing docker network: zion-seeds"
  docker network rm zion-seeds >/dev/null 2>&1 || true
fi

if [[ ${DO_PRUNE} -eq 1 ]]; then
  echo "[clean] Pruning dangling images and builder cache"
  docker image prune -f >/dev/null 2>&1 || true
  docker builder prune -af >/dev/null 2>&1 || true
fi

echo "[clean] Resulting Docker disk usage:"
docker system df || true

echo "[clean] Done. System is ready for frontend-only (Next.js) development."

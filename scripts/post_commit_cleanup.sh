#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[auto-clean] Running workspace cleanup..."
scripts/cleanup_workspace.sh || true

echo "[auto-clean] Running docker cleanup (may take time)..."
scripts/cleanup_docker.sh || true

echo "[auto-clean] Done."

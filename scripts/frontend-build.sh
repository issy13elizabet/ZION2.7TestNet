#!/usr/bin/env bash

# Frontend production build helper
# - Creates/updates frontend/.env.production.local for Ryzen backend
# - Runs `npm ci` (or install) and `npm run build`
#
# Usage:
#   bash scripts/frontend-build.sh [--host <ip-or-dns>] [--pool-port 3333] [--shim-port 18089] [--adapter-port 18099]

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
FRONT_DIR="$ROOT_DIR/frontend"

HOST="91.98.122.165"
POOL_PORT=3333
SHIM_PORT=18089
ADAPTER_PORT=18099

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2;;
    --pool-port) POOL_PORT="$2"; shift 2;;
    --shim-port) SHIM_PORT="$2"; shift 2;;
    --adapter-port) ADAPTER_PORT="$2"; shift 2;;
    *) echo "Unknown option: $1" >&2; exit 1;;
  esac
done

cd "$FRONT_DIR"

echo "[frontend-build] Writing .env.production.local"
cat > .env.production.local <<ENV
NEXT_PUBLIC_ZION_HOST=$HOST
NEXT_PUBLIC_ZION_POOL_PORT=$POOL_PORT
NEXT_PUBLIC_ZION_SHIM_PORT=$SHIM_PORT

ZION_HOST=$HOST
ZION_POOL_PORT=$POOL_PORT
ZION_SHIM_PORT=$SHIM_PORT
ZION_ADAPTER_PORT=$ADAPTER_PORT
ENV

echo "[frontend-build] Installing deps"
if [[ -f package-lock.json ]]; then
  npm ci
else
  npm install
fi

echo "[frontend-build] Building Next.js (production)"
npm run build

echo "[frontend-build] Build complete"

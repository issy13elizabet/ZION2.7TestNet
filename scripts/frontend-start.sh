#!/usr/bin/env bash

# Frontend production start helper
# - Uses prior build in frontend/.next
# - Exposes env via .env.production.local (created by frontend-build.sh) but also accepts overrides
#
# Usage:
#   bash scripts/frontend-start.sh [--port 3000]

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
FRONT_DIR="$ROOT_DIR/frontend"
PORT=3000

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2;;
    *) echo "Unknown option: $1" >&2; exit 1;;
  esac
done

cd "$FRONT_DIR"

if [[ ! -d .next ]]; then
  echo "ERROR: .next build output not found. Run scripts/frontend-build.sh first." >&2
  exit 1
fi

echo "[frontend-start] Starting Next.js on port ${PORT}"
PORT=${PORT} npm run start

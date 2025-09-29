#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VERSION_FILE="$ROOT_DIR/VERSION"
if [[ ! -f "$VERSION_FILE" ]]; then
  echo "[ERR] Chybi VERSION" >&2; exit 1; fi
ROOT_VERSION="$(tr -d '\n' < "$VERSION_FILE")"
echo "[INFO] Root version: $ROOT_VERSION"
if [[ -f "$ROOT_DIR/core/package.json" ]]; then
  CORE_VERSION=$(grep '"version"' "$ROOT_DIR/core/package.json" | head -1 | sed -E 's/.*"version" *: *"([^"]+)".*/\1/')
  if [[ "$CORE_VERSION" != "$ROOT_VERSION" ]]; then
    echo "[WARN] core version ($CORE_VERSION) != root ($ROOT_VERSION)"; else echo "[OK] core version OK"; fi
fi
exit 0

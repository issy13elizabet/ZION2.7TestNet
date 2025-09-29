#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VERSION_FILE="$ROOT_DIR/VERSION"

if [[ ! -f "$VERSION_FILE" ]]; then
  echo "[ERR] Chybi soubor VERSION v rootu." >&2
  exit 1
fi

ROOT_VERSION="$(tr -d '\n' < "$VERSION_FILE")"

echo "[INFO] Root verze: $ROOT_VERSION"

# Kontrola zion-core
CORE_PKG="$ROOT_DIR/zion-core/package.json"
if [[ -f "$CORE_PKG" ]]; then
  CORE_VERSION=$(grep '"version"' "$CORE_PKG" | head -1 | sed -E 's/.*"version" *: *"([^"]+)".*/\1/')
  if [[ "$CORE_VERSION" != "$ROOT_VERSION" ]]; then
    echo "[WARN] zion-core version ($CORE_VERSION) != root VERSION ($ROOT_VERSION)" >&2
  else
    echo "[OK] zion-core verze je synchronizovana."
  fi
fi

# Miner 1.4.0 makro
MINER_CMAKE="$ROOT_DIR/zion-miner-1.4.0/CMakeLists.txt"
if grep -q 'add_definitions(-DZION_VERSION' "$MINER_CMAKE"; then
  MINER_VERSION=$(grep 'add_definitions(-DZION_VERSION' "$MINER_CMAKE" | sed -E 's/.*ZION_VERSION="([^"]+)".*/\1/')
  if [[ "$MINER_VERSION" != "$ROOT_VERSION" ]]; then
    echo "[WARN] miner 1.4.0 ZION_VERSION ($MINER_VERSION) != root VERSION ($ROOT_VERSION)" >&2
  else
    echo "[OK] miner 1.4.0 verze je synchronizovana."
  fi
fi

exit 0

#!/usr/bin/env bash
set -euo pipefail

OS_LABEL="$1" # ubuntu-latest | macos-latest

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
BUILD_DIR="$ROOT_DIR/build"
DIST_DIR="$ROOT_DIR/dist"

mkdir -p "$DIST_DIR"

# Determine target labels
case "$OS_LABEL" in
  ubuntu-latest)
    PLATFORM="linux"
    ;;
  macos-latest)
    PLATFORM="macos"
    ;;
  *)
    PLATFORM="unknown"
    ;;
 esac

ARCH=$(uname -m)
PKG_NAME="zion-${PLATFORM}-${ARCH}"
PKG_DIR="$ROOT_DIR/$PKG_NAME"
rm -rf "$PKG_DIR"
mkdir -p "$PKG_DIR/bin"

# Copy binaries
cp "$BUILD_DIR/ziond" "$PKG_DIR/bin/" || true
cp "$BUILD_DIR/zion_miner" "$PKG_DIR/bin/" || true
cp "$BUILD_DIR/zion_wallet" "$PKG_DIR/bin/" || true
cp "$BUILD_DIR/zion_genesis" "$PKG_DIR/bin/" || true

# Copy sample configs
mkdir -p "$PKG_DIR/config"
cp "$ROOT_DIR/config/mainnet.conf" "$PKG_DIR/config/" || true

# Readme
cat > "$PKG_DIR/README.txt" << EOF
ZION Cryptocurrency - Release Package

Binaries:
  bin/ziond         - Daemon (node)
  bin/zion_miner    - Miner
  bin/zion_wallet   - Wallet (CLI)
  bin/zion_genesis  - Genesis helper tool

Usage:
  1) Edit config/mainnet.conf and set [blockchain].genesis_hash
  2) Start node:
     ./bin/ziond --config=./config/mainnet.conf --datadir=./data

EOF

# Tarball
TARBALL="$DIST_DIR/${PKG_NAME}.tar.gz"
rm -f "$TARBALL"
( cd "$ROOT_DIR" && tar -czf "$TARBALL" "$PKG_NAME" )

# Cleanup staging dir
rm -rf "$PKG_DIR"

echo "Created: $TARBALL"

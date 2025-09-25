#!/usr/bin/env bash
set -euo pipefail

# Fetch latest XMRig release assets or specific version and place into mining/ directory.
# Usage:
#   ./scripts/fetch_xmrig.sh                # fetch default version
#   XMRIG_VERSION=6.21.3 ./scripts/fetch_xmrig.sh

XMRIG_VERSION="${XMRIG_VERSION:-6.21.3}"
DEST_DIR="$(cd "$(dirname "$0")/.." && pwd)/mining"
mkdir -p "$DEST_DIR"

echo "Fetching XMRig v$XMRIG_VERSION assets..."

BASE_URL="https://github.com/xmrig/xmrig/releases/download/v$XMRIG_VERSION"

assets=(
  "xmrig-$XMRIG_VERSION-linux-static-x64.tar.gz"
  "xmrig-$XMRIG_VERSION-macos-arm64.tar.gz"
  "xmrig-$XMRIG_VERSION-macos-x64.tar.gz"
  "xmrig-$XMRIG_VERSION-msvc-win64.zip"
)

for a in "${assets[@]}"; do
  url="$BASE_URL/$a"
  echo "- $url"
  curl -fL "$url" -o "$DEST_DIR/$a"
done

echo "Done. Assets saved to $DEST_DIR"

#!/usr/bin/env bash
# Backup wallet container files with timestamp to ./backups/
# Usage: scripts/backup-wallet.sh [container=z ion-walletd] [dest=./backups]

set -euo pipefail
CONTAINER=${1:-zion-walletd}
DEST=${2:-./backups}
TS=$(date -u +"%Y%m%d-%H%M%S")
mkdir -p "$DEST"
ARCHIVE="$DEST/zion-backup-${TS}.tar.gz"

echo "Creating backup from container: $CONTAINER -> $ARCHIVE"
# Common wallet paths; adjust if needed
FILES=(
  /home/zion/.zion/pool.wallet
  /home/zion/.zion/*.keys
  /home/zion/.zion/*.log
)

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

for f in "${FILES[@]}"; do
  docker cp "$CONTAINER:$f" "$tmpdir" 2>/dev/null || true
done

tar -C "$tmpdir" -czf "$ARCHIVE" .
echo "Backup created: $ARCHIVE"

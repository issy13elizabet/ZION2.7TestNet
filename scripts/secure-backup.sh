#!/usr/bin/env bash
# Securely back up wallet files from a running container and encrypt them with GPG
# Usage: scripts/secure-backup.sh [container=zion-walletd] [recipient=KEYID] [dest=logs/runtime/$(date -u +%Y%m%dT%H%M%SZ)/vault]
set -euo pipefail
CONTAINER=${1:-zion-walletd}
RECIPIENT=${2:-}
TS=${TS:-$(date -u +%Y%m%dT%H%M%SZ)}
DEST=${3:-logs/runtime/${TS}/vault}
if [[ -z "$RECIPIENT" ]]; then
  echo "Recipient KEYID/EMAIL required (GPG public key)." >&2; exit 2
fi
mkdir -p "$DEST"
TMP=$(mktemp -d)
trap 'shred -uz $TMP/* 2>/dev/null || true; rm -rf "$TMP"' EXIT
FILES=(
  /home/zion/.zion/pool.wallet
  /home/zion/.zion/*.keys
  /home/zion/.zion/*.log
)
for f in "${FILES[@]}"; do
  docker cp "$CONTAINER:$f" "$TMP" 2>/dev/null || true
done
ARCHIVE="$TMP/zion-wallet-${TS}.tar.gz"
tar -C "$TMP" -czf "$ARCHIVE" .
OUT="$DEST/zion-wallet-${TS}.tar.gz.gpg"
if ! command -v gpg >/dev/null 2>&1; then
  echo "gpg not found. Install GnuPG." >&2; exit 3
fi
gpg --yes --batch --recipient "$RECIPIENT" --encrypt --output "$OUT" "$ARCHIVE"
echo "Encrypted backup created: $OUT"

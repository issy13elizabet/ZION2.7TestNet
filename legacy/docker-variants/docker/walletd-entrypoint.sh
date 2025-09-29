#!/bin/sh
set -e

WALLET_FILE=${WALLET_FILE:-/home/zion/.zion/pool.wallet}
WALLET_PASS=${WALLET_PASS:-testpass}
DAEMON_HOST=${DAEMON_HOST:-zion-seed1}
DAEMON_PORT=${DAEMON_PORT:-18081}

if [ ! -f "$WALLET_FILE" ]; then
  echo "[walletd] Wallet file not found at $WALLET_FILE. Creating new container..."
  # Create directory if missing
  mkdir -p "$(dirname "$WALLET_FILE")"
  # Generate a new wallet container; Zion wallet CLI flags may differ, try common CryptoNote pattern
  /usr/local/bin/zion_wallet \
    --generate-new-wallet "$WALLET_FILE" \
    --password "$WALLET_PASS" \
    --daemon-host "$DAEMON_HOST" \
    --daemon-port "$DAEMON_PORT" \
    --command exit || true
fi

exec /usr/local/bin/zion_walletd \
  --bind-address 0.0.0.0 \
  --bind-port 8070 \
  --container-file "$WALLET_FILE" \
  --container-password "$WALLET_PASS" \
  --daemon-address "$DAEMON_HOST" \
  --daemon-port "$DAEMON_PORT"

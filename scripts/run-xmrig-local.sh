#!/usr/bin/env bash
set -euo pipefail

# Run XMRig locally on macOS (Apple Silicon) against given pool
# Usage:
#   scripts/run-xmrig-local.sh [--wallet <addr>] [--pool <host:port>] [--tunnel] [--algo rx/0] [--threads N]
# Defaults:
#   wallet: demo address in repo (replace!)
#   pool: 91.98.122.165:3333
#   algo: rx/0

WALLET="Z7v1oJtV3pQAYt2oR9wA3sL5xTnQmBkR6yU8iN2mK4hP9eD3cF7aJ1kL2pQ9wE7rT6"
POOL="91.98.122.165:3333"
ALGO="rx/0"
THREADS=""
TUNNEL=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wallet) WALLET=${2:-}; shift 2 ;;
    --pool) POOL=${2:-}; shift 2 ;;
    --algo) ALGO=${2:-}; shift 2 ;;
    --threads) THREADS=${2:-}; shift 2 ;;
    --tunnel) TUNNEL=true; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if ! command -v xmrig >/dev/null 2>&1; then
  echo "[local] XMRig not found. Install via Homebrew: brew install xmrig" >&2
  exit 2
fi

TARGET="$POOL"
if $TUNNEL; then
  echo "[local] Establishing SSH tunnel localhost:3333 -> ${POOL} (requires SSH to server)…"
  # If pool is on our server, we tunnel to 3333 there; else simple local bind to remote host
  SERVER_HOST=$(echo "$POOL" | cut -d: -f1)
  SERVER_PORT=$(echo "$POOL" | cut -d: -f2)
  if [[ "$SERVER_PORT" != "3333" ]]; then
    echo "[warn] Tunnel assumes remote port 3333; override --pool to <server>:3333 for tunneling." >&2
  fi
  # Use ControlMaster if configured
  ssh -fN -L 3333:localhost:3333 "root@${SERVER_HOST}" || true
  TARGET="localhost:3333"
fi

set -x
exec xmrig \
  --url "stratum+tcp://${TARGET}" \
  --algo "$ALGO" \
  --user "$WALLET" \
  --pass x \
  --keepalive \
  ${THREADS:+--threads "$THREADS"} \
  --rig-id M1LOCAL \
  --print-time 5 \
  --donate-level 0

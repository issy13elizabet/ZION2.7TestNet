#!/usr/bin/env bash

# Collect runtime logs from remote Docker containers into local logs/runtime/<timestamp>/
# Usage:
#   scripts/collect_runtime_logs.sh <host> [user] [since]
# Examples:
#   scripts/collect_runtime_logs.sh 91.98.122.165
#   scripts/collect_runtime_logs.sh 91.98.122.165 root 3h

set -euo pipefail

HOST="${1:-}"
USER="${2:-root}"
SINCE_ARG="${3:-2h}"

if [[ -z "$HOST" ]]; then
  echo "Usage: $0 <host> [user] [since]" >&2
  exit 1
fi

TS=$(date -u +"%Y%m%dT%H%M%SZ")
DEST_DIR="logs/runtime/${TS}"
mkdir -p "$DEST_DIR"

echo "Collecting logs from ${USER}@${HOST} into ${DEST_DIR} (since=${SINCE_ARG})"

# Save docker ps snapshot
ssh -o ConnectTimeout=8 "${USER}@${HOST}" \
  "docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'" \
  > "${DEST_DIR}/docker-ps.txt" || true

echo "Saved docker ps to ${DEST_DIR}/docker-ps.txt"

services=(
  zion-uzi-pool
  zion-rpc-shim
  zion-seed1
  zion-seed2
  zion-redis
  zion-production
  zion-xmrig-test
)

for svc in "${services[@]}"; do
  echo "Fetching logs for ${svc}..."
  # Use --since to limit volume; fall back without since if unsupported
  if ssh -o ConnectTimeout=8 "${USER}@${HOST}" "docker ps --format '{{.Names}}' | grep -qx ${svc}"; then
    if ssh -o ConnectTimeout=8 "${USER}@${HOST}" "docker logs --since ${SINCE_ARG} ${svc} >/dev/null 2>&1"; then
      ssh -o ConnectTimeout=8 "${USER}@${HOST}" "docker logs --since ${SINCE_ARG} ${svc} 2>&1" \
        > "${DEST_DIR}/${svc}.log" || true
    else
      ssh -o ConnectTimeout=8 "${USER}@${HOST}" "docker logs ${svc} 2>&1" \
        > "${DEST_DIR}/${svc}.log" || true
    fi
  else
    echo "Service ${svc} not running" > "${DEST_DIR}/${svc}.log"
  fi
done

echo "Done. Logs stored under ${DEST_DIR}"

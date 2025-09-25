#!/usr/bin/env bash
# collect-pool-diagnostics.sh
# Sběr diagnostiky přímo z hostu (spouští se lokálně a připojí přes SSH)
# Použití: ./scripts/collect-pool-diagnostics.sh root@91.98.122.165 /path/to/key -o ./diagnostics

set -euo pipefail
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 user@server [-i /path/to/key] [-o outdir]"
  exit 1
fi

TARGET=$1
shift
IDENT=""
OUTDIR="./diagnostics"
while [ "$#" -gt 0 ]; do
  case "$1" in
    -i) IDENT="$2"; shift 2; ;;
    -o) OUTDIR="$2"; shift 2; ;;
    *) echo "Unknown arg: $1"; exit 1; ;;
  esac
done

mkdir -p "$OUTDIR"
REMOTE_TMP="/tmp/pool-diagnostics-$(date +%s)"

SSH="ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=accept-new"
if [ -n "$IDENT" ]; then SSH="$SSH -i $IDENT"; fi
SCP="scp -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=accept-new"
if [ -n "$IDENT" ]; then SCP="$SCP -i $IDENT"; fi

echo "Running remote collection on $TARGET..."
$SSH $TARGET bash -lc "set -euo pipefail; mkdir -p $REMOTE_TMP; \
if docker ps --format '{{.Names}}' | grep -q 'zion-uzi-pool'; then docker logs --tail 2000 zion-uzi-pool > $REMOTE_TMP/uzi-pool.log 2>&1 || true; \
 docker exec zion-uzi-pool sh -lc 'cat /app/config.json' > $REMOTE_TMP/uzi-pool.config.json 2>/dev/null || true; \
 docker exec zion-uzi-pool sh -lc \"sed -n '1,2000p' /app/lib/pool.js\" > $REMOTE_TMP/uzi-pool.lib.pool.js 2>/dev/null || true; fi; \
 if command -v curl >/dev/null 2>&1; then curl -sS --max-time 3 http://127.0.0.1:18089/metrics.json > $REMOTE_TMP/rpc-shim.metrics.json || true; fi; \
 if docker ps --format '{{.Names}}' | grep -q 'zion-redis'; then docker exec zion-redis redis-cli -p 6379 --raw KEYS '*' > $REMOTE_TMP/redis-keys.txt 2>/dev/null || true; fi; \
 tar -czf /tmp/pool-diagnostics.tar.gz -C /tmp $(basename $REMOTE_TMP); echo /tmp/pool-diagnostics.tar.gz"

echo "Downloading archive to $OUTDIR..."
$SCP $TARGET:/tmp/pool-diagnostics.tar.gz "$OUTDIR/"
if [ $? -ne 0 ]; then echo "SCP failed"; exit 2; fi

echo "Archive downloaded to $OUTDIR/pool-diagnostics.tar.gz"

exit 0

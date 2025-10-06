#!/usr/bin/env bash
#
# ZION Pool Deployment Script
# Nasazení stratum_pool.py na server s restartom a základní diagnostikou
#

set -euo pipefail

# Konfigurace
SERVER="91.98.122.165"
SERVER_USER="root"
SERVER_PATH="/root/zion27/pool"
LOCAL_POOL_FILE="/media/maitreya/ZION1/2.7/pool/stratum_pool.py"

# Funkce pro timestampy
ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }

echo "$(ts) Starting ZION Pool deployment..."

# 1. Ověření lokálního souboru
if [[ ! -f "$LOCAL_POOL_FILE" ]]; then
    echo "ERROR: Local pool file not found: $LOCAL_POOL_FILE"
    exit 1
fi

echo "$(ts) Local pool file: $(wc -c < "$LOCAL_POOL_FILE") bytes"

# 2. Upload souboru
echo "$(ts) Uploading pool file to server..."
scp "$LOCAL_POOL_FILE" "$SERVER_USER@$SERVER:$SERVER_PATH/stratum_pool.py"

# 3. Backup staré verze na serveru
echo "$(ts) Creating backup on server..."
ssh "$SERVER_USER@$SERVER" "cd $SERVER_PATH && cp stratum_pool.py stratum_pool.py.bak.\$(date +%s) 2>/dev/null || true"

# 4. Zastavení starých procesů
echo "$(ts) Stopping old pool processes..."
ssh "$SERVER_USER@$SERVER" "pgrep -f stratum_pool.py && pgrep -f stratum_pool.py | xargs -r kill -TERM || true"
sleep 2
ssh "$SERVER_USER@$SERVER" "pgrep -f stratum_pool.py && pgrep -f stratum_pool.py | xargs -r kill -9 || true"

# 5. Rotace logu
echo "$(ts) Rotating log files..."
ssh "$SERVER_USER@$SERVER" "cd $SERVER_PATH && [ -f pool.out ] && mv pool.out pool.out.\$(date +%s) || true"

# 6. Start nového procesu
echo "$(ts) Starting new pool process..."
ssh "$SERVER_USER@$SERVER" "cd $SERVER_PATH && nohup python3 stratum_pool.py > pool.out 2>&1 & echo \$! > pool.pid"

# 7. Ověření startu
sleep 3
echo "$(ts) Verifying pool startup..."

PID=$(ssh "$SERVER_USER@$SERVER" "cat $SERVER_PATH/pool.pid 2>/dev/null || echo 'NO_PID'")
if [[ "$PID" == "NO_PID" ]]; then
    echo "ERROR: No PID file found"
    exit 1
fi

echo "$(ts) Pool PID: $PID"

# Ověření procesu
PROCESS_CHECK=$(ssh "$SERVER_USER@$SERVER" "ps -o pid,cmd -p $PID --no-headers 2>/dev/null || echo 'NOT_RUNNING'")
if [[ "$PROCESS_CHECK" == "NOT_RUNNING" ]]; then
    echo "ERROR: Process $PID not running"
    ssh "$SERVER_USER@$SERVER" "cd $SERVER_PATH && tail -n 20 pool.out"
    exit 1
fi

echo "$(ts) Process check: $PROCESS_CHECK"

# Ověření portu
PORT_CHECK=$(ssh "$SERVER_USER@$SERVER" "ss -ltnp | grep :3333 || echo 'PORT_NOT_LISTENING'")
if [[ "$PORT_CHECK" == "PORT_NOT_LISTENING" ]]; then
    echo "WARNING: Port 3333 not listening yet, checking logs..."
    ssh "$SERVER_USER@$SERVER" "cd $SERVER_PATH && tail -n 10 pool.out"
else
    echo "$(ts) Port 3333 is listening: $PORT_CHECK"
fi

# 8. Zobrazení posledních log řádků
echo "$(ts) Recent log output:"
ssh "$SERVER_USER@$SERVER" "cd $SERVER_PATH && tail -n 15 pool.out"

# 9. Kontrola Stratum listening zprávy
LISTENING_CHECK=$(ssh "$SERVER_USER@$SERVER" "cd $SERVER_PATH && grep -c 'Stratum listening' pool.out || echo '0'")
if [[ "$LISTENING_CHECK" -gt 0 ]]; then
    echo "$(ts) SUCCESS: Pool is listening for connections"
else
    echo "WARNING: 'Stratum listening' message not found in log yet"
fi

echo "$(ts) Deployment completed. Pool should be ready for testing."
echo ""
echo "To test with XMRig:"
echo "  timeout 90 xmrig --url $SERVER:3333 --user testWorker --algo rx/0 --keepalive --no-color"
echo ""
echo "To monitor logs:"
echo "  ssh $SERVER_USER@$SERVER 'cd $SERVER_PATH && tail -f pool.out'"
echo ""
echo "To check for VarDiff adjustments:"
echo "  ssh $SERVER_USER@$SERVER 'cd $SERVER_PATH && grep \"VarDiff adjusted\" pool.out'"
#!/usr/bin/env bash
#
# ZION Pool Simple Deployment Script
#

set -e

SERVER="91.98.122.165"
SERVER_USER="root"
SERVER_PATH="/root/zion27/pool"
LOCAL_POOL_FILE="/media/maitreya/ZION1/2.7/pool/stratum_pool.py"

echo "=== ZION Pool Deployment ==="

# 1. Upload
echo "1. Uploading pool file..."
scp "$LOCAL_POOL_FILE" "$SERVER_USER@$SERVER:$SERVER_PATH/stratum_pool.py"

# 2. Kill old processes (separate commands)
echo "2. Stopping old processes..."
ssh "$SERVER_USER@$SERVER" "pkill -f stratum_pool.py" || true
sleep 1

# 3. Backup log
echo "3. Rotating log..."
ssh "$SERVER_USER@$SERVER" "cd $SERVER_PATH && mv pool.out pool.out.old" || true

# 4. Start new process
echo "4. Starting pool..."
ssh "$SERVER_USER@$SERVER" "cd $SERVER_PATH && python3 stratum_pool.py > pool.out 2>&1 &"

# 5. Wait and check
sleep 3
echo "5. Checking startup..."
ssh "$SERVER_USER@$SERVER" "cd $SERVER_PATH && tail -n 10 pool.out"

echo ""
echo "=== Deployment Complete ==="
echo "Test with: timeout 90 xmrig --url $SERVER:3333 --user testWorker --algo rx/0 --keepalive --no-color"
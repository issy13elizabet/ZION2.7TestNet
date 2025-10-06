#!/bin/bash

echo "🚀 ZION Pool VarDiff Final Deployment"

# Stop existing pool processes
echo "⏹️ Stopping existing processes..."
pkill -9 -f "python3.*stratum_pool" 2>/dev/null || true
sleep 2

# Upload latest pool code
echo "📤 Uploading pool code..."
cd /media/maitreya/ZION1/2.7/pool/
scp stratum_pool.py root@91.98.122.165:/root/zion27/pool/stratum_pool.py

# Start pool with logging
echo "🎯 Starting pool with logging..."
ssh root@91.98.122.165 'cd /root/zion27/pool && nohup python3 stratum_pool.py > pool_full.log 2>&1 & echo "Pool PID: $!"'

echo "⏳ Waiting for pool startup..."
sleep 5

echo "🔍 Checking pool status..."
ssh root@91.98.122.165 'ps aux | grep stratum_pool | grep -v grep'
ssh root@91.98.122.165 'ss -tlnp | grep :3333 || echo "Port 3333 not listening"'

echo "📋 Testing XMRig connection (30s)..."
timeout 30 xmrig --url 91.98.122.165:3333 --user testWorker --algo rx/0 --keepalive --no-color | grep -E 'job.*diff|accepted|rejected' | head -20

echo "📊 Pool logs (last 20 lines)..."
ssh root@91.98.122.165 'tail -20 /root/zion27/pool/pool_full.log 2>/dev/null || echo "No logs yet"'

echo "✅ Deployment complete! VarDiff should now work with default_monero_difficulty=32"
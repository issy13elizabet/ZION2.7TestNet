#!/bin/bash
# Deploy XMRig Compatible ZION Pool to SSH server

SSH_HOST="91.98.122.165"
SSH_USER="root"

echo "🚀 Deploying ZION Universal Pool with ZION Address support..."

# Copy pool file
scp zion_universal_pool.py $SSH_USER@$SSH_HOST:~/

# Deploy and start pool
ssh $SSH_USER@$SSH_HOST << 'EOF'
echo "🔧 Stopping old pool processes..."
pkill -f "python3.*pool"

echo "🚀 Starting ZION Universal Pool..."
nohup python3 zion_universal_pool.py > pool.log 2>&1 &

sleep 2

echo "✅ Pool status:"
ps aux | grep -v grep | grep "python3.*zion_universal_pool"
netstat -tlnp | grep :3333

echo "✅ XMRig Compatible Pool deployed to $SSH_HOST:3333"
EOF
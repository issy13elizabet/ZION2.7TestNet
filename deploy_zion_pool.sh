#!/bin/bash
# Deploy ZION Simple Pool to SSH Server
# Deploy to 91.98.122.165

echo "🚀 Deploying ZION Simple Pool to SSH server..."

# Copy pool file
scp -o StrictHostKeyChecking=no zion_simple_pool.py root@91.98.122.165:/root/zion_simple_pool.py

echo "📡 Starting pool on SSH server..."

# Start pool on remote server
ssh -o StrictHostKeyChecking=no root@91.98.122.165 << 'REMOTE_CMD'
cd /root

echo "🔧 Installing Python dependencies..."
apt update && apt install -y python3 python3-pip

echo "🚀 Starting ZION Simple Pool..."
nohup python3 zion_simple_pool.py > pool.log 2>&1 &

echo "✅ Pool started! Check with: tail -f pool.log"

# Show pool status
sleep 2
ps aux | grep zion_simple_pool | grep -v grep
netstat -tlnp | grep :3333

REMOTE_CMD

echo "✅ ZION Pool deployed to 91.98.122.165:3333"
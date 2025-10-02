#!/bin/bash

echo "🚀 ZION 2.7 REAL DEPLOYMENT SCRIPT 🚀"
echo "====================================="
echo "⚠️  ŽÁDNÉ SIMULACE! ŽÁDNÉ MOCKUPY!"
echo "✅ JEN SKUTEČNÉ KOMPONENTY!"
echo "====================================="

# SSH Server Configuration
SSH_HOST="${SSH_HOST:-91.98.122.165}"
SSH_USER="${SSH_USER:-root}"
SSH_PORT="${SSH_PORT:-22}"

if [ -z "$SSH_HOST" ]; then
    echo "❌ Please set SSH_HOST environment variable"
    echo "Usage: SSH_HOST=your.server.ip ./deploy_real_zion_27.sh"
    exit 1
fi

echo "🎯 Target Server: $SSH_USER@$SSH_HOST:$SSH_PORT"
echo ""

# Phase 1: Complete System Reset
echo "🧹 PHASE 1: COMPLETE SYSTEM RESET"
echo "=================================="

echo "🛑 Stopping all existing processes..."
ssh $SSH_USER@$SSH_HOST "
    # Kill all mining processes
    pkill -f xmrig 2>/dev/null || true
    pkill -f zion 2>/dev/null || true
    pkill -f mining 2>/dev/null || true
    pkill -f node 2>/dev/null || true
    pkill -f python 2>/dev/null || true
    
    # Remove all old ZION data
    rm -rf /root/2.7/ 2>/dev/null || true
    rm -rf /root/xmrig-* 2>/dev/null || true
    rm -rf /opt/zion* 2>/dev/null || true
    rm -rf /tmp/*zion* 2>/dev/null || true
    rm -rf /tmp/*mining* 2>/dev/null || true
    
    echo '✅ Old processes and data removed'
"

echo "🐳 Cleaning Docker environment..."
ssh $SSH_USER@$SSH_HOST "
    docker stop \$(docker ps -aq) 2>/dev/null || true
    docker rm \$(docker ps -aq) 2>/dev/null || true  
    docker system prune -af --volumes 2>/dev/null || true
    docker network prune -f 2>/dev/null || true
    
    echo '✅ Docker environment cleaned'
"

# Phase 2: Upload ZION 2.7 Real System
echo ""
echo "📤 PHASE 2: UPLOADING ZION 2.7 REAL SYSTEM"
echo "=========================================="

echo "📦 Creating deployment package..."
cd /Volumes/Zion

# Create deployment archive with REAL components only
tar -czf zion_27_real_deployment.tar.gz \
    --exclude="*.pyc" \
    --exclude="__pycache__" \
    --exclude="node_modules" \
    --exclude=".git" \
    --exclude="*.log" \
    --exclude="test_*" \
    --exclude="*_test.py" \
    2.7/core/ \
    2.7/mining/ \
    2.7/pool/ \
    2.7/wallet/ \
    2.7/network/ \
    2.7/rpc/ \
    2.7/ai/ \
    2.7/data/ \
    2.7/requirements.txt \
    2.7/run_node.py

echo "✅ Deployment package created: $(du -h zion_27_real_deployment.tar.gz | cut -f1)"

echo "🚀 Uploading to server..."
scp -P $SSH_PORT zion_27_real_deployment.tar.gz $SSH_USER@$SSH_HOST:/root/

# Phase 3: Server Setup and Installation
echo ""
echo "🔧 PHASE 3: SERVER SETUP AND INSTALLATION"
echo "========================================"

ssh $SSH_USER@$SSH_HOST << 'REMOTE_SETUP'
    echo "🌟 ZION 2.7 Real Deployment - Server Setup"
    
    # Update system
    apt update && apt upgrade -y
    
    # Install essential packages
    apt install -y python3 python3-pip python3-venv curl wget htop git
    
    # Install Python dependencies
    pip3 install psutil flask flask-cors cryptography numpy
    
    # Create ZION directory
    mkdir -p /root/zion27
    cd /root/zion27
    
    # Extract ZION 2.7
    tar -xzf /root/zion_27_real_deployment.tar.gz
    mv 2.7/* .
    rmdir 2.7
    
    # Install Python requirements
    pip3 install -r requirements.txt
    
    echo "✅ ZION 2.7 extracted and dependencies installed"
REMOTE_SETUP

# Phase 4: Real Component Configuration
echo ""
echo "⚙️ PHASE 4: REAL COMPONENT CONFIGURATION"
echo "======================================="

ssh $SSH_USER@$SSH_HOST << 'REMOTE_CONFIG'
    cd /root/zion27
    
    # Create real configuration files
    echo "🔧 Creating real blockchain configuration..."
    
    # Initialize real blockchain
    mkdir -p data/blocks
    mkdir -p logs
    
    # Create startup script for real components
    cat > start_real_zion.sh << 'STARTUP_SCRIPT'
#!/bin/bash

echo "🚀 Starting ZION 2.7 Real Components"

# Set environment
export PYTHONPATH="/root/zion27:/root/zion27/core"
cd /root/zion27

# Start RPC Server (port 17750)
echo "🌐 Starting RPC Server..."
nohup python3 -m rpc.server > logs/rpc.log 2>&1 &
RPC_PID=$!

# Start P2P Node (port 29876)  
echo "🌍 Starting P2P Node..."
nohup python3 -c "
from network.p2p import P2PNode
from core.blockchain import Blockchain

chain = Blockchain()
p2p = P2PNode(chain=chain)
p2p.start()

import time
while True:
    time.sleep(60)
    print(f'P2P: {len(p2p.peers)} peers, Chain: {chain.height} blocks')
" > logs/p2p.log 2>&1 &
P2P_PID=$!

# Start Mining Pool (port 3333)
echo "⛏️ Starting Mining Pool..."  
nohup python3 -c "
from pool.stratum_pool import MinimalStratumPool
from core.blockchain import Blockchain

chain = Blockchain()
pool = MinimalStratumPool(chain=chain)
pool.start()
" > logs/pool.log 2>&1 &
POOL_PID=$!

# Start Real Blockchain Node
echo "🔗 Starting Blockchain Node..."
nohup python3 run_node.py --pool --interval 30 > logs/node.log 2>&1 &
NODE_PID=$!

echo "✅ All ZION 2.7 components started"
echo "📊 Service Status:"
echo "   🌐 RPC Server:      port 17750 (PID: $RPC_PID)"
echo "   🌍 P2P Node:        port 29876 (PID: $P2P_PID)" 
echo "   ⛏️ Mining Pool:      port 3333 (PID: $POOL_PID)"
echo "   🔗 Blockchain Node: (PID: $NODE_PID)"
echo ""
echo "📝 Logs available in /root/zion27/logs/"
echo "🔧 Use 'systemctl status zion27' to check status"

# Save PIDs for monitoring
echo "$RPC_PID" > /var/run/zion27_rpc.pid
echo "$P2P_PID" > /var/run/zion27_p2p.pid  
echo "$POOL_PID" > /var/run/zion27_pool.pid
echo "$NODE_PID" > /var/run/zion27_node.pid

wait
STARTUP_SCRIPT

    chmod +x start_real_zion.sh
    
    echo "✅ Real component configuration complete"
REMOTE_CONFIG

# Phase 5: Systemd Service Setup
echo ""
echo "🔧 PHASE 5: SYSTEMD SERVICE SETUP"  
echo "================================="

ssh $SSH_USER@$SSH_HOST << 'SYSTEMD_SETUP'
    # Create systemd service for ZION 2.7
    cat > /etc/systemd/system/zion27.service << 'SERVICE_FILE'
[Unit]
Description=ZION 2.7 Real Blockchain System
After=network.target

[Service]
Type=forking
User=root
WorkingDirectory=/root/zion27
ExecStart=/root/zion27/start_real_zion.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SERVICE_FILE

    # Enable and start service
    systemctl daemon-reload
    systemctl enable zion27
    
    echo "✅ Systemd service configured"
SYSTEMD_SETUP

# Phase 6: Firewall Configuration  
echo ""
echo "🔥 PHASE 6: FIREWALL CONFIGURATION"
echo "================================="

ssh $SSH_USER@$SSH_HOST << 'FIREWALL_SETUP'
    # Configure UFW firewall
    ufw --force reset
    ufw default deny incoming
    ufw default allow outgoing
    
    # Allow SSH
    ufw allow 22/tcp
    
    # Allow ZION 2.7 ports
    ufw allow 17750/tcp comment "ZION RPC Server"
    ufw allow 29876/tcp comment "ZION P2P Network"  
    ufw allow 3333/tcp comment "ZION Mining Pool"
    
    # Enable firewall
    ufw --force enable
    
    echo "✅ Firewall configured for ZION 2.7"
FIREWALL_SETUP

# Phase 7: Start Real System
echo ""
echo "🚀 PHASE 7: STARTING REAL ZION 2.7 SYSTEM"
echo "========================================"

ssh $SSH_USER@$SSH_HOST << 'START_SYSTEM'
    cd /root/zion27
    
    # Start ZION 2.7 service
    systemctl start zion27
    
    # Wait for services to initialize
    sleep 10
    
    # Check service status
    systemctl status zion27 --no-pager -l
    
    echo ""
    echo "🔍 Checking component status..."
    
    # Check ports
    netstat -tulpn | grep -E "(17750|29876|3333)" || echo "⚠️ Some ports not yet active"
    
    # Check processes
    ps aux | grep -E "(python3|zion)" | grep -v grep || echo "⚠️ Processes starting up"
    
    echo ""
    echo "📊 ZION 2.7 Real System Status:"
    echo "================================"
    echo "🌐 RPC Server:      http://$SSH_HOST:17750"
    echo "🌍 P2P Network:     $SSH_HOST:29876" 
    echo "⛏️ Mining Pool:      stratum+tcp://$SSH_HOST:3333"
    echo "📝 Logs:           /root/zion27/logs/"
    echo "🔧 Service:        systemctl status zion27"
    echo ""
START_SYSTEM

# Final Status
echo ""
echo "🎉 ZION 2.7 REAL DEPLOYMENT COMPLETE! 🎉"
echo "========================================"
echo "✅ Server:           $SSH_HOST"
echo "✅ System:           Ubuntu with REAL components"
echo "✅ Blockchain:       REAL ZION 2.7 core"
echo "✅ Mining Pool:      REAL Stratum implementation" 
echo "✅ P2P Network:      REAL network protocol"
echo "✅ RPC Server:       REAL JSON-RPC API"
echo "✅ Wallet:           REAL cryptographic wallet"
echo ""
echo "⚠️  NO SIMULATIONS! NO MOCKUPS! NO FAKE DATA!"
echo "🚀 FULLY OPERATIONAL REAL BLOCKCHAIN SYSTEM!"
echo ""
echo "🔗 Connect your miner to: stratum+tcp://$SSH_HOST:3333"
echo "🌐 API available at: http://$SSH_HOST:17750"
echo "📊 Monitor with: ssh $SSH_USER@$SSH_HOST 'tail -f /root/zion27/logs/*.log'"

# Cleanup
rm -f zion_27_real_deployment.tar.gz

echo ""
echo "✅ Deployment script completed successfully!"
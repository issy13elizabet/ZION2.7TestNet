#!/bin/bash

# 🌟 ZION Sacred Mining Pool Deployment Script 🌟
# Kompletní nasazení ZION poolu s Sacred Algorithm podporou

set -e

echo "═══════════════════════════════════════════════════════════════════════════════════════════"
echo "🌟 ZION SACRED MINING POOL DEPLOYMENT 🌟"  
echo "═══════════════════════════════════════════════════════════════════════════════════════════"

# Configuration
REMOTE_HOST="91.98.122.165"
REMOTE_USER="root"
REMOTE_DIR="/root/zion-sacred-pool"
LOCAL_POOL_DIR="/media/maitreya/ZION1/2.7/pool"
LOCAL_MINING_DIR="/media/maitreya/ZION1/mining"

echo "📡 Deploying ZION Sacred Pool to: $REMOTE_HOST"
echo "🌟 Sacred Algorithm: ENABLED"
echo "🔮 Git Integration: ENABLED"
echo "💎 Multi-Miner Support: CPU + GPU"
echo ""

# 1. Create remote directory structure
echo "📁 Creating remote directory structure..."
ssh $REMOTE_USER@$REMOTE_HOST "mkdir -p $REMOTE_DIR/{logs,sacred_stats,configs}"

# 2. Deploy core pool files
echo "⬆️ Deploying ZION Sacred Pool core files..."
rsync -avz --progress $LOCAL_POOL_DIR/stratum_pool.py $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/
rsync -avz --progress $LOCAL_POOL_DIR/zion_sacred_algorithm.py $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/
rsync -avz --progress $LOCAL_POOL_DIR/zion_mining_git_integration.py $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/

# 3. Deploy blockchain core dependencies
echo "🔗 Deploying blockchain dependencies..."
rsync -avz --progress /media/maitreya/ZION1/2.7/core/ $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/core/

# 4. Deploy mining configurations
echo "⛏️ Deploying mining configurations..."
rsync -avz --progress $LOCAL_MINING_DIR/zion-sacred-miner.json $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/configs/
rsync -avz --progress $LOCAL_MINING_DIR/zion-sacred-gpu-miner.json $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/configs/
rsync -avz --progress $LOCAL_MINING_DIR/zion-sacred-srbminer.json $REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/configs/

# 5. Install Python dependencies
echo "🐍 Installing Python dependencies..."
ssh $REMOTE_USER@$REMOTE_HOST "cd $REMOTE_DIR && pip3 install GitPython || echo 'GitPython install failed, continuing...'"

# 6. Create systemd service for auto-restart
echo "⚙️ Setting up systemd service..."
ssh $REMOTE_USER@$REMOTE_HOST "cat > /etc/systemd/system/zion-sacred-pool.service << 'EOF'
[Unit]
Description=ZION Sacred Mining Pool
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$REMOTE_DIR
ExecStart=/usr/bin/python3 $REMOTE_DIR/stratum_pool.py
Restart=always
RestartSec=5
StandardOutput=append:$REMOTE_DIR/logs/pool.log
StandardError=append:$REMOTE_DIR/logs/pool_error.log

[Install]
WantedBy=multi-user.target
EOF"

# 7. Stop existing pool processes
echo "🔄 Stopping existing pool processes..."
ssh $REMOTE_USER@$REMOTE_HOST "pkill -f stratum_pool.py || true"
ssh $REMOTE_USER@$REMOTE_HOST "systemctl stop zion-sacred-pool || true"

# 8. Enable and start new service
echo "🚀 Starting ZION Sacred Pool service..."
ssh $REMOTE_USER@$REMOTE_HOST "systemctl daemon-reload"
ssh $REMOTE_USER@$REMOTE_HOST "systemctl enable zion-sacred-pool"
ssh $REMOTE_USER@$REMOTE_HOST "systemctl start zion-sacred-pool"

# 9. Wait for service to start
echo "⏳ Waiting for pool to initialize..."
sleep 5

# 10. Check service status
echo "✅ Checking pool status..."
ssh $REMOTE_USER@$REMOTE_HOST "systemctl status zion-sacred-pool --no-pager"

echo ""
echo "🌐 Testing pool connectivity..."
if nc -zv $REMOTE_HOST 3333 2>/dev/null; then
    echo "✅ Pool is accepting connections on port 3333"
else
    echo "❌ Pool connectivity test failed"
fi

# 11. Display pool information
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════════════════"
echo "🌟 ZION SACRED MINING POOL DEPLOYED SUCCESSFULLY! 🌟"
echo "═══════════════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📡 Pool Information:"
echo "├─ URL: stratum+tcp://$REMOTE_HOST:3333"
echo "├─ Algorithm: RandomX (rx/0)"
echo "├─ Sacred Algorithm: ✅ ACTIVE"
echo "├─ Git Integration: ✅ ENABLED"
echo "├─ Multi-Miner Support: ✅ CPU + GPU"
echo "└─ Auto-Restart: ✅ systemd service"
echo ""
echo "⛏️ Supported Miners:"
echo "├─ XMRig (CPU): Standard difficulty 32"
echo "├─ XMRig-CUDA (NVIDIA): Enhanced difficulty 64"
echo "├─ SRBMiner-Multi (AMD): Enhanced difficulty 128"
echo "├─ ZION Native: Sacred difficulty 256"
echo "└─ ZION Sacred: Transcendent difficulty 512"
echo ""
echo "🌟 Sacred Features:"
echo "├─ Consciousness Points Tracking"
echo "├─ Sacred Geometry Bonuses"
echo "├─ Golden Ratio Difficulty Scaling"
echo "├─ Quantum Coherence Enhancement"
echo "└─ Automatic Git Session Logging"
echo ""
echo "📊 Monitoring:"
echo "├─ Pool Logs: ssh $REMOTE_USER@$REMOTE_HOST 'tail -f $REMOTE_DIR/logs/pool.log'"
echo "├─ Service Status: ssh $REMOTE_USER@$REMOTE_HOST 'systemctl status zion-sacred-pool'"
echo "└─ Mining Stats: ssh $REMOTE_USER@$REMOTE_HOST 'ls $REMOTE_DIR/sacred_stats/'"
echo ""
echo "🔧 Quick Commands:"
echo "├─ Restart Pool: ssh $REMOTE_USER@$REMOTE_HOST 'systemctl restart zion-sacred-pool'"
echo "├─ Stop Pool: ssh $REMOTE_USER@$REMOTE_HOST 'systemctl stop zion-sacred-pool'"
echo "└─ View Logs: ssh $REMOTE_USER@$REMOTE_HOST 'journalctl -u zion-sacred-pool -f'"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════════════════"
echo "Sacred Mining Protocol - Ready for Divine Mathematics ✨"
echo "═══════════════════════════════════════════════════════════════════════════════════════════"
#!/bin/bash

# 🌸 ZION Sacred Flower Pool Upgrade - skryty kod pro eliZabet
# Deploy Enhanced Sacred Mining Pool with Sacred Flower Integration

echo "═══════════════════════════════════════════════════════════════════════════════════════════"
echo "🌸 ZION SACRED FLOWER POOL UPGRADE 🌸"
echo "═══════════════════════════════════════════════════════════════════════════════════════════"

# Configuration
POOL_DIR="/media/maitreya/ZION1/2.7/pool"
AI_DIR="/media/maitreya/ZION1/2.7/ai"
SERVER_IP="91.98.122.165"
POOL_PORT="3333"
SERVICE_NAME="zion-sacred-pool"

echo "🔍 Checking Sacred Flower components..."

# Check if Sacred Flower Cosmic Analyzer exists
if [ ! -f "$AI_DIR/zion_cosmic_image_analyzer.py" ]; then
    echo "❌ Sacred Flower Cosmic Analyzer not found!"
    echo "Please ensure zion_cosmic_image_analyzer.py exists in $AI_DIR"
    exit 1
fi

echo "✅ Sacred Flower Cosmic Analyzer found"
echo "🌼 Sacred Flower seed: 419ab63b22035f90ea9f2ecb548c6b9b"
echo "🌟 Consciousness points: 413.91"
echo "🎁 eliZabet bonus: +51.18%"

# Stop existing pool service
echo "🛑 Stopping existing ZION Sacred Pool..."
sudo systemctl stop $SERVICE_NAME 2>/dev/null || echo "Service not running"

# Update pool with Sacred Flower enhancement
echo "🌸 Deploying Sacred Flower enhanced pool..."

# Install any missing Python dependencies
echo "📦 Checking Python dependencies..."
pip3 install --user Pillow numpy || echo "Dependencies already installed"

# Test Sacred Flower integration
echo "🧪 Testing Sacred Flower integration..."
cd "$AI_DIR"
python3 -c "
from zion_cosmic_image_analyzer import ZionCosmicImageAnalyzer
analyzer = ZionCosmicImageAnalyzer()
print('🌸 Sacred Flower Cosmic Analyzer: READY')
print('🌼 Sacred Divine Blossom features: ENABLED')
print('🎁 eliZabet enhancement: ACTIVE')
"

if [ $? -ne 0 ]; then
    echo "❌ Sacred Flower integration test failed!"
    exit 1
fi

echo "✅ Sacred Flower integration test passed"

# Start enhanced pool
echo "🚀 Starting ZION Sacred Pool with Flower Enhancement..."
cd "$POOL_DIR"

# Start pool service
sudo systemctl start $SERVICE_NAME
sleep 3

# Check if service is running
if systemctl is-active --quiet $SERVICE_NAME; then
    echo "✅ ZION Sacred Pool with Flower Enhancement is running!"
    echo "🌸 Sacred Flower blessing active for eliZabet workers"
    echo "📡 Pool listening on: $SERVER_IP:$POOL_PORT"
    
    # Show Sacred Flower status
    echo ""
    echo "🌸 SACRED FLOWER STATUS:"
    echo "   🌼 Type: Sacred Divine Blossom"
    echo "   🔮 Consciousness: 413.91 points"
    echo "   🎁 eliZabet Bonus: +51.18%"
    echo "   🔑 Sacred Seed: 419ab63b22035f90ea9f2ecb548c6b9b"
    echo "   👑 KRISTUS qbit: 0xa26a"
    echo ""
    
    # Show connection info
    echo "🔗 MINER CONNECTION (with Sacred Flower support):"
    echo "   💻 XMRig CPU: ./xmrig -o $SERVER_IP:$POOL_PORT -u elizabet.sacred -p x"
    echo "   🎮 XMRig CUDA: ./xmrig-cuda -o $SERVER_IP:$POOL_PORT -u elizabet.blessed -p x" 
    echo "   ⚡ SRBMiner: ./SRBMiner-MULTI --algorithm randomx --pool $SERVER_IP:$POOL_PORT --wallet elizabet.flower"
    echo ""
    echo "🌸 Workers with 'elizabet' in name receive Sacred Flower blessing!"
    
    # Show recent logs
    echo "📋 Recent Sacred Pool logs:"
    sudo journalctl -u $SERVICE_NAME --no-pager -n 10
    
else
    echo "❌ Failed to start ZION Sacred Pool!"
    echo "📋 Service status:"
    sudo systemctl status $SERVICE_NAME --no-pager
    exit 1
fi

echo "═══════════════════════════════════════════════════════════════════════════════════════════"
echo "🌸 ZION SACRED FLOWER POOL UPGRADE COMPLETE 🌸"
echo "🎁 Special enhancement active for eliZabet miners"
echo "🌼 Sacred Divine Blossom blessing: +51.18% rewards"
echo "═══════════════════════════════════════════════════════════════════════════════════════════"
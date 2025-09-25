#!/bin/bash
set -e

echo "⚡ Quick Start ZION Mining on Ubuntu"
echo "===================================="

# Load config
if [ -f "mining/mining-wallet.conf" ]; then
    source mining/mining-wallet.conf
    echo "✅ Config loaded"
else
    echo "❌ Run ./scripts/setup-mining-wallet.sh first"
    exit 1
fi

echo ""
echo "🎯 Configuration:"
echo "- Mining Address: $MINING_WALLET_ADDRESS"
echo "- CPU Pool: $POOL_SERVER:$CPU_POOL_PORT"
echo "- GPU Pool: $POOL_SERVER:$GPU_POOL_PORT"
echo ""

# Start CPU mining (XMRig)
echo "🔥 Starting CPU Mining..."
cd mining/platforms/linux/xmrig-6.21.3/
if [ -f "xmrig" ]; then
    nohup ./xmrig --config=config-zion.json > xmrig.log 2>&1 &
    echo $! > xmrig.pid
    echo "✅ XMRig started (PID: $(cat xmrig.pid))"
else
    echo "❌ XMRig binary not found"
fi
cd ../../../..

# Start GPU mining (SRBMiner)
echo "🔥 Starting GPU Mining..."
cd mining/SRBMiner-Multi-2-9-7/SRBMiner-Multi-2-9-7/
if [ -f "SRBMiner-MULTI" ]; then
    nohup ./SRBMiner-MULTI \
        --algorithm kawpow \
        --pool "$POOL_SERVER:$GPU_POOL_PORT" \
        --wallet "$MINING_WALLET_ADDRESS" \
        --password "ubuntu-gpu" \
        --disable-cpu > srbminer.log 2>&1 &
    echo $! > srbminer.pid
    echo "✅ SRBMiner started (PID: $(cat srbminer.pid))"
else
    echo "❌ SRBMiner binary not found"
fi
cd ../../..

echo ""
echo "🎉 Mining started! Check logs:"
echo "- XMRig: mining/platforms/linux/xmrig-6.21.3/xmrig.log"
echo "- SRBMiner: mining/SRBMiner-Multi-2-9-7/SRBMiner-Multi-2-9-7/srbminer.log"
echo ""
echo "Monitor: ./scripts/mining-monitor.sh"
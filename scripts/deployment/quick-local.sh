#!/bin/bash

# ZION Quick Deployment - Fallback Solution
# =========================================

set -e

echo "🚀 ZION Quick Deployment (Local Binary)"
echo "======================================="

# Check if we have local binary
if [ -f "zion-cryptonote/build/src/ziond" ]; then
    echo "✅ Local binary found - using direct deployment"
    
    # Create data directory
    mkdir -p data logs
    
    # Run directly
    echo "🔥 Starting ZION node locally..."
    cd zion-cryptonote/build/src
    
    ./ziond \
        --data-dir=../../../data \
        --log-file=../../../logs/zion.log \
        --p2p-bind-port=18080 \
        --rpc-bind-port=18081 \
        --rpc-bind-ip=0.0.0.0 \
        --enable-cors=* &
    
    ZION_PID=$!
    echo "📊 ZION node started with PID: $ZION_PID"
    
    # Wait for RPC
    echo "⏳ Waiting for RPC endpoint..."
    for i in {1..30}; do
        if curl -s http://localhost:18081/getinfo &>/dev/null; then
            echo "✅ RPC endpoint ready!"
            break
        fi
        sleep 2
    done
    
    echo "🌟 ZION node is running!"
    echo "🌐 RPC: http://localhost:18081"
    echo "🔗 P2P: Port 18080"
    echo "🛑 Stop: kill $ZION_PID"
    
else
    echo "❌ No local binary found"
    echo "🏗️ Run: cd zion-cryptonote && make"
    echo "🐳 Or: docker-compose -f docker-compose.prod.yml up"
    exit 1
fi
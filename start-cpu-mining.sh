#!/bin/bash

# ZION CPU Mining with XMRig
# Usage: ./start-cpu-mining.sh

echo "🚀 Starting ZION CPU Mining"
echo "==========================="

# Check if XMRig is installed
if ! command -v xmrig &> /dev/null; then
    echo "❌ XMRig not found. Installing..."
    
    # Auto-download XMRig (Ubuntu/Debian)
    if command -v apt &> /dev/null; then
        sudo apt update && sudo apt install -y xmrig
    elif command -v yum &> /dev/null; then
        sudo yum install -y xmrig
    else
        echo "📥 Please download XMRig from: https://xmrig.com/download"
        echo "   Or run: docker run --rm -v \$(pwd):/mnt --network host xmrig/xmrig:latest --config=/mnt/xmrig-zion-cpu.json"
        exit 1
    fi
fi

# Check config file
if [ ! -f "xmrig-zion-cpu.json" ]; then
    echo "❌ Config file xmrig-zion-cpu.json not found!"
    exit 1
fi

# Check pool connectivity
echo "🔗 Testing pool connection..."
if ! nc -z localhost 3333 2>/dev/null; then
    echo "❌ Mining pool not reachable on localhost:3333"
    echo "💡 Make sure ZION bootstrap stack is running:"
    echo "   docker-compose -p zion-bootstrap -f docker-compose-bootstrap.yml up -d"
    exit 1
fi

echo "✅ Pool connection OK"
echo "💎 Starting CPU mining with config: xmrig-zion-cpu.json"
echo "📊 API available on: http://localhost:16000"
echo ""

# Start mining
xmrig --config=xmrig-zion-cpu.json
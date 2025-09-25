#!/bin/bash

# ZION Mining Startup Script - macOS Apple M1/ARM64
# Spouští XMRig pro těžení ZION blockchainu na Apple Silicon

echo "🍎 ZION Mining for macOS Apple M1/ARM64"
echo "Genesis Block Hash: d763b61e4e542a6973c8f649deb228e116bcf3ee099cec92be33efe288829ae1"
echo "Mining Address: ajmqontZjiVUmtNjQu1RNUYq1RZgd5EDodX3qgjcaTMoMzG8EkG4bVPgLhEgudBoH82fQU1iZVw6XPfddKWAHDdA3x92ToH4uo"
echo "Pool: 91.98.122.165:3333"
echo ""

# Check if server is accessible
echo "Checking pool connection..."
if nc -z 91.98.122.165 3333; then
    echo "✅ Pool is reachable"
else
    echo "❌ Pool is not reachable"
    exit 1
fi

# Check architecture
ARCH=$(uname -m)
echo "Detected architecture: $ARCH"

if [[ "$ARCH" == "arm64" ]]; then
    echo "🔥 Starting XMRig for Apple M1/ARM64..."
    cd "$(dirname "$0")/platforms/macos-arm64/xmrig-6.21.3"
    ./xmrig --config=config-zion.json
else
    echo "🔥 Starting XMRig for Intel x64..."
    cd "$(dirname "$0")/platforms/macos-x64/xmrig-6.21.3"
    ./xmrig --config=config-zion.json
fi
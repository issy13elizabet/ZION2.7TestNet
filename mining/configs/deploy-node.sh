#!/bin/bash
# Deploy ZION Miner to remote node
set -euo pipefail

NODE_IP="$1"
BINARY_PATH="$2"

echo "🚀 Deploying ZION Miner to ${NODE_IP}..."

# Copy binary
scp "${BINARY_PATH}" root@${NODE_IP}:/tmp/zion-miner
ssh root@${NODE_IP} "chmod +x /tmp/zion-miner"

# Test connection
ssh root@${NODE_IP} "/tmp/zion-miner --help | head -5"

echo "✅ Deployment to ${NODE_IP} complete"

#!/bin/bash
# ZION Miner v1.4.0 - SSH Deployment Test Script
# Date: 29. září 2025

set -euo pipefail

SSH_SERVER="91.98.122.165"
WALLET_ADDRESS="ZiYgACwXhrYG9iLjRfBgEdgsGsT6DqQ2brtM8j9iR3Rs7geE5kyj7oEGkw9LpjaGX9p1h7uRNJg5BkWKu8HD28EMPpJAYUdJ4"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }
warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }

echo "🎯 ZION Miner v1.4.0 SSH Deployment Test"
echo "=========================================="
echo "Server: ${SSH_SERVER}"
echo "Wallet: ${WALLET_ADDRESS:0:20}..."
echo

# Test 1: SSH Connectivity
log "🔗 Testing SSH connectivity..."
if ssh -o ConnectTimeout=10 root@${SSH_SERVER} "echo 'SSH OK'" >/dev/null 2>&1; then
    success "SSH connection established"
else
    error "SSH connection failed"
    exit 1
fi

# Test 2: Binary Deployment
log "📦 Verifying binary deployment..."
if ssh root@${SSH_SERVER} "test -f /tmp/zion-miner && test -x /tmp/zion-miner"; then
    success "ZION Miner binary deployed and executable"
else
    error "Binary not found or not executable"
    exit 1
fi

# Test 3: Version Check
log "🏷️ Checking miner version..."
VERSION=$(ssh root@${SSH_SERVER} "/tmp/zion-miner --version 2>/dev/null | grep 'ZION MINER' | head -1" || echo "Version check failed")
if [[ "$VERSION" == *"v1.4.0"* ]]; then
    success "Version: $VERSION"
else
    warning "Version check: $VERSION"
fi

# Test 4: ZION Core Services
log "🔍 Testing ZION Core services..."
CORE_STATUS=$(ssh root@${SSH_SERVER} "curl -s localhost:18081/json_rpc -d '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"get_info\"}' 2>/dev/null | grep -o '\"height\":[0-9]*' | cut -d: -f2" || echo "offline")
if [[ "$CORE_STATUS" =~ ^[0-9]+$ ]]; then
    success "ZION Core RPC: Height $CORE_STATUS"
else
    warning "ZION Core RPC: $CORE_STATUS"
fi

# Test 5: Pool Stratum
log "🎱 Testing Pool Stratum..."
POOL_STATUS=$(ssh root@${SSH_SERVER} "timeout 5 telnet localhost 3333 >/dev/null 2>&1 && echo 'listening' || echo 'closed'")
if [[ "$POOL_STATUS" == "listening" ]]; then
    success "Pool Stratum: Port 3333 listening"
else
    warning "Pool Stratum: Port 3333 $POOL_STATUS"
fi

# Test 6: Mining Test (Short Run)
log "⛏️ Running short mining test..."
echo "Starting 15-second mining test..."

ssh root@${SSH_SERVER} "timeout 15 /tmp/zion-miner --cpu-only --threads=2 --pool=localhost:3333 --wallet=${WALLET_ADDRESS} --test-mode" &
SSH_PID=$!

sleep 5
log "Mining test running... checking status"

if kill -0 $SSH_PID 2>/dev/null; then
    success "Mining test: Process running successfully"
    sleep 10
    kill $SSH_PID 2>/dev/null || true
    wait $SSH_PID 2>/dev/null || true
else
    warning "Mining test: Process ended early"
fi

# Test 7: System Resources
log "📊 Checking system resources..."
RESOURCES=$(ssh root@${SSH_SERVER} "echo 'CPU Cores:' \$(nproc); echo 'Memory:' \$(free -h | grep Mem | awk '{print \$2}'); echo 'Load:' \$(uptime | awk -F'load average:' '{print \$2}')")
success "System Resources:"
echo "$RESOURCES" | sed 's/^/  /'

# Test 8: Network Connectivity
log "🌐 Testing external connectivity..."
EXTERNAL_TEST=$(ssh root@${SSH_SERVER} "curl -s --connect-timeout 5 https://api.github.com >/dev/null && echo 'connected' || echo 'failed'")
if [[ "$EXTERNAL_TEST" == "connected" ]]; then
    success "External connectivity: OK"
else
    warning "External connectivity: $EXTERNAL_TEST"
fi

echo
echo "🎉 SSH Deployment Test Complete!"
echo "=================================="
log "Next steps:"
echo "  1. Start persistent mining: ssh root@${SSH_SERVER} 'screen -dmS zion-miner /tmp/zion-miner --cpu-only --pool localhost:3333 --wallet ${WALLET_ADDRESS}'"
echo "  2. Monitor status: ssh root@${SSH_SERVER} 'screen -r zion-miner'"
echo "  3. Check logs: ssh root@${SSH_SERVER} 'journalctl -f | grep zion'"
echo
success "✅ ZION Miner v1.4.0 SSH deployment validated!"
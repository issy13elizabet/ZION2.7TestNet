#!/bin/bash

# ZION Mining Pool SSH Tunnel Script
# ==================================

SERVER_IP="${1:-91.98.122.165}"
LOCAL_PORT="${2:-3333}"
REMOTE_PORT="${3:-3333}"
SSH_USER="${4:-root}"

if [ -z "$1" ]; then
    echo "🌉 ZION Mining Pool SSH Tunnel"
    echo "Usage: $0 <server-ip> [local-port] [remote-port] [ssh-user]"
    echo ""
    echo "Examples:"
    echo "  $0 91.98.122.165                    # Standard tunnel"
    echo "  $0 91.98.122.165 3334 3333 root     # Custom ports"
    echo ""
    echo "📋 This script creates SSH tunnel:"
    echo "   localhost:$LOCAL_PORT -> $SERVER_IP:$REMOTE_PORT"
    echo ""
    echo "⚠️  SSH key or password required for $SSH_USER@$SERVER_IP"
    exit 0
fi

echo "🌉 Creating SSH tunnel for ZION Mining Pool"
echo "==========================================="
echo "📡 Server: $SSH_USER@$SERVER_IP"
echo "🔗 Tunnel: localhost:$LOCAL_PORT -> $SERVER_IP:$REMOTE_PORT"
echo ""

# Check if tunnel already exists
EXISTING_TUNNEL=$(ps aux | grep "ssh.*$LOCAL_PORT:localhost:$REMOTE_PORT.*$SERVER_IP" | grep -v grep)
if [ ! -z "$EXISTING_TUNNEL" ]; then
    echo "⚠️  SSH tunnel already exists:"
    echo "$EXISTING_TUNNEL"
    echo ""
    read -p "Kill existing tunnel and create new one? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        pkill -f "ssh.*$LOCAL_PORT:localhost:$REMOTE_PORT.*$SERVER_IP"
        sleep 2
    else
        echo "❌ Aborted - tunnel already running"
        exit 1
    fi
fi

# Test server connectivity
echo "🔍 Testing server connectivity..."
if ! nc -z "$SERVER_IP" 22 2>/dev/null; then
    echo "❌ Cannot reach $SERVER_IP:22 (SSH)"
    echo "   Check server IP and network connectivity"
    exit 1
fi

echo "✅ Server reachable on SSH"

# Create SSH tunnel
echo "🚀 Creating SSH tunnel..."
echo "   Press Ctrl+C to stop the tunnel"
echo ""

# Use SSH with ControlMaster for better connection handling
ssh -o ControlMaster=yes \
    -o ControlPath=/tmp/zion-tunnel-%r@%h:%p \
    -o ControlPersist=5m \
    -L "$LOCAL_PORT:localhost:$REMOTE_PORT" \
    -N "$SSH_USER@$SERVER_IP"

# Cleanup on exit
echo ""
echo "🧹 Cleaning up SSH tunnel..."
ssh -o ControlPath=/tmp/zion-tunnel-%r@%h:%p -O exit "$SSH_USER@$SERVER_IP" 2>/dev/null || true
echo "✅ Tunnel closed"
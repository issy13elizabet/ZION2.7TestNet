#!/bin/bash

echo "🚀 ZION MINER SSH DEPLOYMENT - REÁLNÝ TEST 🚀"
echo "Deploying zion-core-ui for real blockchain mining"
echo "==============================================="

# Configuration
SSH_HOST="91.98.122.165"
SSH_USER="root"
REMOTE_DIR="/opt/zion-mining"
WALLET="Z3BDEEC2A0AE0F5D81B034308F99ECD8990D9B8B01BD9C7E7429392CA31861C6220DA3B30D74E809FA0A1FE069F1"
POOL_HOST="91.98.122.165"
POOL_PORT="3333"

echo "📡 Target server: $SSH_USER@$SSH_HOST:$REMOTE_DIR"
echo "💰 Wallet: ${WALLET:0:20}...${WALLET: -20}"
echo "🏊 Pool: $POOL_HOST:$POOL_PORT"
echo ""

# Test SSH connection
echo "🔐 Testing SSH connection..."
if ssh -o ConnectTimeout=10 $SSH_USER@$SSH_HOST "echo 'Connection OK'"; then
    echo "✅ SSH connection successful"
else
    echo "❌ SSH connection failed"
    exit 1
fi

# Create remote directory
echo ""
echo "📁 Preparing remote directory..."
ssh $SSH_USER@$SSH_HOST "mkdir -p $REMOTE_DIR/bin"

# Copy binary
echo ""
echo "📦 Uploading zion-core-ui binary..."
scp build-core/bin/Release/zion-core-ui.exe $SSH_USER@$SSH_HOST:$REMOTE_DIR/bin/
if [ $? -eq 0 ]; then
    echo "✅ Binary uploaded successfully"
else
    echo "❌ Binary upload failed"
    exit 1
fi

# Make executable
echo ""
echo "🔧 Setting permissions..."
ssh $SSH_USER@$SSH_HOST "chmod +x $REMOTE_DIR/bin/zion-core-ui.exe"

# Create start script
echo ""
echo "📝 Creating start script..."
ssh $SSH_USER@$SSH_HOST "cat > $REMOTE_DIR/start-mining.sh << 'EOF'
#!/bin/bash
echo '🚀 Starting ZION Interactive Miner'
echo 'Pool: $POOL_HOST:$POOL_PORT'
echo 'Wallet: ${WALLET:0:20}...${WALLET: -20}'
echo '========================='
cd $REMOTE_DIR
./bin/zion-core-ui.exe --pool $POOL_HOST:$POOL_PORT --wallet $WALLET
EOF"

ssh $SSH_USER@$SSH_HOST "chmod +x $REMOTE_DIR/start-mining.sh"

# Test connection to pool
echo ""
echo "🏊 Testing pool connection..."
ssh $SSH_USER@$SSH_HOST "timeout 5 telnet $POOL_HOST $POOL_PORT || echo 'Pool connection test complete'"

echo ""
echo "🎉 DEPLOYMENT COMPLETE!"
echo ""
echo "To start mining on SSH server:"
echo "  ssh $SSH_USER@$SSH_HOST"
echo "  cd $REMOTE_DIR"
echo "  ./start-mining.sh"
echo ""
echo "Mining with:"
echo "  - Pool: $POOL_HOST:$POOL_PORT" 
echo "  - Wallet: ${WALLET:0:30}...${WALLET: -30}"
echo "  - Interactive UI with keyboard controls"
echo ""
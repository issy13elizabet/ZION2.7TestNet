#!/bin/bash

echo "🚀 ZION SSH DEPLOYMENT SCRIPT 🚀"
echo "Deploying cleaned ZION workspace to production server"
echo "====================================================="

# Configuration
SSH_HOST="91.98.122.165"
SSH_USER="root"
REMOTE_DIR="/opt/zion"
LOCAL_DIR="/media/maitreya/ZION1"

echo "📡 Target server: $SSH_USER@$SSH_HOST:$REMOTE_DIR"
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
ssh $SSH_USER@$SSH_HOST "mkdir -p $REMOTE_DIR && cd $REMOTE_DIR"

# Stop any running ZION services
echo ""
echo "🛑 Stopping existing ZION services..."
ssh $SSH_USER@$SSH_HOST "
    docker stop \$(docker ps -q --filter 'name=zion') 2>/dev/null || true
    docker rm \$(docker ps -aq --filter 'name=zion') 2>/dev/null || true
    pkill -f 'zion' 2>/dev/null || true
    pkill -f 'xmrig' 2>/dev/null || true
"

# Create deployment package
echo ""
echo "📦 Creating deployment package..."
cd $LOCAL_DIR
tar -czf /tmp/zion-deployment.tar.gz \
    zion-core/ \
    config/ \
    scripts/ \
    *.md \
    ZION*.md \
    package*.json

echo "✅ Package created: $(du -sh /tmp/zion-deployment.tar.gz | cut -f1)"

# Transfer files
echo ""
echo "📤 Uploading to server..."
scp /tmp/zion-deployment.tar.gz $SSH_USER@$SSH_HOST:/tmp/

# Extract and setup on server
echo ""
echo "📥 Extracting on server..."
ssh $SSH_USER@$SSH_HOST "
    cd $REMOTE_DIR
    tar -xzf /tmp/zion-deployment.tar.gz
    rm /tmp/zion-deployment.tar.gz
    chown -R root:root $REMOTE_DIR
    chmod +x $REMOTE_DIR/scripts/*.sh 2>/dev/null || true
    chmod +x $REMOTE_DIR/mining/*.sh 2>/dev/null || true
"

# Install dependencies
echo ""
echo "🔧 Installing dependencies on server..."
ssh $SSH_USER@$SSH_HOST "
    cd $REMOTE_DIR
    
    # Update system
    apt-get update -y
    
    # Install Node.js if not present
    if ! command -v node &> /dev/null; then
        curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
        apt-get install -y nodejs
    fi
    
    # Install Docker if not present
    if ! command -v docker &> /dev/null; then
        curl -fsSL https://get.docker.com -o get-docker.sh
        sh get-docker.sh
        systemctl enable docker
        systemctl start docker
    fi
    
    # Install other dependencies
    apt-get install -y git curl wget jq bc
    
    echo '✅ Dependencies installed'
"

# Setup ZION Core v2.5 (TypeScript)
echo ""
echo "🏗️ Setting up ZION Core v2.5..."
ssh $SSH_USER@$SSH_HOST "
    cd $REMOTE_DIR/zion-core
    npm install --production
    
    # No Docker build needed - using pure TypeScript ZION Core
    echo '✅ ZION Core v2.5 setup complete (TypeScript native)'
"

# Start ZION Core v2.5 services
echo ""
echo "🚀 Starting ZION Core v2.5..."
ssh $SSH_USER@$SSH_HOST "
    cd $REMOTE_DIR/zion-core
    
    # Stop any existing ZION processes
    pkill -f 'node server.js' || true
    
    # Start ZION Core v2.5 TypeScript system
    ZION_PORT=8888 nohup node server.js > /tmp/zion-core.log 2>&1 &
    
    # Wait for startup
    sleep 10
    
    # Test health
    nc -z localhost 3333 && echo '⛏️ Mining pool running on port 3333' || echo 'Mining pool not available'
    nc -z localhost 8888 && echo '🌐 Web interface running on port 8888' || echo 'Web interface not available'
    
    echo '✅ ZION Core v2.5 services started'
"

# Cleanup
echo ""
echo "🧹 Cleaning up..."
rm /tmp/zion-deployment.tar.gz 2>/dev/null || true

# Final status
echo ""
echo "📊 Deployment Status:"
echo "===================="
ssh $SSH_USER@$SSH_HOST "
    echo 'Server: $SSH_HOST'
    echo 'Directory: $REMOTE_DIR'
    echo 'Services:'
    docker ps --filter 'name=zion' --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
    echo ''
    echo 'Health Check:'
    curl -s http://localhost:8888/health | jq -r '.status // \"unhealthy\"' 2>/dev/null || echo 'Service not responding'
    echo ''
    echo 'Mining Pool:'
    netstat -an | grep ':3333' | head -1 || echo 'Mining pool not listening'
"

echo ""
echo "🎉 DEPLOYMENT COMPLETE!"
echo ""
echo "📡 Access URLs:"
echo "   🌐 Web Interface: http://$SSH_HOST:8888"
echo "   ⛏️ Mining Pool: $SSH_HOST:3333"
echo "   📊 Health Check: http://$SSH_HOST:8888/health"
echo ""
echo "🔗 SSH Access: ssh $SSH_USER@$SSH_HOST"
echo "📁 Remote Directory: $REMOTE_DIR"
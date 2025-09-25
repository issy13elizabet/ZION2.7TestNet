#!/bin/bash

# ZION Production Server Migration Script
# =======================================
# Complete deployment script for VPS/Cloud servers with full automation

set -e

echo "🚀 ZION Server Migration & Production Deployment"
echo "==============================================="

SERVER_IP="${1:-localhost}"
SERVER_USER="${2:-root}"
DEPLOY_DIR="/opt/zion"

if [ "$SERVER_IP" = "localhost" ] || [ -z "$1" ]; then
    echo "📋 Local deployment mode"
    echo "Usage for remote: $0 <server-ip> [user]"
    echo "Example: $0 95.216.123.45 root"
    echo ""
fi

deploy_local() {
    echo "🏠 Local deployment..."
    
    # Stop existing container if running
    docker stop zion-test 2>/dev/null || true
    docker rm zion-test 2>/dev/null || true
    
    # Deploy with Docker Compose
    docker-compose -f docker-compose.prod.yml down 2>/dev/null || true
    docker-compose -f docker-compose.prod.yml up -d
    
    # Wait for health check
    echo "⏳ Waiting for service health check..."
    sleep 10
    
    # Verify deployment
    if curl -s http://localhost:18081/getinfo | grep -q '"status":"OK"'; then
        echo "✅ Local deployment successful!"
        echo "🌐 RPC: http://localhost:18081"
        echo "📊 Monitor: ./prod-monitor.sh monitor"
        return 0
    else
        echo "❌ Deployment verification failed"
        return 1
    fi
}

deploy_remote() {
    echo "🌐 Remote deployment to $SERVER_USER@$SERVER_IP..."
    
    # Test SSH connection first
    if ! ssh -o BatchMode=yes -o ConnectTimeout=5 $SERVER_USER@$SERVER_IP 'echo "SSH OK"' 2>/dev/null; then
        echo "❌ Cannot connect to $SERVER_USER@$SERVER_IP"
        echo "🔧 Check SSH keys or use: ssh-copy-id $SERVER_USER@$SERVER_IP"
        return 1
    fi
    
    # Create deployment package
    echo "📦 Creating deployment package..."
    tar -czf zion-deploy.tar.gz \
        docker-compose.prod.yml \
        docker/ \
        config/ \
        *.sh \
        README.md \
        DEPLOYMENT_LOG_20250919.md \
        PRODUCTION_GUIDE.md 2>/dev/null || true
    
    # Upload and deploy
    echo "⬆️  Uploading to server..."
    scp zion-deploy.tar.gz $SERVER_USER@$SERVER_IP:/tmp/
    
    # Remote deployment commands
    ssh $SERVER_USER@$SERVER_IP << 'REMOTE_COMMANDS'
        echo "🏗️  Setting up ZION on remote server..."
        
        # Detect OS
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            OS=$NAME
        else
            OS=$(uname -s)
        fi
        echo "📋 Detected OS: $OS"
        
        # Install Docker if not present
        if ! command -v docker &> /dev/null; then
            echo "🐳 Installing Docker..."
            curl -fsSL https://get.docker.com | sh
            if command -v systemctl &> /dev/null; then
                systemctl start docker
                systemctl enable docker
            fi
        fi
        
        # Install Docker Compose if not present
        if ! command -v docker-compose &> /dev/null; then
            echo "🐙 Installing Docker Compose..."
            curl -L "https://github.com/docker/compose/releases/download/v2.21.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
            chmod +x /usr/local/bin/docker-compose
        fi
        
        # Extract and deploy
        cd /opt
        rm -rf zion
        mkdir -p zion
        cd zion
        tar -xzf /tmp/zion-deploy.tar.gz
        rm /tmp/zion-deploy.tar.gz
        
        # Make scripts executable
        chmod +x *.sh 2>/dev/null || true
        
        # Create directories
        mkdir -p data logs
        
    # Clone repository (core is vendored, no submodules required for build)
    echo "📥 Cloning ZION repository..."
        if command -v git &> /dev/null; then
            git clone https://github.com/Yose144/Zion.git zion-full || echo "Git clone failed, using uploaded files"
            if [ -d "zion-full" ]; then
                cd zion-full
        # Note: zion-cryptonote is vendored in this repo; submodule init is not required.
                
                # Copy deployment files to full repo
                cp ../docker-compose.prod.yml . 2>/dev/null || true
                cp -r ../docker/ . 2>/dev/null || true
                cp -r ../config/ . 2>/dev/null || true
                cp ../*.sh . 2>/dev/null || true
                chmod +x *.sh 2>/dev/null || true
            fi
        else
            echo "📦 Git not available, using uploaded files only"
            mkdir -p zion-full
            cd zion-full
            cp -r ../* . 2>/dev/null || true
        fi
        
        # Deploy services
        echo "🚀 Deploying ZION services..."
        if [ -f "docker-compose.prod.yml" ]; then
            docker-compose -f docker-compose.prod.yml down 2>/dev/null || true
            docker-compose -f docker-compose.prod.yml up -d
        else
            echo "❌ docker-compose.prod.yml not found"
            exit 1
        fi
        
        # Wait and verify
        echo "⏳ Waiting for services to start..."
        sleep 20
        
        # Check if service is running
        if docker-compose -f docker-compose.prod.yml ps | grep -q "Up"; then
            echo "✅ Container is running"
            
            # Test RPC endpoint
            for i in {1..30}; do
                if curl -s http://localhost:18081/getinfo | grep -q '"status":"OK"' 2>/dev/null; then
                    echo "✅ Remote deployment successful!"
                    echo "🌐 RPC: http://$(curl -s ifconfig.me 2>/dev/null || echo $SERVER_IP):18081"
                    echo "📊 Monitor: ./prod-monitor.sh monitor"
                    break
                fi
                echo "⏳ Waiting for RPC... ($i/30)"
                sleep 2
            done
            
            # Create systemd service for auto-restart (if systemd available)
            if command -v systemctl &> /dev/null; then
                cat > /etc/systemd/system/zion.service << EOF
[Unit]
Description=ZION Cryptocurrency Node
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/zion/zion-full
ExecStart=/usr/local/bin/docker-compose -f docker-compose.prod.yml up -d
ExecStop=/usr/local/bin/docker-compose -f docker-compose.prod.yml down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF
                
                systemctl daemon-reload
                systemctl enable zion
                echo "🔧 Systemd service 'zion' created and enabled"
            fi
            
        else
            echo "❌ Container failed to start"
            docker-compose -f docker-compose.prod.yml logs
            exit 1
        fi
REMOTE_COMMANDS
    
    local remote_exit_code=$?
    
    # Cleanup
    rm zion-deploy.tar.gz
    
    if [ $remote_exit_code -eq 0 ]; then
        echo "🎉 Remote deployment completed!"
        echo "🌐 Server: http://$SERVER_IP:18081"
        return 0
    else
        echo "❌ Remote deployment failed"
        return 1
    fi
}

case "$SERVER_IP" in
    "localhost")
        deploy_local
        ;;
    *)
        deploy_remote
        ;;
esac

deployment_exit_code=$?

echo ""
echo "🌟 ZION Deployment Summary"
echo "========================="
if [ $deployment_exit_code -eq 0 ]; then
    echo "✅ Deployment successful!"
    echo "📱 Frontend connection: Update API endpoint to server IP"
    echo "🔧 Monitor health: ./prod-monitor.sh monitor"
    echo "📊 Check logs: docker-compose logs -f"
    echo "🛑 Stop server: docker-compose down"
    if [ "$SERVER_IP" != "localhost" ]; then
        echo "🔄 Restart: systemctl restart zion (on remote server)"
    fi
else
    echo "❌ Deployment failed!"
    echo "🔧 Check logs and try again"
fi

exit $deployment_exit_code
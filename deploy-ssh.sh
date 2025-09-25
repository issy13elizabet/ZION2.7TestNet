#!/bin/bash

# ZION Quick SSH Deployment Script
# ================================

SERVER_IP="${1:-91.98.122.165}"
SERVER_USER="${2:-root}"

if [ -z "$1" ]; then
    echo "🚀 ZION SSH Deployment Script"
    echo "Usage: $0 <server-ip> [user]"
    echo "Example: $0 91.98.122.165 root"
    echo ""
    echo "📋 This script will:"
    echo "1. Upload deployment files via SCP"
    echo "2. Connect via SSH and deploy ZION"
    echo "3. Start blockchain node"
    echo ""
    echo "⚠️  SSH password will be required"
    exit 0
fi

echo "🚀 ZION SSH Deployment to $SERVER_USER@$SERVER_IP"
echo "================================================"

# Create deployment package
echo "📦 Creating deployment package..."
tar -czf zion-ssh-deploy.tar.gz \
    docker-compose.prod.yml \
    docker/ \
    config/ \
    *.sh \
    README.md 2>/dev/null

echo "⬆️  Uploading files to server (password required)..."
scp zion-ssh-deploy.tar.gz $SERVER_USER@$SERVER_IP:/tmp/

echo "🔗 Connecting to server for deployment..."
ssh $SERVER_USER@$SERVER_IP << 'EOF'
    echo "🏗️  ZION Server Setup Starting..."
    
    # Update system
    apt update
    
    # Install Docker
    if ! command -v docker &> /dev/null; then
        echo "🐳 Installing Docker..."
        curl -fsSL https://get.docker.com | sh
        systemctl start docker
        systemctl enable docker
    fi
    
    # Install Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo "🐙 Installing Docker Compose..."
        curl -L "https://github.com/docker/compose/releases/download/v2.21.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        chmod +x /usr/local/bin/docker-compose
    fi
    
    # Setup UFW firewall for ZION
    echo "🔥 Configuring firewall for ZION..."
    ufw --force reset
    ufw default deny incoming
    ufw default allow outgoing
    ufw allow 22/tcp comment 'SSH'
    ufw allow 18080/tcp comment 'ZION P2P'
    ufw allow 18081/tcp comment 'ZION RPC'
    ufw allow 3333/tcp comment 'ZION Mining Pool'
    ufw --force enable
    
    # Install Git
    if ! command -v git &> /dev/null; then
        echo "📥 Installing Git..."
        apt install -y git
    fi
    
    # Setup deployment directory
    cd /opt
    rm -rf zion
    mkdir -p zion
    cd zion
    
    # Extract uploaded files
    tar -xzf /tmp/zion-ssh-deploy.tar.gz
    chmod +x *.sh
    
    # Clone repository (core is vendored, no submodule init needed)
    echo "📥 Cloning ZION repository..."
    git clone https://github.com/Yose144/Zion.git zion-repo
    cd zion-repo
    
    # Copy deployment files
    cp ../docker-compose.prod.yml .
    cp -r ../docker/ .
    cp -r ../config/ .
    cp ../*.sh .
    chmod +x *.sh
    
    # Deploy ZION
    echo "🚀 Deploying ZION services with mining pool..."
    docker-compose -f docker-compose.prod.yml down 2>/dev/null || true
    docker-compose -f docker-compose.prod.yml --profile pool up -d
    
    # Wait for startup
    echo "⏳ Waiting for services..."
    sleep 20
    
    # Verify deployment
    if curl -s http://localhost:18081/getinfo | grep -q '"status":"OK"'; then
        echo "✅ ZION deployment successful!"
        echo "🌐 RPC: http://$(curl -s ifconfig.me):18081"
        echo "🔗 P2P: Port 18080"
        echo "⛏️  Mining Pool: stratum+tcp://$(curl -s ifconfig.me):3333"
        echo ""
        echo "📊 Pool Connection Test:"
        echo "   nc -zv $(curl -s ifconfig.me) 3333"
        echo ""
        echo "🔧 SSH Tunnel for Local Mining:"
        echo "   ssh -L 3333:localhost:3333 root@$(curl -s ifconfig.me)"
        echo "   Then mine to: stratum+tcp://localhost:3333"
        
        # Create systemd service
        cat > /etc/systemd/system/zion.service << SYSTEMD_EOF
[Unit]
Description=ZION Cryptocurrency Node
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/zion/zion-repo
ExecStart=/usr/local/bin/docker-compose -f docker-compose.prod.yml --profile pool up -d
ExecStop=/usr/local/bin/docker-compose -f docker-compose.prod.yml --profile pool down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
SYSTEMD_EOF
        
        systemctl daemon-reload
        systemctl enable zion
        
        echo "🔧 Auto-restart service enabled"
        echo "🎉 ZION node is now running on the server!"
        
    else
        echo "❌ Deployment verification failed"
        docker-compose -f docker-compose.prod.yml logs
    fi
    
    # Cleanup
    rm /tmp/zion-ssh-deploy.tar.gz
EOF

# Cleanup local files
rm zion-ssh-deploy.tar.gz

echo ""
echo "🌟 SSH Deployment Complete!"
echo "=========================="
echo "🌐 Server URL: http://$SERVER_IP:18081"
echo "⛏️  Mining Pool: stratum+tcp://$SERVER_IP:3333"
echo ""
echo "� Management Commands:"
echo "�📊 Check status: ssh $SERVER_USER@$SERVER_IP 'docker ps'"
echo "🔧 Monitor: ssh $SERVER_USER@$SERVER_IP 'cd /opt/zion/zion-repo && ./prod-monitor.sh monitor'"
echo "🌉 SSH Tunnel: ssh -L 3333:localhost:3333 $SERVER_USER@$SERVER_IP"
echo ""
echo "💎 Happy Mining! Connect your miners to port 3333!"
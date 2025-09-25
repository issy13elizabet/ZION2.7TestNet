#!/bin/bash

# ZION Emergency Fix - Use Pre-built Image
# ========================================

SERVER_IP="${1:-91.98.122.165}"
SERVER_USER="${2:-root}"

echo "🚀 ZION Emergency Fix - Using Pre-built Image"
echo "============================================="

if [ -z "$1" ]; then
    echo "Usage: $0 <server-ip> [user]"
    echo "Example: $0 91.98.122.165 root"
    exit 1
fi

# Export and upload pre-built image
echo "📦 Exporting local Docker image..."
docker save zion:production | gzip > zion-production.tar.gz

echo "⬆️  Uploading image to server..."
scp zion-production.tar.gz $SERVER_USER@$SERVER_IP:/tmp/

echo "🚀 Deploying with pre-built image..."
ssh $SERVER_USER@$SERVER_IP << 'EOF'
    echo "🏗️  Loading pre-built ZION image..."
    
    # Stop any existing services
    cd /opt/zion/zion-repo 2>/dev/null && docker-compose down 2>/dev/null || true
    
    # Load the pre-built image
    cd /tmp
    gunzip -c zion-production.tar.gz | docker load
    rm zion-production.tar.gz
    
    # Create simple deployment
    cd /opt/zion
    mkdir -p zion-prebuilt
    cd zion-prebuilt
    
    # Simple compose with pre-built image
    cat > docker-compose.yml << COMPOSE_EOF
services:
  zion-node:
    image: zion:production
    container_name: zion-production
    ports:
      - "18080:18080"
      - "18081:18081"
      - "8070:8070"
    volumes:
      - ./data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:18081/getinfo || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
COMPOSE_EOF

    # Create data directory
    mkdir -p data
    
    # Deploy the service
    echo "🚀 Starting ZION service..."
    docker-compose up -d
    
    # Wait and verify
    echo "⏳ Waiting for service startup..."
    sleep 20
    
    if docker ps | grep -q "zion-production.*Up"; then
        echo "✅ Container is running!"
        
        # Test RPC with retries
        for i in {1..10}; do
            if curl -s http://localhost:18081/getinfo | grep -q '"status":"OK"' 2>/dev/null; then
                PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "server-ip")
                echo "✅ RPC endpoint responding!"
                echo "🌐 Server: http://$PUBLIC_IP:18081"
                echo "🎉 ZION deployment successful!"
                
                # Create systemd service
                cat > /etc/systemd/system/zion.service << SYSTEMD_EOF
[Unit]
Description=ZION Cryptocurrency Node
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/zion/zion-prebuilt
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
SYSTEMD_EOF
                
                systemctl daemon-reload
                systemctl enable zion
                echo "🔧 Auto-restart service enabled"
                exit 0
            fi
            echo "⏳ Testing RPC... ($i/10)"
            sleep 3
        done
        
        echo "⚠️  Container running but RPC not ready yet"
        echo "📊 Container status:"
        docker ps
        echo "📝 Recent logs:"
        docker logs zion-production --tail=10
    else
        echo "❌ Container failed to start"
        echo "📊 Container status:"
        docker ps -a
        echo "📝 Logs:"
        docker-compose logs
    fi
EOF

# Cleanup
rm zion-production.tar.gz 2>/dev/null || true

echo ""
echo "🌟 Emergency Fix Deployment Complete!"
echo "==================================="
echo "🌐 Test: curl http://$SERVER_IP:18081/getinfo"
echo "📊 Status: ssh $SERVER_USER@$SERVER_IP 'docker ps'"
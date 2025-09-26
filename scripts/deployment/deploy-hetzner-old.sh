#!/bin/bash

# ZION Production Server Deployment to Hetzner VPS
# Updated: 19. září 2025 - Production Ready Version
# Usage: ./deploy-hetzner.sh [server-ip] [user]

set -e

SERVER_IP="${1:-91.98.122.165}"
SERVER_USER="${2:-root}"

echo "🚀 ZION Production Deployment to Hetzner VPS"
echo "============================================="
echo "Target: $SERVER_USER@$SERVER_IP"
echo "Date: $(date)"
echo ""
echo "🚀 Zion Pool Deployment for Hetzner Server"
echo "📍 Server IP: $SERVER_IP"
echo "================================================"

# Update system
echo "📦 Updating system packages..."
apt update && apt upgrade -y

# Create user if not exists
if ! id -u zion > /dev/null 2>&1; then
    echo "👤 Creating zion user..."
    adduser --gecos "" --disabled-password zion
    echo "zion:$(openssl rand -base64 32)" | chpasswd
    usermod -aG sudo zion
    echo "zion ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
fi

# Install UFW and configure firewall
echo "🔥 Configuring firewall..."
apt install ufw -y

# Configure UFW rules
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp comment 'SSH'
ufw allow 3333/tcp comment 'Stratum Pool'
ufw allow 18081/tcp comment 'P2P Network'
ufw allow 18080/tcp comment 'RPC API'
ufw allow 80/tcp comment 'HTTP'
ufw allow 443/tcp comment 'HTTPS'
ufw --force enable

echo "✅ Firewall configured"
ufw status verbose

# Install Docker
if ! command -v docker &> /dev/null; then
    echo "🐳 Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
    usermod -aG docker zion
fi

# Install Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "📦 Installing Docker Compose..."
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
fi

# Create directories
echo "📁 Creating project directories..."
mkdir -p /opt/zion/data/pool /opt/zion/logs/pool
chown -R zion:zion /opt/zion

# Clone repository
echo "📥 Cloning Zion repository..."
cd /opt/zion
if [ ! -d ".git" ]; then
    git clone https://github.com/Yose144/Zion.git .
else
    git pull
fi

# Update submodules
git submodule update --init --recursive

# Build Docker image
echo "🔨 Building Docker image..."
docker build -t zion:latest .

# Create environment file
echo "⚙️ Creating configuration..."
cat > /opt/zion/.env <<EOF
# Zion Pool Configuration
POOL_FEE=1
POOL_DIFFICULTY=1000
SEED_NODES=""
SERVER_IP=$SERVER_IP
EOF

# Create docker-compose override for production
cat > /opt/zion/docker-compose.override.yml <<EOF
version: '3.8'

services:
  pool:
    image: zion:latest
    container_name: zion-pool
    environment:
      - ZION_MODE=pool
      - POOL_PORT=3333
      - POOL_DIFFICULTY=1000
      - POOL_FEE=1
      - ZION_LOG_LEVEL=info
    volumes:
      - ./data/pool:/home/zion/.zion
      - ./logs/pool:/var/log/zion
    ports:
      - "3333:3333"
      - "18080:18080"
      - "18081:18081"
    restart: always
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "10"
EOF

# Create systemd service
echo "🔧 Creating systemd service..."
cat > /etc/systemd/system/zion-pool.service <<EOF
[Unit]
Description=Zion Mining Pool
Requires=docker.service
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=simple
Restart=always
RestartSec=10
User=root
WorkingDirectory=/opt/zion
ExecStart=/usr/local/bin/docker-compose up pool
ExecStop=/usr/local/bin/docker-compose down

[Install]
WantedBy=multi-user.target
EOF

# Start the service
echo "🚀 Starting Zion Pool..."
systemctl daemon-reload
systemctl enable zion-pool.service
systemctl start zion-pool.service

# Install monitoring tools
echo "📊 Installing monitoring tools..."
apt install -y htop iotop ncdu net-tools

# Wait for service to start
sleep 10

# Check status
echo ""
echo "================================================"
echo "✅ DEPLOYMENT COMPLETE!"
echo "================================================"
echo ""
echo "📍 Server IP: $SERVER_IP"
echo "🔗 Stratum Pool: stratum+tcp://$SERVER_IP:3333"
echo "🔗 P2P Port: $SERVER_IP:18081"
echo "🔗 RPC API: http://$SERVER_IP:18080"
echo ""
echo "📋 Useful commands:"
echo "  View logs:        docker logs -f zion-pool"
echo "  Service status:   systemctl status zion-pool"
echo "  Restart service:  systemctl restart zion-pool"
echo "  Check ports:      netstat -tlnp | grep -E '3333|18080|18081'"
echo ""
echo "🎯 Test connection from your local machine:"
echo "  nc -zv $SERVER_IP 3333"
echo "  curl http://$SERVER_IP:18080/status"
echo ""

# Show current status
docker ps
systemctl status zion-pool --no-pager

echo ""
echo "🎉 Your Zion pool is ready at $SERVER_IP:3333!"
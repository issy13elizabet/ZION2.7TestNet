#!/bin/bash

# Zion Docker Deployment Script for Ubuntu Server
# Usage: ./deploy.sh [seed|pool|full]

set -e

MODE=${1:-pool}
SERVER_IP=$(curl -s ifconfig.me)

echo "================================================"
echo "Zion Docker Deployment Script"
echo "Mode: $MODE"
echo "Server IP: $SERVER_IP"
echo "================================================"

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
    echo "Docker installed. Please log out and back in, then run this script again."
    exit 0
fi

# Install Docker Compose if not present
if ! command -v docker-compose &> /dev/null; then
    echo "Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Create deployment directory
DEPLOY_DIR="/opt/zion"
sudo mkdir -p $DEPLOY_DIR
sudo chown -R $USER:$USER $DEPLOY_DIR
cd $DEPLOY_DIR

# Clone repository if not exists
if [ ! -d ".git" ]; then
    echo "Cloning Zion repository..."
    git clone https://github.com/Yose144/Zion.git .
else
    echo "Updating Zion repository..."
    git pull
fi

# No submodules required (zion-cryptonote is vendored). If other submodules are added in future, re-enable as needed.
# git submodule update --init --recursive

# Create data directories
mkdir -p data/pool data/seed data/node logs/pool logs/seed logs/node

# Build Docker image
echo "Building Docker image..."
docker build -t zion:latest .

# Configure firewall
echo "Configuring firewall..."
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS

case "$MODE" in
    seed)
        sudo ufw allow 18080/tcp  # RPC
        sudo ufw allow 18081/tcp  # P2P
        
        # Start seed node
        echo "Starting seed node..."
        docker-compose up -d seed-node
        ;;
    
    pool)
        sudo ufw allow 3333/tcp   # Stratum
        sudo ufw allow 18081/tcp  # P2P
        
        # Create .env file for production
        cat > .env <<EOF
# Zion Pool Configuration
POOL_FEE=1
POOL_DIFFICULTY=1000
SEED_NODES=
GRAFANA_PASSWORD=$(openssl rand -base64 32)
EOF
        
        echo "Starting pool server..."
        docker-compose -f docker-compose.prod.yml up -d pool
        
        echo ""
        echo "Pool server started!"
        echo "Stratum endpoint: stratum+tcp://$SERVER_IP:3333"
        ;;
    
    full)
        sudo ufw allow 18080/tcp  # RPC
        sudo ufw allow 18081/tcp  # P2P
        sudo ufw allow 3333/tcp   # Stratum
        
        echo "Starting full setup (seed + pool + monitoring)..."
        docker-compose -f docker-compose.prod.yml --profile monitoring up -d
        
        echo ""
        echo "Full setup started!"
        echo "Stratum endpoint: stratum+tcp://$SERVER_IP:3333"
        echo "Grafana: http://$SERVER_IP:3000 (admin/admin)"
        ;;
    
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [seed|pool|full]"
        exit 1
        ;;
esac

# Enable UFW
sudo ufw --force enable

# Show container status
echo ""
echo "Container status:"
docker ps

echo ""
echo "View logs with: docker logs -f zion-pool"
echo "Stop with: docker-compose down"
echo ""
echo "Deployment complete!"
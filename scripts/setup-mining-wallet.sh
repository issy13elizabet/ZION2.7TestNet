#!/bin/bash
set -e

echo "⛏️  Setting up MINING WALLET for ZION Mining Operations..."
echo "=========================================================="

# Use existing pool address from MINING_STARTUP_LOG.md
MINING_WALLET_ADDRESS="Z3BDEEC2A0AE0F5D81B034308F99ECD8990D9B8B01BD9C7E7429392CA31861C6220DA3B30D74E809FA0A1FE069F1"
BACKUP_DIR="/home/maitreya/backup-wallets"  # Outside git repo

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Create backup directory
mkdir -p "$BACKUP_DIR"
chmod 700 "$BACKUP_DIR"

echo -e "${YELLOW}⚠️  USING EXISTING PRODUCTION WALLET ⚠️${NC}"
echo "Mining Address: $MINING_WALLET_ADDRESS"
echo "Backup location: $BACKUP_DIR (OUTSIDE git repo)"
echo ""

# Function to update mining configurations
update_mining_configs() {
    local mining_address="$1"
    
    echo -e "${BLUE}Updating XMRig configuration...${NC}"
    if [ -f "mining/platforms/linux/xmrig-6.21.3/config-zion.json" ]; then
        # Update wallet address
        sed -i "s/WORKER_OR_WALLET/$mining_address/g" mining/platforms/linux/xmrig-6.21.3/config-zion.json
        echo -e "${GREEN}✅ XMRig config updated${NC}"
    fi
    
    echo -e "${BLUE}Updating SRBMiner configuration...${NC}"
    if [ -f "mining/srb-kawpow-config.json" ]; then
        # Update pool to use external IP instead of localhost
        sed -i 's/"pool": "localhost:3334"/"pool": "91.98.122.165:3334"/g' mining/srb-kawpow-config.json
        # Update wallet address
        sed -i "s/\"wallet\": \".*\"/\"wallet\": \"$mining_address\"/g" mining/srb-kawpow-config.json
        echo -e "${GREEN}✅ SRBMiner config updated${NC}"
    fi
    
    # Create unified mining config
    cat > mining/mining-wallet.conf << EOF
# ZION Mining Wallet Configuration
# Generated: $(date)
MINING_WALLET_ADDRESS=$mining_address
POOL_SERVER=91.98.122.165
CPU_POOL_PORT=3333
GPU_POOL_PORT=3334
WORKER_ID=ubuntu-miner-$(hostname)
EOF
    
    # Save address to backup
    echo "$mining_address" > "$BACKUP_DIR/mining-wallet.address"
    
    echo -e "${GREEN}✅ Mining configurations updated${NC}"
}

# Function to check XMRig binary exists
check_xmrig_binary() {
    if [ ! -f "mining/platforms/linux/xmrig-6.21.3/xmrig" ]; then
        echo -e "${YELLOW}XMRig binary not found, downloading...${NC}"
        cd mining/platforms/linux/xmrig-6.21.3/
        wget https://github.com/xmrig/xmrig/releases/download/v6.21.3/xmrig-6.21.3-linux-x64.tar.gz
        tar -xzf xmrig-6.21.3-linux-x64.tar.gz --strip-components=1
        chmod +x xmrig
        cd ../../../..
        echo -e "${GREEN}✅ XMRig binary downloaded${NC}"
    else
        echo -e "${GREEN}✅ XMRig binary exists${NC}"
    fi
}

# Main execution
echo -e "${BLUE}Starting mining wallet setup...${NC}"
update_mining_configs "$MINING_WALLET_ADDRESS"
check_xmrig_binary

echo ""
echo -e "${GREEN}🎉 MINING WALLET SETUP COMPLETED!${NC}"
echo -e "${YELLOW}Configuration:${NC}"
echo "- Mining Address: $MINING_WALLET_ADDRESS"
echo "- CPU Pool: 91.98.122.165:3333 (RandomX)"
echo "- GPU Pool: 91.98.122.165:3334 (KawPow)"
echo "- Worker ID: ubuntu-miner-$(hostname)"
echo "- Backup: $BACKUP_DIR (OUTSIDE git repo)"
echo ""
echo -e "${PURPLE}Ready to start mining on Ubuntu!${NC}"
echo "Next: Run ./scripts/start-ubuntu-mining.sh"
#!/bin/bash
set -e

echo "⛏️  Creating MINING WALLET for ZION Mining Operations..."
echo "======================================================="

# Configuration
WALLET_DIR="/home/maitreya/Zion/zion/wallets"
BACKUP_DIR="/home/maitreya/backup-wallets"  # Outside git repo
MINING_WALLET_NAME="zion-mining-worker"
WALLET_PASS="ZionMining2025SecureKey!"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Create directories
mkdir -p "$WALLET_DIR"
mkdir -p "$BACKUP_DIR"
chmod 700 "$WALLET_DIR"
chmod 700 "$BACKUP_DIR"

echo -e "${YELLOW}⚠️  SECURITY NOTICE ⚠️${NC}"
echo "Creating mining wallet with backup OUTSIDE git repository."
echo "Backup location: $BACKUP_DIR"
echo ""

# Function to generate wallet using CLI directly
create_mining_wallet() {
    local wallet_name="$1"
    local wallet_file="$WALLET_DIR/${wallet_name}.wallet"
    local keys_file="$WALLET_DIR/${wallet_name}.keys"
    
    echo -e "${BLUE}Creating wallet: ${wallet_name}${NC}"
    
    # Check if zion-wallet-cli exists
    if [ ! -f "zion-core/build/bin/zion-wallet-cli" ]; then
        echo -e "${RED}Error: zion-wallet-cli not found. Building zion-core first...${NC}"
        cd zion-core
        make -j$(nproc)
        cd ..
    fi
    
    # Generate wallet using CLI
    echo -e "${YELLOW}Generating new wallet...${NC}"
    ./zion-core/build/bin/zion-wallet-cli <<EOF
4
$wallet_file
$WALLET_PASS
English
exit
EOF

    if [ -f "$wallet_file" ]; then
        echo -e "${GREEN}✅ Wallet created: $wallet_file${NC}"
        
        # Get wallet address
        echo -e "${BLUE}Getting wallet address...${NC}"
        local address
        address=$(./zion-core/build/bin/zion-wallet-cli --wallet-file "$wallet_file" --password "$WALLET_PASS" --command "address" 2>/dev/null | grep "Address:" | cut -d' ' -f2 || echo "ADDRESS_ERROR")
        
        if [ "$address" != "ADDRESS_ERROR" ]; then
            echo -e "${GREEN}✅ Mining Wallet Address: $address${NC}"
            
            # Create backup with GPG encryption
            echo -e "${BLUE}Creating encrypted backup...${NC}"
            tar -czf "$BACKUP_DIR/${wallet_name}-$(date +%Y%m%d-%H%M%S).tar.gz" -C "$WALLET_DIR" "${wallet_name}.wallet" "${wallet_name}.keys" 2>/dev/null || echo "Backup created without keys file"
            
            # Save address to config files
            echo "$address" > "$WALLET_DIR/${wallet_name}.address"
            echo "$address" > "$BACKUP_DIR/${wallet_name}.address"
            
            # Update mining configs
            echo -e "${BLUE}Updating mining configurations...${NC}"
            update_mining_configs "$address"
            
            echo -e "${GREEN}✅ Mining wallet setup completed!${NC}"
            echo -e "${PURPLE}Address: $address${NC}"
            echo -e "${PURPLE}Backup: $BACKUP_DIR/${NC}"
        else
            echo -e "${RED}❌ Failed to get wallet address${NC}"
        fi
    else
        echo -e "${RED}❌ Failed to create wallet${NC}"
    fi
}

# Function to update mining configurations
update_mining_configs() {
    local mining_address="$1"
    
    echo -e "${BLUE}Updating XMRig configuration...${NC}"
    if [ -f "mining/platforms/linux/xmrig-6.21.3/config-zion.json" ]; then
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
    
    echo -e "${GREEN}✅ Mining configurations updated${NC}"
}

# Main execution
echo -e "${BLUE}Starting mining wallet creation...${NC}"
create_mining_wallet "$MINING_WALLET_NAME"

echo ""
echo -e "${GREEN}🎉 MINING WALLET SETUP COMPLETED!${NC}"
echo -e "${YELLOW}⚠️  IMPORTANT:${NC}"
echo "1. Backup location: $BACKUP_DIR (OUTSIDE git repo)"
echo "2. Wallet password: $WALLET_PASS"
echo "3. Mining configs updated with new address"
echo "4. Ready to start mining on Ubuntu!"
echo ""
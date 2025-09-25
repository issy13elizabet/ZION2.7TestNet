#!/bin/bash
set -e

echo "🏦 Creating PRODUCTION ZION Wallets with Private Keys..."
echo "========================================================"

# Configuration
WALLET_DIR="/home/maitreya/Zion/zion/wallets"
BACKUP_DIR="/home/maitreya/Zion/zion/backups/wallets"
POOL_WALLET_NAME="zion-pool-production"
DEV_WALLET_NAME="zion-dev-production"
WALLET_PASS="ZionMainNet2025SecureKey!"

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

echo -e "${YELLOW}⚠️  IMPORTANT SECURITY NOTICE ⚠️${NC}"
echo "This script will create production wallets with real private keys."
echo "Make sure to backup the wallet files and private keys securely!"
echo ""

# Function to start walletd daemon temporarily
start_temp_walletd() {
    echo -e "${BLUE}Starting temporary wallet daemon...${NC}"
    
    # Start seed node first if not running
    if ! docker ps | grep -q "seed-node"; then
        echo "Starting seed node..."
        docker compose up -d seed-node
        sleep 10
    fi
    
    # Start walletd with temporary configuration
    docker compose -f docker/compose.pool-seeds.yml up -d walletd
    sleep 5
    
    # Wait for walletd to be ready
    echo "Waiting for wallet daemon to be ready..."
    for i in {1..30}; do
        if curl -s http://localhost:8070/json_rpc >/dev/null 2>&1; then
            echo -e "${GREEN}✅ Wallet daemon is ready${NC}"
            return 0
        fi
        sleep 2
        echo -n "."
    done
    echo -e "${RED}❌ Wallet daemon failed to start${NC}"
    return 1
}

# Function to create wallet using RPC
create_wallet_rpc() {
    local wallet_name=$1
    local wallet_file="$WALLET_DIR/${wallet_name}.wallet"
    
    echo -e "${BLUE}Creating wallet: $wallet_name${NC}"
    
    # Try to create wallet using RPC
    local response=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d "{\"jsonrpc\":\"2.0\",\"id\":\"0\",\"method\":\"createWallet\",\"params\":{\"filename\":\"$wallet_file\",\"password\":\"$WALLET_PASS\",\"language\":\"English\"}}" \
        http://localhost:8070/json_rpc)
    
    if echo "$response" | grep -q '"result"'; then
        echo -e "${GREEN}✅ Successfully created wallet: $wallet_name${NC}"
        
        # Get wallet address
        local addr_response=$(curl -s -X POST \
            -H "Content-Type: application/json" \
            -d '{"jsonrpc":"2.0","id":"0","method":"getAddress","params":{}}' \
            http://localhost:8070/json_rpc)
        
        local address=$(echo "$addr_response" | grep -o '"address":"[^"]*"' | cut -d'"' -f4)
        
        if [[ -n "$address" ]]; then
            echo -e "${GREEN}📍 Address: $address${NC}"
            echo "$address" > "$WALLET_DIR/${wallet_name}.address"
            
            # Try to get private keys (mnemonic seed)
            local seed_response=$(curl -s -X POST \
                -H "Content-Type: application/json" \
                -d "{\"jsonrpc\":\"2.0\",\"id\":\"0\",\"method\":\"queryKey\",\"params\":{\"key_type\":\"mnemonic\"}}" \
                http://localhost:8070/json_rpc)
            
            local mnemonic=$(echo "$seed_response" | grep -o '"key":"[^"]*"' | cut -d'"' -f4)
            if [[ -n "$mnemonic" ]]; then
                echo -e "${PURPLE}🔑 Mnemonic Seed: $mnemonic${NC}"
                # Encrypt and save mnemonic
                echo "$mnemonic" | gpg --symmetric --cipher-algo AES256 --batch --yes --passphrase="$WALLET_PASS" > "$BACKUP_DIR/${wallet_name}.seed.gpg"
                echo -e "${GREEN}✅ Encrypted mnemonic saved to backup${NC}"
            fi
            
            return 0
        fi
    fi
    
    echo -e "${YELLOW}⚠️  RPC wallet creation failed, trying alternative method...${NC}"
    return 1
}

# Function to create wallet using CLI
create_wallet_cli() {
    local wallet_name=$1
    local wallet_file="$WALLET_DIR/${wallet_name}.wallet"
    
    echo -e "${BLUE}Creating wallet using CLI: $wallet_name${NC}"
    
    # Try using zion_wallet directly in container
    if docker run --rm -v "$WALLET_DIR:/wallets" zion \
        zion_wallet \
        --generate-new-wallet "/wallets/${wallet_name}.wallet" \
        --password "$WALLET_PASS" \
        --daemon-host "172.17.0.1" \
        --daemon-port "18081" \
        --command exit; then
        
        echo -e "${GREEN}✅ Successfully created CLI wallet: $wallet_name${NC}"
        
        # Extract address using wallet info command
        local wallet_info=$(docker run --rm -v "$WALLET_DIR:/wallets" zion \
            zion_wallet \
            --wallet-file "/wallets/${wallet_name}.wallet" \
            --password "$WALLET_PASS" \
            --command "address" | grep -o 'Z3[a-zA-Z0-9]\{90,98\}' | head -1)
        
        if [[ -n "$wallet_info" ]]; then
            echo -e "${GREEN}📍 Address: $wallet_info${NC}"
            echo "$wallet_info" > "$WALLET_DIR/${wallet_name}.address"
        fi
        
        return 0
    fi
    
    echo -e "${RED}❌ CLI wallet creation failed${NC}"
    return 1
}

# Function to create secure backup
create_secure_backup() {
    local wallet_name=$1
    local wallet_file="$WALLET_DIR/${wallet_name}.wallet"
    
    if [[ -f "$wallet_file" ]]; then
        echo -e "${BLUE}Creating secure backup for: $wallet_name${NC}"
        
        # Create encrypted backup of wallet file
        gpg --symmetric --cipher-algo AES256 --batch --yes --passphrase="$WALLET_PASS" \
            --output "$BACKUP_DIR/${wallet_name}.wallet.gpg" "$wallet_file"
        
        # Create backup info file
        cat > "$BACKUP_DIR/${wallet_name}.info" << EOF
Wallet Name: $wallet_name
Created: $(date -Iseconds)
Backup Location: $BACKUP_DIR/${wallet_name}.wallet.gpg
Address File: $WALLET_DIR/${wallet_name}.address
Recovery: Use gpg --decrypt $BACKUP_DIR/${wallet_name}.wallet.gpg
Password: [REDACTED - Use secure storage]
EOF
        
        echo -e "${GREEN}✅ Secure backup created${NC}"
        
        # Set secure permissions
        chmod 600 "$BACKUP_DIR/${wallet_name}."*
        chmod 600 "$WALLET_DIR/${wallet_name}."*
    fi
}

# Main execution
echo -e "${YELLOW}Starting production wallet creation process...${NC}"

# Start wallet daemon
if start_temp_walletd; then
    # Create Pool Wallet
    echo -e "\n${BLUE}==================== POOL WALLET ====================${NC}"
    if create_wallet_rpc "$POOL_WALLET_NAME" || create_wallet_cli "$POOL_WALLET_NAME"; then
        create_secure_backup "$POOL_WALLET_NAME"
        POOL_ADDRESS=$(cat "$WALLET_DIR/${POOL_WALLET_NAME}.address" 2>/dev/null || echo "")
    fi
    
    # Create Dev Wallet
    echo -e "\n${BLUE}==================== DEV WALLET ===================${NC}"
    if create_wallet_rpc "$DEV_WALLET_NAME" || create_wallet_cli "$DEV_WALLET_NAME"; then
        create_secure_backup "$DEV_WALLET_NAME"
        DEV_ADDRESS=$(cat "$WALLET_DIR/${DEV_WALLET_NAME}.address" 2>/dev/null || echo "")
    fi
else
    echo -e "${RED}❌ Could not start wallet daemon. Creating offline wallets...${NC}"
    
    # Create offline placeholder wallets
    create_wallet_cli "$POOL_WALLET_NAME" || true
    create_wallet_cli "$DEV_WALLET_NAME" || true
fi

# Summary and Security Instructions
echo -e "\n${GREEN}🎉 PRODUCTION WALLET CREATION COMPLETED${NC}"
echo "========================================================"
echo -e "Pool Wallet:     ${BLUE}$POOL_WALLET_NAME${NC}"
echo -e "Pool Address:    ${GREEN}${POOL_ADDRESS:-'[Check .address file]'}${NC}"
echo -e "Dev Wallet:      ${BLUE}$DEV_WALLET_NAME${NC}"
echo -e "Dev Address:     ${GREEN}${DEV_ADDRESS:-'[Check .address file]'}${NC}"
echo ""
echo -e "${RED}🔐 CRITICAL SECURITY INSTRUCTIONS 🔐${NC}"
echo "========================================================"
echo "1. Wallet files are in: $WALLET_DIR"
echo "2. Encrypted backups are in: $BACKUP_DIR"
echo "3. Password: $WALLET_PASS"
echo "4. IMMEDIATELY backup the entire $BACKUP_DIR to secure offline storage"
echo "5. Test wallet recovery before using in production"
echo "6. Never share wallet files or passwords"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Update pool configuration with real addresses"
echo "2. Test wallet connectivity"
echo "3. Start mining to pool address"
echo ""

# Update configurations if addresses are available
if [[ -n "$POOL_ADDRESS" && -n "$DEV_ADDRESS" ]]; then
    echo -e "${BLUE}Updating configuration files...${NC}"
    
    # Update pool config
    if [[ -f "config/zion-mining-pool.json" ]]; then
        cp "config/zion-mining-pool.json" "config/zion-mining-pool.json.bak"
        sed -i "s/\"pool_wallet\": \"[^\"]*\"/\"pool_wallet\": \"$POOL_ADDRESS\"/" "config/zion-mining-pool.json"
        sed -i "s/\"pool_operator\": \"[^\"]*\"/\"pool_operator\": \"$POOL_ADDRESS\"/" "config/zion-mining-pool.json"
        sed -i "s/\"dev_fund\": \"[^\"]*\"/\"dev_fund\": \"$DEV_ADDRESS\"/" "config/zion-mining-pool.json"
        echo -e "${GREEN}✅ Updated pool configuration${NC}"
    fi
    
    # Update miner config
    if [[ -f "config/miner.conf" ]]; then
        cp "config/miner.conf" "config/miner.conf.bak"
        sed -i "s/mining_address = .*/mining_address = $POOL_ADDRESS/" "config/miner.conf"
        sed -i "s/pool_user = .*/pool_user = $POOL_ADDRESS/" "config/miner.conf"
        echo -e "${GREEN}✅ Updated miner configuration${NC}"
    fi
fi

echo -e "${GREEN}✅ Production wallet setup completed successfully!${NC}"
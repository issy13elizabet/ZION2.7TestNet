#!/bin/bash
# Quick ZION Address Generator for Testing

# Generate valid-looking ZION addresses for testing
generate_zion_address() {
    local prefix="Z3"
    local random_part=$(openssl rand -hex 45 | tr '[:lower:]' '[:upper:]')
    echo "${prefix}${random_part}"
}

# Generate addresses
POOL_ADDRESS=$(generate_zion_address)
DEV_ADDRESS=$(generate_zion_address)

echo "🏦 Generated Test ZION Addresses:"
echo "=================================="
echo "Pool Address: $POOL_ADDRESS"
echo "Dev Address:  $DEV_ADDRESS"
echo ""

# Export for use in other scripts
export ZION_POOL_ADDRESS="$POOL_ADDRESS"
export ZION_DEV_ADDRESS="$DEV_ADDRESS"

# Update pool configuration
if [[ -f "/home/maitreya/Zion/zion/config/zion-mining-pool.json" ]]; then
    echo "Updating pool configuration..."
    cp "/home/maitreya/Zion/zion/config/zion-mining-pool.json" "/home/maitreya/Zion/zion/config/zion-mining-pool.json.bak"
    
    # Update the JSON configuration
    cat > "/home/maitreya/Zion/zion/config/zion-mining-pool.json" << EOF
{
    "zion-mining-pool": {
        "pool_wallet": "$POOL_ADDRESS",
        "pool_fee": 2.5,
        "min_payout": 0.1,
        "network": "zion-mainnet-v2",
        "blockchain_rpc": {
            "host": "localhost",
            "port": 18080,
            "username": "zion-pool",
            "password": "secure-rpc-password-2025"
        },
        "stratum_ports": {
            "randomx": 3333,
            "kawpow": 3334, 
            "ethash": 3335,
            "cryptonight": 3336,
            "octopus": 3337,
            "ergo": 3338
        },
        "difficulty_targets": {
            "randomx": 100000,
            "kawpow": 50000000,
            "ethash": 4000000000,
            "cryptonight": 1000000,
            "octopus": 800000000,
            "ergo": 1500000000
        },
        "block_rewards": {
            "randomx": 2.5,
            "kawpow": 2.5,
            "ethash": 2.5,
            "cryptonight": 2.5,
            "octopus": 2.5,
            "ergo": 2.5
        },
        "payout_addresses": {
            "pool_operator": "$POOL_ADDRESS",
            "dev_fund": "$DEV_ADDRESS"
        }
    }
}
EOF
    
    echo "✅ Updated pool configuration with new addresses"
fi

# Update miner configuration
if [[ -f "/home/maitreya/Zion/zion/config/miner.conf" ]]; then
    echo "Updating miner configuration..."
    sed -i "s/# mining_address = .*/mining_address = $POOL_ADDRESS/" "/home/maitreya/Zion/zion/config/miner.conf"
    sed -i "s/# pool_user = .*/pool_user = $POOL_ADDRESS/" "/home/maitreya/Zion/zion/config/miner.conf"
    echo "✅ Updated miner configuration"
fi

echo ""
echo "✅ Address generation and configuration update completed!"
echo "Pool Address: $POOL_ADDRESS"
echo "Dev Address:  $DEV_ADDRESS"
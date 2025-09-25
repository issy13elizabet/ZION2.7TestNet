#!/bin/bash

echo "⛏️  ZION External Mining Setup"
echo "============================"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Get pool address
POOL_ADDR=$(cat wallets/zion-pool-prod.address 2>/dev/null)
if [[ -z "$POOL_ADDR" ]]; then
    echo -e "${RED}❌ Pool address not found! Run wallet creation first.${NC}"
    exit 1
fi

echo -e "${GREEN}🏦 Pool Address: $POOL_ADDR${NC}"
echo ""

# Function to check if pool is running
check_pool() {
    if nc -z localhost 3333 >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Pool is running on port 3333${NC}"
        return 0
    else
        echo -e "${RED}❌ Pool is not running! Start it first: docker compose up -d pool-server${NC}"
        return 1
    fi
}

# Function to start XMRig CPU mining
start_xmrig() {
    echo -e "${BLUE}🖥️  Starting XMRig CPU Mining...${NC}"
    
    if [[ ! -f "mining/xmrig-6.21.3/xmrig" ]]; then
        echo -e "${RED}❌ XMRig binary not found in mining/xmrig-6.21.3/${NC}"
        echo "Please download and extract XMRig to mining/xmrig-6.21.3/"
        return 1
    fi
    
    cd mining/xmrig-6.21.3
    ./xmrig --config ../xmrig-MAITREYA.json &
    XMRIG_PID=$!
    echo -e "${GREEN}✅ XMRig started with PID: $XMRIG_PID${NC}"
    cd ../..
}

# Function to start SRBMiner GPU mining
start_srbminer() {
    echo -e "${BLUE}🎮 Starting SRBMiner GPU Mining...${NC}"
    
    if [[ ! -f "mining/SRBMiner-Multi-2-9-7/SRBMiner-MULTI" ]]; then
        echo -e "${RED}❌ SRBMiner binary not found in mining/SRBMiner-Multi-2-9-7/${NC}"
        echo "Please download and extract SRBMiner to mining/SRBMiner-Multi-2-9-7/"
        return 1
    fi
    
    cd mining/SRBMiner-Multi-2-9-7
    ./SRBMiner-MULTI --config ../srb-kawpow-config.json &
    SRB_PID=$!
    echo -e "${GREEN}✅ SRBMiner started with PID: $SRB_PID${NC}"
    cd ../..
}

# Function to show mining status
show_status() {
    echo -e "\n${BLUE}📊 Mining Status:${NC}"
    
    # Check XMRig
    if pgrep -f "xmrig" >/dev/null; then
        echo -e "${GREEN}✅ XMRig (CPU): RUNNING${NC}"
    else
        echo -e "${RED}❌ XMRig (CPU): STOPPED${NC}"
    fi
    
    # Check SRBMiner
    if pgrep -f "SRBMiner-MULTI" >/dev/null; then
        echo -e "${GREEN}✅ SRBMiner (GPU): RUNNING${NC}"
    else
        echo -e "${RED}❌ SRBMiner (GPU): STOPPED${NC}"
    fi
    
    # Pool stats
    if nc -z localhost 3333 >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Pool Server: ONLINE${NC}"
    else
        echo -e "${RED}❌ Pool Server: OFFLINE${NC}"
    fi
}

# Function to stop all miners
stop_miners() {
    echo -e "${YELLOW}🛑 Stopping all miners...${NC}"
    pkill -f "xmrig" 2>/dev/null && echo -e "${GREEN}✅ XMRig stopped${NC}"
    pkill -f "SRBMiner-MULTI" 2>/dev/null && echo -e "${GREEN}✅ SRBMiner stopped${NC}"
}

# Main menu
case "${1:-menu}" in
    "xmrig")
        check_pool && start_xmrig
        ;;
    "gpu"|"srb")
        check_pool && start_srbminer
        ;;
    "both"|"all")
        if check_pool; then
            start_xmrig
            sleep 2
            start_srbminer
        fi
        ;;
    "status")
        show_status
        ;;
    "stop")
        stop_miners
        ;;
    "update")
        echo -e "${BLUE}Updating mining configurations...${NC}"
        sed -i "s/\"user\": \"[^\"]*\"/\"user\": \"$POOL_ADDR\"/" mining/xmrig-MAITREYA.json
        sed -i "s/\"wallet\": \"[^\"]*\"/\"wallet\": \"$POOL_ADDR\"/" mining/srb-kawpow-config.json
        echo -e "${GREEN}✅ Configurations updated with pool address${NC}"
        ;;
    *)
        echo -e "${BLUE}ZION External Mining Control${NC}"
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  xmrig     - Start XMRig CPU mining only"
        echo "  gpu|srb   - Start SRBMiner GPU mining only"
        echo "  both|all  - Start both CPU and GPU mining"
        echo "  status    - Show mining status"
        echo "  stop      - Stop all external miners"
        echo "  update    - Update configurations with current pool address"
        echo ""
        echo -e "${YELLOW}Current Pool Address: ${GREEN}$POOL_ADDR${NC}"
        echo ""
        show_status
        ;;
esac
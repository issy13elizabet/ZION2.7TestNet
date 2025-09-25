#!/bin/bash
set -e

echo "🚀 Starting ZION Mining on Ubuntu (CPU + GPU)"
echo "=============================================="

# Configuration
MINING_CONFIG="mining/mining-wallet.conf"
POOL_SERVER="91.98.122.165"
CPU_POOL_PORT="3333"
GPU_POOL_PORT="3334"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Load mining config
if [ -f "$MINING_CONFIG" ]; then
    source "$MINING_CONFIG"
    echo -e "${GREEN}✅ Mining config loaded${NC}"
else
    echo -e "${RED}❌ Mining config not found. Run ./scripts/setup-mining-wallet.sh first${NC}"
    exit 1
fi

# Function to check pool connectivity
check_pool_connectivity() {
    echo -e "${BLUE}Checking pool connectivity...${NC}"
    
    if timeout 3 nc -z "$POOL_SERVER" "$CPU_POOL_PORT" 2>/dev/null; then
        echo -e "${GREEN}✅ CPU Pool ($POOL_SERVER:$CPU_POOL_PORT) is reachable${NC}"
    else
        echo -e "${RED}❌ CPU Pool is not reachable${NC}"
        return 1
    fi
    
    if timeout 3 nc -z "$POOL_SERVER" "$GPU_POOL_PORT" 2>/dev/null; then
        echo -e "${GREEN}✅ GPU Pool ($POOL_SERVER:$GPU_POOL_PORT) is reachable${NC}"
    else
        echo -e "${RED}❌ GPU Pool is not reachable${NC}"
        return 1
    fi
}

# Function to start CPU mining (XMRig)
start_cpu_mining() {
    echo -e "${BLUE}Starting CPU Mining (XMRig)...${NC}"
    
    if [ ! -f "mining/platforms/linux/xmrig-6.21.3/xmrig" ]; then
        echo -e "${RED}❌ XMRig binary not found${NC}"
        return 1
    fi
    
    cd mining/platforms/linux/xmrig-6.21.3/
    
    # Start XMRig in background
    echo -e "${YELLOW}Starting XMRig with config-zion.json...${NC}"
    nohup ./xmrig --config=config-zion.json > xmrig.log 2>&1 &
    XMRIG_PID=$!
    echo $XMRIG_PID > xmrig.pid
    
    cd ../../../..
    
    sleep 3
    if kill -0 $XMRIG_PID 2>/dev/null; then
        echo -e "${GREEN}✅ XMRig started successfully (PID: $XMRIG_PID)${NC}"
        return 0
    else
        echo -e "${RED}❌ XMRig failed to start${NC}"
        return 1
    fi
}

# Function to start GPU mining (SRBMiner)
start_gpu_mining() {
    echo -e "${BLUE}Starting GPU Mining (SRBMiner-MULTI)...${NC}"
    
    if [ ! -f "mining/SRBMiner-Multi-2-9-7/SRBMiner-Multi-2-9-7/SRBMiner-MULTI" ]; then
        echo -e "${RED}❌ SRBMiner binary not found${NC}"
        return 1
    fi
    
    cd mining/SRBMiner-Multi-2-9-7/SRBMiner-Multi-2-9-7/
    
    # Get mining wallet address from config
    WALLET_ADDRESS=$(grep "MINING_WALLET_ADDRESS" ../../mining-wallet.conf | cut -d'=' -f2)
    
    # Start SRBMiner in background  
    echo -e "${YELLOW}Starting SRBMiner-MULTI with KawPow algorithm...${NC}"
    nohup ./SRBMiner-MULTI \
        --algorithm kawpow \
        --pool "$POOL_SERVER:$GPU_POOL_PORT" \
        --wallet "$WALLET_ADDRESS" \
        --password "ubuntu-gpu-miner" \
        --gpu-id 0 \
        --gpu-intensity 23 \
        --disable-cpu \
        --api-enable \
        --api-port 21555 > srbminer.log 2>&1 &
    
    SRBMINER_PID=$!
    echo $SRBMINER_PID > srbminer.pid
    
    cd ../../..
    
    sleep 3
    if kill -0 $SRBMINER_PID 2>/dev/null; then
        echo -e "${GREEN}✅ SRBMiner started successfully (PID: $SRBMINER_PID)${NC}"
        return 0
    else
        echo -e "${RED}❌ SRBMiner failed to start${NC}"
        return 1
    fi
}

# Function to show mining status
show_mining_status() {
    echo ""
    echo -e "${PURPLE}📊 MINING STATUS${NC}"
    echo "================================"
    
    # Check XMRig
    if [ -f "mining/platforms/linux/xmrig-6.21.3/xmrig.pid" ]; then
        XMRIG_PID=$(cat mining/platforms/linux/xmrig-6.21.3/xmrig.pid)
        if kill -0 $XMRIG_PID 2>/dev/null; then
            echo -e "${GREEN}✅ XMRig (CPU): Running (PID: $XMRIG_PID)${NC}"
        else
            echo -e "${RED}❌ XMRig (CPU): Not running${NC}"
        fi
    else
        echo -e "${RED}❌ XMRig (CPU): Not started${NC}"
    fi
    
    # Check SRBMiner
    if [ -f "mining/SRBMiner-Multi-2-9-7/SRBMiner-Multi-2-9-7/srbminer.pid" ]; then
        SRBMINER_PID=$(cat mining/SRBMiner-Multi-2-9-7/SRBMiner-Multi-2-9-7/srbminer.pid)
        if kill -0 $SRBMINER_PID 2>/dev/null; then
            echo -e "${GREEN}✅ SRBMiner (GPU): Running (PID: $SRBMINER_PID)${NC}"
        else
            echo -e "${RED}❌ SRBMiner (GPU): Not running${NC}"
        fi
    else
        echo -e "${RED}❌ SRBMiner (GPU): Not started${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}Configuration:${NC}"
    echo "- Mining Address: $MINING_WALLET_ADDRESS"
    echo "- CPU Pool: $POOL_SERVER:$CPU_POOL_PORT"
    echo "- GPU Pool: $POOL_SERVER:$GPU_POOL_PORT"
    echo "- Worker ID: $WORKER_ID"
    echo ""
    echo -e "${YELLOW}Logs:${NC}"
    echo "- XMRig: mining/platforms/linux/xmrig-6.21.3/xmrig.log"
    echo "- SRBMiner: mining/SRBMiner-Multi-2-9-7/SRBMiner-Multi-2-9-7/srbminer.log"
    echo ""
    echo -e "${PURPLE}Monitor progress: ./scripts/mining-monitor.sh${NC}"
}

# Main execution
echo -e "${BLUE}ZION Ubuntu Mining Startup${NC}"
echo "Target: Mine first 60 blocks"
echo ""

# Check prerequisites
if ! command -v nc > /dev/null; then
    echo -e "${YELLOW}Installing netcat for connectivity testing...${NC}"
    sudo apt update && sudo apt install -y netcat-openbsd
fi

# Check pool connectivity
if ! check_pool_connectivity; then
    echo -e "${RED}❌ Pool connectivity check failed${NC}"
    exit 1
fi

# Start mining
echo -e "${BLUE}Starting dual mining (CPU + GPU)...${NC}"

# Start CPU mining
if start_cpu_mining; then
    echo -e "${GREEN}✅ CPU mining started${NC}"
else
    echo -e "${YELLOW}⚠️  CPU mining failed, continuing with GPU only${NC}"
fi

# Start GPU mining
if start_gpu_mining; then
    echo -e "${GREEN}✅ GPU mining started${NC}"
else
    echo -e "${YELLOW}⚠️  GPU mining failed${NC}"
fi

# Show status
show_mining_status

echo ""
echo -e "${GREEN}🎉 ZION Mining is now running on Ubuntu!${NC}"
echo -e "${YELLOW}Use Ctrl+C to stop monitoring, miners will continue in background${NC}"
echo ""

# Optional: Start monitoring
read -p "Start real-time mining monitor? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    ./scripts/mining-monitor.sh
fi
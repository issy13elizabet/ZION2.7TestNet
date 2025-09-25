#!/bin/bash

echo "⛏️  ZION Mining Monitor - Tracking Progress to 60 Blocks"
echo "======================================================="

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Configuration
TARGET_BLOCKS=60
POOL_RPC="http://localhost:28081/json_rpc"
SEED_RPC="http://localhost:18081/json_rpc"
UPDATE_INTERVAL=10

# Function to get block height from RPC
get_block_height() {
    local rpc_url=$1
    local height=$(curl -s -X POST \
        -H "Content-Type: application/json" \
        -d '{"jsonrpc":"2.0","id":"0","method":"getheight","params":{}}' \
        "$rpc_url" 2>/dev/null | \
        grep -o '"height":[0-9]*' | cut -d':' -f2)
    echo "${height:-0}"
}

# Function to get network info
get_network_info() {
    local rpc_url=$1
    curl -s -X POST \
        -H "Content-Type: application/json" \
        -d '{"jsonrpc":"2.0","id":"0","method":"getinfo","params":{}}' \
        "$rpc_url" 2>/dev/null
}

# Function to get mining stats
get_mining_stats() {
    local rpc_url=$1
    curl -s -X POST \
        -H "Content-Type: application/json" \
        -d '{"jsonrpc":"2.0","id":"0","method":"getmininginfo","params":{}}' \
        "$rpc_url" 2>/dev/null
}

# Function to check pool connectivity
check_pool_status() {
    if nc -z localhost 3333 >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Pool Server: ONLINE${NC}"
        return 0
    else
        echo -e "${RED}❌ Pool Server: OFFLINE${NC}"
        return 1
    fi
}

# Function to check miner status
check_miner_status() {
    if docker ps | grep -q "zion-miner.*Up"; then
        echo -e "${GREEN}✅ Miner: RUNNING${NC}"
        return 0
    else
        echo -e "${RED}❌ Miner: STOPPED${NC}"
        return 1
    fi
}

# Function to display progress bar
progress_bar() {
    local current=$1
    local total=$2
    local width=50
    local percentage=$((current * 100 / total))
    local filled=$((current * width / total))
    local empty=$((width - filled))
    
    printf "["
    printf "%*s" $filled | tr ' ' '█'
    printf "%*s" $empty | tr ' ' '░'
    printf "] %d/%d blocks (%d%%)\n" $current $total $percentage
}

# Main monitoring loop
echo -e "${BLUE}Starting ZION mining monitoring...${NC}"
echo "Target: $TARGET_BLOCKS blocks"
echo "Update interval: ${UPDATE_INTERVAL}s"
echo ""

start_time=$(date +%s)
last_height=0
blocks_mined=0

while true; do
    clear
    echo "⛏️  ZION Mining Monitor - $(date)"
    echo "======================================================="
    
    # Check service status
    echo -e "\n${BLUE}📊 Service Status:${NC}"
    check_pool_status
    check_miner_status
    
    # Get current block heights
    pool_height=$(get_block_height "$POOL_RPC")
    seed_height=$(get_block_height "$SEED_RPC")
    
    # Calculate progress
    current_height=$((pool_height > seed_height ? pool_height : seed_height))
    
    if [ $current_height -gt $last_height ]; then
        blocks_mined=$((blocks_mined + current_height - last_height))
        last_height=$current_height
    fi
    
    echo -e "\n${BLUE}⛏️  Mining Progress:${NC}"
    progress_bar $current_height $TARGET_BLOCKS
    
    echo -e "\n${BLUE}📈 Statistics:${NC}"
    echo "Pool Height:    $pool_height"
    echo "Seed Height:    $seed_height"
    echo "Current Height: $current_height"
    echo "Blocks Mined:   $blocks_mined"
    echo "Target:         $TARGET_BLOCKS"
    echo "Remaining:      $((TARGET_BLOCKS - current_height))"
    
    # Calculate mining rate
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    if [ $elapsed -gt 0 ] && [ $blocks_mined -gt 0 ]; then
        blocks_per_hour=$(echo "scale=2; $blocks_mined * 3600 / $elapsed" | bc 2>/dev/null || echo "0")
        echo "Mining Rate:    $blocks_per_hour blocks/hour"
        
        if [ $current_height -lt $TARGET_BLOCKS ]; then
            remaining_time=$(echo "scale=0; ($TARGET_BLOCKS - $current_height) * $elapsed / $blocks_mined" | bc 2>/dev/null || echo "∞")
            if [ "$remaining_time" != "∞" ] && [ $remaining_time -gt 0 ]; then
                echo "ETA:            $(($remaining_time / 3600))h $(($remaining_time % 3600 / 60))m"
            fi
        fi
    fi
    
    # Show recent docker logs
    echo -e "\n${BLUE}📝 Recent Pool Activity:${NC}"
    docker logs zion-pool --tail 3 2>/dev/null | sed 's/^/  /'
    
    echo -e "\n${BLUE}📝 Recent Miner Activity:${NC}"
    docker logs zion-miner --tail 3 2>/dev/null | sed 's/^/  /'
    
    # Check if target reached
    if [ $current_height -ge $TARGET_BLOCKS ]; then
        echo -e "\n${GREEN}🎉 TARGET REACHED! ${TARGET_BLOCKS} blocks mined!${NC}"
        echo -e "${GREEN}✅ Mining test completed successfully!${NC}"
        
        # Show final statistics
        total_time=$((current_time - start_time))
        echo -e "\n${BLUE}📊 Final Statistics:${NC}"
        echo "Total Time:     $(($total_time / 3600))h $(($total_time % 3600 / 60))m $(($total_time % 60))s"
        echo "Blocks Mined:   $blocks_mined"
        echo "Average Rate:   $(echo "scale=2; $blocks_mined * 3600 / $total_time" | bc 2>/dev/null || echo "N/A") blocks/hour"
        break
    fi
    
    echo -e "\n${YELLOW}Press Ctrl+C to stop monitoring${NC}"
    sleep $UPDATE_INTERVAL
done

echo -e "\n${GREEN}✅ ZION Mining Monitor completed!${NC}"
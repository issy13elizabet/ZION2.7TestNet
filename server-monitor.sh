#!/bin/bash

# ZION Server Monitoring Script
# Real-time monitoring pro production deployment

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    local service="$1"
    local status="$2"
    local message="$3"
    
    case "$status" in
        "ok")
            echo -e "${GREEN}✅ $service: $message${NC}"
            ;;
        "warn")
            echo -e "${YELLOW}⚠️  $service: $message${NC}"
            ;;
        "error")
            echo -e "${RED}❌ $service: $message${NC}"
            ;;
        "info")
            echo -e "${BLUE}ℹ️  $service: $message${NC}"
            ;;
    esac
}

check_docker() {
    echo "=== Docker Status ==="
    if docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -q zion; then
        docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep zion
        print_status "Docker" "ok" "ZION containers running"
    else
        print_status "Docker" "error" "No ZION containers found"
    fi
    echo
}

check_blockchain() {
    echo "=== Blockchain Status ==="
    
    # Check if RPC is responding
    if curl -s --connect-timeout 5 http://localhost:18081/getheight > /dev/null; then
        HEIGHT=$(curl -s http://localhost:18081/getheight | python3 -c "import sys, json; print(json.load(sys.stdin)['height'])" 2>/dev/null || echo "unknown")
        print_status "RPC" "ok" "Height: $HEIGHT"
        
        # Get network info
        if curl -s --connect-timeout 5 http://localhost:18081/getinfo > /dev/null; then
            INFO=$(curl -s http://localhost:18081/getinfo)
            PEERS=$(echo "$INFO" | python3 -c "import sys, json; print(json.load(sys.stdin)['connections_count'])" 2>/dev/null || echo "unknown")
            print_status "Network" "ok" "Peers: $PEERS"
        else
            print_status "Network" "warn" "Cannot get network info"
        fi
    else
        print_status "RPC" "error" "Not responding on port 18081"
    fi
    echo
}

check_p2p() {
    echo "=== P2P Network ==="
    if nc -z localhost 18080 2>/dev/null; then
        print_status "P2P" "ok" "Port 18080 listening"
    else
        print_status "P2P" "error" "Port 18080 not accessible"
    fi
    echo
}

check_resources() {
    echo "=== System Resources ==="
    
    # CPU usage
    CPU=$(top -l 1 | grep "CPU usage" | awk '{print $3}' | sed 's/%//' 2>/dev/null || echo "unknown")
    if [[ "$CPU" != "unknown" && "${CPU%.*}" -lt 80 ]]; then
        print_status "CPU" "ok" "Usage: ${CPU}%"
    else
        print_status "CPU" "warn" "Usage: ${CPU}%"
    fi
    
    # Memory usage
    MEM=$(vm_stat | grep "Pages free" | awk '{print $3}' | sed 's/\.//' 2>/dev/null || echo "unknown")
    print_status "Memory" "info" "Free pages: $MEM"
    
    # Disk usage
    DISK=$(df -h . | awk 'NR==2{print $5}' | sed 's/%//')
    if [[ "$DISK" -lt 90 ]]; then
        print_status "Disk" "ok" "Usage: ${DISK}%"
    else
        print_status "Disk" "warn" "Usage: ${DISK}%"
    fi
    echo
}

check_logs() {
    echo "=== Recent Logs ==="
    if docker logs --tail 5 zion-production 2>/dev/null; then
        print_status "Logs" "ok" "Last 5 lines shown above"
    else
        print_status "Logs" "error" "Cannot access container logs"
    fi
    echo
}

check_frontend() {
    echo "=== Frontend dApp ==="
    if [ -d "frontend" ]; then
        if [ -f "frontend/package.json" ]; then
            FRONTEND_VERSION=$(grep '"version"' frontend/package.json | cut -d'"' -f4)
            print_status "Frontend" "info" "Version: $FRONTEND_VERSION"
            
            if pgrep -f "next dev" > /dev/null; then
                print_status "Dev Server" "ok" "Running (likely on port 3000)"
            else
                print_status "Dev Server" "warn" "Not running"
            fi
        fi
    else
        print_status "Frontend" "warn" "Directory not found"
    fi
    echo
}

main() {
    clear
    echo -e "${BLUE}🔍 ZION Server Monitor - $(date)${NC}"
    echo "=========================================="
    echo
    
    check_docker
    check_blockchain
    check_p2p
    check_resources
    check_logs
    check_frontend
    
    echo "=========================================="
    echo -e "${BLUE}Monitor complete. Run with --watch for continuous monitoring.${NC}"
}

# Continuous monitoring mode
if [[ "$1" == "--watch" ]]; then
    while true; do
        main
        echo -e "\n${YELLOW}Press Ctrl+C to stop monitoring...${NC}"
        sleep 10
    done
else
    main
fi
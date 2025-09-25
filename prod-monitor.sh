#!/bin/bash

# Production Deployment Health Check & Monitor
# ===========================================

set -e

echo "🚀 ZION Production Deployment Monitor"
echo "======================================"

check_docker() {
    echo "🐳 Docker Services Status:"
    if docker-compose -f docker-compose.prod.yml ps; then
        echo "✅ Docker services running"
    else
        echo "❌ Docker services not running"
        return 1
    fi
}

check_blockchain() {
    echo "⛓️  Blockchain Node Status:"
    
    # Wait for RPC to be available
    for i in {1..30}; do
        if curl -s -o /dev/null -w "%{http_code}" http://localhost:18081/getinfo | grep -q "200"; then
            echo "✅ RPC endpoint responding"
            break
        fi
        echo "⏳ Waiting for RPC... (${i}/30)"
        sleep 2
    done
    
    # Get node info
    if response=$(curl -s -X POST http://localhost:18081/json_rpc \
        -H 'Content-Type: application/json' \
        -d '{"jsonrpc":"2.0","id":"0","method":"getinfo"}'); then
        
        height=$(echo "$response" | jq -r '.result.height // 0')
        peers=$(echo "$response" | jq -r '.result.outgoing_connections_count // 0')
        
        echo "📊 Block Height: $height"
        echo "🌐 Connected Peers: $peers"
        
        if [ "$height" -gt 0 ]; then
            echo "✅ Blockchain syncing/synced"
        else
            echo "⚠️  Blockchain starting up"
        fi
    else
        echo "❌ Failed to get blockchain info"
        return 1
    fi
}

check_resources() {
    echo "💾 System Resources:"
    
    # Memory usage (macOS compatible)
    if command -v free >/dev/null 2>&1; then
        mem_used=$(free -h | awk '/^Mem:/ {print $3}')
        mem_total=$(free -h | awk '/^Mem:/ {print $2}')
        echo "🧠 Memory: $mem_used / $mem_total"
    else
        # macOS memory check
        mem_info=$(vm_stat)
        echo "🧠 Memory: $(echo "$mem_info" | grep "Pages active" | awk '{print $3}' | sed 's/\.//')MB active"
    fi
    
    # Disk usage
    disk_used=$(df -h / | awk 'NR==2 {print $3}')
    disk_total=$(df -h / | awk 'NR==2 {print $2}')
    disk_percent=$(df -h / | awk 'NR==2 {print $5}')
    echo "💿 Disk: $disk_used / $disk_total ($disk_percent)"
    
    # CPU load
    if command -v uptime >/dev/null 2>&1; then
        load=$(uptime | awk -F'load average:' '{print $2}' | cut -d, -f1 | xargs)
        echo "⚡ CPU Load: $load"
    else
        echo "⚡ CPU Load: Available"
    fi
}

check_logs() {
    echo "📝 Recent Logs:"
    echo "--- Last 5 container logs ---"
    docker-compose -f docker-compose.prod.yml logs --tail=5 zion-node
}

monitor_loop() {
    while true; do
        clear
        echo "🕐 $(date)"
        echo "========================"
        
        check_docker && echo
        check_blockchain && echo
        check_resources && echo
        check_logs
        
        echo "========================"
        echo "⏳ Next check in 30 seconds... (Ctrl+C to exit)"
        sleep 30
    done
}

case "${1:-check}" in
    "monitor")
        monitor_loop
        ;;
    "check")
        check_docker && echo
        check_blockchain && echo
        check_resources
        ;;
    "logs")
        docker-compose -f docker-compose.prod.yml logs -f zion-node
        ;;
    *)
        echo "Usage: $0 [check|monitor|logs]"
        echo "  check   - Run single health check"
        echo "  monitor - Continuous monitoring loop"
        echo "  logs    - Follow live logs"
        ;;
esac
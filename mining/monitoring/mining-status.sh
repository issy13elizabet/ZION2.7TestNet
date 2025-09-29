#!/bin/bash
# ZION Mining Status Monitor
while true; do
    clear
    echo "🎯 ZION Mining Status - $(date)"
    echo "=================================="
    
    echo "📡 Core Services:"
    curl -s localhost:18081/json_rpc -d '{"jsonrpc":"2.0","id":1,"method":"get_info"}' | jq -r '.result.height // "❌ Offline"' | sed 's/^/  Height: /'
    
    echo "⛏️ Mining Processes:"
    pgrep -f zion-miner && echo "  ZION Miner: ✅ Running" || echo "  ZION Miner: ❌ Stopped"
    
    echo "🔗 Network:"
    ss -tlnp | grep :3333 >/dev/null && echo "  Pool Port: ✅ Open" || echo "  Pool Port: ❌ Closed"
    
    sleep 30
done

#!/bin/bash

# ZION Bootstrap Test Script
# Testuje funkcionalnost bootstrap patch a mining stack

set -e

echo "🚀 ZION Bootstrap Stack Test"
echo "============================"

# Funkce pro čekání na healthy službu
wait_for_healthy() {
    local service=$1
    local timeout=${2:-60}
    
    echo "⏳ Čekám na $service..."
    local count=0
    while [ $count -lt $timeout ]; do
        if docker-compose -p zion-bootstrap ps $service | grep -q "healthy"; then
            echo "✅ $service je healthy!"
            return 0
        fi
        sleep 2
        ((count+=2))
    done
    echo "❌ $service timeout po ${timeout}s"
    return 1
}

# Test REST endpoint (ZION daemon uses REST API)
test_rest() {
    local host=$1
    local port=$2
    local endpoint=$3
    
    echo "📡 Testuju REST: $endpoint na $host:$port"
    
    local response=$(curl -s -m 10 "http://$host:$port/$endpoint" 2>/dev/null || echo "FAILED")
    
    if [[ "$response" == "FAILED" ]]; then
        echo "❌ REST selhalo: $endpoint"
        return 1
    elif echo "$response" | grep -q '"status":"OK"'; then
        echo "✅ REST $endpoint OK"
        return 0
    else
        echo "⚠️  Neočekávaná REST odpověď pro $endpoint: $response"
        return 1
    fi
}

# Test JSON-RPC endpoint (RPC shim)
test_rpc() {
    local host=$1
    local port=$2
    local method=$3
    
    echo "📡 Testuju JSON-RPC: $method na $host:$port"
    
    local response=$(curl -s -m 10 "http://$host:$port/json_rpc" \
        -H "Content-Type: application/json" \
        -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"$method\",\"params\":{}}" 2>/dev/null || echo "FAILED")
    
    if [[ "$response" == "FAILED" ]]; then
        echo "❌ JSON-RPC selhalo: $method"
        return 1
    elif echo "$response" | grep -q '"error"'; then
        echo "⚠️  JSON-RPC error pro $method: $(echo "$response" | jq -r '.error.message' 2>/dev/null || echo "unknown")"
        return 1
    else
        echo "✅ JSON-RPC $method OK"
        return 0
    fi
}

# Test getblocktemplate (hlavní bootstrap test)
test_bootstrap_gbt() {
    local wallet_addr="Z3222ywic7fGUZHv9EfFwEKf1VJRrEqPbLrHgRiBb43LWaS1Cz2gVwgdF2kvUPsGb9jSvUUf31oNCSZgNEtUiGDT4sBLtXmGzc"
    
    echo "🎯 BOOTSTRAP TEST: getblocktemplate"
    
    local response=$(curl -s -m 15 "http://localhost:18089/json_rpc" \
        -H "Content-Type: application/json" \
        -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"getblocktemplate\",\"params\":{\"wallet_address\":\"$wallet_addr\",\"reserve_size\":0}}" 2>/dev/null || echo "FAILED")
    
    if [[ "$response" == "FAILED" ]]; then
        echo "❌ Bootstrap GBT komunikační selhání"
        return 1
    elif echo "$response" | grep -q '"Core is busy"'; then
        echo "❌ Bootstrap patch NEFUNGUJE - stále 'Core is busy'"
        return 1
    elif echo "$response" | grep -q '"blocktemplate_blob"'; then
        echo "✅ Bootstrap patch FUNGUJE - dostali jsme block template!"
        echo "📊 Template height: $(echo "$response" | jq -r '.result.height' 2>/dev/null || echo "unknown")"
        echo "📊 Template difficulty: $(echo "$response" | jq -r '.result.difficulty' 2>/dev/null || echo "unknown")"
        return 0
    else
        echo "⚠️  Neočekávaná odpověď GBT: $response"
        return 1
    fi
}

# Main test sequence
main() {
    echo "🔍 Kontrola Docker Compose stacku..."
    
    if ! docker-compose -p zion-bootstrap ps >/dev/null 2>&1; then
        echo "❌ Docker Compose stack není spuštěný"
        echo "💡 Spusť: docker-compose -p zion-bootstrap -f docker-compose-production.yml up -d"
        exit 1
    fi
    
    # Čekání na základní služby
    wait_for_healthy "seed1" 90 || exit 1
    wait_for_healthy "rpc-shim" 60 || exit 1
    
    # Základní testy - REST API na daemon, JSON-RPC na shim
    test_rest "localhost" "18081" "getinfo" || exit 1
    test_rpc "localhost" "18089" "getheight" || exit 1
    
    # Klíčový bootstrap test
    test_bootstrap_gbt || exit 1
    
    echo ""
    echo "🎉 VŠECHNY TESTY PROŠLY!"
    echo "🚀 ZION Bootstrap stack je připravený na mining!"
    echo ""
    echo "📋 Další kroky:"
    echo "   1. Spusť XMRig: xmrig --url stratum+tcp://localhost:3333 --user Z3... --algo rx/0"
    echo "   2. Sleduj height: watch 'curl -s localhost:18089/json_rpc -d \"{\\\"method\\\":\\\"getheight\\\"}\" | jq'"
    echo "   3. Monitor pool: curl localhost:8117/stats"
}

# Spuštění
main "$@"
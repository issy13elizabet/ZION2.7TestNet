#!/bin/bash

# ZION Mining Monitor - macOS
# Sleduje stav těžby a poskytuje real-time informace

echo "🔮 ZION Cosmic Mining Monitor - $(date)"
echo "==============================================="

# Kontrola připojení k poolu
echo "📡 Pool Connection Status:"
if nc -z 91.98.122.165 3333 2>/dev/null; then
    echo "   ✅ Pool 91.98.122.165:3333 - ONLINE"
else
    echo "   ❌ Pool 91.98.122.165:3333 - OFFLINE"
fi

echo ""

# XMRig proces
echo "⚡ XMRig Process Status:"
if pgrep -x "xmrig" > /dev/null; then
    echo "   ✅ XMRig is RUNNING"
    echo "   PID: $(pgrep xmrig)"
    
    # CPU usage
    cpu_usage=$(top -l 1 -pid $(pgrep xmrig) -stats cpu 2>/dev/null | tail -1 | awk '{print $3}')
    if [[ ! -z "$cpu_usage" ]]; then
        echo "   CPU Usage: ${cpu_usage}%"
    fi
else
    echo "   ❌ XMRig is NOT RUNNING"
fi

echo ""

# Logy z XMRig (posledních 10 řádků)
echo "📊 Recent Mining Logs:"
if [[ -f "/Users/yose/Desktop/TestNet/Zion-2.6-TestNet/mining/xmrig/xmrig-zion.log" ]]; then
    tail -10 "/Users/yose/Desktop/TestNet/Zion-2.6-TestNet/mining/xmrig/xmrig-zion.log"
else
    echo "   No log file found yet..."
fi

echo ""

# System resources
echo "💻 System Resources:"
echo "   $(top -l 1 -n 0 | grep "CPU usage")"
echo "   $(top -l 1 -n 0 | grep "PhysMem")"

echo ""
echo "🚀 Mining for ZION - Jai Ram Ram Ram! 🚀"
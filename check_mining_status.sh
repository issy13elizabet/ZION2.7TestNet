#!/bin/bash
# 📊 Jednorázový ZION Mining Status Check

SSH_SERVER="91.98.122.165"
SSH_USER="root"

echo "📊 ZION Mining Status na SSH serveru ${SSH_SERVER}"
echo "=================================================="

ssh -o StrictHostKeyChecking=no "${SSH_USER}@${SSH_SERVER}" << 'MINING_STATUS'
cd /root/zion-2.7.1

echo "🔍 Mining Process Status:"
ps aux | grep zion_wrapper | grep -v grep || echo "❌ Žádný mining proces nenalezen"

echo ""
echo "📋 Mining Log (posledních 15 řádků):"
if [ -f mining.log ]; then
    tail -15 mining.log
else
    echo "❌ Mining log neexistuje"
fi

echo ""
echo "📊 ZION Blockchain Stats:"
python3 zion_wrapper.py stats | tail -20

echo ""
echo "💰 Wallet Balances:"
python3 zion_wrapper.py wallet list | grep ZION

echo ""
echo "🖥️  System Resources:"
echo "CPU cores: $(nproc)"
echo "Memory usage: $(free | awk 'NR==2{printf "%.2f%%", $3*100/$2 }')"
echo "Load average: $(uptime | cut -d',' -f3-)"

MINING_STATUS
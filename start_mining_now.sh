#!/bin/bash
# 🚀 Start ZION Mining na SSH
echo "🚀 Spouštím ZION mining na SSH serveru..."

ssh -o StrictHostKeyChecking=no root@91.98.122.165 << 'START_MINING'
cd /root/zion-2.7.1

echo "💰 Získávám wallet adresy..."
python3 zion_wrapper.py wallet list

echo ""
echo "🔍 Nebo vytvořím novou adresu..."
python3 zion_wrapper.py wallet create

echo ""
echo "🚀 Spouštím těžbu s první dostupnou adresou..."

# Získání první adresy
MINING_ADDRESS=$(python3 zion_wrapper.py wallet list 2>/dev/null | grep -o 'Zx[A-Za-z0-9]*' | head -1)

if [ -n "$MINING_ADDRESS" ]; then
    echo "💰 Mining adresa: $MINING_ADDRESS"
    echo "🔥 Spouštím těžbu..."
    nohup python3 zion_wrapper.py mine $MINING_ADDRESS > mining.log 2>&1 &
    
    echo "🎯 Mining PID: $!"
    echo "📊 Status:"
    ps aux | grep zion_wrapper | grep -v grep
    
    echo ""
    echo "📋 Prvních několik řádků z mining logu:"
    sleep 3
    tail -10 mining.log 2>/dev/null || echo "Log se ještě vytváří..."
    
else
    echo "⚠️  Žádná mining adresa nenalezena - používám fallback"
    echo "🔥 Spouštím těžbu s fallback adresou..."
    nohup python3 zion_wrapper.py mine ZxChainFallback123456789 > mining.log 2>&1 &
    echo "🎯 Mining PID: $!"
fi

echo ""
echo "✅ ZION mining spuštěn na SSH serveru!"
echo "📱 Pro sledování: tail -f /root/zion-2.7.1/mining.log"

START_MINING
#!/bin/bash
# 🚀 Spustit ZION Mining se správnými parametry

SSH_SERVER="91.98.122.165"
SSH_USER="root"

echo "🚀 Spouštím ZION mining se správnými parametry..."

ssh -o StrictHostKeyChecking=no "${SSH_USER}@${SSH_SERVER}" << 'START_CORRECT_MINING'
cd /root/zion-2.7.1

echo "💰 Získávám mining adresu z wallet..."
MINING_ADDRESS=$(python3 zion_wrapper.py wallet list | grep "Miner Address" | head -1 | awk '{print $1}')

if [ -z "$MINING_ADDRESS" ]; then
    echo "⚠️  Používám první dostupnou adresu..."
    MINING_ADDRESS=$(python3 zion_wrapper.py wallet list | grep "ZION_" | head -1 | awk '{print $1}')
fi

echo "💰 Mining adresa: $MINING_ADDRESS"

if [ -n "$MINING_ADDRESS" ]; then
    echo "🔥 Spouštím ZION těžbu..."
    nohup python3 zion_wrapper.py mine --address "$MINING_ADDRESS" --blocks 10 --algorithm argon2 > mining.log 2>&1 &
    
    MINING_PID=$!
    echo "🎯 Mining PID: $MINING_PID"
    
    echo "⏳ Čekám 5 sekund na spuštění..."
    sleep 5
    
    echo "🔍 Status mining procesu:"
    ps aux | grep zion_wrapper | grep -v grep || echo "❌ Proces již není aktivní"
    
    echo ""
    echo "📋 Mining log:"
    if [ -f mining.log ]; then
        tail -10 mining.log
    else
        echo "❌ Log soubor neexistuje"
    fi
    
else
    echo "❌ Nepodařilo se získat mining adresu!"
    echo "🔄 Zkusím vytvořit novou..."
    python3 zion_wrapper.py wallet create
fi

echo ""
echo "✅ Mining startup dokončen!"

START_CORRECT_MINING
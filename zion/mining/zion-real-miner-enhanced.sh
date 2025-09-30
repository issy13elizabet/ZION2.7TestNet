#!/bin/bash

# ZION Real Miner 1.4.0 - Enhanced version startup script
# Temperature monitoring integrated + NiceHash support

echo "🌟 ==========================================  🌟"
echo "🔥    ZION REAL MINER 1.4.0 (Enhanced)       🔥"
echo "🌡️    + Temperature Monitoring                 🌡️" 
echo "💎    + NiceHash Support                      💎"
echo "🌟 ==========================================  🌟"
echo

# Check Python3
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 není nainstalován!"
    echo "💡 Nainstaluj: sudo apt-get install python3"
    read -p "Stiskni Enter pro ukončení..."
    exit 1
fi

# Check tkinter
if ! python3 -c "import tkinter" 2>/dev/null; then
    echo "❌ tkinter není nainstalován!"
    echo "💡 Nainstaluj: sudo apt-get install python3-tk"
    read -p "Stiskni Enter pro ukončení..."
    exit 1
fi

# Check lm-sensors for temperature monitoring
if ! command -v sensors &> /dev/null; then
    echo "⚠️  Temperature monitoring není k dispozici!"
    echo "💡 Nainstaluj: sudo apt-get install lm-sensors"
    echo "💡 Nastav: sudo sensors-detect --auto"
    echo "⚠️  Mining může pokračovat bez temperature monitoring"
    echo
fi

# Show current CPU temperature if available
if command -v sensors &> /dev/null; then
    echo "🌡️  Aktuální CPU teplota:"
    sensors | grep -i "tctl\|package\|core" | head -3
    echo
fi

# Final warning for REAL mining
echo "⚠️  REAL MINING UPOZORNĚNÍ:"
echo "🔥 100% zatížení CPU cores"
echo "⚡ Vysoká spotřeba elektřiny"  
echo "🌡️ Možné přehřátí systému"
echo "💻 Zpomalení ostatních aplikací"
echo "🎯 Skutečná těžba kryptoměn"
echo

# Confirm start
echo "🚀 Pokračovat ve spuštění REAL Mineru?"
read -p "Stiskni Enter pro pokračování nebo Ctrl+C pro zrušení..."

# Change to script directory
cd "$(dirname "$0")"

# Start enhanced real miner
echo "🚀 Spouštím ZION Real Miner Enhanced..."
python3 zion-real-miner-v2.py

echo "⏹️  Miner ukončen."
echo "🌟 Děkujeme za použití ZION Real Miner!"
read -p "Stiskni Enter pro ukončení..."
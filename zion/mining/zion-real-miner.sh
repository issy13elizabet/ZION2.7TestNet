#!/bin/bash

# ZION REAL Miner 1.4.0 - Skutečný mining launcher

echo "🔥 ZION REAL Miner 1.4.0 - Skutečný RandomX Mining"
echo "=================================================="
echo ""
echo "⚠️  UPOZORNĚNÍ: Toto je SKUTEČNÝ mining client!"
echo "🔥 Spotřebovává 100% CPU výkon a elektřinu"
echo "🌡️  Sleduj teplotu CPU během mining"
echo ""

# Kontrola CPU cores
CPU_CORES=$(nproc)
echo "🖥️  Detekováno CPU cores: $CPU_CORES"

# Doporučení pro threads
if [ $CPU_CORES -le 4 ]; then
    RECOMMENDED=$CPU_CORES
else
    RECOMMENDED=$((CPU_CORES - 1))
fi

echo "💡 Doporučené threads: $RECOMMENDED (nechává 1 core pro systém)"

# Kontrola teploty (pokud je dostupná)
if command -v sensors &> /dev/null; then
    echo ""
    echo "🌡️  Aktuální teplota CPU:"
    sensors | grep -E "(Core|Package|Tctl)" | head -3
else
    echo "💡 Pro monitoring teploty: sudo apt-get install lm-sensors"
fi

echo ""
read -p "🚀 Spustit REAL mining? [y/N]: " confirm

if [[ $confirm =~ ^[Yy]$ ]]; then
    echo "🔥 Spouštění REAL mining..."
    
    # Kontrola Python3 a tkinter
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python3 není nainstalován!"
        exit 1
    fi
    
    python3 -c "import tkinter" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "📦 Instalace python3-tk..."
        sudo apt-get update && sudo apt-get install -y python3-tk
    fi
    
    # Spuštění real mining GUI
    cd "$(dirname "$0")"
    python3 zion-real-miner.py
    
else
    echo "👋 Mining zrušen"
    echo ""
    echo "💡 Pro simulovaný test mining použij: python3 zion-miner.py"
fi
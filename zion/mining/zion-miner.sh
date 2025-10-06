#!/bin/bash

# ZION Miner 1.4.0 Launcher for Ubuntu
# Automaticky nainstaluje závislosti a spustí GUI miner

echo "🌟 ZION Miner 1.4.0 - Professional Mining Client"
echo "================================================"

# Kontrola Python3
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 není nainstalován!"
    echo "💡 Instalace: sudo apt-get update && sudo apt-get install python3"
    exit 1
fi

# Kontrola tkinter
python3 -c "import tkinter" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Instalace python3-tk..."
    sudo apt-get update
    sudo apt-get install -y python3-tk
fi

# Kontrola dalších závislostí
echo "🔧 Kontrola závislostí..."

# Vytvoření desktop shortcut pokud neexistuje
DESKTOP_FILE="$HOME/Desktop/zion-miner.desktop"
if [ ! -f "$DESKTOP_FILE" ]; then
    cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=ZION Miner 1.4.0
Comment=Professional Mining Client pro ZION, NiceHash a další pooly
Exec=bash "$PWD/zion-miner.sh"
Icon=utilities-terminal
Terminal=false
Categories=Utility;Network;
EOF
    chmod +x "$DESKTOP_FILE"
    echo "🖥️ Desktop shortcut vytvořen"
fi

# Spuštění GUI
echo "🚀 Spouštění ZION Miner GUI..."
cd "$(dirname "$0")"
python3 zion-miner.py

echo "👋 ZION Miner ukončen"
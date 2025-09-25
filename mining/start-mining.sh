#!/bin/bash

# ZION Mining Startup Script - Auto-detect Platform
# Automaticky detekuje platformu a spustí správnou verzi XMRig

echo "🚀 ZION Mining Platform Auto-Detection"
echo "Genesis Block Hash: d763b61e4e542a6973c8f649deb228e116bcf3ee099cec92be33efe288829ae1"
echo "Mining Address: ajmqontZjiVUmtNjQu1RNUYq1RZgd5EDodX3qgjcaTMoMzG8EkG4bVPgLhEgudBoH82fQU1iZVw6XPfddKWAHDdA3x92ToH4uo"
echo "Server: 91.98.122.165:18081"
echo ""

# Detect platform
OS=$(uname -s)
ARCH=$(uname -m)

echo "Detected OS: $OS"
echo "Detected Architecture: $ARCH"

case "$OS" in
    "Darwin")
        echo "🍎 macOS detected"
        exec "$(dirname "$0")/start-mining-macos.sh"
        ;;
    "Linux")
        echo "🐧 Linux detected"
        exec "$(dirname "$0")/start-mining-linux.sh"
        ;;
    "MINGW"*|"CYGWIN"*|"MSYS"*)
        echo "🪟 Windows detected"
        echo "Please use start-mining-windows.bat instead"
        exit 1
        ;;
    *)
        echo "❌ Unsupported platform: $OS"
        echo "Available options:"
        echo "- macOS: ./start-mining-macos.sh"
        echo "- Linux: ./start-mining-linux.sh" 
        echo "- Windows: start-mining-windows.bat"
        echo "- Mobile: Open mobile/index.html in browser"
        exit 1
        ;;
esac
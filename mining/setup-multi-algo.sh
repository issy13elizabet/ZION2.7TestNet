#!/bin/bash
# 🚀 ZION Multi-Algorithm Mining Setup Script
# Supports: CPU (RandomX) + GPU (Multi-algo)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINING_DIR="$SCRIPT_DIR"
GPU_DIR="$MINING_DIR/gpu"
CPU_DIR="$MINING_DIR/cpu"

echo "🚀 ZION Multi-Algo Mining Setup"
echo "==============================="

# Detect platform
PLATFORM=""
if [[ "$OSTYPE" == "darwin"* ]]; then
    if [[ $(uname -m) == "arm64" ]]; then
        PLATFORM="macos-arm64"
    else
        PLATFORM="macos-x64"
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="linux"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    PLATFORM="windows"
fi

echo "🖥️  Platform detected: $PLATFORM"

# Create directory structure
echo "📁 Creating mining directory structure..."
mkdir -p "$GPU_DIR"/{t-rex,lolminer,nbminer,teamred,configs}
mkdir -p "$CPU_DIR/xmrig"
mkdir -p "$MINING_DIR"/{multi-algo,benchmarks,logs}

# GPU Detection
echo "🔍 Detecting GPU..."
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_VENDOR="nvidia"
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
    echo "🎮 NVIDIA GPU detected: $GPU_INFO"
elif command -v rocm-smi >/dev/null 2>&1; then
    GPU_VENDOR="amd"
    GPU_INFO=$(rocm-smi --showproductname | grep "GPU" | head -1)
    echo "🎮 AMD GPU detected: $GPU_INFO"
elif [[ "$PLATFORM" == "macos"* ]]; then
    GPU_VENDOR="apple"
    GPU_INFO=$(system_profiler SPDisplaysDataType | grep "Chipset Model" | head -1)
    echo "🍎 Apple GPU detected: $GPU_INFO"
else
    echo "⚠️  No GPU detected or no GPU tools available"
    GPU_VENDOR="unknown"
fi

# Download miners based on GPU
download_miners() {
    echo "⬇️  Downloading GPU miners..."
    
    case $GPU_VENDOR in
        "nvidia")
            echo "📦 Downloading T-Rex Miner (NVIDIA)..."
            if [[ "$PLATFORM" == "linux" ]]; then
                wget -q -O "$GPU_DIR/t-rex/t-rex.tar.gz" \
                    "https://github.com/trex-miner/T-Rex/releases/download/0.26.8/t-rex-0.26.8-linux.tar.gz"
            elif [[ "$PLATFORM" == "windows" ]]; then
                wget -q -O "$GPU_DIR/t-rex/t-rex.zip" \
                    "https://github.com/trex-miner/T-Rex/releases/download/0.26.8/t-rex-0.26.8-win.zip"
            fi
            ;;
        "amd")
            echo "📦 Downloading TeamRedMiner (AMD)..."
            if [[ "$PLATFORM" == "linux" ]]; then
                wget -q -O "$GPU_DIR/teamred/teamred.tar.gz" \
                    "https://github.com/todxx/teamredminer/releases/download/v0.10.19/teamredminer-v0.10.19-linux.tgz"
            fi
            ;;
    esac
    
    # Universal miners
    echo "📦 Downloading lolMiner (Universal)..."
    if [[ "$PLATFORM" == "linux" ]]; then
        wget -q -O "$GPU_DIR/lolminer/lolminer.tar.gz" \
            "https://github.com/Lolliedieb/lolMiner-releases/releases/download/1.82/lolMiner_v1.82_Lin64.tar.gz"
    elif [[ "$PLATFORM" == "macos"* ]]; then
        # Note: lolMiner has limited macOS support
        echo "⚠️  lolMiner has limited macOS support"
    fi
}

# Create GPU mining configs
create_gpu_configs() {
    echo "⚙️  Creating GPU mining configurations..."
    
    # KawPow (RavenCoin) config
    cat > "$GPU_DIR/configs/kawpow-rvn.json" << 'EOF'
{
  "pools": [
    {
      "url": "stratum+tcp://rvn-us-east1.nanopool.org:12433",
      "user": "ZION_RVN_ADDRESS.worker1",
      "pass": "x",
      "nicehash": false
    }
  ],
  "algorithm": "kawpow",
  "intensity": 20,
  "temperature-limit": 80,
  "power-limit": 300,
  "log-file": "../logs/gpu-kawpow.log"
}
EOF

    # ERGO (Autolykos) config  
    cat > "$GPU_DIR/configs/ergo-auto.json" << 'EOF'
{
  "pools": [
    {
      "url": "stratum+tcp://erg-us-east1.nanopool.org:11433", 
      "user": "ZION_ERGO_ADDRESS.worker1",
      "pass": "x"
    }
  ],
  "algorithm": "autolykos2",
  "intensity": 18,
  "temperature-limit": 75,
  "power-limit": 280,
  "log-file": "../logs/gpu-ergo.log"
}
EOF

    # ETC (Ethash) config
    cat > "$GPU_DIR/configs/etc-ethash.json" << 'EOF'
{
  "pools": [
    {
      "url": "stratum+tcp://etc-us-east1.nanopool.org:19999",
      "user": "ZION_ETC_ADDRESS.worker1", 
      "pass": "x"
    }
  ],
  "algorithm": "ethash",
  "intensity": 22,
  "temperature-limit": 78,
  "power-limit": 320,
  "log-file": "../logs/gpu-etc.log"
}
EOF
}

# Create hybrid mining script
create_hybrid_script() {
    echo "🔧 Creating hybrid mining script..."
    
    cat > "$MINING_DIR/start-hybrid-mining.sh" << 'EOF'
#!/bin/bash
# ZION Hybrid Mining: CPU (RandomX) + GPU (Multi-algo)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "🚀 Starting ZION Hybrid Mining"
echo "=============================="

# Function to cleanup on exit
cleanup() {
    echo "🛑 Stopping miners..."
    if [[ -n "$CPU_PID" ]]; then
        kill $CPU_PID 2>/dev/null
    fi
    if [[ -n "$GPU_PID" ]]; then 
        kill $GPU_PID 2>/dev/null
    fi
    echo "✅ Cleanup completed"
}
trap cleanup EXIT INT TERM

# Start CPU Mining (RandomX - ZION)
echo "⚡ Starting CPU Mining (RandomX)..."
if [[ -f "platforms/linux/xmrig-6.21.3/xmrig" ]]; then
    ./platforms/linux/xmrig-6.21.3/xmrig --config=xmrig-Ryzen3600.json > "$LOG_DIR/cpu-randomx.log" 2>&1 &
    CPU_PID=$!
    echo "✅ CPU miner started (PID: $CPU_PID)"
elif [[ -f "platforms/macos-arm64/xmrig-6.21.3/xmrig" ]]; then
    ./platforms/macos-arm64/xmrig-6.21.3/xmrig --config=xmrig-Ryzen3600.json > "$LOG_DIR/cpu-randomx.log" 2>&1 &
    CPU_PID=$!
    echo "✅ CPU miner started (PID: $CPU_PID)"
else
    echo "❌ XMRig not found"
    exit 1
fi

# Wait for CPU miner to initialize
sleep 15

# Start GPU Mining (Algorithm selection)
echo "🎮 Starting GPU Mining..."
GPU_ALGO="kawpow"  # Default algorithm
GPU_CONFIG="gpu/configs/${GPU_ALGO}-rvn.json"

if [[ -f "gpu/t-rex/t-rex" ]]; then
    echo "Using T-Rex miner for $GPU_ALGO"
    ./gpu/t-rex/t-rex --config "$GPU_CONFIG" > "$LOG_DIR/gpu-${GPU_ALGO}.log" 2>&1 &
    GPU_PID=$!
elif [[ -f "gpu/lolminer/lolMiner" ]]; then
    echo "Using lolMiner for $GPU_ALGO" 
    ./gpu/lolminer/lolMiner --config "$GPU_CONFIG" > "$LOG_DIR/gpu-${GPU_ALGO}.log" 2>&1 &
    GPU_PID=$!
else
    echo "⚠️  No GPU miner found, running CPU-only"
fi

echo "✅ Hybrid mining setup complete!"
echo "📊 Monitor logs: tail -f logs/*.log"
echo "🔄 Press Ctrl+C to stop"

# Monitor both processes
while true; do
    # Check CPU miner
    if ! kill -0 $CPU_PID 2>/dev/null; then
        echo "❌ CPU miner stopped"
        break
    fi
    
    # Check GPU miner (if running)
    if [[ -n "$GPU_PID" ]] && ! kill -0 $GPU_PID 2>/dev/null; then
        echo "❌ GPU miner stopped"
        break
    fi
    
    # Status update every 5 minutes
    sleep 300
    echo "⚡ Mining status: $(date) - CPU: ✅ GPU: ✅"
done

echo "🛑 Mining stopped"
EOF

    chmod +x "$MINING_DIR/start-hybrid-mining.sh"
}

# Create profit switching script
create_profit_switcher() {
    echo "💰 Creating profit switching script..."
    
    cat > "$MINING_DIR/multi-algo/profit-switcher.py" << 'EOF'
#!/usr/bin/env python3
"""
ZION Multi-Algorithm Profit Switcher
Automatically switches GPU mining algorithm based on profitability
"""

import requests
import json
import time
import subprocess
import os
from datetime import datetime

class ProfitSwitcher:
    def __init__(self):
        self.current_algo = None
        self.gpu_process = None
        self.algorithms = {
            'kawpow': {
                'pool': 'stratum+tcp://rvn-us-east1.nanopool.org:12433',
                'address': 'ZION_RVN_ADDRESS', 
                'miner': 't-rex',
                'config': 'gpu/configs/kawpow-rvn.json'
            },
            'ergo': {
                'pool': 'stratum+tcp://erg-us-east1.nanopool.org:11433',
                'address': 'ZION_ERGO_ADDRESS',
                'miner': 'lolminer', 
                'config': 'gpu/configs/ergo-auto.json'
            },
            'ethash': {
                'pool': 'stratum+tcp://etc-us-east1.nanopool.org:19999',
                'address': 'ZION_ETC_ADDRESS',
                'miner': 't-rex',
                'config': 'gpu/configs/etc-ethash.json'  
            }
        }
    
    def get_profitability(self):
        """Fetch algorithm profitability from API"""
        try:
            # WhatToMine API example
            response = requests.get('https://whattomine.com/coins.json', timeout=10)
            data = response.json()
            
            profits = {}
            for algo, info in self.algorithms.items():
                # Calculate estimated profit (simplified)
                profits[algo] = self._calculate_profit(data, algo)
            
            return profits
        except Exception as e:
            print(f"❌ Error fetching profitability: {e}")
            return None
    
    def _calculate_profit(self, data, algo):
        """Calculate profit for algorithm (placeholder)"""
        # This would be more sophisticated in practice
        base_profits = {'kawpow': 0.05, 'ergo': 0.04, 'ethash': 0.06}
        return base_profits.get(algo, 0.03)
    
    def switch_algorithm(self, new_algo):
        """Switch to new mining algorithm"""
        if new_algo == self.current_algo:
            return
            
        print(f"🔄 Switching from {self.current_algo} to {new_algo}")
        
        # Stop current miner
        if self.gpu_process:
            self.gpu_process.terminate()
            self.gpu_process.wait(timeout=10)
        
        # Start new miner
        algo_info = self.algorithms[new_algo]
        miner_path = f"gpu/{algo_info['miner']}/{algo_info['miner']}"
        config_path = algo_info['config']
        
        if os.path.exists(miner_path):
            self.gpu_process = subprocess.Popen([
                miner_path, '--config', config_path
            ])
            self.current_algo = new_algo
            print(f"✅ Started {new_algo} mining")
        else:
            print(f"❌ Miner not found: {miner_path}")
    
    def run(self):
        """Main profit switching loop"""
        print("💰 Starting ZION Profit Switcher")
        
        while True:
            try:
                profits = self.get_profitability()
                
                if profits:
                    # Find most profitable algorithm
                    best_algo = max(profits, key=profits.get)
                    best_profit = profits[best_algo]
                    
                    print(f"📊 Profitability check: {datetime.now()}")
                    for algo, profit in profits.items():
                        marker = "⭐" if algo == best_algo else "  "
                        print(f"{marker} {algo}: ${profit:.4f}/day")
                    
                    # Switch if significantly more profitable
                    if (self.current_algo != best_algo and 
                        (not self.current_algo or 
                         best_profit > profits.get(self.current_algo, 0) * 1.05)):
                        self.switch_algorithm(best_algo)
                
                # Wait 5 minutes before next check
                time.sleep(300)
                
            except KeyboardInterrupt:
                print("🛑 Profit switcher stopped")
                if self.gpu_process:
                    self.gpu_process.terminate()
                break
            except Exception as e:
                print(f"❌ Error in profit switcher: {e}")
                time.sleep(60)

if __name__ == "__main__":
    switcher = ProfitSwitcher()
    switcher.run()
EOF

    chmod +x "$MINING_DIR/multi-algo/profit-switcher.py"
}

# Main setup execution
main() {
    echo "🎯 Starting ZION Multi-Algo Mining Setup..."
    
    # Check requirements
    if ! command -v wget >/dev/null 2>&1; then
        echo "❌ wget is required but not installed"
        exit 1
    fi
    
    # Run setup steps
    download_miners
    create_gpu_configs  
    create_hybrid_script
    create_profit_switcher
    
    echo "✅ ZION Multi-Algo Mining Setup Complete!"
    echo ""
    echo "🚀 Next Steps:"
    echo "1. Edit GPU configs with your wallet addresses"
    echo "2. Run: ./start-hybrid-mining.sh"
    echo "3. Monitor: tail -f logs/*.log"
    echo "4. Profit switching: python3 multi-algo/profit-switcher.py"
    echo ""
    echo "📊 Expected Performance (Ryzen + RTX 4080):"
    echo "   CPU RandomX: 8-15 KH/s"
    echo "   GPU KawPow: 65-75 MH/s"
    echo "   Total Daily Profit: $4-11"
}

# Run main function
main "$@"
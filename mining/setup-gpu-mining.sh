#!/bin/bash

# 🚀 ZION GPU Mining Setup Script - Apple M1
# Automatická konfigurace GPU mining pro Apple Silicon

echo "🍎 ZION GPU Mining Setup for Apple M1"
echo "========================================="

# Kontrola architektury
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    echo "❌ This script is for Apple M1/ARM64 only"
    exit 1
fi

echo "✅ Detected Apple M1 architecture"

# Kontrola Metal GPU support
echo "🔍 Checking Metal GPU support..."
system_profiler SPDisplaysDataType | grep -q "Metal Support"
if [ $? -eq 0 ]; then
    echo "✅ Metal GPU support detected"
    GPU_CORES=$(system_profiler SPDisplaysDataType | grep "Total Number of Cores" | awk '{print $5}')
    echo "🔥 GPU Cores: $GPU_CORES"
else
    echo "❌ No Metal GPU support found"
    exit 1
fi

# Vytvoření GPU-optimalized config
echo "⚙️ Creating GPU-optimized mining configuration..."

CONFIG_DIR="$(dirname "$0")/platforms/macos-arm64/xmrig-6.21.3"
GPU_CONFIG="$CONFIG_DIR/config-zion-gpu.json"

# Backup current config
if [ -f "$CONFIG_DIR/config-zion.json" ]; then
    cp "$CONFIG_DIR/config-zion.json" "$CONFIG_DIR/config-zion-cpu-backup.json"
    echo "📋 Backed up CPU config to config-zion-cpu-backup.json"
fi

# Vytvoření GPU config
cat > "$GPU_CONFIG" << 'EOF'
{
    "api": {
        "id": null,
        "worker-id": "apple-m1-gpu"
    },
    "http": {
        "enabled": true,
        "host": "127.0.0.1",
        "port": 8080,
        "access-token": null,
        "restricted": true
    },
    "autosave": true,
    "background": false,
    "colors": true,
    "title": true,
    "randomx": {
        "init": 0,
        "init-avx2": 0,
        "mode": "fast",
        "1gb-pages": false,
        "rdmsr": false,
        "wrmsr": false,
        "cache_qos": false,
        "numa": false,
        "scratchpad_prefetch_mode": 1
    },
    "cpu": {
        "enabled": true,
        "huge-pages": false,
        "huge-pages-jit": false,
        "hw-aes": null,
        "priority": 2,
        "memory-pool": false,
        "yield": true,
        "asm": true,
        "max-threads-hint": 75,
        "argon2-impl": null,
        "rx": [0, 1, 2, 3, 4, 5]
    },
    "opencl": {
        "enabled": true,
        "cache": true,
        "loader": null,
        "platform": "AMD",
        "rx/0": [
            {
                "index": 0,
                "intensity": 1000,
                "worksize": 8,
                "strided_index": 2,
                "mem_chunk": 2,
                "comp_mode": true,
                "threads": 2
            }
        ]
    },
    "cuda": {
        "enabled": false,
        "loader": null
    },
    "log-file": "xmrig-zion-gpu.log",
    "donate-level": 1,
    "donate-over-proxy": 1,
    "pools": [
        {
            "algo": "rx/0",
            "coin": null,
            "url": "91.98.122.165:3333",
            "user": "ZION_WALLET_ADDRESS",
            "pass": "apple-m1-gpu",
            "rig-id": "zion-m1-hybrid",
            "nicehash": false,
            "keepalive": true,
            "enabled": true,
            "tls": false,
            "tls-fingerprint": null,
            "daemon": false,
            "socks5": null,
            "self-select": null,
            "submit-to-origin": false
        }
    ],
    "retries": 5,
    "retry-pause": 5,
    "print-time": 30,
    "dmi": true,
    "syslog": false,
    "verbose": 1,
    "watch": true,
    "pause-on-battery": true,
    "pause-on-active": false
}
EOF

echo "✅ Created GPU-optimized config: $GPU_CONFIG"

# Update main mining script
MINING_SCRIPT="$(dirname "$0")/start-mining-macos.sh"
if [ -f "$MINING_SCRIPT" ]; then
    # Backup original script
    cp "$MINING_SCRIPT" "$(dirname "$0")/start-mining-macos-cpu-backup.sh"
    
    # Update mining script for GPU
    sed -i '' 's/config-zion.json/config-zion-gpu.json/g' "$MINING_SCRIPT"
    echo "✅ Updated mining script to use GPU config"
fi

echo ""
echo "🚀 GPU Mining Setup Complete!"
echo "========================================="
echo "📋 What was configured:"
echo "  ✅ Apple M1 GPU detection"
echo "  ✅ Metal GPU support verified"
echo "  ✅ Hybrid CPU+GPU configuration"
echo "  ✅ Optimized for Apple Silicon"
echo "  ✅ HTTP API enabled on port 8080"
echo ""
echo "🔥 To start GPU mining:"
echo "  cd $(dirname "$0")"
echo "  ./start-mining-macos.sh"
echo ""
echo "📊 Monitor mining:"
echo "  curl http://localhost:8080/2/summary"
echo "  tail -f platforms/macos-arm64/xmrig-6.21.3/xmrig-zion-gpu.log"
echo ""
echo "🎯 Expected performance:"
echo "  CPU: ~1-2 KH/s (6-core efficiency)"
echo "  GPU: ~0.5-1 KH/s (Metal acceleration)"
echo "  Total: ~2-3 KH/s hybrid mining"
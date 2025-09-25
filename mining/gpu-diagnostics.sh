#!/bin/bash

# 🔧 ZION GPU Mining Troubleshooting & Testing Script

echo "🔧 ZION GPU Mining Diagnostics"
echo "============================="

# 1. GPU Hardware Check
echo "1️⃣ GPU Hardware Detection:"
system_profiler SPDisplaysDataType | grep -A 10 "Chipset Model"
echo ""

# 2. Metal Support Check  
echo "2️⃣ Metal Framework Support:"
if command -v xcrun >/dev/null 2>&1; then
    xcrun metal --version 2>/dev/null || echo "Metal compiler not available"
else
    echo "Xcode Command Line Tools not installed"
fi
echo ""

# 3. XMRig GPU Support Check
echo "3️⃣ XMRig GPU Capabilities:"
XMRIG_PATH="./platforms/macos-arm64/xmrig-6.21.3/xmrig"
if [ -f "$XMRIG_PATH" ]; then
    echo "✅ XMRig binary found"
    $XMRIG_PATH --help | grep -i "opencl\|gpu" || echo "No OpenCL support in this XMRig build"
else
    echo "❌ XMRig binary not found"
fi
echo ""

# 4. Config Validation
echo "4️⃣ Mining Configuration:"
GPU_CONFIG="./platforms/macos-arm64/xmrig-6.21.3/config-zion-gpu.json"
if [ -f "$GPU_CONFIG" ]; then
    echo "✅ GPU config exists"
    echo "OpenCL enabled: $(jq -r '.opencl.enabled' "$GPU_CONFIG" 2>/dev/null || echo "unknown")"
    echo "CPU enabled: $(jq -r '.cpu.enabled' "$GPU_CONFIG" 2>/dev/null || echo "unknown")"
else
    echo "❌ GPU config not found"
fi
echo ""

# 5. Pool Connectivity
echo "5️⃣ Mining Pool Connection:"
if nc -z 91.98.122.165 3333 2>/dev/null; then
    echo "✅ Pool is reachable (91.98.122.165:3333)"
else
    echo "❌ Pool is not reachable"
    echo "   Try alternative pools or check network"
fi
echo ""

# 6. Performance Test (if pool available)
echo "6️⃣ Performance Test Options:"
echo "🔹 CPU-only test:"
echo "   ./xmrig --config=config-zion-cpu-backup.json --max-cpu-usage=50 --dry-run"
echo ""
echo "🔹 GPU+CPU hybrid test:"
echo "   ./xmrig --config=config-zion-gpu.json --max-cpu-usage=75 --dry-run"  
echo ""
echo "🔹 Benchmark mode (no pool needed):"
echo "   ./xmrig --bench=1M --threads=6"
echo ""

# 7. Monitoring Commands
echo "7️⃣ Monitoring Commands:"
echo "📊 Real-time stats:"
echo "   curl http://localhost:8080/2/summary | jq ."
echo ""
echo "📋 Hashrate tracking:"
echo "   watch -n 5 'curl -s http://localhost:8080/2/summary | jq .hashrate'"
echo ""
echo "📝 Log monitoring:" 
echo "   tail -f platforms/macos-arm64/xmrig-6.21.3/xmrig-zion-gpu.log"

# 8. Expected Results
echo ""
echo "8️⃣ Expected Apple M1 Performance:"
echo "┌─────────────────┬──────────────┬─────────────┐"
echo "│ Component       │ Hashrate     │ Power Usage │"
echo "├─────────────────┼──────────────┼─────────────┤"
echo "│ CPU (6 cores)   │ 1.0-2.0 KH/s │ ~10W        │"
echo "│ GPU (8 cores)   │ 0.5-1.0 KH/s │ ~8W         │"  
echo "│ Total Hybrid    │ 1.5-3.0 KH/s │ ~18W        │"
echo "└─────────────────┴──────────────┴─────────────┘"
echo ""
echo "💡 Tips for optimization:"
echo "   • Monitor CPU temperature (< 80°C)"
echo "   • Use --pause-on-battery for laptops"
echo "   • Adjust intensity based on thermal throttling"
echo "   • Test different thread configurations"

#!/bin/bash
# 🎯 ZION GPU Algorithm Benchmark Script
# Test různé algoritmy pro nalezení nejlepšího pro tvou GPU

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$SCRIPT_DIR/benchmarks"
LOG_DIR="$SCRIPT_DIR/logs"

mkdir -p "$BENCHMARK_DIR" "$LOG_DIR"

echo "🎯 ZION GPU Algorithm Benchmark"
echo "=============================="

# Benchmark config
BENCHMARK_DURATION=60  # seconds per algorithm
ALGORITHMS=("kawpow" "ergo" "ethash" "octopus")

# GPU Detection
detect_gpu() {
    echo "🔍 Detecting GPU..."
    
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_VENDOR="nvidia"
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        echo "🎮 NVIDIA GPU: $GPU_NAME"
        
        # Get GPU specs
        MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        POWER=$(nvidia-smi --query-gpu=power.max_limit --format=csv,noheader,nounits | head -1)
        echo "   Memory: ${MEMORY} MB"
        echo "   Power Limit: ${POWER} W"
        
    elif command -v rocm-smi >/dev/null 2>&1; then
        GPU_VENDOR="amd"  
        GPU_NAME=$(rocm-smi --showproductname | grep "Card series" | awk -F': ' '{print $2}')
        echo "🎮 AMD GPU: $GPU_NAME"
        
    else
        echo "❌ No compatible GPU found"
        exit 1
    fi
}

# Create benchmark configs
create_benchmark_configs() {
    echo "⚙️  Creating benchmark configurations..."
    
    # KawPow benchmark (RVN)
    cat > "$BENCHMARK_DIR/bench-kawpow.json" << EOF
{
  "algorithm": "kawpow",
  "benchmark": true,
  "benchmark-hashorder": true,
  "pools": [{"url": "benchmark", "user": "benchmark", "pass": "x"}],
  "intensity": [18, 20, 22, 24],
  "log-file": "$LOG_DIR/bench-kawpow.log"
}
EOF

    # Ergo benchmark (ERGO)  
    cat > "$BENCHMARK_DIR/bench-ergo.json" << EOF
{
  "algorithm": "autolykos2", 
  "benchmark": true,
  "benchmark-hashorder": true,
  "pools": [{"url": "benchmark", "user": "benchmark", "pass": "x"}],
  "intensity": [16, 18, 20, 22],
  "log-file": "$LOG_DIR/bench-ergo.log"  
}
EOF

    # Ethash benchmark (ETC)
    cat > "$BENCHMARK_DIR/bench-ethash.json" << EOF
{
  "algorithm": "ethash",
  "benchmark": true, 
  "benchmark-hashorder": true,
  "pools": [{"url": "benchmark", "user": "benchmark", "pass": "x"}],
  "intensity": [20, 22, 24, 26],
  "log-file": "$LOG_DIR/bench-ethash.log"
}
EOF

    # Octopus benchmark (CFX)
    cat > "$BENCHMARK_DIR/bench-octopus.json" << EOF  
{
  "algorithm": "octopus",
  "benchmark": true,
  "benchmark-hashorder": true, 
  "pools": [{"url": "benchmark", "user": "benchmark", "pass": "x"}],
  "intensity": [18, 20, 22, 24],
  "log-file": "$LOG_DIR/bench-octopus.log"
}
EOF
}

# Run benchmark for algorithm
benchmark_algorithm() {
    local algo=$1
    local config="$BENCHMARK_DIR/bench-${algo}.json"
    local log="$LOG_DIR/bench-${algo}.log"
    
    echo "⚡ Benchmarking $algo..."
    
    # Find miner executable
    MINER=""
    if [[ "$GPU_VENDOR" == "nvidia" ]] && [[ -f "gpu/t-rex/t-rex" ]]; then
        MINER="gpu/t-rex/t-rex"
    elif [[ -f "gpu/lolminer/lolMiner" ]]; then
        MINER="gpu/lolminer/lolMiner"  
    elif [[ -f "gpu/nbminer/nbminer" ]]; then
        MINER="gpu/nbminer/nbminer"
    else
        echo "❌ No compatible miner found for $algo"
        return 1
    fi
    
    # Clear previous log
    > "$log"
    
    # Run benchmark
    echo "   Using: $(basename "$MINER")"
    echo "   Duration: ${BENCHMARK_DURATION}s"
    echo "   Config: $config"
    
    timeout $BENCHMARK_DURATION "$MINER" --config "$config" >> "$log" 2>&1 &
    local miner_pid=$!
    
    # Wait and monitor
    for i in $(seq 1 $BENCHMARK_DURATION); do
        if ! kill -0 $miner_pid 2>/dev/null; then
            break
        fi
        if (( i % 10 == 0 )); then
            echo "   Progress: ${i}/${BENCHMARK_DURATION}s"
        fi
        sleep 1
    done
    
    # Stop miner
    kill $miner_pid 2>/dev/null
    wait $miner_pid 2>/dev/null
    
    # Parse results
    parse_benchmark_results "$algo" "$log"
}

# Parse benchmark results from log
parse_benchmark_results() {
    local algo=$1 
    local log=$2
    
    if [[ ! -f "$log" ]]; then
        echo "   ❌ No benchmark log found"
        return
    fi
    
    # Extract hashrate (varies by miner)
    local hashrate=""
    local power=""
    local temp=""
    
    # T-Rex miner parsing
    if grep -q "T-Rex" "$log" 2>/dev/null; then
        hashrate=$(grep -E "Total.*H/s" "$log" | tail -1 | grep -oE '[0-9]+\.[0-9]+.*H/s' | head -1)
        power=$(grep -E "Power.*W" "$log" | tail -1 | grep -oE '[0-9]+W' | head -1)
        temp=$(grep -E "Temp.*C" "$log" | tail -1 | grep -oE '[0-9]+C' | head -1)
    fi
    
    # lolMiner parsing  
    if grep -q "lolMiner" "$log" 2>/dev/null; then
        hashrate=$(grep -E "Total.*h/s" "$log" | tail -1 | grep -oE '[0-9]+\.[0-9]+.*h/s' | head -1)
        power=$(grep -E "[0-9]+W" "$log" | tail -1 | grep -oE '[0-9]+W' | head -1)
        temp=$(grep -E "[0-9]+°C" "$log" | tail -1 | grep -oE '[0-9]+°C' | head -1)
    fi
    
    # Store results
    echo "$algo,$hashrate,$power,$temp,$(date)" >> "$BENCHMARK_DIR/results.csv"
    
    # Display results
    printf "   ✅ %-10s: %-15s %8s %6s\n" "$algo" "${hashrate:-N/A}" "${power:-N/A}" "${temp:-N/A}"
}

# Generate benchmark report
generate_report() {
    local report="$BENCHMARK_DIR/benchmark-report.md"
    
    echo "📊 Generating benchmark report..."
    
    cat > "$report" << EOF
# 🎯 ZION GPU Benchmark Report

**GPU**: $GPU_NAME  
**Date**: $(date)  
**Benchmark Duration**: ${BENCHMARK_DURATION}s per algorithm

## Results Summary

| Algorithm | Hashrate | Power | Temp | Efficiency |
|-----------|----------|-------|------|------------|
EOF

    # Add results from CSV
    if [[ -f "$BENCHMARK_DIR/results.csv" ]]; then
        while IFS=, read -r algo hashrate power temp timestamp; do
            # Calculate efficiency (hashrate per watt)
            local efficiency="N/A"
            if [[ -n "$hashrate" && -n "$power" ]]; then
                local hr_num=$(echo "$hashrate" | grep -oE '[0-9]+\.[0-9]+')
                local power_num=$(echo "$power" | grep -oE '[0-9]+')
                if [[ -n "$hr_num" && -n "$power_num" && "$power_num" -gt 0 ]]; then
                    efficiency=$(echo "scale=2; $hr_num / $power_num" | bc -l 2>/dev/null || echo "N/A")
                fi
            fi
            
            echo "| $algo | $hashrate | $power | $temp | $efficiency |" >> "$report"
        done < "$BENCHMARK_DIR/results.csv"
    fi
    
    cat >> "$report" << EOF

## Recommendations

Based on benchmark results:

1. **Most Profitable**: Check current market prices
2. **Most Efficient**: Algorithm with highest hashrate/power ratio  
3. **Coolest Running**: Algorithm with lowest temperature
4. **Most Stable**: Algorithm with consistent hashrate

## Next Steps

1. Choose target algorithm based on profitability
2. Fine-tune intensity settings
3. Set up profit switching  
4. Monitor 24/7 performance

## Logs

- Detailed logs available in: \`logs/bench-*.log\`
- Raw results: \`benchmarks/results.csv\`
EOF

    echo "✅ Report saved: $report"
    echo "📋 View with: cat $report"
}

# Main benchmark execution
main() {
    echo "🚀 Starting GPU benchmark process..."
    
    # Setup  
    detect_gpu
    create_benchmark_configs
    
    # Initialize results
    echo "algorithm,hashrate,power,temp,timestamp" > "$BENCHMARK_DIR/results.csv"
    
    echo ""
    echo "📊 Benchmark Results:"
    echo "   Algorithm    Hashrate        Power   Temp"
    echo "   ─────────────────────────────────────────"
    
    # Run benchmarks
    for algo in "${ALGORITHMS[@]}"; do
        benchmark_algorithm "$algo"
        sleep 5  # Cool down between tests
    done
    
    echo ""
    
    # Generate report
    generate_report
    
    echo ""
    echo "✅ Benchmark complete!"
    echo "📊 Results summary available in: benchmarks/benchmark-report.md"
    echo "🔄 Run profit switching with: python3 multi-algo/profit-switcher.py" 
}

# Run if called directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
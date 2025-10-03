#!/bin/bash
# ZION 2.7.1 - Startup Script
# Clean launch of the complete ZION 2.7.1 blockchain system

echo "🌟 Starting ZION 2.7.1 - Pure RandomX Blockchain"
echo "=================================================="

# Set working directory
cd "$(dirname "$0")"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Check if requirements are installed
echo "📦 Checking dependencies..."
python3 -c "import sys; sys.path.insert(0, '.'); import core.blockchain, mining.algorithms, mining.config" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Installing dependencies..."
    pip3 install -r requirements.txt
fi

echo "✅ Dependencies OK"

# Run tests first
echo "🧪 Running tests..."
python3 tests/run_tests.py
if [ $? -ne 0 ]; then
    echo "❌ Tests failed! Check the output above."
    exit 1
fi

echo "✅ Tests passed"

# Show system info
echo ""
echo "📊 ZION 2.7.1 System Information:"
echo "----------------------------------"
python3 zion_cli.py info

echo ""
echo "🔧 Available Commands:"
echo "----------------------"
echo "python3 zion_cli.py info                    - Show blockchain info"
echo "python3 zion_cli.py test                    - Run test suite"
echo "python3 zion_cli.py algorithms list         - List mining algorithms"
echo "python3 zion_cli.py algorithms benchmark    - Benchmark algorithms"
echo "python3 zion_cli.py mine --address <addr>   - Start mining"
echo "python3 zion_cli.py benchmark --blocks 5    - Run mining benchmark"

echo ""
echo "🚀 ZION 2.7.1 is ready!"
echo "Run 'python3 zion_cli.py --help' for more options."
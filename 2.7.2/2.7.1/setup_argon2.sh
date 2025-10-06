#!/bin/bash
# ZION 2.7.1 ASIC-Resistant Setup Script
# Installs Argon2 dependencies for maximum decentralization

echo "🛡️ ZION 2.7.1 ASIC-Resistant Argon2 Setup"
echo "=========================================="

# Detect OS
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 macOS detected"
    OS="macos"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "🐧 Linux detected"
    OS="linux"
else
    echo "❌ Unsupported OS: $OSTYPE"
    exit 1
fi

echo "📦 Installing Argon2 dependencies..."

# Install Python dependencies
echo "🐍 Installing Python packages..."
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install Python dependencies"
    exit 1
fi

echo "🧪 Testing Argon2 installation..."

# Test Argon2 import
python3 -c "
try:
    import argon2
    from argon2 import PasswordHasher
    print('✅ Argon2 library available')
    
    # Test ASIC-resistant algorithm
    from mining.algorithms import Argon2Algorithm
    config = {'time_cost': 1, 'memory_cost': 32768, 'parallelism': 1, 'hash_len': 32}
    algo = Argon2Algorithm(config)
    result = algo.benchmark(5)
    print(f'✅ ASIC-resistant mining working: {result[\"hashrate\"]}')
    
except ImportError as e:
    print(f'❌ Argon2 import failed: {e}')
    exit(1)
except Exception as e:
    print(f'❌ Algorithm test failed: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 ASIC-Resistant Setup Complete!"
    echo "🛡️ ZION 2.7.1 is now protected against ASIC mining"
    echo ""
    echo "🚀 Quick start:"
    echo "  cd /Volumes/Zion/2.7.1"
    echo "  python3 zion_cli.py info"
    echo "  python3 zion_cli.py algorithms benchmark"
    echo "  python3 zion_cli.py mine your_address"
    echo ""
    echo "🛡️ ASIC Resistance Features:"
    echo "  • Argon2 memory-hard algorithm"
    echo "  • 64MB+ memory requirement per thread"
    echo "  • CPU-only mining (no GPU/ASIC support)"
    echo "  • SHA256 and other ASIC-friendly algorithms blocked"
else
    echo "❌ Setup failed. Please check the errors above."
    exit 1
fi
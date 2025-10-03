#!/bin/bash
# ZION 2.7.1 - RandomX ASIC-Resistant Setup Script
# Install RandomX library for maximum decentralization

echo "🛡️ ZION 2.7.1 ASIC-Resistant RandomX Setup"
echo "==========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "🐧 Linux detected"

    # Check if we're on Ubuntu/Debian
    if command -v apt &> /dev/null; then
        print_info "Ubuntu/Debian detected - installing RandomX dependencies"

        # Update package list
        sudo apt update

        # Install build dependencies
        sudo apt install -y build-essential cmake libhwloc-dev libssl-dev git

        # Install Python RandomX
        print_info "Installing randomx-python..."
        pip3 install randomx-python

        if [ $? -eq 0 ]; then
            print_status "randomx-python installed successfully"
        else
            print_warning "randomx-python installation failed, trying alternative..."

            # Try installing from source
            git clone https://github.com/tevador/randomx-python.git
            cd randomx-python
            pip3 install .
            cd ..
            rm -rf randomx-python
        fi

    elif [[ -f /etc/redhat-release ]]; then
        print_info "Red Hat/CentOS detected"

        # Install build dependencies
        sudo dnf install -y gcc gcc-c++ make cmake hwloc-devel openssl-devel git

        # Install Python RandomX
        pip3 install randomx-python

    else
        print_warning "Unsupported Linux distribution"
        print_info "Please install RandomX manually from: https://github.com/tevador/RandomX"
    fi

elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 macOS detected"

    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        print_error "Homebrew not found. Please install Homebrew first:"
        echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi

    print_info "Installing RandomX dependencies via Homebrew..."

    # Install build dependencies
    brew install cmake openssl hwloc git

    # Install Python RandomX
    print_info "Installing randomx-python..."
    pip3 install randomx-python

    if [ $? -eq 0 ]; then
        print_status "RandomX installed successfully on macOS"
    else
        print_warning "Installation failed. You may need to install RandomX manually."
    fi

elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo "🪟 Windows detected"

    print_info "Installing RandomX for Windows..."

    # Check if we're using conda
    if command -v conda &> /dev/null; then
        print_info "Conda detected - installing via conda"
        conda install -c conda-forge randomx-python
    else
        # Try pip installation
        pip3 install randomx-python

        if [ $? -ne 0 ]; then
            print_warning "Automatic installation failed."
            print_info "Please install RandomX manually:"
            echo "1. Download from: https://github.com/tevador/RandomX/releases"
            echo "2. Or use conda: conda install -c conda-forge randomx-python"
        fi
    fi

else
    print_error "Unsupported OS: $OSTYPE"
    print_info "Please install RandomX manually from: https://github.com/tevador/RandomX"
    exit 1
fi

echo ""
print_info "Testing RandomX installation..."

# Test RandomX
python3 -c "
try:
    import randomx
    print('✅ RandomX library available')

    # Test basic functionality
    key = b'ZION_TEST_KEY'
    flags = randomx.get_flags()
    cache = randomx.create_cache(flags, key)
    vm = randomx.create_vm(flags, cache, None)

    test_data = b'ZION ASIC RESISTANT TEST'
    hash_result = randomx.calculate_hash(vm, test_data)

    print('✅ RandomX hash calculation works')
    print(f'   Test hash: {hash_result.hex()[:32]}...')

    # Cleanup
    randomx.destroy_vm(vm)
    randomx.destroy_cache(cache)

    print('✅ RandomX test completed successfully')

except ImportError as e:
    print('❌ RandomX library not available:', e)
    print('💡 Install with: pip3 install randomx-python')
    exit(1)
except Exception as e:
    print('❌ RandomX test failed:', e)
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    print_status "RandomX ASIC-resistant mining is ready!"
    echo ""
    echo "🎯 Why RandomX?"
    echo "==============="
    echo "• ASIC Resistant - Maximum decentralization"
    echo "• CPU Optimized - Fair mining for everyone"
    echo "• Memory Hard - Prevents specialized hardware"
    echo "• Verified Security - Battle-tested algorithm"
    echo ""
    echo "🚀 Usage:"
    echo "========"
    echo "cd /Volumes/Zion/2.7.1"
    echo "python3 zion_cli.py algorithms benchmark  # Test RandomX performance"
    echo "python3 zion_cli.py mine --address <addr>  # Start ASIC-resistant mining"
else
    print_error "RandomX setup failed. Please check the errors above."
    exit 1
fi
#!/bin/bash
# ZION 2.7.1 - Argon2 ASIC-Resistant Setup Script
# Install Argon2 library     # Install Python Argon2
    print_info "Installing argon2-cffi..."
    pip3 install argon2-cffi

    if [ $? -eq 0 ]; then
        print_status "Argon2 installed successfully on macOS"
    else
        print_warning "Installation failed. You may need to install Argon2 manually."
    fium decentralization

echo "🛡️ ZION 2.7.1 ASIC-Resistant Argon2 Setup"
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

        # Install Python Argon2
        print_info "Installing argon2-cffi..."
        pip3 install argon2-cffi

        if [ $? -eq 0 ]; then
            print_status "argon2-cffi installed successfully"
        else
            print_warning "argon2-cffi installation failed, trying alternative..."

            # Try installing from source
            pip3 install argon2-cffi --no-cache-dir
        fi

    elif [[ -f /etc/redhat-release ]]; then
        print_info "Red Hat/CentOS detected"

        # Install build dependencies
        sudo dnf install -y gcc gcc-c++ make cmake hwloc-devel openssl-devel git

        # Install Python Argon2
        pip3 install argon2-cffi

    else
        print_warning "Unsupported Linux distribution"
        print_info "Please install Argon2 manually: pip3 install argon2-cffi"
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

    print_info "Installing Argon2 for Windows..."

    # Check if we're using conda
    if command -v conda &> /dev/null; then
        print_info "Conda detected - installing via conda"
        conda install -c conda-forge argon2-cffi
    else
        # Try pip installation
        pip3 install argon2-cffi

        if [ $? -ne 0 ]; then
            print_warning "Automatic installation failed."
            print_info "Please install Argon2 manually:"
            echo "1. Use pip: pip3 install argon2-cffi"
            echo "2. Or use conda: conda install -c conda-forge argon2-cffi"
        fi
    fi

else
    print_error "Unsupported OS: $OSTYPE"
    print_info "Please install Argon2 manually: pip3 install argon2-cffi"
    exit 1
fi

echo ""
print_info "Testing Argon2 installation..."

# Test Argon2
python3 -c "
try:
    import argon2
    print('✅ Argon2 library available')

    # Test basic functionality
    from argon2 import PasswordHasher
    hasher = PasswordHasher()
    test_data = b'ZION ASIC RESISTANT TEST'
    hash_result = hasher.hash(test_data)

    print('✅ Argon2 hash calculation works')
    print(f'   Test hash: {hash_result.hex()[:32]}...')

    print('✅ Argon2 test completed successfully')

except ImportError as e:
    print('❌ Argon2 library not available:', e)
    print('💡 Install with: pip3 install argon2-cffi')
    exit(1)
except Exception as e:
    print('❌ Argon2 test failed:', e)
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    print_status "Argon2 ASIC-resistant mining is ready!"
    echo ""
    echo "🎯 Why Argon2?"
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
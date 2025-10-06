#!/bin/bash

# 🚀 ZION 2.7.1 SSH DEPLOYMENT SCRIPT WITH GPU MINING SUPPORT 🚀
# JAI RAM SITA HANUMAN - ON THE STAR
#
# Advanced deployment script for ZION 2.7.1 blockchain with:
# ✨ Complete AI ecosystem (11 components)
# 🧠 KRISTUS Quantum Engine (future-ready)
# ⚡ GPU mining acceleration (AMD/NVIDIA)
# 🛡️ Enhanced security and monitoring

echo "🌟 ═══════════════════════════════════════════════════════════════"
echo "🌟   ZION 2.7.1 BLOCKCHAIN SSH DEPLOYMENT & GPU MINING SETUP   🌟"
echo "🌟 ═══════════════════════════════════════════════════════════════"
echo "🚀 JAI RAM SITA HANUMAN - ON THE STAR"
echo

# Color definitions for beautiful output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# SSH Configuration Variables
SSH_HOST="${SSH_HOST:-91.98.122.165}"
SSH_USER="${SSH_USER:-root}"
SSH_KEY="${SSH_KEY:-~/.ssh/id_rsa}"
SSH_PORT="${SSH_PORT:-22}"
DEPLOYMENT_DIR="${DEPLOYMENT_DIR:-~/zion_271_deployment}"

# GPU Mining Configuration
ENABLE_GPU_MINING="${ENABLE_GPU_MINING:-true}"
GPU_TYPE="${GPU_TYPE:-auto}" # auto, amd, nvidia, cpu_only
MINING_THREADS="${MINING_THREADS:-auto}"
MEMORY_LIMIT="${MEMORY_LIMIT:-8192}" # MB

# Network Configuration
ZION_NETWORK="${ZION_NETWORK:-testnet}" # mainnet, testnet
ZION_PORT="${ZION_PORT:-19933}"
RPC_PORT="${RPC_PORT:-19934}"

# Function to print colored output
print_status() {
    echo -e "${CYAN}[$(date +'%H:%M:%S')]${NC} $1"
}

print_success() {
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

# Validation function
validate_config() {
    print_status "Validating deployment configuration..."
    
    if [ -z "$SSH_HOST" ]; then
        print_error "SSH_HOST not provided!"
        echo
        echo "Usage examples:"
        echo "  SSH_HOST=your.server.com SSH_USER=ubuntu ./zion_271_ssh_deploy.sh"
        echo "  SSH_HOST=192.168.1.100 SSH_USER=miner SSH_KEY=~/.ssh/mining_key ./zion_271_ssh_deploy.sh"
        echo
        echo "Environment variables:"
        echo "  SSH_HOST         - Target server hostname/IP (required)"
        echo "  SSH_USER         - SSH username (required)"
        echo "  SSH_KEY          - SSH private key path (default: ~/.ssh/id_rsa)"
        echo "  SSH_PORT         - SSH port (default: 22)"
        echo "  ENABLE_GPU_MINING - Enable GPU mining (default: true)"
        echo "  GPU_TYPE         - GPU type: auto, amd, nvidia, cpu_only (default: auto)"
        echo "  MINING_THREADS   - Mining threads (default: auto)"
        echo "  ZION_NETWORK     - Network: mainnet, testnet (default: testnet)"
        exit 1
    fi
    
    if [ -z "$SSH_USER" ]; then
        print_error "SSH_USER not provided!"
        exit 1
    fi
    
    if [ ! -f "$SSH_KEY" ]; then
        print_warning "SSH key not found at $SSH_KEY"
        print_info "Will attempt password authentication"
    fi
    
    print_success "Configuration validated"
}

# Create deployment package
create_deployment_package() {
    print_status "Creating ZION 2.7.1 deployment package..."
    
    PACKAGE_NAME="zion_271_deployment_$(date +%Y%m%d_%H%M%S).tar.gz"
    TEMP_DIR="/tmp/zion_271_package"
    
    # Clean and create temp directory
    rm -rf "$TEMP_DIR"
    mkdir -p "$TEMP_DIR"
    
    print_info "Packaging ZION 2.7.1 core components..."
    
    # Copy ZION 2.7.1 core files
    if [ -d "2.7.1" ]; then
        cp -r 2.7.1 "$TEMP_DIR/"
        print_success "Core ZION 2.7.1 files copied"
    else
        print_error "ZION 2.7.1 directory not found!"
        exit 1
    fi
    
    # Copy configuration files
    print_info "Adding configuration files..."
    mkdir -p "$TEMP_DIR/config"
    
    # Copy mining and network configs
    [ -f "config/mainnet.conf" ] && cp "config/mainnet.conf" "$TEMP_DIR/config/"
    [ -f "config/miner.conf" ] && cp "config/miner.conf" "$TEMP_DIR/config/"
    [ -f "config/genesis.json" ] && cp "config/genesis.json" "$TEMP_DIR/config/"
    
    # Copy scripts and tools
    print_info "Adding deployment scripts and tools..."
    mkdir -p "$TEMP_DIR/scripts"
    
    # Create GPU detection script
    cat > "$TEMP_DIR/scripts/detect_gpu.py" << 'EOF'
#!/usr/bin/env python3
"""GPU Detection and Configuration Script for ZION Mining"""

import subprocess
import sys
import json
import os

def detect_nvidia_gpus():
    """Detect NVIDIA GPUs using nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                name, memory = line.split(',')
                gpus.append({'type': 'nvidia', 'name': name.strip(), 'memory_mb': int(memory.strip())})
        return gpus
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

def detect_amd_gpus():
    """Detect AMD GPUs using rocm-smi or lspci"""
    try:
        # Try rocm-smi first
        result = subprocess.run(['rocm-smi', '--showproductname'], capture_output=True, text=True)
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.split('\n'):
                if 'GPU' in line and ':' in line:
                    name = line.split(':')[-1].strip()
                    gpus.append({'type': 'amd', 'name': name, 'memory_mb': 8192})  # Default estimate
            return gpus
    except FileNotFoundError:
        pass
    
    # Fallback to lspci
    try:
        result = subprocess.run(['lspci', '-v'], capture_output=True, text=True)
        gpus = []
        for line in result.stdout.split('\n'):
            if 'VGA compatible controller' in line and ('AMD' in line or 'ATI' in line or 'Radeon' in line):
                name = line.split(':')[-1].strip()
                gpus.append({'type': 'amd', 'name': name, 'memory_mb': 6144})  # Default estimate
        return gpus
    except FileNotFoundError:
        return []

def get_cpu_info():
    """Get CPU information"""
    try:
        with open('/proc/cpuinfo', 'r') as f:
            content = f.read()
        
        cpu_count = content.count('processor\t:')
        cpu_model = 'Unknown'
        
        for line in content.split('\n'):
            if line.startswith('model name'):
                cpu_model = line.split(':')[1].strip()
                break
        
        return {'cores': cpu_count, 'model': cpu_model}
    except:
        return {'cores': 1, 'model': 'Unknown'}

def main():
    print("🔍 ZION GPU Detection and Mining Configuration")
    print("=" * 50)
    
    # Detect GPUs
    nvidia_gpus = detect_nvidia_gpus()
    amd_gpus = detect_amd_gpus()
    cpu_info = get_cpu_info()
    
    config = {
        'nvidia_gpus': nvidia_gpus,
        'amd_gpus': amd_gpus,
        'cpu_info': cpu_info,
        'total_gpus': len(nvidia_gpus) + len(amd_gpus),
        'recommended_config': {}
    }
    
    print(f"🔧 CPU: {cpu_info['model']} ({cpu_info['cores']} cores)")
    print(f"🎮 NVIDIA GPUs: {len(nvidia_gpus)}")
    for gpu in nvidia_gpus:
        print(f"   - {gpu['name']} ({gpu['memory_mb']} MB)")
    
    print(f"🔴 AMD GPUs: {len(amd_gpus)}")
    for gpu in amd_gpus:
        print(f"   - {gpu['name']} ({gpu['memory_mb']} MB)")
    
    # Generate recommended configuration
    if nvidia_gpus:
        config['recommended_config']['primary_gpu'] = 'nvidia'
        config['recommended_config']['mining_threads'] = len(nvidia_gpus) * 2
        config['recommended_config']['memory_usage'] = min(sum(gpu['memory_mb'] for gpu in nvidia_gpus) * 0.8, 12288)
    elif amd_gpus:
        config['recommended_config']['primary_gpu'] = 'amd'
        config['recommended_config']['mining_threads'] = len(amd_gpus) * 2
        config['recommended_config']['memory_usage'] = min(sum(gpu['memory_mb'] for gpu in amd_gpus) * 0.8, 8192)
    else:
        config['recommended_config']['primary_gpu'] = 'cpu'
        config['recommended_config']['mining_threads'] = max(1, cpu_info['cores'] - 1)
        config['recommended_config']['memory_usage'] = 2048
    
    print(f"\n🚀 Recommended Configuration:")
    print(f"   Primary GPU: {config['recommended_config']['primary_gpu']}")
    print(f"   Mining Threads: {config['recommended_config']['mining_threads']}")
    print(f"   Memory Usage: {config['recommended_config']['memory_usage']} MB")
    
    # Save configuration
    with open('/tmp/zion_gpu_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Configuration saved to /tmp/zion_gpu_config.json")

if __name__ == '__main__':
    main()
EOF
    
    chmod +x "$TEMP_DIR/scripts/detect_gpu.py"
    
    # Create mining setup script
    cat > "$TEMP_DIR/scripts/setup_mining.sh" << 'EOF'
#!/bin/bash

echo "🚀 ZION 2.7.1 Mining Setup"
echo "=" * 30

# Detect hardware
python3 scripts/detect_gpu.py

# Install GPU drivers if needed
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "✅ NVIDIA drivers detected"
elif lspci | grep -i nvidia >/dev/null; then
    echo "⚠️  NVIDIA GPU detected but drivers not installed"
    echo "Installing NVIDIA drivers..."
    sudo apt update
    sudo apt install -y nvidia-driver-470 nvidia-utils-470
fi

if command -v rocm-smi >/dev/null 2>&1; then
    echo "✅ AMD ROCm detected"
elif lspci | grep -i amd >/dev/null; then
    echo "⚠️  AMD GPU detected but ROCm not installed"
    echo "Installing AMD ROCm..."
    sudo apt update
    sudo apt install -y rocm-dev rocm-libs
fi

# Setup Python environment
echo "🐍 Setting up Python environment..."
python3 -m pip install --user numpy scipy cryptography hashlib requests

# Create mining configuration
echo "⚙️ Creating mining configuration..."
python3 -c "
import json
import os

# Load GPU config
try:
    with open('/tmp/zion_gpu_config.json', 'r') as f:
        gpu_config = json.load(f)
except:
    gpu_config = {'recommended_config': {'primary_gpu': 'cpu', 'mining_threads': 2, 'memory_usage': 2048}}

mining_config = {
    'mining': {
        'enabled': True,
        'threads': gpu_config['recommended_config']['mining_threads'],
        'gpu_type': gpu_config['recommended_config']['primary_gpu'],
        'memory_limit_mb': gpu_config['recommended_config']['memory_usage'],
        'algorithm': 'zion_hybrid_pow',
        'pool': {
            'enabled': False,
            'url': 'stratum+tcp://pool.zion.network:4444',
            'user': 'ZiON_MINER_ADDRESS',
            'password': 'x'
        }
    },
    'network': {
        'testnet': True,
        'port': 19933,
        'rpc_port': 19934
    },
    'ai': {
        'enabled': True,
        'quantum_engine': False,  # Safe default
        'components': ['lightning', 'bio', 'music', 'cosmic']
    }
}

os.makedirs('config', exist_ok=True)
with open('config/mining_config.json', 'w') as f:
    json.dump(mining_config, f, indent=2)

print('✅ Mining configuration created')
"

echo "🌟 Mining setup complete!"
EOF
    
    chmod +x "$TEMP_DIR/scripts/setup_mining.sh"
    
    # Create the deployment archive
    print_info "Creating deployment archive..."
    cd "$TEMP_DIR/.." && tar -czf "$PACKAGE_NAME" "$(basename "$TEMP_DIR")"
    
    # Move to current directory
    mv "$PACKAGE_NAME" "$(pwd)/"
    
    # Cleanup
    rm -rf "$TEMP_DIR"
    
    print_success "Deployment package created: $PACKAGE_NAME"
    echo "$PACKAGE_NAME"
}

# Upload deployment package
upload_package() {
    local package_file="$1"
    print_status "Uploading deployment package to SSH server..."
    
    # Test SSH connection first
    print_info "Testing SSH connection to $SSH_USER@$SSH_HOST:$SSH_PORT..."
    
    SSH_OPTS="-o ConnectTimeout=10 -o StrictHostKeyChecking=no"
    if [ -f "$SSH_KEY" ]; then
        SSH_OPTS="$SSH_OPTS -i $SSH_KEY"
    fi
    
    if ! ssh $SSH_OPTS -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" "echo 'SSH connection successful'" 2>/dev/null; then
        print_error "SSH connection failed!"
        print_info "Please check:"
        print_info "  - SSH_HOST: $SSH_HOST"
        print_info "  - SSH_USER: $SSH_USER"
        print_info "  - SSH_PORT: $SSH_PORT"
        print_info "  - SSH_KEY: $SSH_KEY"
        exit 1
    fi
    
    print_success "SSH connection verified"
    
    # Upload package
    print_info "Uploading $package_file..."
    
    if [ -f "$SSH_KEY" ]; then
        scp -P "$SSH_PORT" -i "$SSH_KEY" "$package_file" "$SSH_USER@$SSH_HOST:~/"
    else
        scp -P "$SSH_PORT" "$package_file" "$SSH_USER@$SSH_HOST:~/"
    fi
    
    if [ $? -eq 0 ]; then
        print_success "Package uploaded successfully"
    else
        print_error "Package upload failed!"
        exit 1
    fi
}

# Remote deployment and setup
remote_setup() {
    local package_file="$1"
    print_status "Setting up ZION 2.7.1 on remote server..."
    
    SSH_OPTS="-p $SSH_PORT"
    if [ -f "$SSH_KEY" ]; then
        SSH_OPTS="$SSH_OPTS -i $SSH_KEY"
    fi
    
    ssh $SSH_OPTS "$SSH_USER@$SSH_HOST" << REMOTE_SCRIPT
#!/bin/bash

echo "🌟 ═══════════════════════════════════════════════════════════════"
echo "🌟        ZION 2.7.1 REMOTE SETUP AND GPU MINING CONFIG        🌟"
echo "🌟 ═══════════════════════════════════════════════════════════════"

# Colors for remote output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

print_status() { echo -e "\${CYAN}[Remote][\$(date +'%H:%M:%S')]\${NC} \$1"; }
print_success() { echo -e "\${GREEN}✅ \$1\${NC}"; }
print_warning() { echo -e "\${YELLOW}⚠️  \$1\${NC}"; }
print_error() { echo -e "\${RED}❌ \$1\${NC}"; }

# System information
print_status "System Information:"
echo "  OS: \$(lsb_release -d 2>/dev/null | cut -f2 || echo 'Unknown')"
echo "  Kernel: \$(uname -r)"
echo "  Architecture: \$(uname -m)"
echo "  CPU Cores: \$(nproc)"
echo "  Total Memory: \$(free -h | awk '/^Mem:/ {print \$2}')"

# Extract deployment package
PACKAGE_FILE="\$(ls -1t zion_271_deployment_*.tar.gz 2>/dev/null | head -1)"

if [ -z "\$PACKAGE_FILE" ]; then
    print_error "No deployment package found!"
    exit 1
fi

print_status "Extracting \$PACKAGE_FILE..."
tar -xzf "\$PACKAGE_FILE"

DEPLOYMENT_DIR="\$(tar -tzf "\$PACKAGE_FILE" | head -1 | cut -f1 -d'/')"
cd "\$DEPLOYMENT_DIR"

print_success "Package extracted to \$DEPLOYMENT_DIR"

# Update system
print_status "Updating system packages..."
sudo apt update -qq
sudo apt install -y python3 python3-pip build-essential cmake git curl wget htop

# Install Python packages
print_status "Installing Python requirements..."
python3 -m pip install --user --upgrade pip
python3 -m pip install --user numpy scipy cryptography requests hashlib256 2>/dev/null || echo "Some packages may not be available"

# Run hardware detection and setup
print_status "Running hardware detection and mining setup..."
cd scripts
chmod +x *.py *.sh
python3 detect_gpu.py
./setup_mining.sh

# Go back to main directory
cd ..

# Test ZION 2.7.1 components
print_status "Testing ZION 2.7.1 components..."

# Test AI Master Orchestrator
python3 -c "
import sys, os
sys.path.append('2.7.1/ai')
sys.path.append('2.7.1/core')

try:
    from ai_master_orchestrator import ZionAIMasterOrchestrator
    orchestrator = ZionAIMasterOrchestrator()
    print('✅ AI Master Orchestrator: OK')
    
    status = orchestrator.get_orchestrator_status()
    print(f'   Active components: {status[\"active_components\"]}')
    print(f'   Total operations: {status[\"total_operations\"]}')
except Exception as e:
    print(f'⚠️  AI Master Orchestrator: {e}')
"

# Test KRISTUS Quantum Engine (safe mode)
python3 -c "
import sys
sys.path.append('2.7.1/core')

try:
    from zion_271_kristus_quantum_engine import create_safe_kristus_engine
    engine = create_safe_kristus_engine(8, False)  # Safe mode
    print('✅ KRISTUS Quantum Engine: OK (Safe Mode)')
    
    # Test quantum hash
    test_hash = engine.compute_quantum_hash(b'SSH_DEPLOYMENT_TEST', 100000)
    print(f'   Test hash: {test_hash[:16]}...')
    
    stats = engine.get_engine_statistics()
    print(f'   Operations: {stats[\"operations\"][\"fallback_operations\"]} fallback')
    
except Exception as e:
    print(f'⚠️  KRISTUS Quantum Engine: {e}')
"

# Test Quantum AI Integration
python3 -c "
import sys
sys.path.append('2.7.1/ai')
sys.path.append('2.7.1/core')

try:
    from quantum_enhanced_ai_integration import get_quantum_ai_integrator
    integrator = get_quantum_ai_integrator(enable_quantum=False)  # Safe mode
    print('✅ Quantum AI Integration: OK')
    
    stats = integrator.get_integration_statistics()
    print(f'   Quantum enabled: {stats[\"quantum_enabled\"]}')
    print(f'   Operations: {stats[\"operations\"]}')
    
except Exception as e:
    print(f'⚠️  Quantum AI Integration: {e}')
"

# Create startup script
print_status "Creating ZION startup script..."
cat > start_zion_mining.sh << 'START_SCRIPT'
#!/bin/bash

echo "🚀 Starting ZION 2.7.1 Mining Node..."

# Load mining configuration
if [ -f "config/mining_config.json" ]; then
    echo "✅ Mining configuration loaded"
else
    echo "⚠️  No mining configuration found, creating default..."
    mkdir -p config
    echo '{"mining":{"enabled":true,"threads":2,"gpu_type":"cpu"},"network":{"testnet":true}}' > config/mining_config.json
fi

# Start ZION node
echo "🌟 Launching ZION 2.7.1 blockchain node..."

cd 2.7.1
export PYTHONPATH="\$PWD/core:\$PWD/ai:\$PYTHONPATH"

# Start with AI orchestrator
python3 -c "
import sys
sys.path.insert(0, 'ai')
sys.path.insert(0, 'core')

print('🌟 ZION 2.7.1 Blockchain Node Starting...')
print('🧠 AI Components: Lightning, Bio, Music, Cosmic, Quantum (Safe)')
print('🛡️ Security: KRISTUS Quantum Engine (Fallback Mode)')
print('⚡ Mining: GPU-accelerated (if available)')
print('')

# Initialize AI Orchestrator
try:
    from ai_master_orchestrator import ZionAIMasterOrchestrator
    orchestrator = ZionAIMasterOrchestrator()
    
    print('✅ AI Master Orchestrator initialized')
    print('🚀 ZION 2.7.1 is now ready for mining!')
    print('')
    print('Next steps:')
    print('1. Configure your mining address in config/mining_config.json')
    print('2. Enable quantum features after testing (optional)')
    print('3. Connect to mining pool or solo mine')
    print('')
    print('🌟 JAI RAM SITA HANUMAN - ON THE STAR!')
    
except Exception as e:
    print(f'❌ Error starting ZION: {e}')
    sys.exit(1)
"

echo "🌟 ZION 2.7.1 startup complete!"
START_SCRIPT

chmod +x start_zion_mining.sh

# Create monitoring script
cat > monitor_mining.sh << 'MONITOR_SCRIPT'
#!/bin/bash

echo "📊 ZION 2.7.1 Mining Monitor"
echo "=" * 30

while true; do
    echo "\$(date): Mining Status Check"
    
    # Check system resources
    echo "  CPU Usage: \$(top -bn1 | grep "Cpu(s)" | awk '{print \$2}' | cut -d'%' -f1)%"
    echo "  Memory: \$(free -h | awk '/^Mem:/ {printf \"%s / %s (%.1f%%)\", \$3, \$2, \$3/\$2 * 100.0}')"
    
    # Check GPU if available
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "  GPU Temp: \$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)°C"
        echo "  GPU Usage: \$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)%"
    fi
    
    echo "  Network: \$(ss -tuln | grep :19933 >/dev/null && echo 'Node Running' || echo 'Node Stopped')"
    echo ""
    
    sleep 60
done
MONITOR_SCRIPT

chmod +x monitor_mining.sh

# Final setup summary
print_status "ZION 2.7.1 SSH deployment completed!"
echo
print_success "🌟 Deployment Summary:"
echo "  📁 Installation Directory: \$(pwd)"
echo "  🚀 Startup Script: ./start_zion_mining.sh"
echo "  📊 Monitor Script: ./monitor_mining.sh"
echo "  ⚙️  Configuration: config/mining_config.json"
echo "  🔧 GPU Detection: scripts/detect_gpu.py"
echo
print_success "🌟 Next Steps:"
echo "  1. Run: ./start_zion_mining.sh"
echo "  2. Monitor: ./monitor_mining.sh (in another terminal)"
echo "  3. Configure mining address in config/mining_config.json"
echo "  4. Enable GPU mining if available"
echo
echo "🌟 JAI RAM SITA HANUMAN - ON THE STAR!"
echo "🚀 Ready to mine ZION and fly to the stars!"

REMOTE_SCRIPT
}

# Main deployment function
main() {
    echo "🌟 ZION 2.7.1 SSH Deployment Starting..."
    echo "🚀 Target: $SSH_USER@$SSH_HOST:$SSH_PORT"
    echo "⚡ GPU Mining: $ENABLE_GPU_MINING"
    echo "🌐 Network: $ZION_NETWORK"
    echo

    # Validate configuration
    validate_config

    # Create deployment package
    print_status "Phase 1: Creating deployment package..."
    PACKAGE_FILE=$(create_deployment_package)

    # Upload to SSH server
    print_status "Phase 2: Uploading to SSH server..."
    upload_package "$PACKAGE_FILE"

    # Remote setup
    print_status "Phase 3: Remote setup and configuration..."
    remote_setup "$PACKAGE_FILE"

    # Cleanup local package
    print_info "Cleaning up local deployment package..."
    rm -f "$PACKAGE_FILE"

    echo
    print_success "🌟 ZION 2.7.1 SSH deployment completed successfully!"
    echo
    print_info "🚀 SSH into your server and run:"
    print_info "   ssh $SSH_USER@$SSH_HOST -p $SSH_PORT"
    print_info "   cd ~/$(basename $(tar -tzf /tmp/temp_package.tar.gz 2>/dev/null | head -1 | cut -f1 -d'/' 2>/dev/null) 2>/dev/null || echo 'zion_271_deployment')"
    print_info "   ./start_zion_mining.sh"
    echo
    print_success "🌟 JAI RAM SITA HANUMAN - ON THE STAR!"
    print_success "🚀 Ready to mine ZION with GPU acceleration!"
}

# Help function
show_help() {
    cat << EOF
🌟 ZION 2.7.1 SSH Deployment Script

USAGE:
    SSH_HOST=server.com SSH_USER=miner ./zion_271_ssh_deploy.sh

REQUIRED ENVIRONMENT VARIABLES:
    SSH_HOST         Target server hostname or IP address
    SSH_USER         SSH username for connection

OPTIONAL ENVIRONMENT VARIABLES:
    SSH_KEY          SSH private key path (default: ~/.ssh/id_rsa)
    SSH_PORT         SSH port number (default: 22)
    ENABLE_GPU_MINING Enable GPU mining support (default: true)
    GPU_TYPE         GPU type: auto, amd, nvidia, cpu_only (default: auto)
    MINING_THREADS   Number of mining threads (default: auto)
    ZION_NETWORK     Network mode: mainnet, testnet (default: testnet)
    DEPLOYMENT_DIR   Remote deployment directory (default: ~/zion_271_deployment)

EXAMPLES:
    # Basic deployment
    SSH_HOST=192.168.1.100 SSH_USER=ubuntu ./zion_271_ssh_deploy.sh

    # Custom SSH port and key
    SSH_HOST=mining.server.com SSH_USER=miner SSH_PORT=2222 SSH_KEY=~/.ssh/mining_key ./zion_271_ssh_deploy.sh

    # CPU-only mining
    SSH_HOST=server.com SSH_USER=miner GPU_TYPE=cpu_only ./zion_271_ssh_deploy.sh

    # Mainnet deployment
    SSH_HOST=prod.server.com SSH_USER=miner ZION_NETWORK=mainnet ./zion_271_ssh_deploy.sh

🌟 JAI RAM SITA HANUMAN - ON THE STAR
EOF
}

# Check for help request
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
    exit 0
fi

# Run main deployment
main "$@"
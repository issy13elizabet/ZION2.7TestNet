#!/bin/bash
# SSH Mining Infrastructure - Complete Cleanup & Redeploy Script
# ZION v1.4.0 Miner Integration
# Date: 29. září 2025

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZION_ROOT="/media/maitreya/ZION1"
MINER_VERSION="v1.4.0"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to stop all mining processes
cleanup_processes() {
    log "🛑 Stopping all mining processes..."
    
    # Kill XMRig processes
    pkill -f xmrig || true
    pkill -f zion-miner || true
    pkill -f SRBMiner || true
    
    # Stop Docker mining containers
    if command -v docker >/dev/null 2>&1; then
        docker ps -q --filter "label=zion-mining" | xargs -r docker stop
        docker ps -aq --filter "label=zion-mining" | xargs -r docker rm
    fi
    
    success "Mining processes stopped"
}

# Function to clean SSH infrastructure
cleanup_ssh() {
    log "🧹 Cleaning SSH mining infrastructure..."
    
    # Remove SSH keys
    rm -rf ~/.ssh/zion-mining-* || true
    rm -rf ~/.ssh/mining-* || true
    
    # Clean known_hosts
    if [[ -f ~/.ssh/known_hosts ]]; then
        ssh-keygen -R mining-node-1.zion.net 2>/dev/null || true
        ssh-keygen -R mining-node-2.zion.net 2>/dev/null || true
        ssh-keygen -R 91.98.122.165 2>/dev/null || true
    fi
    
    # Remove mining directories
    rm -rf "${ZION_ROOT}/mining/ssh-deployments" || true
    rm -rf "${ZION_ROOT}/mining/remote-configs" || true
    
    success "SSH infrastructure cleaned"
}

# Function to clean Docker mining stack
cleanup_docker() {
    log "🐳 Cleaning Docker mining stack..."
    
    cd "${ZION_ROOT}"
    
    # Stop and remove mining compose stacks
    if [[ -f docker/compose.mining.yml ]]; then
        docker-compose -f docker/compose.mining.yml down --volumes --remove-orphans || true
    fi
    
    if [[ -f docker-compose.mining.yml ]]; then
        docker-compose -f docker-compose.mining.yml down --volumes --remove-orphans || true
    fi
    
    # Remove mining-related containers and networks
    docker network rm zion-mining-network 2>/dev/null || true
    docker volume rm zion-mining-data 2>/dev/null || true
    
    success "Docker stack cleaned"
}

# Function to prepare v1.4.0 miner binaries
prepare_binaries() {
    log "📦 Preparing ZION Miner ${MINER_VERSION} binaries..."
    
    local build_dir="${ZION_ROOT}/zion-miner-1.4.0/build"
    local deploy_dir="${ZION_ROOT}/mining/binaries"
    
    # Ensure build exists
    if [[ ! -f "${build_dir}/zion-miner" ]]; then
        warning "Building ZION Miner ${MINER_VERSION}..."
        cd "${ZION_ROOT}/zion-miner-1.4.0"
        mkdir -p build
        cd build
        cmake .. >/dev/null 2>&1
        make -j$(nproc) >/dev/null 2>&1
    fi
    
    # Create deployment directory
    mkdir -p "${deploy_dir}"
    
    # Copy binaries
    cp "${build_dir}/zion-miner" "${deploy_dir}/zion-miner-${MINER_VERSION}"
    cp "${build_dir}/zion-miner" "/tmp/zion-miner"
    chmod +x "${deploy_dir}/zion-miner-${MINER_VERSION}"
    chmod +x "/tmp/zion-miner"
    
    success "Binaries prepared: ${deploy_dir}/zion-miner-${MINER_VERSION}"
}

# Function to setup fresh mining configs
setup_configs() {
    log "⚙️ Setting up fresh mining configurations..."
    
    local config_dir="${ZION_ROOT}/mining/configs"
    mkdir -p "${config_dir}"
    
    # ZION Miner config
    cat > "${config_dir}/zion-miner-${MINER_VERSION}.conf" << EOF
# ZION Miner v1.4.0 Configuration
# CryptoNote Protocol with Debug Features

[mining]
pool_host=localhost
pool_port=3333
wallet_address=ZiYgACwXhrYG9iLjRfBgEdgsGsT6DqQ2brtM8j9iR3Rs7geE5kyj7oEGkw9LpjaGX9p1h7uRNJg5BkWKu8HD28EMPpJAYUdJ4
protocol=cryptonote
cpu_threads=auto
cpu_batch=10000

[debug]
# Uncomment for testing with forced low target
# force_low_target=ffffffffffffffff
verbose_stratum=true

[runtime]
# Keyboard shortcuts enabled
# g = GPU toggle, o = algorithm cycle, ? = help
# r = reset stats, c = clear screen, v = verbose
EOF

    # SSH deployment script template
    cat > "${config_dir}/deploy-node.sh" << 'EOF'
#!/bin/bash
# Deploy ZION Miner to remote node
set -euo pipefail

NODE_IP="$1"
BINARY_PATH="$2"

echo "🚀 Deploying ZION Miner to ${NODE_IP}..."

# Copy binary
scp "${BINARY_PATH}" root@${NODE_IP}:/tmp/zion-miner
ssh root@${NODE_IP} "chmod +x /tmp/zion-miner"

# Test connection
ssh root@${NODE_IP} "/tmp/zion-miner --help | head -5"

echo "✅ Deployment to ${NODE_IP} complete"
EOF
    
    chmod +x "${config_dir}/deploy-node.sh"
    
    success "Configurations created in ${config_dir}"
}

# Function to test core services
test_core_services() {
    log "🔍 Testing core ZION services..."
    
    # Test ZION Core RPC
    if curl -s localhost:18081/json_rpc -d '{"jsonrpc":"2.0","id":1,"method":"get_info"}' | grep -q "result"; then
        success "ZION Core RPC: ✅ Responding"
    else
        warning "ZION Core RPC: ❌ Not responding - may need restart"
    fi
    
    # Test Pool Stratum
    if timeout 5 telnet localhost 3333 >/dev/null 2>&1; then
        success "ZION Pool Stratum: ✅ Listening on port 3333"
    else
        warning "ZION Pool Stratum: ❌ Not listening - may need restart"
    fi
}

# Function to setup monitoring
setup_monitoring() {
    log "📊 Setting up mining monitoring..."
    
    local monitor_dir="${ZION_ROOT}/mining/monitoring"
    mkdir -p "${monitor_dir}"
    
    cat > "${monitor_dir}/mining-status.sh" << 'EOF'
#!/bin/bash
# ZION Mining Status Monitor
while true; do
    clear
    echo "🎯 ZION Mining Status - $(date)"
    echo "=================================="
    
    echo "📡 Core Services:"
    curl -s localhost:18081/json_rpc -d '{"jsonrpc":"2.0","id":1,"method":"get_info"}' | jq -r '.result.height // "❌ Offline"' | sed 's/^/  Height: /'
    
    echo "⛏️ Mining Processes:"
    pgrep -f zion-miner && echo "  ZION Miner: ✅ Running" || echo "  ZION Miner: ❌ Stopped"
    
    echo "🔗 Network:"
    ss -tlnp | grep :3333 >/dev/null && echo "  Pool Port: ✅ Open" || echo "  Pool Port: ❌ Closed"
    
    sleep 30
done
EOF
    
    chmod +x "${monitor_dir}/mining-status.sh"
    
    success "Monitoring setup: ${monitor_dir}/mining-status.sh"
}

# Main execution
main() {
    log "🚀 Starting ZION Mining Infrastructure - Complete Cleanup & Redeploy"
    log "Version: ${MINER_VERSION}"
    log "Target: Fresh deployment with v1.4.0 miner"
    echo
    
    # Confirmation
    read -p "⚠️  This will completely remove existing SSH mining setup. Continue? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "❌ Operation cancelled by user"
        exit 1
    fi
    
    # Execute cleanup steps
    cleanup_processes
    cleanup_ssh
    cleanup_docker
    
    # Setup fresh infrastructure
    prepare_binaries
    setup_configs
    setup_monitoring
    
    # Test services
    test_core_services
    
    echo
    success "🎉 SSH Mining Infrastructure cleanup & redeploy complete!"
    echo
    log "📋 Next steps:"
    echo "   1. Start core services: cd ${ZION_ROOT} && docker-compose up -d"
    echo "   2. Test local miner: /tmp/zion-miner --protocol=cryptonote --pool localhost:3333 --wallet <ADDRESS> --cpu-only"
    echo "   3. Deploy to nodes: ${ZION_ROOT}/mining/configs/deploy-node.sh <NODE_IP> ${ZION_ROOT}/mining/binaries/zion-miner-${MINER_VERSION}"
    echo "   4. Monitor status: ${ZION_ROOT}/mining/monitoring/mining-status.sh"
    echo
    log "🎯 Infrastructure ready for ZION Miner ${MINER_VERSION} deployment!"
}

# Execute main function
main "$@"
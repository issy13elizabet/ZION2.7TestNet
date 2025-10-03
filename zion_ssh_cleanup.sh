#!/bin/bash

# 🧹 ZION SSH SERVER CLEANUP SCRIPT 🧹
# Kompletní vymazání všech starých souborů před novým ZION 2.7.1 deploymentem
# JAI RAM SITA HANUMAN - ON THE STAR

echo "🧹 ═══════════════════════════════════════════════════════════════"
echo "🧹   ZION SSH SERVER CLEANUP - COMPLETE WIPE FOR FRESH START   🧹"
echo "🧹 ═══════════════════════════════════════════════════════════════"
echo "🚨 WARNING: This will DELETE ALL existing ZION files on SSH server!"
echo

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# SSH Configuration
SSH_HOST="${SSH_HOST:-91.98.122.165}"
SSH_USER="${SSH_USER:-root}"
SSH_KEY="${SSH_KEY:-~/.ssh/id_rsa}"
SSH_PORT="${SSH_PORT:-22}"

# Functions
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

# Validation
if [ -z "$SSH_HOST" ]; then
    print_error "SSH_HOST not provided!"
    echo "Usage: SSH_HOST=server.com SSH_USER=username ./zion_ssh_cleanup.sh"
    exit 1
fi

if [ -z "$SSH_USER" ]; then
    print_error "SSH_USER not provided!"
    exit 1
fi

print_warning "🚨 This will COMPLETELY WIPE all ZION files on $SSH_USER@$SSH_HOST:$SSH_PORT"
echo
print_info "Files and directories that will be deleted:"
echo "  - All zion* files and directories"
echo "  - All 2.7* directories"
echo "  - All pool* files"
echo "  - All stratum* files"
echo "  - All mining* files"
echo "  - All deployment packages"
echo "  - All log files"
echo "  - All backup files"
echo

read -p "Are you ABSOLUTELY SURE you want to proceed? Type 'YES DELETE ALL': " confirmation

if [ "$confirmation" != "YES DELETE ALL" ]; then
    print_info "Cleanup cancelled by user"
    exit 0
fi

echo
print_status "🧹 Starting SSH server cleanup..."

# Test SSH connection
print_status "Testing SSH connection to $SSH_USER@$SSH_HOST:$SSH_PORT..."

SSH_OPTS="-o ConnectTimeout=10 -o StrictHostKeyChecking=no"
if [ -f "$SSH_KEY" ]; then
    SSH_OPTS="$SSH_OPTS -i $SSH_KEY"
fi

if ! ssh $SSH_OPTS -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" "echo 'SSH connection successful'" 2>/dev/null; then
    print_error "SSH connection failed!"
    exit 1
fi

print_success "SSH connection verified"

# Execute cleanup on remote server
print_status "🧹 Executing comprehensive cleanup on remote server..."

ssh $SSH_OPTS -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" << 'CLEANUP_SCRIPT'
#!/bin/bash

echo "🧹 ═══════════════════════════════════════════════════════════════"
echo "🧹        REMOTE SERVER CLEANUP - DELETING ALL ZION FILES       🧹"
echo "🧹 ═══════════════════════════════════════════════════════════════"

# Colors for remote
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

print_status() { echo -e "${CYAN}[Remote]$(date +'%H:%M:%S')${NC} $1"; }
print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }

print_status "Current directory: $(pwd)"
print_status "User: $(whoami)"
print_status "Available space before cleanup: $(df -h . | tail -1 | awk '{print $4}')"

# Stop all ZION-related processes first
print_status "🛑 Stopping all ZION-related processes..."

# Kill Python processes related to ZION
pkill -f "stratum_pool.py" 2>/dev/null || true
pkill -f "zion" 2>/dev/null || true  
pkill -f "pool" 2>/dev/null || true
pkill -f "mining" 2>/dev/null || true
sleep 2

# Force kill if still running
pkill -9 -f "stratum_pool.py" 2>/dev/null || true
pkill -9 -f "zion" 2>/dev/null || true
pkill -9 -f "pool" 2>/dev/null || true
sleep 1

print_success "All processes stopped"

# List what will be deleted (for confirmation)
print_status "🔍 Scanning for ZION files to delete..."

echo "Files and directories found:"
find ~ -maxdepth 2 -name "*zion*" -o -name "*2.7*" -o -name "*pool*" -o -name "*stratum*" -o -name "*mining*" 2>/dev/null | head -20
echo

# Delete ZION directories and files
print_status "🗑️ Deleting ZION directories..."

# Delete version directories
rm -rf ~/2.7/ 2>/dev/null || true
rm -rf ~/2.7.1/ 2>/dev/null || true
rm -rf ~/zion/ 2>/dev/null || true
rm -rf ~/ZION/ 2>/dev/null || true

# Delete deployment directories
rm -rf ~/zion_*deployment* 2>/dev/null || true
rm -rf ~/deployment/ 2>/dev/null || true

print_success "Directories deleted"

# Delete ZION files
print_status "🗑️ Deleting ZION files..."

# Delete all zion-related files
rm -f ~/zion* 2>/dev/null || true
rm -f ~/ZION* 2>/dev/null || true

# Delete pool files
rm -f ~/*pool* 2>/dev/null || true
rm -f ~/*stratum* 2>/dev/null || true

# Delete mining files  
rm -f ~/*mining* 2>/dev/null || true
rm -f ~/*miner* 2>/dev/null || true

# Delete deployment packages
rm -f ~/*.tar.gz 2>/dev/null || true
rm -f ~/*.zip 2>/dev/null || true

# Delete log files
rm -f ~/*.log 2>/dev/null || true
rm -f ~/*.out 2>/dev/null || true
rm -f ~/pool.pid 2>/dev/null || true

# Delete backup files
rm -f ~/*.bak* 2>/dev/null || true
rm -f ~/*.old 2>/dev/null || true

print_success "Files deleted"

# Clean up configuration files
print_status "🗑️ Cleaning configuration files..."

rm -f ~/.zion* 2>/dev/null || true
rm -f ~/config.json 2>/dev/null || true
rm -f ~/mining_config.json 2>/dev/null || true

print_success "Configuration cleaned"

# Clean up temporary files
print_status "🗑️ Cleaning temporary files..."

rm -rf /tmp/zion* 2>/dev/null || true
rm -rf /tmp/mining* 2>/dev/null || true
rm -rf /tmp/pool* 2>/dev/null || true

print_success "Temporary files cleaned"

# Clean up Python cache and compiled files
print_status "🗑️ Cleaning Python cache..."

find ~ -name "*.pyc" -delete 2>/dev/null || true
find ~ -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find ~ -name "*.pyo" -delete 2>/dev/null || true

print_success "Python cache cleaned"

# Final verification
print_status "🔍 Final verification..."

remaining_files=$(find ~ -maxdepth 2 -name "*zion*" -o -name "*2.7*" -o -name "*pool*" -o -name "*stratum*" -o -name "*mining*" 2>/dev/null | wc -l)

if [ "$remaining_files" -eq 0 ]; then
    print_success "✨ Complete cleanup successful - no ZION files remaining!"
else
    print_warning "Some files may still remain:"
    find ~ -maxdepth 2 -name "*zion*" -o -name "*2.7*" -o -name "*pool*" -o -name "*stratum*" -o -name "*mining*" 2>/dev/null | head -5
fi

print_status "Available space after cleanup: $(df -h . | tail -1 | awk '{print $4}')"

print_status "🧹 Remote cleanup completed!"

# Show final system state
echo
print_status "📊 Final system state:"
echo "  Disk usage: $(df -h . | tail -1)"
echo "  Memory: $(free -h | grep Mem)"  
echo "  Processes: $(ps aux | grep -v grep | grep -E 'zion|pool|stratum|mining' | wc -l) ZION processes running"

echo
print_success "🌟 SSH server is now clean and ready for fresh ZION 2.7.1 deployment!"
print_success "🚀 JAI RAM SITA HANUMAN - ON THE STAR!"

CLEANUP_SCRIPT

# Local cleanup completion
if [ $? -eq 0 ]; then
    print_success "🌟 SSH server cleanup completed successfully!"
else
    print_error "❌ SSH server cleanup encountered errors"
    exit 1
fi

echo
print_success "🧹 Complete SSH Server Cleanup Summary:"
echo "  🎯 Target: $SSH_USER@$SSH_HOST:$SSH_PORT"
echo "  ✨ Status: All ZION files deleted"
echo "  🚀 Ready: For fresh ZION 2.7.1 deployment"
echo
print_info "Next steps:"
echo "  1. Run: SSH_HOST=$SSH_HOST SSH_USER=$SSH_USER ./zion_271_ssh_deploy.sh"
echo "  2. Monitor: SSH into server and check deployment"
echo "  3. Start: Launch new ZION 2.7.1 mining"
echo
print_success "🌟 JAI RAM SITA HANUMAN - ON THE STAR!"
print_success "🧹 Server is pristine clean for stellar journey!"
#!/bin/bash

# ZION Cryptocurrency Deployment Script
# Automated deployment for mainnet/testnet nodes

set -e

# Configuration
ZION_VERSION="1.0.0"
ZION_HOME="${ZION_HOME:-$HOME/.zion}"
ZION_DATA="${ZION_DATA:-$ZION_HOME/data}"
ZION_CONFIG="${ZION_CONFIG:-$ZION_HOME/config}"
ZION_LOGS="${ZION_LOGS:-$ZION_HOME/logs}"
ZION_WALLETS="${ZION_WALLETS:-$ZION_HOME/wallets}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# ASCII Art Banner
print_banner() {
    echo -e "${PURPLE}"
    cat << "EOF"
╔══════════════════════════════════════════════════════╗
║                                                      ║
║     ███████╗██╗ ██████╗ ███╗   ██╗                 ║
║     ╚══███╔╝██║██╔═══██╗████╗  ██║                 ║
║       ███╔╝ ██║██║   ██║██╔██╗ ██║                 ║
║      ███╔╝  ██║██║   ██║██║╚██╗██║                 ║
║     ███████╗██║╚██████╔╝██║ ╚████║                 ║
║     ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝                 ║
║                                                      ║
║         CRYPTOCURRENCY DEPLOYMENT SYSTEM            ║
║                    Version 1.0.0                     ║
╚══════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# Check system requirements
check_requirements() {
    echo -e "${BLUE}Checking system requirements...${NC}"
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    else
        echo -e "${RED}Unsupported OS: $OSTYPE${NC}"
        exit 1
    fi
    
    # Check CPU
    CPU_CORES=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 1)
    echo "  CPU Cores: $CPU_CORES"
    
    # Check RAM
    if [[ "$OS" == "macos" ]]; then
        TOTAL_RAM=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
    else
        TOTAL_RAM=$(( $(grep MemTotal /proc/meminfo | awk '{print $2}') / 1024 / 1024 ))
    fi
    echo "  Total RAM: ${TOTAL_RAM}GB"
    
    if [ "$TOTAL_RAM" -lt 2 ]; then
        echo -e "${YELLOW}Warning: Less than 2GB RAM. Mining may be slow.${NC}"
    fi
    
    # Check disk space
    DISK_SPACE=$(df -h . | tail -1 | awk '{print $4}')
    echo "  Available Disk: $DISK_SPACE"
    
    echo -e "${GREEN}✓ System requirements check passed${NC}\n"
}

# Install dependencies
install_dependencies() {
    echo -e "${BLUE}Installing dependencies...${NC}"
    
    if [[ "$OS" == "macos" ]]; then
        if ! command -v brew &> /dev/null; then
            echo "Installing Homebrew..."
            /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        fi
        
        echo "Installing required packages..."
        brew install cmake openssl leveldb
        
    elif [[ "$OS" == "linux" ]]; then
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y build-essential cmake libssl-dev libleveldb-dev
        elif command -v yum &> /dev/null; then
            sudo yum install -y gcc-c++ cmake openssl-devel leveldb-devel
        fi
    fi
    
    echo -e "${GREEN}✓ Dependencies installed${NC}\n"
}

# Setup directories
setup_directories() {
    echo -e "${BLUE}Setting up ZION directories...${NC}"
    
    mkdir -p "$ZION_HOME"
    mkdir -p "$ZION_DATA"
    mkdir -p "$ZION_CONFIG"
    mkdir -p "$ZION_LOGS"
    mkdir -p "$ZION_WALLETS"
    
    echo "  ZION_HOME: $ZION_HOME"
    echo "  ZION_DATA: $ZION_DATA"
    echo "  ZION_CONFIG: $ZION_CONFIG"
    echo "  ZION_LOGS: $ZION_LOGS"
    echo "  ZION_WALLETS: $ZION_WALLETS"
    
    echo -e "${GREEN}✓ Directories created${NC}\n"
}

# Build from source
build_zion() {
    echo -e "${BLUE}Building ZION from source...${NC}"
    
    cd "$(dirname "$0")"
    
    if [ ! -d "build" ]; then
        mkdir build
    fi
    
    cd build
    cmake ..
    make -j"$CPU_CORES"
    
    echo -e "${GREEN}✓ ZION built successfully${NC}\n"
}

# Install binaries
install_binaries() {
    echo -e "${BLUE}Installing ZION binaries...${NC}"
    
    cd "$(dirname "$0")/build"
    
    # Create bin directory
    mkdir -p "$ZION_HOME/bin"
    
    # Copy binaries
    cp ziond "$ZION_HOME/bin/" 2>/dev/null || true
    cp zion_wallet "$ZION_HOME/bin/" 2>/dev/null || true
    cp zion_miner "$ZION_HOME/bin/" 2>/dev/null || true
    
    # Make executable
    chmod +x "$ZION_HOME/bin/"* 2>/dev/null || true
    
    # Create symlinks
    if [[ "$OS" == "macos" ]]; then
        ln -sf "$ZION_HOME/bin/ziond" /usr/local/bin/ziond 2>/dev/null || true
        ln -sf "$ZION_HOME/bin/zion_wallet" /usr/local/bin/zion_wallet 2>/dev/null || true
        ln -sf "$ZION_HOME/bin/zion_miner" /usr/local/bin/zion_miner 2>/dev/null || true
    else
        sudo ln -sf "$ZION_HOME/bin/ziond" /usr/local/bin/ziond 2>/dev/null || true
        sudo ln -sf "$ZION_HOME/bin/zion_wallet" /usr/local/bin/zion_wallet 2>/dev/null || true
        sudo ln -sf "$ZION_HOME/bin/zion_miner" /usr/local/bin/zion_miner 2>/dev/null || true
    fi
    
    echo -e "${GREEN}✓ Binaries installed${NC}\n"
}

# Setup configuration
setup_config() {
    local NETWORK=$1
    echo -e "${BLUE}Setting up $NETWORK configuration...${NC}"
    
    # Copy configuration
    if [ "$NETWORK" == "mainnet" ]; then
        cp "$(dirname "$0")/config/mainnet.conf" "$ZION_CONFIG/zion.conf"
    else
        # Create testnet config
        cat > "$ZION_CONFIG/zion.conf" << EOF
# ZION Testnet Configuration
[network]
network_id = "zion-testnet-v1"
chain_id = 2
p2p_port = 28080
rpc_port = 28081

[blockchain]
genesis_timestamp = $(date +%s)
genesis_difficulty = 100
initial_block_reward = 50000000

[mining]
enable_mining = true
mining_threads = 1
EOF
    fi
    
    echo -e "${GREEN}✓ Configuration created${NC}\n"
}

# Create systemd service
create_service() {
    if [[ "$OS" != "linux" ]]; then
        echo -e "${YELLOW}Systemd service only available on Linux${NC}"
        return
    fi
    
    echo -e "${BLUE}Creating systemd service...${NC}"
    
    sudo tee /etc/systemd/system/ziond.service > /dev/null << EOF
[Unit]
Description=ZION Cryptocurrency Daemon
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$ZION_HOME
ExecStart=$ZION_HOME/bin/ziond --config=$ZION_CONFIG/zion.conf --datadir=$ZION_DATA
ExecStop=/bin/kill -TERM \$MAINPID
Restart=on-failure
RestartSec=30
StandardOutput=append:$ZION_LOGS/ziond.log
StandardError=append:$ZION_LOGS/ziond.error.log

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable ziond
    
    echo -e "${GREEN}✓ Systemd service created${NC}\n"
}

# Create launch agent for macOS
create_launchd() {
    if [[ "$OS" != "macos" ]]; then
        return
    fi
    
    echo -e "${BLUE}Creating launchd service...${NC}"
    
    cat > ~/Library/LaunchAgents/com.zion.daemon.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.zion.daemon</string>
    <key>ProgramArguments</key>
    <array>
        <string>$ZION_HOME/bin/ziond</string>
        <string>--config=$ZION_CONFIG/zion.conf</string>
        <string>--datadir=$ZION_DATA</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$ZION_LOGS/ziond.log</string>
    <key>StandardErrorPath</key>
    <string>$ZION_LOGS/ziond.error.log</string>
</dict>
</plist>
EOF
    
    launchctl load ~/Library/LaunchAgents/com.zion.daemon.plist
    
    echo -e "${GREEN}✓ Launchd service created${NC}\n"
}

# Start node
start_node() {
    echo -e "${BLUE}Starting ZION node...${NC}"
    
    if [[ "$OS" == "linux" ]]; then
        sudo systemctl start ziond
        sleep 3
        if systemctl is-active --quiet ziond; then
            echo -e "${GREEN}✓ ZION daemon started${NC}"
        else
            echo -e "${RED}Failed to start daemon${NC}"
            sudo journalctl -u ziond -n 50
        fi
    else
        "$ZION_HOME/bin/ziond" --config="$ZION_CONFIG/zion.conf" --datadir="$ZION_DATA" &
        sleep 3
        if ps aux | grep -q "[z]iond"; then
            echo -e "${GREEN}✓ ZION daemon started${NC}"
        else
            echo -e "${RED}Failed to start daemon${NC}"
        fi
    fi
}

# Create wallet
create_wallet() {
    echo -e "${BLUE}Creating new wallet...${NC}"
    
    "$ZION_HOME/bin/zion_wallet" new > "$ZION_WALLETS/wallet_info.txt"
    
    echo -e "${GREEN}✓ Wallet created${NC}"
    echo -e "${YELLOW}IMPORTANT: Wallet information saved to $ZION_WALLETS/wallet_info.txt${NC}"
    echo -e "${YELLOW}Make sure to backup this file!${NC}\n"
}

# Show status
show_status() {
    echo -e "${BLUE}ZION Node Status${NC}"
    echo "═══════════════════════════════════════"
    
    if ps aux | grep -q "[z]iond"; then
        echo -e "Daemon: ${GREEN}Running${NC}"
        
        # Get PID
        PID=$(ps aux | grep "[z]iond" | awk '{print $2}')
        echo "PID: $PID"
        
        # Check ports
        if lsof -i :18080 &>/dev/null; then
            echo -e "P2P Port: ${GREEN}Listening${NC}"
        else
            echo -e "P2P Port: ${YELLOW}Not listening${NC}"
        fi
        
        if lsof -i :18081 &>/dev/null; then
            echo -e "RPC Port: ${GREEN}Listening${NC}"
        else
            echo -e "RPC Port: ${YELLOW}Not listening${NC}"
        fi
    else
        echo -e "Daemon: ${RED}Not running${NC}"
    fi
    
    echo ""
    echo "Data directory: $ZION_DATA"
    echo "Config file: $ZION_CONFIG/zion.conf"
    echo "Log files: $ZION_LOGS"
    echo ""
}

# Main menu
main_menu() {
    print_banner
    
    echo "Welcome to ZION Deployment System"
    echo "══════════════════════════════════"
    echo ""
    echo "1) Deploy Mainnet Node"
    echo "2) Deploy Testnet Node"
    echo "3) Check Node Status"
    echo "4) Create Wallet"
    echo "5) Start Mining"
    echo "6) Stop Node"
    echo "7) View Logs"
    echo "8) Uninstall"
    echo "9) Exit"
    echo ""
    
    read -p "Select option: " choice
    
    case $choice in
        1)
            deploy_node "mainnet"
            ;;
        2)
            deploy_node "testnet"
            ;;
        3)
            show_status
            ;;
        4)
            create_wallet
            ;;
        5)
            start_mining
            ;;
        6)
            stop_node
            ;;
        7)
            view_logs
            ;;
        8)
            uninstall
            ;;
        9)
            exit 0
            ;;
        *)
            echo -e "${RED}Invalid option${NC}"
            ;;
    esac
}

# Deploy node
deploy_node() {
    local NETWORK=$1
    
    echo -e "${PURPLE}Deploying ZION $NETWORK node...${NC}\n"
    
    check_requirements
    install_dependencies
    setup_directories
    build_zion
    install_binaries
    setup_config "$NETWORK"
    
    if [[ "$OS" == "linux" ]]; then
        create_service
    else
        create_launchd
    fi
    
    start_node
    create_wallet
    
    echo ""
    echo -e "${GREEN}════════════════════════════════════════${NC}"
    echo -e "${GREEN}    ZION $NETWORK NODE DEPLOYED!       ${NC}"
    echo -e "${GREEN}════════════════════════════════════════${NC}"
    echo ""
    
    show_status
}

# Start mining
start_mining() {
    echo -e "${BLUE}Starting ZION miner...${NC}"
    
    read -p "Number of threads (0=auto): " threads
    read -p "Mining address: " address
    
    "$ZION_HOME/bin/zion_miner" --threads="$threads" --address="$address" &
    
    echo -e "${GREEN}✓ Miner started${NC}"
}

# Stop node
stop_node() {
    echo -e "${BLUE}Stopping ZION node...${NC}"
    
    if [[ "$OS" == "linux" ]]; then
        sudo systemctl stop ziond
    else
        killall ziond 2>/dev/null || true
    fi
    
    echo -e "${GREEN}✓ Node stopped${NC}"
}

# View logs
view_logs() {
    if [[ "$OS" == "linux" ]]; then
        sudo journalctl -u ziond -f
    else
        tail -f "$ZION_LOGS/ziond.log"
    fi
}

# Uninstall
uninstall() {
    echo -e "${RED}This will remove all ZION data!${NC}"
    read -p "Are you sure? (y/N): " confirm
    
    if [[ "$confirm" != "y" ]]; then
        return
    fi
    
    stop_node
    
    if [[ "$OS" == "linux" ]]; then
        sudo systemctl disable ziond
        sudo rm /etc/systemd/system/ziond.service
    else
        launchctl unload ~/Library/LaunchAgents/com.zion.daemon.plist
        rm ~/Library/LaunchAgents/com.zion.daemon.plist
    fi
    
    rm -rf "$ZION_HOME"
    rm -f /usr/local/bin/zion*
    
    echo -e "${GREEN}✓ ZION uninstalled${NC}"
}

# Run main menu if no arguments
if [ $# -eq 0 ]; then
    main_menu
else
    case $1 in
        mainnet)
            deploy_node "mainnet"
            ;;
        testnet)
            deploy_node "testnet"
            ;;
        status)
            show_status
            ;;
        *)
            echo "Usage: $0 [mainnet|testnet|status]"
            exit 1
            ;;
    esac
fi

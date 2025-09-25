#!/bin/bash
# ZION Health Monitoring Setup Script
# Installs and configures automated health monitoring

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
ZION_PATH="/home/maitreya/Zion/zion"
SERVICE_FILE="zion-health-monitor.service"
LOG_DIR="/var/log"
SYSTEMD_DIR="/etc/systemd/system"

echo -e "${BLUE}🔧 Setting up ZION Health Monitoring System${NC}"
echo "================================================="

# Check if running as root for systemd installation
if [ "$EUID" -eq 0 ]; then
    INSTALL_SYSTEMD=true
    echo -e "${GREEN}Running as root - will install systemd service${NC}"
else
    INSTALL_SYSTEMD=false
    echo -e "${YELLOW}Not running as root - manual systemd setup required${NC}"
fi

# Ensure scripts are executable
echo -n "Making scripts executable... "
chmod +x "$ZION_PATH/scripts/health-check.sh"
chmod +x "$ZION_PATH/scripts/health-monitor.sh"
echo -e "${GREEN}✅ Done${NC}"

# Test health check script
echo -n "Testing health check script... "
if "$ZION_PATH/scripts/health-check.sh" >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Working${NC}"
else
    echo -e "${YELLOW}⚠️  Issues detected (expected if services not running)${NC}"
fi

# Create log directory if needed
if [ "$INSTALL_SYSTEMD" = true ]; then
    echo -n "Setting up logging... "
    touch "$LOG_DIR/zion-monitor.log"
    chown maitreya:maitreya "$LOG_DIR/zion-monitor.log" 2>/dev/null || true
    echo -e "${GREEN}✅ Done${NC}"
fi

# Install systemd service
if [ "$INSTALL_SYSTEMD" = true ]; then
    echo -n "Installing systemd service... "
    cp "$ZION_PATH/scripts/$SERVICE_FILE" "$SYSTEMD_DIR/"
    systemctl daemon-reload
    echo -e "${GREEN}✅ Installed${NC}"
    
    echo -n "Enabling service... "
    systemctl enable zion-health-monitor.service
    echo -e "${GREEN}✅ Enabled${NC}"
    
    echo -n "Starting service... "
    systemctl start zion-health-monitor.service
    echo -e "${GREEN}✅ Started${NC}"
    
    # Check service status
    echo -e "\n${BLUE}Service Status:${NC}"
    systemctl status zion-health-monitor.service --no-pager -l
    
else
    echo -e "\n${YELLOW}Manual systemd setup required:${NC}"
    echo "1. Copy service file: sudo cp $ZION_PATH/scripts/$SERVICE_FILE /etc/systemd/system/"
    echo "2. Reload systemd: sudo systemctl daemon-reload"
    echo "3. Enable service: sudo systemctl enable zion-health-monitor.service"
    echo "4. Start service: sudo systemctl start zion-health-monitor.service"
fi

echo -e "\n${BLUE}📋 Usage Instructions:${NC}"
echo "- Manual health check: $ZION_PATH/scripts/health-check.sh"
echo "- Check monitor status: $ZION_PATH/scripts/health-monitor.sh status"
echo "- Restart services: $ZION_PATH/scripts/health-monitor.sh restart"
echo "- View logs: journalctl -u zion-health-monitor.service -f"
echo "- Stop monitoring: sudo systemctl stop zion-health-monitor.service"

echo -e "\n${BLUE}🔔 Alert Configuration:${NC}"
echo "Set DISCORD_WEBHOOK_URL environment variable for Discord alerts:"
echo "sudo systemctl edit zion-health-monitor.service"
echo "Add: [Service]"
echo "     Environment=\"DISCORD_WEBHOOK_URL=your_webhook_url_here\""

echo -e "\n${GREEN}🎉 ZION Health Monitoring setup complete!${NC}"
echo -e "${GREEN}The system will now automatically monitor ZION services every 5 minutes.${NC}"

if [ "$INSTALL_SYSTEMD" = true ]; then
    echo -e "\n${BLUE}Monitoring service is running. Check status with:${NC}"
    echo "sudo systemctl status zion-health-monitor.service"
fi
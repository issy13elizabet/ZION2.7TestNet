#!/bin/bash
# ZION Continuous Health Monitor
# Runs health checks every 5 minutes and handles alerts

# Configuration
HEALTH_CHECK_SCRIPT="/home/maitreya/Zion/zion/scripts/health-check.sh"
LOG_FILE="/var/log/zion-monitor.log"
ALERT_FILE="/tmp/zion-alerts.log"
CHECK_INTERVAL=300  # 5 minutes
MAX_FAILURES=3
FAILURE_COUNT=0

# Discord webhook for alerts (optional)
DISCORD_WEBHOOK_URL="${DISCORD_WEBHOOK_URL:-}"

# Colors for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging function
log() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$LOG_FILE"
}

# Send Discord alert
send_discord_alert() {
    local message=$1
    if [ -n "$DISCORD_WEBHOOK_URL" ]; then
        curl -s -H "Content-Type: application/json" \
             -d "{\"content\": \"🚨 ZION Alert: ${message}\"}" \
             "$DISCORD_WEBHOOK_URL" >/dev/null 2>&1
    fi
}

# Send alert to production server
send_production_alert() {
    local message=$1
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Log alert
    echo "$timestamp - $message" >> "$ALERT_FILE"
    
    # Send to Discord if configured
    send_discord_alert "$message"
    
    # You could also send email, SMS, or other notifications here
    log "ALERT" "$message"
}

# Main monitoring loop
monitor_health() {
    log "INFO" "Starting ZION health monitoring (interval: ${CHECK_INTERVAL}s)"
    
    while true; do
        log "INFO" "Running health check..."
        
        if "$HEALTH_CHECK_SCRIPT" >/dev/null 2>&1; then
            if [ $FAILURE_COUNT -gt 0 ]; then
                log "INFO" "Health check passed - system recovered"
                send_production_alert "System recovered after $FAILURE_COUNT failures"
                FAILURE_COUNT=0
            else
                log "INFO" "Health check passed - all systems operational"
            fi
        else
            FAILURE_COUNT=$((FAILURE_COUNT + 1))
            log "WARN" "Health check failed (failure $FAILURE_COUNT/$MAX_FAILURES)"
            
            if [ $FAILURE_COUNT -ge $MAX_FAILURES ]; then
                log "ERROR" "Maximum failures reached - sending alerts"
                send_production_alert "ZION TestNet health check failed $FAILURE_COUNT times consecutively"
                
                # Try to restart failed services
                log "INFO" "Attempting to restart services..."
                if command -v docker-compose >/dev/null 2>&1; then
                    cd /home/maitreya/Zion/zion
                    docker-compose restart >/dev/null 2>&1
                    log "INFO" "Docker compose restart attempted"
                fi
                
                # Reset failure count after restart attempt
                FAILURE_COUNT=0
            fi
        fi
        
        sleep $CHECK_INTERVAL
    done
}

# Handle signals for graceful shutdown
cleanup() {
    log "INFO" "ZION health monitor stopping..."
    exit 0
}

trap cleanup SIGTERM SIGINT

# Start monitoring
case "${1:-monitor}" in
    "monitor")
        monitor_health
        ;;
    "check")
        exec "$HEALTH_CHECK_SCRIPT"
        ;;
    "status")
        if [ -f "$ALERT_FILE" ]; then
            echo "Recent alerts:"
            tail -10 "$ALERT_FILE"
        else
            echo "No alerts recorded"
        fi
        ;;
    "restart")
        log "INFO" "Manual service restart requested"
        cd /home/maitreya/Zion/zion
        docker-compose restart
        ;;
    *)
        echo "Usage: $0 {monitor|check|status|restart}"
        echo "  monitor - Start continuous monitoring (default)"
        echo "  check   - Run single health check"
        echo "  status  - Show recent alerts"
        echo "  restart - Restart all services"
        exit 1
        ;;
esac
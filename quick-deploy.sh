#!/bin/bash

# ZION Production Server Deployment Script
# Quick deployment with health checks

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

print_banner() {
    echo -e "${BLUE}"
    cat << "EOF"
╔════════════════════════════════════════════════════════╗
║                   ZION DEPLOYMENT                     ║
║              Production Server Setup                  ║
╚════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker not found. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose not found. Please install Docker Compose first."
        exit 1
    fi
    
    # Check Git
    if ! command -v git &> /dev/null; then
        error "Git not found. Please install Git first."
        exit 1
    fi
    
    success "Prerequisites checked"
}

setup_directories() {
    log "Setting up directories..."
    
    mkdir -p logs
    mkdir -p data/blockchain
    mkdir -p data/wallet
    
    success "Directories created"
}

deploy_containers() {
    log "Deploying ZION containers..."
    
    # Stop any existing containers
    docker-compose -f docker-compose.prod.yml down 2>/dev/null || true
    
    # Start services
    docker-compose -f docker-compose.prod.yml up -d
    
    success "Containers deployed"
}

wait_for_services() {
    log "Waiting for services to start..."
    
    # Wait for main node
    local attempts=0
    local max_attempts=60
    
    while [ $attempts -lt $max_attempts ]; do
        if docker ps | grep -q "zion-production.*healthy"; then
            success "ZION node is healthy"
            break
        fi
        
        echo -n "."
        sleep 2
        attempts=$((attempts + 1))
    done
    
    if [ $attempts -eq $max_attempts ]; then
        error "Timeout waiting for services to start"
        exit 1
    fi
}

verify_deployment() {
    log "Verifying deployment..."
    
    # Check containers
    if docker ps --format "table {{.Names}}\t{{.Status}}" | grep -q zion; then
        success "Containers running:"
        docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep zion
    else
        error "No ZION containers running"
        exit 1
    fi
    
    # Check RPC endpoint
    log "Testing RPC endpoint..."
    sleep 5  # Give RPC time to start
    
    if curl -s --connect-timeout 10 http://localhost:18081/getheight > /dev/null; then
        HEIGHT=$(curl -s http://localhost:18081/getheight | python3 -c "import sys, json; print(json.load(sys.stdin)['height'])" 2>/dev/null || echo "unknown")
        success "RPC responding - Blockchain height: $HEIGHT"
    else
        warn "RPC not yet responding (this is normal on first start)"
    fi
    
    # Check P2P port
    if nc -z localhost 18080 2>/dev/null; then
        success "P2P port 18080 listening"
    else
        warn "P2P port not yet accessible"
    fi
}

show_status() {
    echo
    echo -e "${BLUE}=== DEPLOYMENT COMPLETE ===${NC}"
    echo
    echo "🚀 ZION Production Server is running!"
    echo
    echo "📊 Service URLs:"
    echo "   • RPC API: http://localhost:18081"
    echo "   • P2P Network: localhost:18080"
    echo "   • Frontend: cd frontend && npm run dev (port 3000)"
    echo
    echo "🔍 Monitoring commands:"
    echo "   • ./server-monitor.sh              # One-time status check"
    echo "   • ./server-monitor.sh --watch      # Continuous monitoring"
    echo "   • docker logs zion-production      # View logs"
    echo
    echo "🛠️  Management commands:"
    echo "   • docker-compose -f docker-compose.prod.yml logs -f     # Follow logs"
    echo "   • docker-compose -f docker-compose.prod.yml restart     # Restart services"
    echo "   • docker-compose -f docker-compose.prod.yml down        # Stop services"
    echo
    echo "💡 Next steps:"
    echo "   1. Wait for blockchain sync (check height with: curl localhost:18081/getheight)"
    echo "   2. Start frontend: cd frontend && npm run dev"
    echo "   3. Configure firewall for external access (ports 18080, 18081)"
    echo
}

main() {
    print_banner
    check_prerequisites
    setup_directories
    deploy_containers
    wait_for_services
    verify_deployment
    show_status
}

# Handle command line arguments
case "${1:-}" in
    "stop")
        log "Stopping ZION services..."
        docker-compose -f docker-compose.prod.yml down
        success "Services stopped"
        ;;
    "restart")
        log "Restarting ZION services..."
        docker-compose -f docker-compose.prod.yml restart
        success "Services restarted"
        ;;
    "status")
        ./server-monitor.sh
        ;;
    "logs")
        docker-compose -f docker-compose.prod.yml logs -f
        ;;
    "")
        main
        ;;
    *)
        echo "Usage: $0 [stop|restart|status|logs]"
        echo
        echo "Commands:"
        echo "  (no args)  - Deploy/start ZION server"
        echo "  stop       - Stop all services"
        echo "  restart    - Restart all services"
        echo "  status     - Show server status"
        echo "  logs       - Follow service logs"
        exit 1
        ;;
esac
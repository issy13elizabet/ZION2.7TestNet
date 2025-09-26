#!/bin/bash

# ZION Production Server Deployment
# =================================

set -e

COMPOSE_FILE="docker-compose.prod.yml"
SERVICE_NAME="zion-node"
RPC_PORT="18081"
P2P_PORT="18080"

echo "🚀 ZION Production Server Deployment"
echo "===================================="

check_prerequisites() {
    echo "🔍 Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker not found"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo "❌ Docker Compose not found"
        exit 1
    fi
    
    # Check ports
    if lsof -i :$RPC_PORT &> /dev/null; then
        echo "⚠️  Port $RPC_PORT already in use"
    fi
    
    if lsof -i :$P2P_PORT &> /dev/null; then
        echo "⚠️  Port $P2P_PORT already in use"
    fi
    
    echo "✅ Prerequisites OK"
}

deploy_services() {
    echo "🏗️  Deploying ZION services..."
    
    # Stop existing services
    echo "🛑 Stopping existing services..."
    docker-compose -f $COMPOSE_FILE down || true
    
    # Start services
    echo "🚀 Starting production services..."
    docker-compose -f $COMPOSE_FILE up -d
    
    echo "✅ Services started"
}

verify_deployment() {
    echo "🔍 Verifying deployment..."
    
    # Wait for services to start
    echo "⏳ Waiting for services to initialize..."
    sleep 10
    
    # Check container status
    if docker-compose -f $COMPOSE_FILE ps | grep -q "Up"; then
        echo "✅ Containers running"
    else
        echo "❌ Container startup failed"
        docker-compose -f $COMPOSE_FILE logs
        exit 1
    fi
    
    # Check RPC endpoint
    echo "🌐 Testing RPC endpoint..."
    for i in {1..30}; do
        if curl -s -f http://localhost:$RPC_PORT/getinfo &> /dev/null; then
            echo "✅ RPC endpoint responding"
            break
        fi
        echo "⏳ Waiting for RPC... ($i/30)"
        sleep 2
    done
    
    # Get deployment status
    echo "📊 Deployment Status:"
    ./prod-monitor.sh check
}

show_deployment_info() {
    echo "🎉 ZION Production Server Deployed!"
    echo "=================================="
    echo "🌐 RPC Endpoint: http://localhost:$RPC_PORT"
    echo "🔗 P2P Port: $P2P_PORT"
    echo "📝 Logs: docker-compose -f $COMPOSE_FILE logs -f"
    echo "📊 Monitor: ./prod-monitor.sh monitor"
    echo "🛑 Stop: docker-compose -f $COMPOSE_FILE down"
    echo ""
    echo "🔥 Užitečné příkazy:"
    echo "  ./prod-monitor.sh check    - Rychlá kontrola"
    echo "  ./prod-monitor.sh monitor  - Kontinuální monitoring"
    echo "  ./prod-monitor.sh logs     - Live logy"
    echo ""
    echo "🌟 Server je připraven pro produkci!"
}

main() {
    case "${1:-deploy}" in
        "deploy"|"start")
            check_prerequisites
            deploy_services
            verify_deployment
            show_deployment_info
            ;;
        "stop")
            echo "🛑 Stopping ZION production services..."
            docker-compose -f $COMPOSE_FILE down
            echo "✅ Services stopped"
            ;;
        "restart")
            echo "🔄 Restarting ZION production services..."
            docker-compose -f $COMPOSE_FILE restart
            verify_deployment
            ;;
        "status")
            ./prod-monitor.sh check
            ;;
        *)
            echo "Usage: $0 [deploy|start|stop|restart|status]"
            echo "  deploy/start - Deploy production server"
            echo "  stop         - Stop all services"
            echo "  restart      - Restart all services"
            echo "  status       - Check deployment status"
            ;;
    esac
}

main "$@"
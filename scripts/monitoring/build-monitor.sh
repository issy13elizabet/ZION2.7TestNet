#!/bin/bash

# ZION Build & Deployment Monitor
# ===============================

set -e

BUILD_ID=""
DEPLOYMENT_LOG="/tmp/zion-deployment.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$DEPLOYMENT_LOG"
}

check_build_status() {
    log "🏗️  Checking Docker build status..."
    
    # Get build process
    if build_ps=$(docker ps -a --filter "label=com.docker.compose.service=zion-node" 2>/dev/null); then
        if echo "$build_ps" | grep -q "Up"; then
            log "✅ Build completed - containers running"
            return 0
        elif echo "$build_ps" | grep -q "Exited"; then
            log "❌ Build failed - container exited"
            return 1
        fi
    fi
    
    # Check if build is in progress
    if docker images | grep -q "zion.*zion-node"; then
        log "✅ Build completed - image created"
        return 0
    fi
    
    # Check build logs
    if docker-compose -f docker-compose.prod.yml ps 2>/dev/null | grep -q "Up"; then
        log "✅ Services are running"
        return 0
    fi
    
    log "⏳ Build still in progress..."
    return 2
}

monitor_build() {
    log "🚀 Starting ZION build monitoring..."
    
    while true; do
        status=$(check_build_status)
        case $? in
            0) 
                log "🎉 Build completed successfully!"
                deploy_services
                break
                ;;
            1)
                log "💥 Build failed!"
                show_build_logs
                exit 1
                ;;
            2)
                log "⏳ Build in progress... waiting 30s"
                sleep 30
                ;;
        esac
    done
}

show_build_logs() {
    log "📝 Last build logs:"
    echo "--- Docker Compose Logs ---"
    docker-compose -f docker-compose.prod.yml logs --tail=20
}

deploy_services() {
    log "🚀 Deploying ZION production services..."
    
    # Ensure services are up
    docker-compose -f docker-compose.prod.yml up -d
    
    # Wait for RPC
    log "⏳ Waiting for RPC endpoint..."
    for i in {1..60}; do
        if curl -s -o /dev/null -w "%{http_code}" http://localhost:18081/getinfo | grep -q "200"; then
            log "✅ RPC endpoint ready!"
            break
        fi
        sleep 2
        if [ $i -eq 60 ]; then
            log "⚠️  RPC endpoint not responding after 2 minutes"
        fi
    done
    
    # Final health check
    ./prod-monitor.sh check
    
    log "🌟 ZION server deployment completed!"
    echo "
🎯 Next steps:
   • Monitor: ./prod-monitor.sh monitor
   • Logs: ./prod-monitor.sh logs  
   • RPC: curl http://localhost:18081/getinfo
   • Stop: docker-compose -f docker-compose.prod.yml down
"
}

case "${1:-monitor}" in
    "monitor")
        monitor_build
        ;;
    "status")
        check_build_status
        ;;
    "logs")
        show_build_logs
        ;;
    "deploy")
        deploy_services
        ;;
    *)
        echo "Usage: $0 [monitor|status|logs|deploy]"
        echo "  monitor - Watch build and auto-deploy when ready"
        echo "  status  - Check current build status"
        echo "  logs    - Show recent build logs"
        echo "  deploy  - Deploy services (assumes build complete)"
        ;;
esac
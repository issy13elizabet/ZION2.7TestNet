#!/bin/bash
# ZION 2.6.75 Production Deployment Script
# Sacred Technology + Battle-Tested Infrastructure
# 🕉️ Complete deployment automation 🕉️

set -e

echo "🕉️ ZION 2.6.75 PRODUCTION DEPLOYMENT 🕉️"
echo "Sacred Technology meets Production Infrastructure"
echo "=" * 80

# Configuration
ZION_VERSION="2.6.75"
COMPOSE_FILE="docker-compose.sacred-production.yml"
ENV_FILE=".env.production"
BACKUP_DIR="./backups"
LOG_DIR="./logs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

sacred() {
    echo -e "${PURPLE}🕉️ [SACRED]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check GPU support (optional)
    if command -v nvidia-smi &> /dev/null; then
        info "NVIDIA GPU detected - AI-GPU bridge will be enabled"
        GPU_ENABLED=true
    else
        warning "No NVIDIA GPU detected - AI-GPU bridge will run in CPU mode"
        GPU_ENABLED=false
    fi
    
    # Check available ports
    check_ports=(80 443 3000 8000 8080 3333 9735 6379 27017 9090 3001)
    for port in "${check_ports[@]}"; do
        if netstat -tuln | grep ":$port " > /dev/null; then
            warning "Port $port is already in use"
        fi
    done
    
    log "Prerequisites check completed ✅"
}

# Create directories
create_directories() {
    log "Creating necessary directories..."
    
    directories=(
        "$BACKUP_DIR"
        "$LOG_DIR"
        "./config/sacred"
        "./config/api"
        "./config/nginx"
        "./config/prometheus"
        "./config/grafana"
        "./config/lightning"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        info "Created directory: $dir"
    done
    
    log "Directories created ✅"
}

# Generate SSL certificates
generate_ssl_certs() {
    log "Generating SSL certificates..."
    
    SSL_DIR="./ssl"
    mkdir -p "$SSL_DIR"
    
    # Generate self-signed certificate for development
    if [ ! -f "$SSL_DIR/zion.crt" ]; then
        openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
            -keyout "$SSL_DIR/zion.key" \
            -out "$SSL_DIR/zion.crt" \
            -subj "/C=US/ST=Cosmos/L=Sacred/O=ZION/CN=zion.lol" \
            2>/dev/null || warning "SSL certificate generation failed"
        
        info "SSL certificates generated"
    else
        info "SSL certificates already exist"
    fi
}

# Create configuration files
create_config_files() {
    log "Creating configuration files..."
    
    # Nginx configuration
    cat > "./config/nginx/default.conf.template" << 'EOF'
upstream frontend {
    server ${FRONTEND_HOST}:3000;
}

upstream api {
    server ${API_HOST}:8000;
}

server {
    listen 80;
    server_name ${DOMAIN} www.${DOMAIN};
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name ${DOMAIN} www.${DOMAIN};
    
    ssl_certificate /etc/ssl/certs/zion.crt;
    ssl_certificate_key /etc/ssl/certs/zion.key;
    
    # Frontend
    location / {
        proxy_pass http://frontend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # API
    location /api/ {
        proxy_pass http://api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # WebSocket support
    location /ws {
        proxy_pass http://api;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF

    # Prometheus configuration
    cat > "./config/prometheus/prometheus.yml" << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "zion_rules.yml"

scrape_configs:
  - job_name: 'zion-core'
    static_configs:
      - targets: ['zion-sacred-core:8601']
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: 'zion-api'
    static_configs:
      - targets: ['zion-production-server:8000']
    metrics_path: /metrics
    scrape_interval: 15s

  - job_name: 'zion-mining-pool'
    static_configs:
      - targets: ['zion-mining-pool:3334']
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: 'zion-lightning'
    static_configs:
      - targets: ['zion-lightning:8181']
    metrics_path: /metrics
    scrape_interval: 15s

  - job_name: 'zion-bridges'
    static_configs:
      - targets: ['zion-bridge-manager:8080']
    metrics_path: /metrics
    scrape_interval: 15s

  - job_name: 'zion-ai-gpu'
    static_configs:
      - targets: ['zion-ai-gpu:8888']
    metrics_path: /metrics
    scrape_interval: 20s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
EOF
    
    info "Configuration files created"
}

# Build and deploy
deploy_zion() {
    log "Starting ZION 2.6.75 deployment..."
    
    # Load environment variables
    if [ -f "$ENV_FILE" ]; then
        set -a
        source "$ENV_FILE"
        set +a
        info "Environment variables loaded from $ENV_FILE"
    else
        warning "Environment file $ENV_FILE not found, using defaults"
    fi
    
    # Pull latest images
    sacred "Pulling latest Docker images..."
    docker-compose -f "$COMPOSE_FILE" pull
    
    # Build custom images
    sacred "Building ZION sacred containers..."
    docker-compose -f "$COMPOSE_FILE" build
    
    # Create and start services
    sacred "Starting ZION sacred infrastructure..."
    docker-compose -f "$COMPOSE_FILE" up -d
    
    log "ZION 2.6.75 deployment completed ✅"
}

# Health checks
run_health_checks() {
    log "Running health checks..."
    
    sleep 30  # Wait for services to start
    
    services=(
        "zion-legacy-daemon:18081"
        "zion-sacred-core:8601"
        "zion-production-server:8000"
        "zion-frontend:3000"
        "zion-redis:6379"
        "zion-mongo:27017"
    )
    
    for service in "${services[@]}"; do
        container=$(echo "$service" | cut -d':' -f1)
        port=$(echo "$service" | cut -d':' -f2)
        
        if docker ps | grep "$container" > /dev/null; then
            info "✅ $container is running"
            
            # Check if port is responding
            if timeout 5 bash -c "</dev/tcp/localhost/$port" 2>/dev/null; then
                info "✅ $container port $port is responding"
            else
                warning "⚠️ $container port $port not responding yet"
            fi
        else
            error "❌ $container is not running"
        fi
    done
}

# Show status
show_status() {
    log "ZION 2.6.75 Deployment Status:"
    echo
    sacred "🕉️ Sacred Technology Services:"
    echo "   📱 Frontend: http://localhost:3000"
    echo "   🔗 API: http://localhost:8000"
    echo "   ⛏️ Mining Pool: stratum+tcp://localhost:3334"
    echo "   ⚡ Lightning: http://localhost:8181"
    echo "   🌈 Bridges: http://localhost:8080"
    echo "   🤖 AI-GPU: http://localhost:8888"
    echo
    info "📊 Monitoring Services:"
    echo "   📈 Prometheus: http://localhost:9090"
    echo "   📊 Grafana: http://localhost:3001"
    echo
    info "🐳 Docker Services:"
    docker-compose -f "$COMPOSE_FILE" ps
    echo
    sacred "🌟 ZION 2.6.75 Sacred Technology is now operational! 🌟"
    echo "   Liberation through technology has begun."
}

# Backup function
backup_data() {
    log "Creating backup..."
    
    BACKUP_NAME="zion-backup-$(date +%Y%m%d-%H%M%S).tar.gz"
    BACKUP_PATH="$BACKUP_DIR/$BACKUP_NAME"
    
    # Stop services for consistent backup
    docker-compose -f "$COMPOSE_FILE" stop
    
    # Create backup
    tar -czf "$BACKUP_PATH" \
        --exclude='node_modules' \
        --exclude='*.log' \
        . || warning "Backup creation failed"
    
    # Restart services
    docker-compose -f "$COMPOSE_FILE" start
    
    info "Backup created: $BACKUP_PATH"
}

# Main deployment function
main() {
    case "${1:-deploy}" in
        "deploy"|"")
            check_prerequisites
            create_directories
            generate_ssl_certs
            create_config_files
            deploy_zion
            run_health_checks
            show_status
            ;;
        "status")
            show_status
            ;;
        "stop")
            log "Stopping ZION services..."
            docker-compose -f "$COMPOSE_FILE" down
            ;;
        "restart")
            log "Restarting ZION services..."
            docker-compose -f "$COMPOSE_FILE" restart
            ;;
        "logs")
            docker-compose -f "$COMPOSE_FILE" logs -f "${2:-}"
            ;;
        "backup")
            backup_data
            ;;
        "update")
            log "Updating ZION..."
            docker-compose -f "$COMPOSE_FILE" pull
            docker-compose -f "$COMPOSE_FILE" up -d
            ;;
        "clean")
            warning "This will remove all ZION containers and volumes!"
            read -p "Are you sure? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                docker-compose -f "$COMPOSE_FILE" down -v
                docker system prune -f
                info "Cleanup completed"
            fi
            ;;
        *)
            echo "Usage: $0 {deploy|status|stop|restart|logs|backup|update|clean}"
            echo "Commands:"
            echo "  deploy  - Deploy ZION 2.6.75 production infrastructure"
            echo "  status  - Show deployment status"
            echo "  stop    - Stop all services"
            echo "  restart - Restart all services"  
            echo "  logs    - Show logs (optionally specify service)"
            echo "  backup  - Create backup of all data"
            echo "  update  - Update to latest images"
            echo "  clean   - Remove all containers and volumes"
            exit 1
            ;;
    esac
}

# Execute main function
main "$@"
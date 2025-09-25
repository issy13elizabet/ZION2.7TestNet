# Zion Docker Deployment Guide

## Quick Start

### Local Development

```bash
# Build the image
docker build -t zion:latest .

# Start seed node and pool server
docker-compose up -d

# View logs
docker-compose logs -f

# Stop everything
docker-compose down
```

### Production Deployment on Ubuntu Server

1. **SSH to your server:**
```bash
ssh root@your-server-ip
```

2. **Run the deployment script:**
```bash
# Download and run deployment script
curl -sSL https://raw.githubusercontent.com/Yose144/Zion/master/docker/deploy.sh | bash -s pool
```

Or manually:

```bash
# Clone repository
git clone https://github.com/Yose144/Zion.git /opt/zion
cd /opt/zion

# Run deployment
chmod +x docker/deploy.sh
./docker/deploy.sh pool
```

## Docker Images

### Build from source
```bash
docker build -t zion:latest .
```

### Multi-platform build (for ARM servers)
```bash
docker buildx build --platform linux/amd64,linux/arm64 -t zion:latest .
```

## Container Modes

The Docker container supports multiple modes via the `ZION_MODE` environment variable:

- **daemon**: Standard full node
- **pool**: Mining pool server
- **miner**: CPU miner
- **seed**: Seed node (no mining)
- **wallet**: Wallet CLI
- **genesis**: Genesis block generator

## Docker Compose Configurations

### Development (docker-compose.yml)
- Seed node on ports 18080/18081
- Pool server on port 3333
- Optional miner (use `--profile mining`)
- Optional full node (use `--profile full`)

### Production (docker-compose.prod.yml)
- Optimized for production use
- Resource limits and logging
- Optional monitoring stack (Grafana, Loki, Prometheus)
- Nginx reverse proxy support

## Environment Variables

### Common
- `ZION_MODE`: Operating mode (daemon/pool/miner/seed)
- `ZION_CONFIG`: Config file path
- `ZION_DATA_DIR`: Data directory path
- `ZION_LOG_LEVEL`: Logging level (debug/info/warn/error)

### Pool Server
- `POOL_PORT`: Stratum port (default: 3333)
- `POOL_DIFFICULTY`: Initial difficulty (default: 1000)
- `POOL_FEE`: Pool fee percentage (default: 1%)

### Miner
- `MINER_THREADS`: Number of mining threads
- `MINER_POOL`: Pool address (host:port)
- `MINER_WALLET`: Wallet address for payouts

### Network
- `P2P_PORT`: P2P port (default: 18081)
- `RPC_PORT`: RPC port (default: 18080)
- `SEED_NODES`: Comma-separated seed nodes

## Monitoring

Enable monitoring stack:
```bash
docker-compose -f docker-compose.prod.yml --profile monitoring up -d
```

Access:
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus Node Exporter: http://localhost:9100/metrics
- Loki: http://localhost:3100

## Networking

### Expose Ports
- **3333**: Stratum mining protocol
- **18080**: JSON-RPC API
- **18081**: P2P network

### Firewall Rules (UFW)
```bash
# Pool server
sudo ufw allow 3333/tcp
sudo ufw allow 18081/tcp

# Seed node
sudo ufw allow 18080/tcp
sudo ufw allow 18081/tcp
```

## Volumes

### Data Persistence
```yaml
volumes:
  - ./data:/home/zion/.zion  # Blockchain data
  - ./logs:/var/log/zion     # Log files
```

### Backup
```bash
# Backup blockchain data
docker run --rm -v zion_seed-data:/data -v $(pwd):/backup alpine tar czf /backup/zion-backup.tar.gz /data

# Restore
docker run --rm -v zion_seed-data:/data -v $(pwd):/backup alpine tar xzf /backup/zion-backup.tar.gz -C /
```

## Troubleshooting

### View logs
```bash
docker logs -f zion-pool
docker-compose logs -f pool-server
```

### Shell access
```bash
docker exec -it zion-pool bash
```

### Reset data
```bash
docker-compose down -v  # Removes volumes
docker-compose up -d
```

### Health check
```bash
docker inspect zion-pool --format='{{.State.Health.Status}}'
curl http://localhost:18080/status
```

## Security Recommendations

1. **Use non-root user** (already configured in Dockerfile)
2. **Enable firewall** (UFW recommended)
3. **Use SSL/TLS** for public endpoints (nginx proxy)
4. **Regular updates**:
   ```bash
   docker pull zion:latest
   docker-compose up -d
   ```
5. **Resource limits** (configured in docker-compose.prod.yml)
6. **Log rotation** (configured with json-file driver)

## CI/CD Integration

GitHub Actions workflow will be added for:
- Automatic Docker image builds
- Push to GitHub Container Registry
- Multi-platform builds (amd64, arm64)
- Security scanning with Trivy

## Support

- GitHub Issues: https://github.com/Yose144/Zion/issues
- Docker Hub: https://hub.docker.com/r/yose144/zion (coming soon)
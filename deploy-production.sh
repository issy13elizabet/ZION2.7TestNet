#!/bin/bash

# 🌟 ZION PRODUCTION DEPLOYMENT SCRIPT
# Simple and functional deployment

set -e

echo "🚀 ZION Production Deployment Starting..."

# Stop old containers
echo "🛑 Stopping old containers..."
docker-compose -f docker-compose.production.yml down 2>/dev/null || true

# Clean up old images
echo "🧹 Cleaning up old images..."
docker system prune -f

# Build new image
echo "🔨 Building ZION Production Image..."
docker-compose -f docker-compose.production.yml build --no-cache

# Start production server
echo "🚀 Starting ZION Production Server..."
docker-compose -f docker-compose.production.yml up -d

# Wait for startup
echo "⏳ Waiting for server startup..."
sleep 10

# Test health
echo "🩺 Testing server health..."
if curl -f http://localhost:8888/health > /dev/null 2>&1; then
    echo "✅ ZION Production Server is HEALTHY!"
    echo ""
    echo "🌐 API Endpoints:"
    echo "   📊 Health: http://localhost:8888/health"
    echo "   🔗 Bridge Status: http://localhost:8888/api/bridge/status"  
    echo "   🌈 Rainbow Bridge: http://localhost:8888/api/rainbow-bridge/status"
    echo "   🌌 Galaxy Debug: http://localhost:8888/api/galaxy/debug"
    echo ""
    echo "🎉 DEPLOYMENT SUCCESSFUL!"
else
    echo "❌ Health check failed!"
    echo "📋 Container logs:"
    docker-compose -f docker-compose.production.yml logs
    exit 1
fi
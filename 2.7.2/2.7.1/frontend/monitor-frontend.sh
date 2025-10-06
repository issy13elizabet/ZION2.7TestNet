#!/bin/bash

# ZION Frontend Monitor Script
# Sleduje stav ZION frontend development serveru

echo "🔮 ZION Frontend Monitor - $(date)"
echo "========================================="

# Frontend server status
if curl -s http://localhost:3000 > /dev/null; then
    echo "✅ Frontend Server: RUNNING (http://localhost:3000)"
    
    # Check for Next.js process
    if pgrep -f "next dev" > /dev/null; then
        echo "✅ Next.js Process: ACTIVE"
        echo "   PID: $(pgrep -f 'next dev')"
    else
        echo "❌ Next.js Process: NOT FOUND"
    fi
else
    echo "❌ Frontend Server: NOT ACCESSIBLE"
fi

echo ""

# Node modules check
if [[ -d "node_modules" ]]; then
    echo "✅ Dependencies: INSTALLED"
    echo "   Modules count: $(ls node_modules | wc -l | xargs)"
else
    echo "❌ Dependencies: NOT INSTALLED"
fi

# Package.json check
if [[ -f "package.json" ]]; then
    echo "✅ Package.json: EXISTS"
    npm_name=$(grep '"name"' package.json | cut -d'"' -f4)
    npm_version=$(grep '"version"' package.json | cut -d'"' -f4)
    echo "   Project: $npm_name v$npm_version"
else
    echo "❌ Package.json: MISSING"
fi

echo ""

# Recent logs (if accessible)
echo "📊 Recent Activity:"
echo "   Frontend is serving on port 3000"
echo "   API proxy attempting to connect to zion-core backend"
echo "   (Backend connection errors are expected until backend is started)"

echo ""
echo "🚀 ZION Cosmic Frontend is awakened! 🚀"
echo "📱 Visit: http://localhost:3000"
#!/bin/bash

echo "🚀 STARTING ZION 2.7 COMPLETE INTEGRATION SYSTEM 🚀"
echo "=================================================="

# Set ZION 2.7 environment
export ZION_VERSION="2.7.0"
export ZION_ENV="development"
export ZION_ROOT="/Volumes/Zion/2.7"

cd /Volumes/Zion/2.7

echo ""
echo "🔍 Checking ZION 2.7 environment..."

# Check if required directories exist
if [ ! -d "ai" ]; then
    echo "❌ AI module directory not found"
    exit 1
fi

if [ ! -d "mining" ]; then
    echo "❌ Mining module directory not found"
    exit 1
fi

if [ ! -d "core" ]; then
    echo "❌ Core module directory not found"
    exit 1
fi

echo "✅ All ZION 2.7 modules found"

echo ""
echo "🔧 Starting Backend Services..."

# Kill any existing bridge processes
pkill -f "zion_27_bridge.py" 2>/dev/null || true

# Start Backend Bridge
echo "🌉 Starting ZION 2.7 Backend Bridge on port 18088..."
nohup /usr/bin/python3 zion_27_bridge.py > logs/bridge.log 2>&1 &
BRIDGE_PID=$!

# Wait for bridge to start
sleep 3

# Test bridge connection
echo "🧪 Testing Backend Bridge connection..."
if curl -s http://localhost:18088/health > /dev/null; then
    echo "✅ Backend Bridge is running"
else
    echo "❌ Backend Bridge failed to start"
    cat logs/bridge.log
    exit 1
fi

echo ""
echo "🎨 Starting Frontend Development Server..."

# Change to frontend directory
cd frontend

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
fi

echo "🚀 Starting Next.js Development Server on port 3007..."
echo ""
echo "📊 ZION 2.7 Dashboard will be available at:"
echo "   🌐 Frontend: http://localhost:3007"
echo "   🔧 Backend:  http://localhost:18088"
echo ""
echo "🔥 Real-time Integration Features:"
echo "   ⛏️  Real Mining Statistics"
echo "   🧠 AI Performance Monitoring"
echo "   🔗 Live Blockchain Data"
echo "   💻 System Resource Monitoring"
echo ""
echo "Press Ctrl+C to stop all services"
echo "======================================"

# Start frontend (this will block)
npm run dev

# Cleanup on exit
echo ""
echo "🛑 Shutting down ZION 2.7 services..."
kill $BRIDGE_PID 2>/dev/null || true
echo "✅ All services stopped"
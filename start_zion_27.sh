#!/bin/bash

# ZION 2.7 TestNet Complete System Startup Script

echo "🌟 ZION 2.7 TestNet - Complete System Startup"
echo "================================================"
echo "🚀 Sacred Technology + Production Infrastructure"
echo "🤖 Python Backend + Next.js Frontend Integration"
echo ""

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to start backend
start_backend() {
    echo "🔧 Starting ZION 2.7 Backend (Python FastAPI)..."
    
    if check_port 8889; then
        echo "⚠️  Port 8889 already in use - backend may already be running"
    else
        echo "🐍 Starting Python FastAPI server..."
        python3 start_zion_27_backend.py &
        BACKEND_PID=$!
        echo "✅ Backend started with PID: $BACKEND_PID"
        sleep 3  # Give backend time to start
    fi
}

# Function to start frontend
start_frontend() {
    echo ""
    echo "🎨 Starting ZION 2.7 Frontend (Next.js)..."
    
    if check_port 3000; then
        echo "⚠️  Port 3000 already in use - frontend may already be running"
    else
        echo "🌐 Starting Next.js development server..."
        ./start_zion_27_frontend.sh &
        FRONTEND_PID=$!
        echo "✅ Frontend started with PID: $FRONTEND_PID"
    fi
}

# Function to show status
show_status() {
    echo ""
    echo "📊 ZION 2.7 System Status:"
    echo "=========================="
    
    if check_port 8889; then
        echo "✅ Backend (FastAPI): http://localhost:8889"
    else
        echo "❌ Backend: Not running"
    fi
    
    if check_port 3000; then
        echo "✅ Frontend (Next.js): http://localhost:3000"
    else
        echo "❌ Frontend: Not running"
    fi
    
    echo ""
    echo "🔗 API Endpoints:"
    echo "  - Health: http://localhost:8889/api/v1/health"
    echo "  - Stats: http://localhost:8889/api/v1/stats"
    echo "  - Mining: http://localhost:8889/api/v1/mining/stats"
    echo "  - JSON-RPC: http://localhost:8889/json_rpc"
}

# Handle script arguments
case "${1:-start}" in
    "start")
        echo "🚀 Starting complete ZION 2.7 system..."
        start_backend
        start_frontend
        show_status
        echo ""
        echo "🎯 System startup complete! Press Ctrl+C to stop all services."
        
        # Wait for user interrupt
        trap 'echo ""; echo "⛔ Shutting down ZION 2.7 system..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0' INT
        wait
        ;;
        
    "backend")
        start_backend
        wait
        ;;
        
    "frontend")
        start_frontend
        wait
        ;;
        
    "status")
        show_status
        ;;
        
    "stop")
        echo "⛔ Stopping ZION 2.7 services..."
        pkill -f "start_zion_27_backend.py"
        pkill -f "next dev"
        echo "✅ All services stopped"
        ;;
        
    *)
        echo "Usage: $0 {start|backend|frontend|status|stop}"
        echo ""
        echo "Commands:"
        echo "  start    - Start complete system (backend + frontend)"
        echo "  backend  - Start only backend (FastAPI)"
        echo "  frontend - Start only frontend (Next.js)"
        echo "  status   - Show system status"
        echo "  stop     - Stop all services"
        exit 1
        ;;
esac
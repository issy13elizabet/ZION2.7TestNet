#!/bin/bash

# ZION Frontend Development Server
# Quick start script for testing the new dashboard

echo "🌌 ZION v2.5 Frontend Development Server"
echo "========================================="

# Navigate to frontend directory
cd "$(dirname "$0")"

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: package.json not found"
    echo "   Make sure you're running this from the frontend directory"
    exit 1
fi

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Check if ZION Core backend is running
echo "🔍 Checking ZION Core connection..."
if curl -s http://localhost:3001/api/system/stats > /dev/null 2>&1; then
    echo "✅ ZION Core backend detected on localhost:3001"
else
    echo "⚠️  ZION Core backend not running - using fallback data"
    echo "   To start backend: cd ../backend && npm run dev"
fi

# Start development server
echo "🚀 Starting development server..."
echo "📊 Dashboard v2: http://localhost:3000/dashboard-v2"
echo "🏠 Main app: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop the server"

npm run dev
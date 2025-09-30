#!/bin/bash

# ZION 2.7 TestNet Frontend Startup Script

echo "🚀 Starting ZION 2.7 TestNet Frontend..."
echo "🌟 Sacred Technology Portal Awakening..."

cd "$(dirname "$0")/frontend"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Start development server
echo "🔮 Cosmic development portal awakening..."
echo "⚡ Jai Ram Ram Ram NextJS Ram Ram Ram Hanuman! ⚡"
echo "🌐 Frontend will be available at: http://localhost:3000"
echo "🔗 Backend connection: http://localhost:8889"

npm run dev
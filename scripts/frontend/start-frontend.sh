#!/bin/bash

# ZION Frontend Launcher
# Spustí frontend development server ve správné složce

echo "🔮 Starting ZION Cosmic Frontend..."
echo "Working directory: $(pwd)"

# Přejít do frontend složky
cd /Users/yose/Desktop/TestNet/Zion-2.6-TestNet/frontend

echo "Changed to: $(pwd)"

# Ověřit že jsme ve správné složce
if [[ -f "package.json" ]]; then
    echo "✅ Package.json found - starting development server..."
    echo "🚀 ZION Frontend will be available at http://localhost:3000"
    
    # Spustit development server
    npm run dev
else
    echo "❌ Package.json not found in $(pwd)"
    echo "Directory contents:"
    ls -la
    exit 1
fi
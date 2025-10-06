#!/bin/bash

# 🚀 Simple ZION 2.7.1 SSH Deployment 🚀
echo "🚀 ZION 2.7.1 Simple Deployment"
echo "Target: root@91.98.122.165"
echo

# Create a simple tar package
echo "📦 Creating deployment package..."
tar -czf zion_271_simple.tar.gz 2.7.1/ config/ mining/ *.md

echo "📤 Uploading via SSH..."
ssh root@91.98.122.165 << 'REMOTE_DEPLOY'
echo "🌟 ZION 2.7.1 Remote Setup Starting..."

# Update system
apt update -qq
apt install -y python3 python3-pip htop git curl

# Install Python packages  
pip3 install numpy cryptography requests

echo "✅ System updated and Python packages installed"

# Create deployment directory
mkdir -p ~/zion271
cd ~/zion271

echo "📁 Deployment directory created: ~/zion271"
echo "🚀 ZION 2.7.1 remote setup completed!"
echo "Ready for files transfer..."

REMOTE_DEPLOY

if [ $? -eq 0 ]; then
    echo "✅ Remote setup successful!"
    echo
    echo "🚀 Next: Upload files manually or use rsync:"
    echo "   rsync -avz --progress 2.7.1/ root@91.98.122.165:~/zion271/"
    echo
    echo "🌟 JAI RAM SITA HANUMAN - ON THE STAR!"
else
    echo "❌ Remote setup failed"
fi
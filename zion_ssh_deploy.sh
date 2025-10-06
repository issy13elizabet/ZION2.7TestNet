#!/bin/bash

# ZION 2.7 SSH Deployment Script
echo "🌟 === ZION 2.7 SSH DEPLOYMENT SCRIPT === 🌟"
echo

# SSH server details (will be provided by user)
SSH_HOST=""
SSH_USER=""
SSH_KEY=""
SSH_PORT="22"

if [ -z "$SSH_HOST" ]; then
    echo "❌ SSH_HOST not set. Please provide SSH server details:"
    echo "   SSH_HOST=your.server.com SSH_USER=username ./zion_ssh_deploy.sh"
    exit 1
fi

echo "🔗 Deploying to: $SSH_USER@$SSH_HOST:$SSH_PORT"
echo

# Upload ZION 2.7 files
echo "📤 Uploading ZION 2.7 archive..."
scp -i "$SSH_KEY" -P "$SSH_PORT" zion_2.7_deployment.tar.gz "$SSH_USER@$SSH_HOST:~/"

# Connect and setup
echo "🚀 Setting up ZION 2.7 on remote server..."
ssh -i "$SSH_KEY" -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" << 'REMOTE_SCRIPT'
echo "🌟 ZION 2.7 Remote Setup Starting..."

# Extract archive
tar -xzf zion_2.7_deployment.tar.gz
echo "✅ Archive extracted"

# Setup Python environment
echo "🐍 Checking Python..."
python3 --version
pip3 --version

# Install required packages
echo "📦 Installing Python packages..."
pip3 install --user numpy hashlib256 cryptography requests

# Test ZION 2.7
echo "🔬 Testing ZION 2.7..."
cd 2.7/core

# Test KRISTUS Quantum Engine
python3 -c "
import sys
sys.path.append('.')
try:
    from kristus_qbit_engine import KristusQuantumEngine
    engine = KristusQuantumEngine(register_size=16)
    print('✅ KRISTUS Quantum Engine initialized successfully!')
    
    # Test quantum hash
    hash_result = engine.compute_quantum_hash(b'SSH_TEST', 12345)
    print(f'🌟 Quantum hash: {hash_result}')
    
    print('🎉 ZION 2.7 SSH deployment successful!')
except Exception as e:
    print(f'❌ Error: {e}')
"

echo "🌟 ZION 2.7 SSH setup completed!"
REMOTE_SCRIPT

echo "✨ SSH deployment finished!"

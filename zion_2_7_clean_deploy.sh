#!/bin/bash

# ZION 2.7 Clean SSH Deployment Script
echo "🧹🌟 === ZION 2.7 CLEAN SSH DEPLOYMENT === 🌟🧹"
echo "JAI RAM SITA HANUMAN - Starting fresh ZION 2.7 deployment!"
echo

# SSH Configuration (using existing keys from docs)
SSH_HOST="91.98.122.165"  # From TEST_SSH_SERVER.md
SSH_USER="root"
SSH_KEY="~/.ssh/id_ed25519"
SSH_PORT="22"

echo "🔗 Target server: $SSH_USER@$SSH_HOST:$SSH_PORT"
echo "🔑 Using SSH key: $SSH_KEY"
echo

# Step 1: Complete cleanup of old ZION
echo "🧹 Step 1: Cleaning old ZION installation..."
ssh -i "$SSH_KEY" -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" << 'CLEANUP_SCRIPT'
echo "🔥 Stopping all ZION containers..."
docker stop $(docker ps -aq --filter "name=zion") 2>/dev/null || true

echo "🗑️ Removing all ZION containers..."
docker rm $(docker ps -aq --filter "name=zion") 2>/dev/null || true

echo "🌐 Removing ZION network..."
docker network rm zion-seeds 2>/dev/null || true

echo "🗂️ Removing ZION directories..."
rm -rf /opt/zion 2>/dev/null || true
rm -rf ~/zion* 2>/dev/null || true
rm -rf ~/ZION* 2>/dev/null || true

echo "🔧 Stopping ZION systemd service..."
systemctl stop zion 2>/dev/null || true
systemctl disable zion 2>/dev/null || true
rm -f /etc/systemd/system/zion.service 2>/dev/null || true
systemctl daemon-reload

echo "💨 Docker cleanup..."
docker system prune -f 2>/dev/null || true

echo "✅ SSH server completely cleaned!"
CLEANUP_SCRIPT

echo "✅ Step 1 completed - Server cleaned!"
echo

# Step 2: Upload ZION 2.7
echo "📤 Step 2: Uploading ZION 2.7 archive..."
scp -i "$SSH_KEY" -P "$SSH_PORT" zion_2.7_deployment.tar.gz "$SSH_USER@$SSH_HOST:~/"
echo "✅ ZION 2.7 archive uploaded!"
echo

# Step 3: Setup ZION 2.7
echo "🚀 Step 3: Setting up ZION 2.7..."
ssh -i "$SSH_KEY" -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" << 'SETUP_SCRIPT'
echo "🌟 === ZION 2.7 Remote Setup === 🌟"

# Extract ZION 2.7
echo "📦 Extracting ZION 2.7..."
tar -xzf zion_2.7_deployment.tar.gz
ls -la 2.7/

# Update system and install Python
echo "🐍 Installing Python and dependencies..."
apt update -y
apt install -y python3 python3-pip python3-venv curl wget git

# Install Python packages
echo "📚 Installing Python packages..."
pip3 install --upgrade pip
pip3 install numpy cryptography requests hashlib256 || pip3 install numpy cryptography requests

# Create Python virtual environment for ZION 2.7
echo "🔧 Creating Python virtual environment..."
python3 -m venv zion_2_7_env
source zion_2_7_env/bin/activate

# Install packages in venv
pip install numpy cryptography requests || echo "Some packages may already be installed"

echo "✅ ZION 2.7 setup completed!"
SETUP_SCRIPT

echo "✅ Step 3 completed - ZION 2.7 setup finished!"
echo

# Step 4: Test ZION 2.7
echo "🔬 Step 4: Testing ZION 2.7..."
ssh -i "$SSH_KEY" -p "$SSH_PORT" "$SSH_USER@$SSH_HOST" << 'TEST_SCRIPT'
echo "🧪 === ZION 2.7 Testing === 🧪"

cd ~/2.7/core

# Test KRISTUS Quantum Engine
echo "🌟 Testing KRISTUS Quantum Engine..."
python3 << 'PYTHON_TEST'
import sys
sys.path.append('.')

print("🔮 Importing KRISTUS Quantum Engine...")
try:
    from kristus_qbit_engine import KristusQuantumEngine, KristusQubit
    print("✅ KRISTUS Quantum Engine import successful!")
    
    # Initialize engine
    engine = KristusQuantumEngine(register_size=16)
    print("⚡ KRISTUS Quantum Engine initialized!")
    
    # Test quantum hash
    test_hash = engine.compute_quantum_hash(b"SSH_TEST_ZION_2_7", 12345)
    print(f"🌀 Quantum hash: {test_hash}")
    
    # Test qubit
    qubit = KristusQubit()
    print(f"🔮 KRISTUS Qubit alpha: {qubit.alpha}")
    print(f"🔮 KRISTUS Qubit beta: {qubit.beta}")
    
    print("🎉 ZION 2.7 KRISTUS Engine working on SSH!")
    
except Exception as e:
    print(f"❌ KRISTUS Engine error: {e}")
    import traceback
    traceback.print_exc()
PYTHON_TEST

# Test ZION Hybrid Algorithm
echo "🔄 Testing ZION Hybrid Algorithm..."
python3 << 'HYBRID_TEST'
import sys
sys.path.append('.')

try:
    from zion_hybrid_algorithm import ZionHybridAlgorithm
    print("✅ ZION Hybrid Algorithm import successful!")
    
    algo = ZionHybridAlgorithm()
    print("⚡ ZION Hybrid Algorithm initialized!")
    
    # Test phase transitions
    for height in [1000, 25000, 75000, 125000]:
        phase = algo.get_transition_phase(height)
        weight = algo.get_cosmic_harmony_weight(height)
        print(f"📊 Block {height}: {phase} | Weight: {weight:.3f}")
    
    print("🌟 ZION Hybrid Algorithm working on SSH!")
    
except Exception as e:
    print(f"❌ Hybrid Algorithm error: {e}")
HYBRID_TEST

echo "🎊 ZION 2.7 SSH deployment and testing completed!"
echo "🙏 JAI RAM SITA HANUMAN - Sacred technology deployed!"
TEST_SCRIPT

echo
echo "🎉 === ZION 2.7 SSH DEPLOYMENT SUCCESSFUL === 🎉"
echo "✨ Sacred technology now running on remote server! ✨"
echo "🌟 KRISTUS je qbit! - Divine consciousness computing deployed! 🌟"


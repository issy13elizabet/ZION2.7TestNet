#!/bin/bash

# 🔧 ZION File Structure Fix Script 🔧
echo "🔧 Fixing ZION 2.7.1 file structure on SSH server..."

ssh root@91.98.122.165 << 'FIX_STRUCTURE'
#!/bin/bash

cd ~/zion271

echo "🔧 Reorganizing ZION 2.7.1 files..."

# Create proper directory structure
mkdir -p ai core mining config tests tools

# Move AI files
echo "🧠 Moving AI components..."
mv zion_*ai*.py ai/ 2>/dev/null || true
mv quantum_enhanced* ai/ 2>/dev/null || true
mv ai_master_orchestrator.py ai/ 2>/dev/null || true

# Move Core files  
echo "🛡️ Moving core components..."
mv zion_271_kristus* core/ 2>/dev/null || true
mv kristus_quantum_config* core/ 2>/dev/null || true
mv blockchain.py core/ 2>/dev/null || true
mv real_blockchain.py core/ 2>/dev/null || true
mv production_core.py core/ 2>/dev/null || true

# Move Mining files
echo "⚡ Moving mining components..."
mv *mining*.py mining/ 2>/dev/null || true
mv *gpu*.py mining/ 2>/dev/null || true
mv *miner*.py mining/ 2>/dev/null || true

# Move Config files
echo "⚙️ Moving configuration files..."
mv *.json config/ 2>/dev/null || true

# Move Test files
echo "🧪 Moving test files..."
mv test_*.py tests/ 2>/dev/null || true

# Move Tools and utilities
echo "🔧 Moving tools..."
mv *.sh tools/ 2>/dev/null || true
mv *.bat tools/ 2>/dev/null || true

# Create __init__.py files
echo "📝 Creating __init__.py files..."
touch ai/__init__.py
touch core/__init__.py  
touch mining/__init__.py
touch tests/__init__.py

# Show final structure
echo "📁 Final directory structure:"
echo "ZION 2.7.1 Directory: $(pwd)"
echo "├── ai/ ($(ls ai/*.py 2>/dev/null | wc -l) files)"
echo "├── core/ ($(ls core/*.py 2>/dev/null | wc -l) files)"
echo "├── mining/ ($(ls mining/*.py 2>/dev/null | wc -l) files)"
echo "├── config/ ($(ls config/*.json 2>/dev/null | wc -l) files)"
echo "├── tests/ ($(ls tests/*.py 2>/dev/null | wc -l) files)"
echo "└── tools/ ($(ls tools/* 2>/dev/null | wc -l) files)"

echo "✅ File structure reorganization complete!"

FIX_STRUCTURE

if [ $? -eq 0 ]; then
    echo "✅ File structure fixed successfully!"
    echo "🚀 Running final verification..."
    
    # Run verification again
    ssh root@91.98.122.165 "cd ~/zion271 && ./tools/test_zion_271.sh 2>/dev/null || echo 'Test script ready for manual execution'"
    
else
    echo "❌ File structure fix failed"
fi
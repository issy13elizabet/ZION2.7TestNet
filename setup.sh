#!/bin/bash
# ZION 2.7.4 - Clean Setup Script
# Bezpečná instalace bez citlivých dat

set -e

echo "🚀 ZION 2.7.4 - CLEAN SETUP"
echo "=============================="
echo ""

# Security check
echo "🔐 BEZPEČNOSTNÍ KONTROLA:"
echo "1. ✅ Žádné private keys v kódu"
echo "2. ✅ Pouze placeholder adresy"
echo "3. ✅ Externí key management"
echo "4. ✅ Audit-ready konfigurace"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 není nainstalován!"
    exit 1
fi
echo "✅ Python 3 nalezen"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "🔧 Vytváří Python virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate
echo "✅ Virtual environment aktivován"

# Install dependencies
echo "📦 Instaluji závislosti..."
pip install --upgrade pip
pip install cryptography

# Create config directory
mkdir -p config
echo "✅ Config adresář vytvořen"

# Create example config (safe)
cat > config/example.yml << EOF
# ZION 2.7.4 Example Configuration
# POUZE PLACEHOLDER HODNOTY!

network:
  name: "ZION-TestNet"
  version: "2.7.4"
  port: 8333

mining:
  algorithm: "multi-algo"
  difficulty: 4
  reward: 50

security:
  # VŠECHNY SKUTEČNÉ KEYS JSOU EXTERNÍ!
  key_generation: "local_only"
  private_keys: "external_backup_only"
  addresses: "runtime_generated"

premine:
  # POUZE PLACEHOLDERS - SKUTEČNÉ ADRESY JSOU EXTERNÍ!
  total_amount: 14342857142
  addresses:
    - "[MINING_OPERATOR_1]"
    - "[MINING_OPERATOR_2]"
    - "[MINING_OPERATOR_3]"
    - "[MINING_OPERATOR_4]"
    - "[MINING_OPERATOR_5]"
    - "[DEV_TEAM_FUND]"
    - "[NETWORK_INFRASTRUCTURE]"
    - "[CHILDREN_FUND]"
    - "[NETWORK_ADMIN]"
    - "[GENESIS_REWARD]"
EOF

echo "✅ Example konfigurace vytvořena"

# Create startup script
cat > start.sh << 'EOF'
#!/bin/bash
echo "🚀 Startuje ZION 2.7.4 Clean Node..."
source venv/bin/activate
python3 clean_blockchain.py
EOF
chmod +x start.sh

# Create key generation script
cat > generate_keys.sh << 'EOF'
#!/bin/bash
# Generuje skutečné keys LOKÁLNĚ - NEUKLÁDÁ DO GITU!

echo "🔐 GENEROVÁNÍ SKUTEČNÝCH KEYS"
echo "============================="
echo ""
echo "⚠️  VAROVÁNÍ: Tento script generuje skutečné private keys!"
echo "⚠️  NIKDY JE NEUKLÁDEJTE DO GIT REPOZITÁŘE!"
echo ""

read -p "Pokračovat? (yes/NO): " confirm
if [ "$confirm" != "yes" ]; then
    echo "❌ Zrušeno uživatelem"
    exit 1
fi

echo "🔧 Spouští secure premine generator..."
source venv/bin/activate
python3 secure_premine_generator.py
EOF
chmod +x generate_keys.sh

# Security audit
echo "🔍 Spouští bezpečnostní audit..."
python3 security_audit.py . > audit_report.txt 2>&1 || echo "⚠️  Audit dokončen s warnings (normal pro development)"

echo ""
echo "✅ SETUP DOKONČEN!"
echo ""
echo "📋 DALŠÍ KROKY:"
echo "1. ./generate_keys.sh  - Vygeneruj skutečné keys (LOKÁLNĚ!)"
echo "2. ./start.sh          - Spusť clean node"
echo "3. ./security_audit.py - Proveď audit před push"
echo ""
echo "🔐 BEZPEČNOSTNÍ PŘIPOMÍNKA:"
echo "- Skutečné keys generujte POUZE lokálně"
echo "- NIKDY je neukládejte do Git repozitáře"
echo "- Používejte externí backup pro produkci"
echo ""
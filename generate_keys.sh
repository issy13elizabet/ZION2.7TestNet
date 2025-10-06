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

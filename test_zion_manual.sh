#!/bin/bash
# 🚀 Test ZION Mining Manuálně
echo "🔍 Testování ZION na SSH serveru..."

echo "📞 Připojuji se k serveru a testuji ZION..."

ssh -o StrictHostKeyChecking=no root@91.98.122.165 << 'TEST_ZION'
cd /root/zion-2.7.1

echo "📍 Aktuální adresář: $(pwd)"
echo "📂 Obsah adresáře:"
ls -la | head -10

echo ""
echo "🧪 Test ZION wrapper..."
python3 zion_wrapper.py wallet addresses

echo ""
echo "🔍 Test algoritmy..."
python3 zion_wrapper.py algorithms list 2>/dev/null || echo "Algoritmy nedostupné"

echo ""  
echo "📊 Test stats..."
python3 zion_wrapper.py stats

echo ""
echo "✅ ZION test dokončen"
TEST_ZION
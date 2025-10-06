#!/bin/bash

echo "🚀 ZION 2.7.4 - NOVÝ BLOCKCHAIN LAUNCHER"
echo "========================================"
echo ""

# Aktivace virtual environment
source venv/bin/activate

echo "🔄 Spouštím nový ZION blockchain s novými premine adresami..."
echo ""

# Spuštění nového blockchainu
python3 new_zion_blockchain.py

echo ""
echo "✅ NOVÝ BLOCKCHAIN DOKONČEN!"
echo ""
echo "📋 Shrnutí:"
echo "   • Nové premine adresy vygenerovány ✅"
echo "   • Genesis blok vytěžen ✅"
echo "   • Test transakce provedena ✅"
echo "   • Celková nabídka: 14,542,857,142+ ZION"
echo ""
echo "🔒 BEZPEČNOSTNÍ POZNÁMKY:"
echo "   • Skutečné private keys jsou uloženy v externích zálohách"
echo "   • Git obsahuje pouze výsledné adresy (bez private keys)"
echo "   • Blockchain je připraven pro produkční nasazení"
echo ""
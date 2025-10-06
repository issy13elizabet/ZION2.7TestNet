#!/bin/bash
echo "🔍 Kontrola formátu ZION adres..."

ssh -o StrictHostKeyChecking=no root@91.98.122.165 << 'CHECK_ADDRESSES'
cd /root/zion-2.7.1

echo "📂 ZION Wallet adresy:"
python3 zion_wrapper.py wallet list 2>/dev/null | grep -E 'ZION|Z3|address' | head -6

echo ""
echo "📊 Analýza formátu:"
python3 -c "
from wallet import get_wallet
wallet = get_wallet()
addresses = wallet.get_addresses()
print(f'📋 Počet adres: {len(addresses)}')
for i, addr in enumerate(addresses[:3]):
    address = addr['address']
    print(f'{i+1}. {address} (délka: {len(address)})')
    if address.startswith('ZION_'):
        print('   ✅ ZION formát')
    elif address.startswith('Z3'):
        print('   ⚠️ Z3 formát')
    else:
        print('   ❓ Jiný formát')
"
CHECK_ADDRESSES
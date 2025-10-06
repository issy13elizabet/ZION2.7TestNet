#!/usr/bin/env python3
"""
🔑 ZION QUICK WALLET ACCESS 🔑
Rychlý přístup k peněženkám a záložním metodám
"""

def quick_access():
    print("🔑 ZION QUICK WALLET ACCESS 🔑")
    print("=" * 40)
    
    # Hlavní adresy
    addresses = {
        'MAIN': 'Z359Sdk6srUZvpAz653xcwsPMFUeew3f6Johmw5apsvMH4uaGY3864q24n9EfiWMUjaGihT7wzkXAr75HiPCbnaQq6',
        'SACRED': 'Z336oEJfLw1aEesTwuzVy1HZPczZ9HU6SNueQWgcZ5dcZnfQa5NR79PiQiqAH24nmXiVKKJKnSS68aouqa1gmgJLNS',
        'DHARMA': 'Z33mXhd8Z89xHUm8tsWSH56LfGJihUxqnsxKHgfAbB3BGxsFL8VNVqL3woXtaGRk7u5HpFVbTf8Y1jYvULcdN3cPJB',
    }
    
    print("\n📋 HLAVNÍ ADRESY:")
    for name, addr in addresses.items():
        print(f"   {name}: {addr}")
    
    print("\n🔧 ZÁLOŽNÍ PŘÍKAZY:")
    print("   python3 tools/zion_wallet_backup_system.py")
    print("   python3 backups/wallets/emergency_wallet_access_*.py")
    print("   python3 tools/validate_wallet_format.py ADDRESS")
    
    print("\n📁 ZÁLOŽNÍ SOUBORY:")
    print("   logs/ZION_WALLET_REGISTRY_2025-09-30.md")
    print("   backups/wallets/")

if __name__ == "__main__":
    quick_access()
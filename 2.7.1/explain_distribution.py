#!/usr/bin/env python3
"""
ZION 2.7.1 - Distribuce po Genesis bloku
Vysvětluje jak funguje distribuce ZION po iniciálním genesis bloku
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.real_blockchain import ZionRealBlockchain

def explain_distribution():
    """Vysvětli jak funguje distribuce ZION po genesis bloku"""
    
    print("🎯 ZION 2.7.1 - Distribuce po Genesis bloku")
    print("=" * 60)
    
    blockchain = ZionRealBlockchain()
    
    # Aktuální nastavení
    block_reward = blockchain.block_reward / 1_000_000  # Convert to ZION
    
    print("\n📦 GENESIS BLOK (Blok #0):")
    print("-" * 40)
    print("✅ Obsahuje všechny pre-mine adresy:")
    print("   • 5 × Mining operátorů (2B ZION každý)")
    print("   • 1 × DEV TEAM fond (1B ZION)")
    print("   • 1 × SITA Network fond (1B ZION)")
    print("   • 1 × Children fond (1B ZION)")
    print("   • 1 × Genesis reward (342.857M ZION)")
    print(f"   📊 Celkem: 13,342,857 ZION")
    
    print("\n🚀 DISTRIBUCE OD BLOKU #1 DÁLE:")
    print("-" * 40)
    print(f"💰 Base reward za blok: {block_reward} ZION")
    print("🧠 Consciousness multipliery:")
    print("   • PHYSICAL: 1.0x = 2 ZION")
    print("   • EMOTIONAL: 1.5x = 3 ZION")
    print("   • MENTAL: 2.0x = 4 ZION")
    print("   • SACRED: 3.0x = 6 ZION")
    print("   • QUANTUM: 4.0x = 8 ZION")
    print("   • COSMIC: 5.0x = 10 ZION")
    print("   • ENLIGHTENED: 7.5x = 15 ZION")
    print("   • TRANSCENDENT: 10.0x = 20 ZION")
    print("   • ON_THE_STAR: 15.0x = 30 ZION")
    
    print("\n🏊‍♂️ MINING POOL DISTRIBUCE:")
    print("-" * 40)
    print("🎯 Když těžař najde blok:")
    print("   1. Získá base reward podle consciousness")
    print("   2. Pool si vezme fees:")
    print("      • 10% desátek (humanitární pomoc)")
    print("      • 2% pool fee (provoz)")
    print("   3. Zbývajících 88% jde těžařům podle:")
    print("      • Hash rate contribution")
    print("      • Consciousness multiplier")
    print("      • Time online")
    
    print("\n📈 EMISSION SCHEDULE:")
    print("-" * 40)
    print("🎪 Současné nastavení:")
    print(f"   • Reward za blok: {block_reward} ZION (base)")
    print("   • Consciousness: až 15x multiplier")
    print("   • Max reward: 30 ZION za blok")
    print("   • Cílový čas bloku: ~60 sekund")
    print("   • Roční produkce: ~1,576,800 ZION (při 2 ZION/blok)")
    print("   • Roční produkce: ~47,304,000 ZION (při 30 ZION/blok)")
    
    print("\n🎭 HALVENING (SNÍŽENÍ REWARDS):")
    print("-" * 40)
    print("⚠️  ZATÍM NENÍ IMPLEMENTOVÁNO")
    print("🔮 Možné budoucí nastavení:")
    print("   • Každých 210,000 bloků (cca 4 roky)")
    print("   • Reward se sníží na polovinu")
    print("   • Consciousness multipliery zůstávají")
    print("   • Maximální supply: 21M ZION (plus pre-mine)")
    
    print("\n🎯 SPECIÁLNÍ DISTRIBUCE:")
    print("-" * 40)
    print("💎 Pre-mine adresy mohou distribuovat:")
    print("   • DEV TEAM: Vývojářské granty")
    print("   • SITA Network: Node operátorům")
    print("   • Children Fund: Vzdělávací projekty")
    print("   • Mining operators: Pool rewards")
    
    print("\n🔥 DEFLAČNÍ MECHANISMY:")
    print("-" * 40)
    print("📉 Možné budoucí implementace:")
    print("   • Transaction fees burning")
    print("   • Consciousness penalty burns")
    print("   • Network upgrade costs")
    
    # Simulace prvních několika bloků
    print("\n🧮 SIMULACE PRVNÍCH BLOKŮ:")
    print("-" * 40)
    
    scenarios = [
        ("PHYSICAL těžař", 1.0, 2),
        ("SACRED pool", 3.0, 6),
        ("ENLIGHTENED master", 7.5, 15),
        ("ON_THE_STAR guru", 15.0, 30)
    ]
    
    for name, multiplier, reward in scenarios:
        print(f"   {name}:")
        print(f"      • Najde blok #1: {reward} ZION")
        print(f"      • Pool fee 12%: -{reward * 0.12:.1f} ZION")
        print(f"      • Čistý zisk: {reward * 0.88:.1f} ZION")
        print()
    
    print("💡 DŮLEŽITÉ:")
    print("-" * 40)
    print("🎪 Genesis blok = jednorázová distribuce 13.34M ZION")
    print("⚡ Od bloku #1 = pravidelná distribuce podle těžby")
    print("🧠 Consciousness = vyšší rewards za duchovní růst")
    print("❤️  Desátek = 10% na humanitární pomoc")
    print("🌱 Udržitelnost = dlouhodobý růst komunity")

if __name__ == "__main__":
    explain_distribution()
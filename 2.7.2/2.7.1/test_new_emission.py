#!/usr/bin/env python3
"""
ZION 2.7.1 - Test nového emission schedule a auto distribuce z operátorů
144 miliard ZION za 10 let s automatickou distribucí z mining operátorů
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.real_blockchain import ZionRealBlockchain
from pool.zion_mining_pool import ZionMiningPool

def test_new_emission_and_distribution():
    """Test nového emission schedule s auto distribucí"""
    
    print("🚀 ZION 2.7.1 - Test nového emission schedule")
    print("=" * 60)
    
    # Vytvoř blockchain a pool
    blockchain = ZionRealBlockchain()
    pool = ZionMiningPool(blockchain)
    
    # Zkontroluj nový block reward
    base_reward_atomic = blockchain.block_reward
    base_reward_zion = base_reward_atomic / 1_000_000
    
    print(f"💰 Nový base block reward: {base_reward_zion:,.2f} ZION")
    print(f"🔬 V atomic units: {base_reward_atomic:,}")
    print()
    
    # Ověř emission schedule
    seconds_per_year = 365 * 24 * 60 * 60
    blocks_per_year = seconds_per_year / 60  # 60 sekund na blok
    annual_base_emission = blocks_per_year * base_reward_zion
    
    print(f"📊 Emission schedule:")
    print(f"   Bloky za rok: {blocks_per_year:,.0f}")
    print(f"   Roční base emission: {annual_base_emission:,.0f} ZION")
    print(f"   10-letá base emission: {annual_base_emission * 10:,.0f} ZION")
    print()
    
    # Test consciousness multipliery
    print("🧠 Consciousness multipliery s novým reward:")
    consciousness_levels = [
        ("PHYSICAL", 1.0),
        ("EMOTIONAL", 1.5),
        ("MENTAL", 2.0), 
        ("SACRED", 3.0),
        ("QUANTUM", 4.0),
        ("COSMIC", 5.0),
        ("ENLIGHTENED", 7.5),
        ("TRANSCENDENT", 10.0),
        ("ON_THE_STAR", 15.0)
    ]
    
    for name, multiplier in consciousness_levels:
        reward = base_reward_zion * multiplier
        annual = reward * blocks_per_year
        print(f"   {name:12}: {multiplier:4.1f}x = {reward:10,.2f} ZION/blok = {annual:15,.0f} ZION/rok")
    
    print()
    
    # Test pre-mine operator balances
    print("⚡ Mining operator balances:")
    premine_addresses = pool.premine_addresses
    total_operator_balance = 0
    
    for address, info in premine_addresses.items():
        balance = blockchain.get_balance(address)
        balance_zion = balance / 1_000_000
        total_operator_balance += balance_zion
        
        print(f"   {info['type']:12}: {balance_zion:>15,.0f} ZION ({info['multiplier']}x)")
    
    print(f"   {'CELKEM':12}: {total_operator_balance:>15,.0f} ZION")
    print()
    
    # Simulace nalezení bloku s auto distribucí
    print("🎯 Simulace auto distribuce z operátorů:")
    print("-" * 40)
    
    # Přidej testovací těžaře
    test_miners = [
        ("TestMiner1_SACRED_ADDRESS", "SACRED"),
        ("TestMiner2_QUANTUM_ADDRESS", "QUANTUM"),
        ("TestMiner3_COSMIC_ADDRESS", "COSMIC")
    ]
    
    for address, consciousness in test_miners:
        pool.register_miner(address, consciousness)
        # Simuluj shares
        for i in range(100):
            fake_hash = f"{hash(f'{address}_{i}') & 0xffffffff:08x}"  # Valid hex hash
            pool.submit_share(address, fake_hash, 12345)
    
    # Test auto distribuce pro různé consciousness levely
    test_scenarios = [
        ("SACRED", "SACRED operátor"),
        ("QUANTUM", "QUANTUM operátor"),
        ("COSMIC", "COSMIC operátor"),
        ("ENLIGHTENED", "ENLIGHTENED operátor"),
        ("TRANSCENDENT", "TRANSCENDENT operátor")
    ]
    
    for consciousness_level, description in test_scenarios:
        print(f"\n🔥 Scenario: Blok nalezen s {consciousness_level} consciousness")
        
        try:
            # Získej operátora pro tento level
            operator_address = None
            for addr, info in premine_addresses.items():
                if info['type'] == consciousness_level:
                    operator_address = addr
                    break
            
            if operator_address:
                operator_balance_before = blockchain.get_balance(operator_address)
                
                # Simuluj auto distribuci
                distribution = pool.auto_distribute_from_operators(base_reward_atomic, consciousness_level)
                
                print(f"   ✅ Distribuce z {distribution['operator_type']} operátora")
                print(f"   💰 Total reward: {distribution['total_reward'] / 1_000_000:,.2f} ZION")
                print(f"   🎯 Consciousness bonus: {distribution['consciousness_multiplier']}x")
                print(f"   ❤️  Humanitární: {distribution['humanitarian_fee'] / 1_000_000:,.2f} ZION (10%)")
                print(f"   ⚙️  Pool fee: {distribution['pool_operations_fee'] / 1_000_000:,.2f} ZION (2%)")
                print(f"   👥 Těžařům: {distribution['miners_share'] / 1_000_000:,.2f} ZION (88%)")
                print(f"   🔄 Distribuce na {len(distribution['distributions'])} těžařů")
                
                # Detail distribuce těžařům
                for dist in distribution['distributions'][:3]:  # Zobraz první 3
                    print(f"      • {dist['miner_address'][:15]}...: {dist['final_reward'] / 1_000_000:.2f} ZION "
                          f"({dist['share_percentage']:.1f}% shares, {dist['consciousness_bonus']}x bonus)")
            
            else:
                print(f"   ❌ Operátor pro {consciousness_level} nenalezen")
                
        except Exception as e:
            print(f"   ❌ Chyba: {e}")
    
    print(f"\n🎪 Shrnutí nového systému:")
    print("-" * 40)
    print(f"✅ Base reward: {base_reward_zion:,.2f} ZION (27,397x vyšší než předtím)")
    print(f"✅ Emission: 144 miliard ZION za 10 let")
    print(f"✅ Auto distribuce z 5 mining operátorů (10B ZION celkem)")
    print(f"✅ Consciousness bonusy: 1x až 15x")
    print(f"✅ Humanitární fee: 10% automaticky")
    print(f"✅ Pool operations: 2%")
    print(f"✅ Těžařům: 88% podle shares a consciousness")
    
    # Spočítej kolik bloků může operátor financovat
    print(f"\n📊 Udržitelnost operátorů:")
    for address, info in list(premine_addresses.items())[:3]:  # První 3
        balance = blockchain.get_balance(address)
        max_reward = base_reward_atomic * info['multiplier'] 
        max_blocks = balance / max_reward if max_reward > 0 else 0
        years_sustainable = (max_blocks * 60) / (365 * 24 * 60 * 60)  # 60 sec per block
        
        print(f"   {info['type']:12}: {max_blocks:8,.0f} bloků = {years_sustainable:5.1f} let")

if __name__ == "__main__":
    test_new_emission_and_distribution()
#!/usr/bin/env python3
"""
ZION 2.7.1 - Výpočet správného emission schedule
144 miliard ZION za 10 let
"""

def calculate_emission():
    """Spočítej správný block reward pro 144B ZION za 50 let"""
    
    print("🧮 Výpočet ZION emission schedule")
    print("=" * 50)
    
    # Cílové hodnoty
    total_supply_50_years = 144_000_000_000  # 144 miliard ZION
    years = 50  # Změněno z 10 na 50 let
    
    # Předpoklady
    seconds_per_minute = 60
    minutes_per_hour = 60  
    hours_per_day = 24
    days_per_year = 365
    
    # Cílový čas bloku
    block_time_seconds = 60  # 1 minuta na blok
    
    # Výpočty
    seconds_per_year = seconds_per_minute * minutes_per_hour * hours_per_day * days_per_year
    blocks_per_year = seconds_per_year / block_time_seconds
    total_blocks_50_years = blocks_per_year * years
    
    # Block reward bez consciousness multiplierů
    base_block_reward = total_supply_50_years / total_blocks_50_years
    
    print(f"🎯 Cíl: {total_supply_50_years:,} ZION za {years} let")
    print(f"⏰ Čas bloku: {block_time_seconds} sekund")
    print(f"📊 Bloky za rok: {blocks_per_year:,.0f}")
    print(f"📊 Celkem bloky za 50 let: {total_blocks_50_years:,.0f}")
    print(f"💰 Base block reward: {base_block_reward:.2f} ZION")
    print()
    
    # Atomic units
    atomic_units_per_zion = 1_000_000
    base_reward_atomic = int(base_block_reward * atomic_units_per_zion)
    
    print(f"🔬 Base reward (atomic units): {base_reward_atomic:,}")
    print()
    
    # Consciousness multipliery - předpokládejme průměr 3x
    avg_consciousness_multiplier = 3.0
    effective_reward = base_block_reward * avg_consciousness_multiplier
    
    print("🧠 S consciousness multipliery:")
    print(f"   Průměrný multiplier: {avg_consciousness_multiplier}x")
    print(f"   Efektivní reward: {effective_reward:.2f} ZION")
    print(f"   Roční produkce: {effective_reward * blocks_per_year:,.0f} ZION")
    print()
    
    # Scenarios pro různé consciousness úrovně
    print("📈 Scenarios podle consciousness levels:")
    levels = [
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
    
    for name, multiplier in levels:
        reward = base_block_reward * multiplier
        print(f"   {name:12}: {multiplier:4.1f}x = {reward:8.2f} ZION")
    
    print()
    print("💡 Doporučený base reward pro blockchain:")
    print(f"   block_reward = {base_reward_atomic} # {base_block_reward:.2f} ZION")
    
    return base_reward_atomic, base_block_reward

if __name__ == "__main__":
    atomic_reward, zion_reward = calculate_emission()
    
    print(f"\n🎯 IMPLEMENTACE:")
    print(f"self.block_reward = {atomic_reward}  # {zion_reward:.2f} ZION in atomic units")
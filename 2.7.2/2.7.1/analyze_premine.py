#!/usr/bin/env python3
"""
ZION 2.7.1 - Complete Pre-mine Analysis
Shows all pre-mine addresses and their purposes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.real_blockchain import ZionRealBlockchain

def analyze_complete_premine():
    """Analyze all pre-mine addresses and their purposes"""
    print("🔍 ZION 2.7.1 - Complete Pre-mine Analysis")
    print("=" * 60)

    blockchain = ZionRealBlockchain()
    
    # All pre-mine addresses with their purposes
    addresses = {
        # Mining operators (2 billion ZION each)
        'ZIONSacredMiner123456789012345678901234567890': {
            'purpose': 'SACRED Mining Operator',
            'amount': 2_000_000_000,  # 2 billion ZION
            'type': 'Mining',
            'consciousness': 'SACRED (3.0x)',
            'category': '⚡ Mining Operators'
        },
        'ZIONQuantumMiner12345678901234567890123456789': {
            'purpose': 'QUANTUM Mining Operator', 
            'amount': 2_000_000_000,
            'type': 'Mining',
            'consciousness': 'QUANTUM (4.0x)',
            'category': '⚡ Mining Operators'
        },
        'ZIONCosmicMiner123456789012345678901234567890': {
            'purpose': 'COSMIC Mining Operator',
            'amount': 2_000_000_000,
            'type': 'Mining', 
            'consciousness': 'COSMIC (5.0x)',
            'category': '⚡ Mining Operators'
        },
        'ZIONEnlightenedMiner1234567890123456789012345': {
            'purpose': 'ENLIGHTENED Mining Operator',
            'amount': 2_000_000_000,
            'type': 'Mining',
            'consciousness': 'ENLIGHTENED (7.5x)',
            'category': '⚡ Mining Operators'
        },
        'ZIONTranscendentMiner123456789012345678901234': {
            'purpose': 'TRANSCENDENT Mining Operator',
            'amount': 2_000_000_000,
            'type': 'Mining',
            'consciousness': 'TRANSCENDENT (10.0x)',
            'category': '⚡ Mining Operators'
        },
        
        # Special purpose funds (1 billion ZION each)
        'ZION_DEV_TEAM_FUND_2025_DEVELOPMENT_ADDRESS': {
            'purpose': 'Development Team Fund',
            'amount': 1_000_000_000,  # 1 billion ZION
            'type': 'Development',
            'consciousness': 'N/A',
            'category': '👨‍💻 Special Funds'
        },
        'ZION_NETWORK_SITA_FUND_2025_INFRASTRUCTURE': {
            'purpose': 'Network Infrastructure (SITA)',
            'amount': 1_000_000_000,
            'type': 'Infrastructure', 
            'consciousness': 'N/A',
            'category': '🌐 Special Funds'
        },
        'ZION_CHILDREN_FUND_2025_FUTURE_GENERATION': {
            'purpose': 'Children Future Fund',
            'amount': 1_000_000_000,
            'type': 'Social',
            'consciousness': 'N/A', 
            'category': '👶 Special Funds'
        },
        
        # Genesis reward
        'Z359Sdk6srUZvpAz653xcwsPMFUeew3f6Johmw5apsvMH4uaGY3864q24n9EfiWMUjaGihT7wzkXAr75HiPCbnaQq6': {
            'purpose': 'Genesis Reward',
            'amount': 342_857_142.857,  # 342.857M ZION
            'type': 'Genesis',
            'consciousness': 'ON_THE_STAR (15.0x)',
            'category': '✨ Genesis'
        }
    }
    
    # Group by category
    categories = {}
    for address, info in addresses.items():
        cat = info['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((address, info))
    
    total_premine = 0
    
    # Display by category
    for category, addr_list in categories.items():
        print(f"\n{category}")
        print("-" * 60)
        
        category_total = 0
        for address, info in addr_list:
            balance = blockchain.get_balance(address)
            balance_zion = balance / 1_000_000  # Convert to ZION
            
            print(f"📍 {info['purpose']}")
            print(f"   Address: {address[:25]}...")
            print(f"   Balance: {balance_zion:,.0f} ZION")
            print(f"   Type: {info['type']}")
            if info['consciousness'] != 'N/A':
                print(f"   Consciousness: {info['consciousness']}")
            print()
            
            category_total += balance_zion
            total_premine += balance_zion
        
        print(f"   Category Total: {category_total:,.0f} ZION")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 ZION PRE-MINE SUMMARY")
    print("=" * 60)
    print(f"⚡ Mining Operators (5 × 2B):  {10_000_000_000:,} ZION")
    print(f"👨‍💻 DEV TEAM Fund:              {1_000_000_000:,} ZION")
    print(f"🌐 SITA Network Fund:          {1_000_000_000:,} ZION") 
    print(f"👶 Children Fund:              {1_000_000_000:,} ZION")
    print(f"✨ Genesis Reward:             {342_857_142:,} ZION")
    print("-" * 60)
    print(f"🎯 TOTAL PRE-MINE:             {total_premine:,.0f} ZION")
    
    # Percentages
    print(f"\n📈 Distribution Percentages:")
    print(f"   Mining Operators: {(10_000_000_000 / total_premine * 100):.1f}%")
    print(f"   Special Funds: {(3_000_000_000 / total_premine * 100):.1f}%") 
    print(f"   Genesis: {(342_857_142 / total_premine * 100):.1f}%")
    
    # Purpose explanation
    print(f"\n🎯 Pre-mine Purposes:")
    print(f"   ⚡ Mining Operators: Pool operations, consciousness mining")
    print(f"   👨‍💻 DEV TEAM: Development, maintenance, upgrades")
    print(f"   🌐 SITA Network: Infrastructure, nodes, network stability")
    print(f"   👶 Children Fund: Future generation, education, growth")
    print(f"   ✨ Genesis: Initial distribution, founders reward")
    
    return True

if __name__ == "__main__":
    analyze_complete_premine()
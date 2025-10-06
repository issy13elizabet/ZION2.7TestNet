#!/usr/bin/env python3
"""
Test script to verify pre-mine addresses have correct balances
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.real_blockchain import ZionRealBlockchain

def test_premine_addresses():
    """Test that pre-mine addresses have 2 billion ZION each"""
    print("🔍 Testing pre-mine addresses...")

    # Initialize blockchain (this will create genesis with pre-mine)
    blockchain = ZionRealBlockchain()

    # Pre-mine addresses
    premine_addresses = [
        'ZIONSacredMiner123456789012345678901234567890',
        'ZIONQuantumMiner12345678901234567890123456789',
        'ZIONCosmicMiner123456789012345678901234567890',
        'ZIONEnlightenedMiner1234567890123456789012345',
        'ZIONTranscendentMiner123456789012345678901234'
    ]
    
    # Special purpose addresses (1 billion each)
    special_addresses = [
        ('ZION_DEV_TEAM_FUND_2025_DEVELOPMENT_ADDRESS', 'DEV TEAM'),
        ('ZION_NETWORK_SITA_FUND_2025_INFRASTRUCTURE', 'SITA NETWORK'),
        ('ZION_CHILDREN_FUND_2025_FUTURE_GENERATION', 'CHILDREN FUND')
    ]

    expected_balance_2b = 2000000000000  # 2 billion ZION in atomic units
    expected_balance_1b = 1000000000000  # 1 billion ZION in atomic units
    total_expected = 0

    print("\n📊 Mining operator addresses (2 billion ZION each):")
    print("-" * 70)

    for address in premine_addresses:
        balance = blockchain.get_balance(address)
        balance_zion = balance / 1000000  # Convert to ZION (1 ZION = 1M atomic units)

        print(f"   {address[:20]}... = {balance_zion:,.0f} ZION")
        total_expected += expected_balance_2b

        if balance != expected_balance_2b:
            print(f"❌ Balance mismatch for {address}")
            return False
    
    print("\n📊 Special purpose addresses (1 billion ZION each):")
    print("-" * 70)

    for address, purpose in special_addresses:
        balance = blockchain.get_balance(address)
        balance_zion = balance / 1000000

        print(f"   {address[:20]}... = {balance_zion:,.0f} ZION ({purpose})")
        total_expected += expected_balance_1b

        if balance != expected_balance_1b:
            print(f"❌ Balance mismatch for {address} ({purpose})")
            return False

    # Check genesis address
    genesis_address = 'Z359Sdk6srUZvpAz653xcwsPMFUeew3f6Johmw5apsvMH4uaGY3864q24n9EfiWMUjaGihT7wzkXAr75HiPCbnaQq6'
    genesis_balance = blockchain.get_balance(genesis_address)
    genesis_balance_zion = genesis_balance / 1000000

    print(f"   {genesis_address[:20]}... = {genesis_balance_zion:,.0f} ZION (Genesis)")
    total_expected += genesis_balance

    # Check total supply
    all_addresses = premine_addresses + [addr for addr, _ in special_addresses] + [genesis_address]
    total_supply = sum(blockchain.get_balance(addr) for addr in all_addresses)

    print("\n💰 Total supply check:")
    print(f"   Mining operators (5 × 2B): {5 * expected_balance_2b / 1000000:,.0f} ZION")
    print(f"   Special funds (3 × 1B): {3 * expected_balance_1b / 1000000:,.0f} ZION") 
    print(f"   Genesis reward: {genesis_balance / 1000000:,.0f} ZION")
    print(f"   Expected total: {total_expected / 1000000:,.0f} ZION")
    print(f"   Actual total: {total_supply / 1000000:,.0f} ZION")

    if total_supply == total_expected:
        print("✅ All pre-mine addresses verified successfully!")
        print(f"🎯 Total pre-mine: {total_supply / 1000000:,.0f} ZION")
        return True
    else:
        print("❌ Total supply mismatch!")
        return False

if __name__ == "__main__":
    success = test_premine_addresses()
    sys.exit(0 if success else 1)
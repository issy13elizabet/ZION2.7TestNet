#!/usr/bin/env python3
"""
Test ZION Mining Pool with Humanitarian Aid (Desátek)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pool.zion_mining_pool import ZionMiningPool
from core.real_blockchain import ZionRealBlockchain
import json

def test_humanitarian_pool():
    """Test mining pool with 10% humanitarian aid"""
    print("🏊 Testing ZION Pool with Humanitarian Aid (Desátek)")
    
    # Initialize blockchain and pool
    blockchain = ZionRealBlockchain()
    pool = ZionMiningPool(blockchain)
    
    print(f"\n⚖️ Pool Fee Structure:")
    print(f"   Humanitarian Aid (Desátek): {pool.humanitarian_fee * 100}%")
    print(f"   Pool Operations: 2%")
    print(f"   Total Fees: {(pool.humanitarian_fee + 0.02) * 100}%")
    print(f"   Miner Gets: {(1 - pool.humanitarian_fee - 0.02) * 100}%")
    
    # Register a test miner
    test_miner = "ZION_TEST_MINER_FOR_HUMANITARIAN"
    pool.register_miner(test_miner, "SACRED")
    
    # Simulate finding a block
    print(f"\n🎯 Simulating block found by {test_miner[:20]}...")
    
    # Get base reward and calculate with consciousness
    base_reward = blockchain.block_reward  # 2 ZION
    consciousness_multiplier = 3.0  # SACRED level
    total_reward = base_reward * consciousness_multiplier  # 6 ZION
    
    # Calculate fees
    humanitarian_amount = total_reward * pool.humanitarian_fee  # 10%
    operations_amount = total_reward * 0.02  # 2%
    miner_amount = total_reward - humanitarian_amount - operations_amount  # 88%
    
    print(f"\n💰 Block Reward Breakdown:")
    print(f"   Base reward: {base_reward/1000000:.1f} ZION")
    print(f"   Consciousness bonus (SACRED): {consciousness_multiplier}x")
    print(f"   Total reward: {total_reward/1000000:.1f} ZION")
    print(f"   ")
    print(f"   🤲 Humanitarian aid: {humanitarian_amount/1000000:.2f} ZION (10%)")
    print(f"   🏊 Pool operations: {operations_amount/1000000:.2f} ZION (2%)")
    print(f"   ⛏️  Miner reward: {miner_amount/1000000:.2f} ZION (88%)")
    
    # Show humanitarian impact
    print(f"\n❤️ Humanitarian Impact:")
    print(f"   Every block helps humanitarian causes!")
    print(f"   From 6 ZION reward → {humanitarian_amount/1000000:.2f} ZION goes to aid")
    print(f"   Pool address: {pool.humanitarian_address}")
    
    # Show different consciousness levels impact
    print(f"\n🧘 Consciousness Levels & Humanitarian Contribution:")
    levels = ["PHYSICAL", "EMOTIONAL", "MENTAL", "SACRED", "QUANTUM", "COSMIC", "ENLIGHTENED", "TRANSCENDENT", "ON_THE_STAR"]
    
    for level in levels:
        multiplier = pool.consciousness_levels[level]
        level_reward = base_reward * multiplier
        level_humanitarian = level_reward * pool.humanitarian_fee
        print(f"   {level:12} ({multiplier:4.1f}x): {level_humanitarian/1000000:6.2f} ZION humanitarian aid")
    
    # Pool stats
    print(f"\n📊 Pool Configuration:")
    stats = pool.get_pool_stats()
    print(f"   Humanitarian fee: {stats['pool_info']['humanitarian_fee']}%")
    print(f"   Operations fee: {stats['pool_info']['operations_fee']}%")
    print(f"   Pre-mine operators: {stats['premine_operators']}")
    print(f"   Consciousness levels: {len(pool.consciousness_levels)}")

if __name__ == "__main__":
    test_humanitarian_pool()
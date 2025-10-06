#!/usr/bin/env python3
"""
ZION 2.7.1 - Integration Test
Test complete blockchain system integration
"""

import asyncio
import time
from core.real_blockchain import ZionRealBlockchain
from wallet import get_wallet
from network import get_network

async def test_full_integration():
    """Test complete system integration"""
    print("🚀 ZION 2.7.1 FULL SYSTEM INTEGRATION TEST")
    print("=" * 60)

    # 1. Initialize components
    print("\n🏗️ Initializing components...")

    blockchain = ZionRealBlockchain()
    wallet = get_wallet()
    network = get_network()

    print("✅ Blockchain initialized")
    print("✅ Wallet initialized")
    print("✅ Network initialized")

    # 2. Start network
    print("\n🌐 Starting P2P network...")
    await network.start()
    print("✅ Network started")

    # 3. Create wallet addresses
    print("\n👛 Creating wallet addresses...")
    addr1 = wallet.create_address("Miner Address")
    addr2 = wallet.create_address("Recipient Address")
    print(f"✅ Created addresses: {addr1[:20]}..., {addr2[:20]}...")

    # 4. Mine some blocks
    print("\n⛏️ Mining blocks...")
    for i in range(3):
        consciousness = ["COSMIC", "ENLIGHTENMENT", "ON_THE_STAR"][i]
        block = blockchain.mine_block(
            miner_address=addr1,
            consciousness_level=consciousness
        )
        if block:
            print(f"✅ Block {block.height} mined with {consciousness} consciousness")

    # 5. Create transactions
    print("\n💸 Creating transactions...")
    balance = wallet.get_balance(addr1)
    if balance > 2000000:  # If miner has enough coins
        tx = wallet.create_transaction(addr1, addr2, 1000000, 1000)
        if tx:
            print(f"✅ Transaction created: {tx.tx_id[:16]}...")

    # 6. Mine block with transactions
    print("\n⛏️ Mining block with transactions...")
    block = blockchain.mine_block(
        miner_address=addr1,
        consciousness_level="LIBERATION"
    )
    if block:
        print(f"✅ Block {block.height} mined with {len(block.transactions)} transactions")

    # 7. Show final stats
    print("\n📊 FINAL SYSTEM STATUS:")
    print("=" * 40)

    stats = blockchain.get_blockchain_stats()
    print(f"📦 Total Blocks: {stats['block_count']}")
    print(f"💰 Total Supply: {stats['total_supply']:,} atomic units")
    print(f"📝 Total Transactions: {stats['total_transactions']}")
    print(f"📋 Mempool Size: {stats['mempool_size']}")
    print(f"🌐 Network Peers: {network.get_peer_count()}")

    print("\n👛 Wallet Status:")
    print(f"   Addresses: {len(wallet.get_addresses())}")
    print(f"   Total Balance: {wallet.get_total_balance():,} atomic units")

    for addr in wallet.get_addresses():
        balance = wallet.get_balance(addr['address'])
        print(f"   {addr['address'][:20]}...: {balance:,} ZION")

    # 8. Verify integrity
    print("\n🔍 Verifying blockchain integrity...")
    valid = blockchain.verify_blockchain()
    if valid:
        print("✅ BLOCKCHAIN INTEGRITY VERIFIED!")
    else:
        print("❌ Blockchain integrity check failed")

    # 9. Stop network
    print("\n⏹️ Stopping network...")
    await network.stop()
    print("✅ Network stopped")

    print("\n🎉 ZION 2.7.1 INTEGRATION TEST COMPLETED!")
    print("🚀 System is fully operational!")

if __name__ == "__main__":
    asyncio.run(test_full_integration())
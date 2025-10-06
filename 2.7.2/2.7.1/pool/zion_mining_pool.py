#!/usr/bin/env python3
"""
ZION 2.7.1 - Enhanced Mining Pool with Pre-mine Integration
Consciousness-Based Pool Mining with Sacred Addresses
"""

import asyncio
import json
import time
import hashlib
import logging
import sys
import os
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.real_blockchain import ZionRealBlockchain, RealTransaction


@dataclass
class MinerStats:
    """Statistics for individual miner"""
    address: str
    hashrate: float = 0.0
    shares_submitted: int = 0
    shares_accepted: int = 0
    shares_rejected: int = 0
    blocks_found: int = 0
    last_activity: datetime = field(default_factory=datetime.now)
    consciousness_level: str = "PHYSICAL"
    sacred_multiplier: float = 1.0
    earnings: int = 0  # In atomic units


@dataclass
class PoolBlock:
    """Block found by pool"""
    height: int
    hash: str
    finder: str  # Miner address who found the block
    reward: int
    timestamp: datetime
    consciousness_boost: float = 1.0


class ZionMiningPool:
    """
    ZION 2.7.1 Mining Pool with Consciousness Integration
    
    Features:
    - Pre-mine address integration
    - Consciousness-based rewards
    - Sacred algorithm support
    - Fair share distribution
    - Real-time monitoring
    """
    
    def __init__(self, blockchain: ZionRealBlockchain):
        self.blockchain = blockchain
        self.miners: Dict[str, MinerStats] = {}
        self.active_miners: Set[str] = set()
        self.pool_blocks: List[PoolBlock] = []
        
        # Pool configuration
        self.pool_fee = 0.02  # 2% pool operations fee  
        self.humanitarian_fee = 0.10  # 10% humanitarian aid (desátek)
        self.total_fee = self.pool_fee + self.humanitarian_fee  # 12% total
        self.miners_share = 1.0 - self.total_fee  # 88% for miners
        self.min_payout = 1000000  # 1 ZION minimum payout
        self.share_difficulty = 1000
        
        # Consciousness levels and multipliers
        self.consciousness_levels = {
            "PHYSICAL": 1.0,
            "EMOTIONAL": 1.5,
            "MENTAL": 2.0,
            "INTUITIVE": 2.5,
            "SACRED": 3.0,
            "QUANTUM": 4.0,
            "COSMIC": 5.0,
            "ENLIGHTENED": 7.5,
            "TRANSCENDENT": 10.0,
            "ON_THE_STAR": 15.0
        }
        
        # Pre-mine addresses (operators/founders)
        self.premine_addresses = {
            'ZIONSacredMiner123456789012345678901234567890': {
                'type': 'SACRED',
                'multiplier': 3.0,
                'balance': 2000000000000000,  # 2 billion ZION in atomic units
                'operator': True,
                'auto_distribute': True
            },
            'ZIONQuantumMiner12345678901234567890123456789': {
                'type': 'QUANTUM', 
                'multiplier': 4.0,
                'balance': 2000000000000000,  # 2 billion ZION in atomic units
                'operator': True,
                'auto_distribute': True
            },
            'ZIONCosmicMiner123456789012345678901234567890': {
                'type': 'COSMIC',
                'multiplier': 5.0,
                'balance': 2000000000000000,  # 2 billion ZION in atomic units
                'operator': True,
                'auto_distribute': True
            },
            'ZIONEnlightenedMiner1234567890123456789012345': {
                'type': 'ENLIGHTENED',
                'multiplier': 7.5,
                'balance': 2000000000000000,  # 2 billion ZION in atomic units
                'operator': True,
                'auto_distribute': True
            },
            'ZIONTranscendentMiner123456789012345678901234': {
                'type': 'TRANSCENDENT',
                'multiplier': 10.0,
                'balance': 2000000000000000,  # 2 billion ZION in atomic units
                'operator': True,
                'auto_distribute': True
            }
        }
        
        # Pool wallet (for collecting fees)
        self.pool_address = "ZION_POOL_OPERATIONS_ADDRESS_2025"
        self.humanitarian_address = "ZION_HUMANITARIAN_AID_FUND_2025"
        
        # Statistics
        self.total_hashrate = 0.0
        self.blocks_found = 0
        self.total_shares = 0
        self.start_time = datetime.now()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("ZionPool")

    def register_miner(self, address: str, consciousness_level: str = "PHYSICAL") -> bool:
        """Register new miner to pool"""
        if address not in self.miners:
            multiplier = self.consciousness_levels.get(consciousness_level, 1.0)
            
            self.miners[address] = MinerStats(
                address=address,
                consciousness_level=consciousness_level,
                sacred_multiplier=multiplier
            )
            
            self.logger.info(f"🔮 New miner registered: {address[:20]}... (Level: {consciousness_level})")
            return True
        return False

    def submit_share(self, miner_address: str, hash_result: str, nonce: int) -> Dict:
        """Process share submission from miner"""
        if miner_address not in self.miners:
            return {"status": "error", "message": "Miner not registered"}
        
        miner = self.miners[miner_address]
        miner.shares_submitted += 1
        miner.last_activity = datetime.now()
        self.active_miners.add(miner_address)
        
        # Validate share difficulty
        hash_int = int(hash_result, 16)
        target = 2 ** 256 // self.share_difficulty
        
        if hash_int < target:
            # Valid share
            miner.shares_accepted += 1
            self.total_shares += 1
            
            # Check if it's a block (blockchain difficulty)
            block_target = 2 ** 256 // self.blockchain.difficulty
            if hash_int < block_target:
                return self._process_block_found(miner_address, hash_result, nonce)
            
            # Calculate share reward based on consciousness level
            base_share_reward = 100  # Base atomic units per share
            consciousness_bonus = miner.sacred_multiplier
            share_reward = int(base_share_reward * consciousness_bonus)
            
            miner.earnings += share_reward
            
            return {
                "status": "accepted",
                "reward": share_reward,
                "consciousness_bonus": consciousness_bonus,
                "total_earnings": miner.earnings
            }
        else:
            # Invalid share
            miner.shares_rejected += 1
            return {"status": "rejected", "message": "Share difficulty too low"}

    def _process_block_found(self, finder_address: str, block_hash: str, nonce: int) -> Dict:
        """Process when miner finds a valid block"""
        try:
            # Create block reward transaction
            base_reward = self.blockchain.block_reward
            finder = self.miners[finder_address]
            
            # Apply consciousness multiplier to block reward
            consciousness_bonus = finder.sacred_multiplier
            total_reward = int(base_reward * consciousness_bonus)
            
            # Humanitarian fee (desátek - 10%)
            humanitarian_fee_amount = int(total_reward * self.humanitarian_fee)
            pool_operations_fee = int(total_reward * 0.02)  # 2% for pool operations
            total_fees = humanitarian_fee_amount + pool_operations_fee
            miner_reward = total_reward - total_fees
            
            # Create transactions
            reward_tx = RealTransaction(
                tx_id=f"block_reward_{int(time.time())}",
                from_address="ZION_BLOCK_REWARD_SYSTEM",
                to_address=finder_address,
                amount=miner_reward,
                fee=0,
                timestamp=int(time.time()),
                consciousness_boost=consciousness_bonus
            )
            
            # Humanitarian aid transaction (desátek)
            humanitarian_tx = RealTransaction(
                tx_id=f"humanitarian_aid_{int(time.time())}",
                from_address="ZION_BLOCK_REWARD_SYSTEM", 
                to_address=self.humanitarian_address,
                amount=humanitarian_fee_amount,
                fee=0,
                timestamp=int(time.time())
            )
            
            # Pool operations fee
            operations_tx = RealTransaction(
                tx_id=f"pool_operations_{int(time.time())}",
                from_address="ZION_BLOCK_REWARD_SYSTEM", 
                to_address=self.pool_address,
                amount=pool_operations_fee,
                fee=0,
                timestamp=int(time.time())
            )
            
            # Update statistics
            finder.blocks_found += 1
            finder.earnings += miner_reward
            self.blocks_found += 1
            
            # Record pool block
            pool_block = PoolBlock(
                height=len(self.blockchain.blocks),
                hash=block_hash,
                finder=finder_address,
                reward=total_reward,
                timestamp=datetime.now(),
                consciousness_boost=consciousness_bonus
            )
            self.pool_blocks.append(pool_block)
            
            self.logger.info(f"🎉 BLOCK FOUND by {finder_address[:20]}...")
            self.logger.info(f"   Miner reward: {miner_reward/1000000:.2f} ZION")
            self.logger.info(f"   Humanitarian aid: {humanitarian_fee_amount/1000000:.2f} ZION (10%)")
            self.logger.info(f"   Pool operations: {pool_operations_fee/1000000:.2f} ZION (2%)")
            self.logger.info(f"   Consciousness bonus: {consciousness_bonus}x")
            
            return {
                "status": "block_found",
                "reward": miner_reward,
                "humanitarian_aid": humanitarian_fee_amount,
                "pool_operations": pool_operations_fee,
                "consciousness_bonus": consciousness_bonus,
                "block_height": pool_block.height
            }
            
        except Exception as e:
            self.logger.error(f"Error processing block: {e}")
            return {"status": "error", "message": str(e)}

    def get_pool_stats(self) -> Dict:
        """Get comprehensive pool statistics"""
        active_miners = len([m for m in self.miners.values() 
                           if (datetime.now() - m.last_activity).seconds < 300])
        
        # Calculate total pool hashrate
        total_hashrate = sum(miner.hashrate for miner in self.miners.values())
        
        # Get consciousness distribution
        consciousness_dist = defaultdict(int)
        for miner in self.miners.values():
            consciousness_dist[miner.consciousness_level] += 1
            
        # Recent blocks (last 10)
        recent_blocks = self.pool_blocks[-10:] if self.pool_blocks else []
        
        return {
            "pool_info": {
                "total_miners": len(self.miners),
                "active_miners": active_miners,
                "total_hashrate": total_hashrate,
                "blocks_found": self.blocks_found,
                "total_shares": self.total_shares,
                "pool_fee": self.humanitarian_fee * 100,  # As percentage (10% humanitarian)
                "humanitarian_fee": self.humanitarian_fee * 100,
                "operations_fee": 2.0,  # 2% for pool operations
                "uptime_hours": (datetime.now() - self.start_time).total_seconds() / 3600
            },
            "consciousness_distribution": dict(consciousness_dist),
            "premine_operators": len(self.premine_addresses),
            "recent_blocks": [
                {
                    "height": block.height,
                    "finder": block.finder[:20] + "...",
                    "reward": block.reward / 1000000,  # In ZION
                    "consciousness_boost": block.consciousness_boost,
                    "time": block.timestamp.strftime("%H:%M:%S")
                } for block in recent_blocks
            ],
            "top_miners": self._get_top_miners(5)
        }

    def _get_top_miners(self, limit: int) -> List[Dict]:
        """Get top miners by earnings"""
        sorted_miners = sorted(
            self.miners.values(),
            key=lambda m: m.earnings,
            reverse=True
        )[:limit]
        
        return [
            {
                "address": miner.address[:20] + "...",
                "earnings": miner.earnings / 1000000,  # In ZION
                "shares": miner.shares_accepted,
                "blocks": miner.blocks_found,
                "consciousness": miner.consciousness_level,
                "multiplier": miner.sacred_multiplier
            } for miner in sorted_miners
        ]

    def get_miner_stats(self, address: str) -> Optional[Dict]:
        """Get individual miner statistics"""
        if address not in self.miners:
            return None
            
        miner = self.miners[address]
        is_premine = address in self.premine_addresses
        
        return {
            "address": address,
            "consciousness_level": miner.consciousness_level,
            "sacred_multiplier": miner.sacred_multiplier,
            "hashrate": miner.hashrate,
            "shares_submitted": miner.shares_submitted,
            "shares_accepted": miner.shares_accepted,
            "shares_rejected": miner.shares_rejected,
            "acceptance_rate": miner.shares_accepted / max(miner.shares_submitted, 1) * 100,
            "blocks_found": miner.blocks_found,
            "earnings": miner.earnings / 1000000,  # In ZION
            "last_activity": miner.last_activity.strftime("%Y-%m-%d %H:%M:%S"),
            "is_premine_operator": is_premine,
            "premine_info": self.premine_addresses.get(address) if is_premine else None
        }

    def process_payouts(self) -> Dict:
        """Process pending payouts to miners"""
        payouts_processed = 0
        total_paid = 0
        
        for address, miner in self.miners.items():
            if miner.earnings >= self.min_payout:
                # Create payout transaction
                payout_amount = miner.earnings
                
                payout_tx = RealTransaction(
                    tx_id=f"payout_{address}_{int(time.time())}",
                    from_address=self.pool_address,
                    to_address=address,
                    amount=payout_amount,
                    fee=1000,  # Small network fee
                    timestamp=int(time.time())
                )
                
                # Add to blockchain mempool
                if self.blockchain.add_transaction(payout_tx):
                    miner.earnings = 0  # Reset earnings
                    payouts_processed += 1
                    total_paid += payout_amount
                    
                    self.logger.info(f"💰 Payout {payout_amount/1000000:.6f} ZION to {address[:20]}...")
        
        return {
            "payouts_processed": payouts_processed,
            "total_paid_zion": total_paid / 1000000,
            "timestamp": datetime.now().isoformat()
        }

    def auto_distribute_from_operators(self, block_reward: int, consciousness_level: str) -> Dict:
        """
        Automaticky distribuuje odměny z mining operator adres na základě těžby
        
        Args:
            block_reward: Základní block reward v atomic units
            consciousness_level: Úroveň consciousness nalezeného bloku
            
        Returns:
            Dictionary s detaily distribuce
        """
        # Najdi vhodného operátora podle consciousness level
        operator_address = None
        operator_info = None
        
        for address, info in self.premine_addresses.items():
            if info['type'] == consciousness_level and info.get('auto_distribute', False):
                operator_address = address
                operator_info = info
                break
        
        if not operator_address:
            # Fallback na SACRED operátora
            for address, info in self.premine_addresses.items():
                if info['type'] == 'SACRED' and info.get('auto_distribute', False):
                    operator_address = address
                    operator_info = info
                    break
        
        if not operator_address:
            raise ValueError("Žádný dostupný mining operator pro distribuci")
        
        # Spočítej total reward s consciousness multiplierem
        consciousness_multiplier = operator_info['multiplier']
        total_reward = int(block_reward * consciousness_multiplier)
        
        # Rozdělení fees
        humanitarian_amount = int(total_reward * self.humanitarian_fee)  # 10%
        pool_operations_amount = int(total_reward * self.pool_fee)        # 2%
        miners_total = total_reward - humanitarian_amount - pool_operations_amount  # 88%
        
        # Check operator balance
        current_balance = self.blockchain.get_balance(operator_address)
        if current_balance < total_reward:
            raise ValueError(f"Nedostatečný balance operátora {operator_address[:20]}...")
        
        distribution_details = {
            'operator_address': operator_address,
            'operator_type': operator_info['type'],
            'consciousness_multiplier': consciousness_multiplier,
            'base_reward': block_reward,
            'total_reward': total_reward,
            'humanitarian_fee': humanitarian_amount,
            'pool_operations_fee': pool_operations_amount,
            'miners_share': miners_total,
            'operator_balance_before': current_balance,
            'operator_balance_after': current_balance - total_reward,
            'distributions': []
        }
        
        # Distribuuj těžařům podle share contribution
        active_miners = [(addr, miner) for addr, miner in self.miners.items() if miner.shares_accepted > 0]
        
        if active_miners:
            total_shares = sum(miner.shares_accepted for _, miner in active_miners)
            
            for miner_address, miner in active_miners:
                # Spočítej podíl těžaře
                miner_share_percentage = miner.shares_accepted / total_shares if total_shares > 0 else 0
                miner_reward = int(miners_total * miner_share_percentage)
                
                # Bonus za consciousness level
                miner_consciousness_bonus = miner.sacred_multiplier
                final_miner_reward = int(miner_reward * miner_consciousness_bonus)
                
                if final_miner_reward > 0:
                    distribution_details['distributions'].append({
                        'miner_address': miner_address,
                        'shares': miner.shares_accepted,
                        'share_percentage': miner_share_percentage * 100,
                        'base_reward': miner_reward,
                        'consciousness_bonus': miner_consciousness_bonus,
                        'final_reward': final_miner_reward,
                        'consciousness_level': miner.consciousness_level
                    })
                    
                    # Aktualizuj earnings
                    miner.earnings += final_miner_reward
        
        return distribution_details

    def process_block_with_operator_distribution(self, finder_address: str, consciousness_level: str):
        """
        Zpracuje nalezený blok s automatickou distribucí z operátora
        """
        # Získej base reward
        base_reward = self.blockchain.block_reward
        
        # Automatická distribuce z operátora
        distribution = self.auto_distribute_from_operators(base_reward, consciousness_level)
        
        # Log distribuce
        print(f"🎉 Blok nalezen! Automatická distribuce z {distribution['operator_type']} operátora")
        print(f"   💰 Celková odměna: {distribution['total_reward'] / 1_000_000:.2f} ZION")
        print(f"   ❤️  Humanitární: {distribution['humanitarian_fee'] / 1_000_000:.2f} ZION")
        print(f"   ⚙️  Pool fee: {distribution['pool_operations_fee'] / 1_000_000:.2f} ZION")
        print(f"   👥 Těžařům: {distribution['miners_share'] / 1_000_000:.2f} ZION")
        print(f"   🔄 Distribuce na {len(distribution['distributions'])} těžařů")
        
        # Update pool statistiky
        self.blocks_found += 1
        if finder_address in self.miners:
            self.miners[finder_address].blocks_found += 1
        
        return distribution

    async def start_pool_server(self, host: str = "0.0.0.0", port: int = 8888):
        """Start the mining pool server"""
        self.logger.info(f"🏊 Starting ZION Mining Pool on {host}:{port}")
        self.logger.info(f"🔮 Consciousness levels supported: {list(self.consciousness_levels.keys())}")
        self.logger.info(f"👑 Pre-mine operators: {len(self.premine_addresses)}")
        
        # In real implementation, this would start a Stratum server
        # For now, just log that pool is ready
        self.logger.info("✅ ZION Mining Pool ready for miners!")
        
        # Simulate pool running
        while True:
            await asyncio.sleep(60)  # Update every minute
            self._update_pool_stats()

    def _update_pool_stats(self):
        """Update pool statistics periodically"""
        # Remove inactive miners from active set
        cutoff_time = datetime.now() - timedelta(minutes=5)
        inactive_miners = {
            addr for addr in self.active_miners 
            if self.miners[addr].last_activity < cutoff_time
        }
        self.active_miners -= inactive_miners
        
        # Log pool status
        if len(self.active_miners) > 0:
            total_hashrate = sum(
                self.miners[addr].hashrate 
                for addr in self.active_miners
            )
            self.logger.info(f"📊 Pool: {len(self.active_miners)} miners, {total_hashrate:.2f} H/s")


def main():
    """Demo of mining pool functionality"""
    print("🏊 ZION 2.7.1 Mining Pool Demo")
    
    # Initialize blockchain and pool
    blockchain = ZionRealBlockchain()
    pool = ZionMiningPool(blockchain)
    
    # Demo: Register some miners
    test_miners = [
        ("ZION_MINER_ALICE_001", "EMOTIONAL"),
        ("ZION_MINER_BOB_002", "MENTAL"), 
        ("ZION_MINER_CHARLIE_003", "SACRED"),
        ("ZIONSacredMiner123456789012345678901234567890", "SACRED")  # Pre-mine operator
    ]
    
    for address, level in test_miners:
        pool.register_miner(address, level)
    
    # Demo: Submit some shares
    print("\n📊 Simulating share submissions...")
    for i in range(5):
        for address, _ in test_miners:
            # Simulate share hash (would be real in production)
            share_hash = hashlib.sha256(f"{address}_{i}_{time.time()}".encode()).hexdigest()
            result = pool.submit_share(address, share_hash, i)
            print(f"Share from {address[:15]}...: {result['status']}")
    
    # Show pool stats
    print("\n🏊 Pool Statistics:")
    stats = pool.get_pool_stats()
    print(json.dumps(stats, indent=2))
    
    # Show individual miner stats
    print("\n👤 Miner Statistics:")
    for address, _ in test_miners:
        miner_stats = pool.get_miner_stats(address)
        if miner_stats:
            print(f"\n{address[:20]}...")
            print(f"  Consciousness: {miner_stats['consciousness_level']}")
            print(f"  Multiplier: {miner_stats['sacred_multiplier']}x")
            print(f"  Shares: {miner_stats['shares_accepted']}/{miner_stats['shares_submitted']}")
            print(f"  Earnings: {miner_stats['earnings']:.6f} ZION")


if __name__ == "__main__":
    main()
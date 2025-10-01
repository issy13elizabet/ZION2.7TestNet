#!/usr/bin/env python3
"""
ZION 2.7.1 Real Blockchain Implementation
Production-Ready Blockchain Core with No Simulations
🌟 JAI RAM SITA HANUMAN - ON THE STAR
"""

import json
import time
import hashlib
import sqlite3
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class RealBlock:
    """Real production block"""
    height: int
    hash: str
    previous_hash: str
    timestamp: int
    nonce: int
    difficulty: int
    transactions: List[Dict[str, Any]]
    reward: int
    miner_address: str
    consciousness_level: str = "PHYSICAL"
    sacred_multiplier: float = 1.0
    
    def calculate_hash(self) -> str:
        """Calculate block hash"""
        block_string = json.dumps({
            'height': self.height,
            'previous_hash': self.previous_hash,
            'timestamp': self.timestamp,
            'nonce': self.nonce,
            'transactions': self.transactions,
            'miner_address': self.miner_address
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()


@dataclass 
class RealTransaction:
    """Real production transaction"""
    tx_id: str
    from_address: str
    to_address: str
    amount: int  # atomic units
    fee: int
    timestamp: int
    signature: str
    consciousness_boost: float = 1.0
    
    def calculate_hash(self) -> str:
        """Calculate transaction hash"""
        tx_string = json.dumps({
            'from_address': self.from_address,
            'to_address': self.to_address,
            'amount': self.amount,
            'fee': self.fee,
            'timestamp': self.timestamp
        }, sort_keys=True)
        return hashlib.sha256(tx_string.encode()).hexdigest()


class ZionRealBlockchain:
    """ZION 2.7.1 Real Blockchain - No Simulations"""
    
    def __init__(self, db_file: str = "zion_real_blockchain.db"):
        self.db_file = db_file
        self.blocks: List[RealBlock] = []
        self.mempool: List[RealTransaction] = []
        self.difficulty = 1000
        self.block_reward = 100000000  # 1 ZION in atomic units
        self._lock = threading.Lock()
        
        # Initialize database
        self._init_database()
        
        # Load existing blocks
        self._load_blocks_from_db()
        
        # Create genesis if needed
        if not self.blocks:
            self._create_genesis_block()
    
    def _init_database(self):
        """Initialize blockchain database"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS real_blocks (
                height INTEGER PRIMARY KEY,
                hash TEXT UNIQUE NOT NULL,
                previous_hash TEXT,
                timestamp INTEGER,
                nonce INTEGER,
                difficulty INTEGER,
                transactions_json TEXT,
                reward INTEGER,
                miner_address TEXT,
                consciousness_level TEXT,
                sacred_multiplier REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS real_transactions (
                tx_id TEXT PRIMARY KEY,
                block_height INTEGER,
                from_address TEXT,
                to_address TEXT,
                amount INTEGER,
                fee INTEGER,
                timestamp INTEGER,
                signature TEXT,
                consciousness_boost REAL,
                FOREIGN KEY (block_height) REFERENCES real_blocks (height)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_blocks_from_db(self):
        """Load existing blocks from database"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM real_blocks ORDER BY height')
        rows = cursor.fetchall()
        
        for row in rows:
            transactions = json.loads(row[6])  # transactions_json
            
            block = RealBlock(
                height=row[0],
                hash=row[1], 
                previous_hash=row[2],
                timestamp=row[3],
                nonce=row[4],
                difficulty=row[5],
                transactions=transactions,
                reward=row[7],
                miner_address=row[8],
                consciousness_level=row[9],
                sacred_multiplier=row[10]
            )
            
            self.blocks.append(block)
        
        conn.close()
        
        if self.blocks:
            print(f"📦 Loaded {len(self.blocks)} real blocks from database")
    
    def _create_genesis_block(self):
        """Create genesis block"""
        genesis_transactions = [{
            'type': 'genesis',
            'amount': 342857142857,  # Genesis reward
            'to_address': 'ZION_GENESIS_REWARD_ADDRESS'
        }]
        
        genesis_block = RealBlock(
            height=0,
            hash="",
            previous_hash="0" * 64,
            timestamp=int(time.time()),
            nonce=0,
            difficulty=1,
            transactions=genesis_transactions,
            reward=342857142857,
            miner_address="ZION_GENESIS_MINER",
            consciousness_level="ON_THE_STAR",
            sacred_multiplier=10.0
        )
        
        # Calculate genesis hash
        genesis_block.hash = genesis_block.calculate_hash()
        
        # Add to blockchain
        self.blocks.append(genesis_block)
        
        # Save to database
        self._save_block_to_db(genesis_block)
        
        print("✨ Genesis block created with real blockchain hash")
        print(f"   Hash: {genesis_block.hash}")
        print(f"   Reward: {genesis_block.reward} atomic units")
    
    def add_transaction(self, tx: RealTransaction) -> bool:
        """Add transaction to mempool"""
        with self._lock:
            # Validate transaction
            if not self._validate_transaction(tx):
                return False
            
            # Add to mempool
            self.mempool.append(tx)
            
            # Save to database
            self._save_transaction_to_db(tx, None)  # No block yet
            
            print(f"📝 Transaction added to mempool: {tx.tx_id[:16]}...")
            return True
    
    def _validate_transaction(self, tx: RealTransaction) -> bool:
        """Validate transaction (basic checks)"""
        # Check amounts
        if tx.amount <= 0 or tx.fee < 0:
            return False
        
        # Check addresses
        if not tx.from_address or not tx.to_address:
            return False
        
        # Check signature exists
        if not tx.signature:
            return False
        
        return True
    
    def mine_block(self, miner_address: str, consciousness_level: str = "PHYSICAL") -> Optional[RealBlock]:
        """Mine a new block"""
        with self._lock:
            # Get transactions from mempool
            transactions = self.mempool[:10]  # Max 10 transactions per block
            
            # Calculate sacred multiplier
            sacred_multipliers = {
                "PHYSICAL": 1.0,
                "EMOTIONAL": 1.1,
                "MENTAL": 1.2,
                "INTUITIVE": 1.3,
                "SPIRITUAL": 1.5,
                "COSMIC": 2.0,
                "UNITY": 2.5,
                "ENLIGHTENMENT": 3.0,
                "LIBERATION": 5.0,
                "ON_THE_STAR": 10.0
            }
            sacred_multiplier = sacred_multipliers.get(consciousness_level, 1.0)
            
            # Create new block
            new_block = RealBlock(
                height=len(self.blocks),
                hash="",
                previous_hash=self.blocks[-1].hash if self.blocks else "0" * 64,
                timestamp=int(time.time()),
                nonce=0,
                difficulty=self.difficulty,
                transactions=[asdict(tx) for tx in transactions],
                reward=int(self.block_reward * sacred_multiplier),
                miner_address=miner_address,
                consciousness_level=consciousness_level,
                sacred_multiplier=sacred_multiplier
            )
            
            # Mine the block (proof of work)
            start_time = time.time()
            while True:
                new_block.nonce += 1
                new_block.hash = new_block.calculate_hash()
                
                # Check if hash meets difficulty
                if int(new_block.hash, 16) < (2 ** 256) // self.difficulty:
                    break
                
                # Prevent infinite mining in production
                if time.time() - start_time > 30:  # 30 second timeout
                    print(f"⚠️  Mining timeout for block {new_block.height}")
                    return None
            
            # Add block to blockchain
            self.blocks.append(new_block)
            
            # Remove transactions from mempool
            self.mempool = self.mempool[len(transactions):]
            
            # Save to database
            self._save_block_to_db(new_block)
            
            # Update transaction records with block height
            for tx in transactions:
                self._update_transaction_block(tx.tx_id, new_block.height)
            
            mining_time = time.time() - start_time
            print(f"⛏️  Block {new_block.height} mined!")
            print(f"   Hash: {new_block.hash}")
            print(f"   Reward: {new_block.reward} atomic units")
            print(f"   Mining time: {mining_time:.2f}s")
            print(f"   Transactions: {len(transactions)}")
            print(f"   🧠 Consciousness: {consciousness_level}")
            print(f"   🌟 Sacred multiplier: {sacred_multiplier:.2f}x")
            
            return new_block
    
    def _save_block_to_db(self, block: RealBlock):
        """Save block to database"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO real_blocks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            block.height, block.hash, block.previous_hash, block.timestamp,
            block.nonce, block.difficulty, json.dumps(block.transactions),
            block.reward, block.miner_address, block.consciousness_level,
            block.sacred_multiplier
        ))
        
        conn.commit()
        conn.close()
    
    def _save_transaction_to_db(self, tx: RealTransaction, block_height: Optional[int]):
        """Save transaction to database"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO real_transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            tx.tx_id, block_height, tx.from_address, tx.to_address,
            tx.amount, tx.fee, tx.timestamp, tx.signature, tx.consciousness_boost
        ))
        
        conn.commit()
        conn.close()
    
    def _update_transaction_block(self, tx_id: str, block_height: int):
        """Update transaction with block height"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute(
            'UPDATE real_transactions SET block_height = ? WHERE tx_id = ?',
            (block_height, tx_id)
        )
        
        conn.commit()
        conn.close()
    
    def get_balance(self, address: str) -> int:
        """Get address balance (atomic units)"""
        balance = 0
        
        for block in self.blocks:
            # Check mining rewards
            if block.miner_address == address:
                balance += block.reward
            
            # Check transactions
            for tx_data in block.transactions:
                if tx_data.get('to_address') == address:
                    balance += tx_data.get('amount', 0)
                elif tx_data.get('from_address') == address:
                    balance -= tx_data.get('amount', 0)
                    balance -= tx_data.get('fee', 0)
        
        return max(0, balance)
    
    def get_block_count(self) -> int:
        """Get current block count"""
        return len(self.blocks)
    
    def get_latest_block(self) -> Optional[RealBlock]:
        """Get latest block"""
        return self.blocks[-1] if self.blocks else None
    
    def verify_blockchain(self) -> bool:
        """Verify blockchain integrity"""
        for i in range(1, len(self.blocks)):
            current_block = self.blocks[i]
            previous_block = self.blocks[i-1]
            
            # Check previous hash
            if current_block.previous_hash != previous_block.hash:
                print(f"❌ Block {i} has invalid previous hash")
                return False
            
            # Check hash calculation
            if current_block.hash != current_block.calculate_hash():
                print(f"❌ Block {i} has invalid hash")
                return False
        
        print("✅ Blockchain integrity verified")
        return True
    
    def get_blockchain_stats(self) -> Dict[str, Any]:
        """Get blockchain statistics"""
        total_supply = sum(block.reward for block in self.blocks)
        total_transactions = sum(len(block.transactions) for block in self.blocks)
        
        consciousness_distribution = {}
        for block in self.blocks:
            level = block.consciousness_level
            consciousness_distribution[level] = consciousness_distribution.get(level, 0) + 1
        
        return {
            'block_count': len(self.blocks),
            'total_supply': total_supply,
            'total_transactions': total_transactions,
            'mempool_size': len(self.mempool),
            'difficulty': self.difficulty,
            'consciousness_distribution': consciousness_distribution
        }


if __name__ == "__main__":
    print("🚀 ZION 2.7.1 Real Blockchain Test")
    print("JAI RAM SITA HANUMAN - ON THE STAR! ⭐")
    
    # Initialize real blockchain
    blockchain = ZionRealBlockchain()
    
    # Mine some real blocks
    for i in range(3):
        miner_address = f"ZION_REAL_MINER_{i}"
        consciousness_levels = ["SPIRITUAL", "COSMIC", "ENLIGHTENMENT"]
        
        block = blockchain.mine_block(
            miner_address=miner_address,
            consciousness_level=consciousness_levels[i]
        )
        
        if block:
            print(f"✅ Real block {block.height} added to blockchain")
    
    # Show stats
    stats = blockchain.get_blockchain_stats()
    print(f"\n📊 Blockchain Stats:")
    print(f"   Blocks: {stats['block_count']}")
    print(f"   Total Supply: {stats['total_supply']:,} atomic units")
    print(f"   Consciousness Distribution: {stats['consciousness_distribution']}")
    
    # Verify integrity
    blockchain.verify_blockchain()
    
    print("\n✅ Real blockchain operational!")
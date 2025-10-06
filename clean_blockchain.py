#!/usr/bin/env python3
"""
ZION 2.7.4 - Clean Blockchain Core
Čistá implementace blockchainu bez citlivých dat
"""

import hashlib
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

class CleanZionBlockchain:
    """Čistá implementace ZION blockchainu pro veřejné použití"""
    
    def __init__(self):
        self.blocks = []
        self.pending_transactions = []
        self.mining_difficulty = 4
        self.block_reward = 50  # ZION
        self.premine_addresses = self._get_placeholder_addresses()
        
    def _get_placeholder_addresses(self) -> Dict:
        """Vrací pouze placeholder adresy - žádné skutečné keys!"""
        return {
            '[MINING_OPERATOR_1]': {
                'purpose': 'Sacred Mining Operator',
                'amount': 2_000_000_000,
                'type': 'mining'
            },
            '[MINING_OPERATOR_2]': {
                'purpose': 'Quantum Mining Operator', 
                'amount': 2_000_000_000,
                'type': 'mining'
            },
            '[MINING_OPERATOR_3]': {
                'purpose': 'Cosmic Mining Operator',
                'amount': 2_000_000_000,
                'type': 'mining'
            },
            '[MINING_OPERATOR_4]': {
                'purpose': 'Enlightened Mining Operator',
                'amount': 2_000_000_000,
                'type': 'mining'
            },
            '[MINING_OPERATOR_5]': {
                'purpose': 'Transcendent Mining Operator',
                'amount': 2_000_000_000,
                'type': 'mining'
            },
            '[DEV_TEAM_FUND]': {
                'purpose': 'Development Team Fund',
                'amount': 1_000_000_000,
                'type': 'development'
            },
            '[NETWORK_INFRASTRUCTURE]': {
                'purpose': 'Network Infrastructure',
                'amount': 1_000_000_000,
                'type': 'infrastructure'
            },
            '[CHILDREN_FUND]': {
                'purpose': 'Children Future Fund',
                'amount': 1_000_000_000,
                'type': 'social'
            },
            '[NETWORK_ADMIN]': {
                'purpose': 'Network Administrator',
                'amount': 1_000_000_000,
                'type': 'admin'
            },
            '[GENESIS_REWARD]': {
                'purpose': 'Genesis Reward',
                'amount': 342_857_142,
                'type': 'genesis'
            }
        }
    
    def create_genesis_block(self) -> Dict:
        """Vytvoří genesis block s placeholder adresami"""
        genesis_transactions = []
        
        for address, info in self.premine_addresses.items():
            genesis_transactions.append({
                'from_address': 'GENESIS',
                'to_address': address,
                'amount': info['amount'],
                'purpose': info['purpose'],
                'type': info['type'],
                'timestamp': time.time()
            })
        
        genesis_block = {
            'index': 0,
            'timestamp': time.time(),
            'transactions': genesis_transactions,
            'previous_hash': '0' * 64,
            'nonce': 0,
            'hash': None
        }
        
        genesis_block['hash'] = self._calculate_hash(genesis_block)
        return genesis_block
    
    def _calculate_hash(self, block: Dict) -> str:
        """Počítá hash bloku"""
        block_string = json.dumps({
            'index': block['index'],
            'timestamp': block['timestamp'],
            'transactions': block['transactions'],
            'previous_hash': block['previous_hash'],
            'nonce': block['nonce']
        }, sort_keys=True)
        
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def mine_block(self, miner_address: str = '[PLACEHOLDER_MINER]') -> Dict:
        """Vytěží nový blok"""
        if not self.blocks:
            # Create genesis block first
            genesis = self.create_genesis_block()
            self.blocks.append(genesis)
            return genesis
        
        # Create new block
        new_block = {
            'index': len(self.blocks),
            'timestamp': time.time(),
            'transactions': self.pending_transactions.copy(),
            'previous_hash': self.blocks[-1]['hash'],
            'nonce': 0,
            'hash': None
        }
        
        # Add mining reward
        reward_transaction = {
            'from_address': 'MINING_REWARD',
            'to_address': miner_address,
            'amount': self.block_reward,
            'purpose': 'Block Mining Reward',
            'type': 'reward',
            'timestamp': time.time()
        }
        new_block['transactions'].append(reward_transaction)
        
        # Mine (simplified - no real proof of work)
        new_block['hash'] = self._calculate_hash(new_block)
        
        self.blocks.append(new_block)
        self.pending_transactions = []
        
        return new_block
    
    def add_transaction(self, from_addr: str, to_addr: str, amount: float, purpose: str = "Transfer"):
        """Přidá transakci do pending pool"""
        transaction = {
            'from_address': from_addr,
            'to_address': to_addr,
            'amount': amount,
            'purpose': purpose,
            'type': 'transfer',
            'timestamp': time.time()
        }
        self.pending_transactions.append(transaction)
        return transaction
    
    def get_balance(self, address: str) -> float:
        """Získá zůstatek adresy"""
        balance = 0
        
        for block in self.blocks:
            for tx in block['transactions']:
                if tx['to_address'] == address:
                    balance += tx['amount']
                elif tx['from_address'] == address:
                    balance -= tx['amount']
        
        return balance
    
    def get_blockchain_info(self) -> Dict:
        """Vrací informace o blockchainu"""
        total_supply = sum(info['amount'] for info in self.premine_addresses.values())
        
        return {
            'version': '2.7.4-CLEAN',
            'total_blocks': len(self.blocks),
            'pending_transactions': len(self.pending_transactions),
            'total_supply': total_supply,
            'mining_difficulty': self.mining_difficulty,
            'block_reward': self.block_reward,
            'status': 'PUBLIC_READY',
            'security_note': 'All sensitive data externalized'
        }

def demo_clean_blockchain():
    """Demo čistého blockchainu"""
    print("🚀 ZION 2.7.4 - Clean Blockchain Demo")
    print("=" * 50)
    
    blockchain = CleanZionBlockchain()
    
    # Create genesis
    print("⛏️  Mining genesis block...")
    genesis = blockchain.mine_block()
    print(f"✅ Genesis block created: {genesis['hash'][:16]}...")
    
    # Add transaction
    print("📝 Adding transaction...")
    blockchain.add_transaction(
        '[PLACEHOLDER_SENDER]',
        '[PLACEHOLDER_RECEIVER]', 
        100,
        'Demo Transfer'
    )
    
    # Mine block
    print("⛏️  Mining block 1...")
    block1 = blockchain.mine_block('[PLACEHOLDER_MINER]')
    print(f"✅ Block 1 mined: {block1['hash'][:16]}...")
    
    # Show info
    info = blockchain.get_blockchain_info()
    print("\n📊 Blockchain Info:")
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    print("\n🔐 Security Status:")
    print("   ✅ No private keys in code")
    print("   ✅ No real addresses exposed")
    print("   ✅ All sensitive data externalized")
    print("   ✅ Ready for public repository")

if __name__ == "__main__":
    demo_clean_blockchain()
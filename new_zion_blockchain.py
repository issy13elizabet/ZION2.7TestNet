#!/usr/bin/env python3
"""
ZION 2.7.4 - Nový Blockchain s Novými Premine Adresami
Implementace blockchainu s čerstvě vygenerovanými adresami
"""

import hashlib
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

class NewZionBlockchain:
    """Nový ZION blockchain s novými premine adresami"""
    
    def __init__(self):
        self.blocks = []
        self.pending_transactions = []
        self.mining_difficulty = 4
        self.block_reward = 50  # ZION
        self.premine_addresses = self._get_new_addresses()
        self.balances = {}
        
        # Inicializace s genesis blockem
        self._create_genesis_block()
        
    def _get_new_addresses(self) -> Dict:
        """Nově vygenerované adresy pro premine"""
        return {
            'ZION_SACRED_B0FA7E2A234D8C2F08545F02295C98': {
                'purpose': 'Sacred Mining Operator',
                'amount': 2_000_000_000,
                'type': 'mining'
            },
            'ZION_QUANTUM_89D80B129682D41AD76DAE3F90C3E2': {
                'purpose': 'Quantum Mining Operator', 
                'amount': 2_000_000_000,
                'type': 'mining'
            },
            'ZION_COSMIC_397B032D6E2D3156F6F709E8179D36': {
                'purpose': 'Cosmic Mining Operator',
                'amount': 2_000_000_000,
                'type': 'mining'
            },
            'ZION_ENLIGHTENED_004A5DBD12FDCAACEDCB5384DDC035': {
                'purpose': 'Enlightened Mining Operator',
                'amount': 2_000_000_000,
                'type': 'mining'
            },
            'ZION_TRANSCENDENT_6BD30CB1835013503A8167D9CD86E0': {
                'purpose': 'Transcendent Mining Operator',
                'amount': 2_000_000_000,
                'type': 'mining'
            },
            'ZION_DEVELOPMENT_TEAM_FUND_378614887FEA27791540F45': {
                'purpose': 'Development Team Fund',
                'amount': 2_142_857_142,
                'type': 'development'
            },
            'ZION_NETWORK_INFRASTRUCTURE_SITA_B5F3BE9968A1D90': {
                'purpose': 'Network Infrastructure (SITA)',
                'amount': 1_000_000_000,
                'type': 'infrastructure'
            },
            'ZION_CHILDREN_FUTURE_FUND_1ECCB72BC30AADD086656A59': {
                'purpose': 'Children Future Fund',
                'amount': 1_000_000_000,
                'type': 'charity'
            },
            'ZION_MAITREYA_BUDDHA_D7A371ABD1FF1C5D42AB02AAE4F57': {
                'purpose': 'Network Administrator',
                'amount': 999_000_000,
                'type': 'admin'
            },
            'ZION_ON_THE_STAR_0B461AB5BCACC40D1ECE95A2D82030': {
                'purpose': 'Genesis Reward',
                'amount': 200_000_000,
                'type': 'genesis'
            }
        }
    
    def _create_genesis_block(self):
        """Vytvoří genesis blok s premine distribucí"""
        genesis_transactions = []
        
        # Vytvoří premine transakce
        for address, info in self.premine_addresses.items():
            genesis_transactions.append({
                'from': 'GENESIS',
                'to': address,
                'amount': info['amount'],
                'purpose': info['purpose'],
                'timestamp': time.time()
            })
            
            # Inicializace zůstatků
            self.balances[address] = info['amount']
        
        genesis_block = {
            'index': 0,
            'timestamp': time.time(),
            'transactions': genesis_transactions,
            'previous_hash': '0000000000000000000000000000000000000000000000000000000000000000',
            'nonce': 0,
            'hash': None
        }
        
        # Mining genesis bloku
        genesis_block['hash'] = self._mine_block(genesis_block)
        self.blocks.append(genesis_block)
        
        print(f"✅ Genesis block created: {genesis_block['hash'][:16]}...")
        
    def _mine_block(self, block) -> str:
        """Těží blok pomocí Proof of Work"""
        target = "0" * self.mining_difficulty
        
        while True:
            block['nonce'] += 1
            block_string = json.dumps(block, sort_keys=True, separators=(',', ':'))
            block_hash = hashlib.sha256(block_string.encode()).hexdigest()
            
            if block_hash.startswith(target):
                return block_hash
                
    def create_transaction(self, from_address: str, to_address: str, amount: float, purpose: str = ""):
        """Vytvoří novou transakci"""
        if from_address not in self.balances:
            self.balances[from_address] = 0
        if to_address not in self.balances:
            self.balances[to_address] = 0
            
        if self.balances[from_address] < amount:
            raise ValueError(f"Insufficient balance: {self.balances[from_address]} < {amount}")
        
        transaction = {
            'from': from_address,
            'to': to_address,
            'amount': amount,
            'purpose': purpose,
            'timestamp': time.time()
        }
        
        self.pending_transactions.append(transaction)
        return transaction
    
    def mine_pending_transactions(self, mining_reward_address: str) -> str:
        """Vytěží nový blok s čekajícími transakcemi"""
        block = {
            'index': len(self.blocks),
            'timestamp': time.time(),
            'transactions': self.pending_transactions.copy(),
            'previous_hash': self.blocks[-1]['hash'] if self.blocks else '0',
            'nonce': 0,
            'hash': None
        }
        
        # Přidání mining reward transakce
        if mining_reward_address:
            reward_transaction = {
                'from': 'MINING_REWARD',
                'to': mining_reward_address,
                'amount': self.block_reward,
                'purpose': 'Block Mining Reward',
                'timestamp': time.time()
            }
            block['transactions'].append(reward_transaction)
        
        # Mining
        print(f"⛏️  Mining block {block['index']}...")
        block['hash'] = self._mine_block(block)
        
        # Aktualizace zůstatků
        for tx in block['transactions']:
            if tx['from'] != 'GENESIS' and tx['from'] != 'MINING_REWARD':
                self.balances[tx['from']] -= tx['amount']
            
            if tx['to'] not in self.balances:
                self.balances[tx['to']] = 0
            self.balances[tx['to']] += tx['amount']
        
        self.blocks.append(block)
        self.pending_transactions = []
        
        print(f"✅ Block {block['index']} mined: {block['hash'][:16]}...")
        return block['hash']
    
    def get_balance(self, address: str) -> float:
        """Vrátí zůstatek adresy"""
        return self.balances.get(address, 0)
    
    def get_total_supply(self) -> float:
        """Vrátí celkovou nabídku ZION"""
        return sum(self.balances.values())
    
    def validate_chain(self) -> bool:
        """Validuje celý blockchain"""
        for i in range(1, len(self.blocks)):
            current_block = self.blocks[i]
            previous_block = self.blocks[i-1]
            
            # Kontrola hash řetězce
            if current_block['previous_hash'] != previous_block['hash']:
                return False
                
            # Kontrola hash bloku
            block_copy = current_block.copy()
            block_hash = block_copy.pop('hash')
            calculated_hash = self._mine_block(block_copy)
            
            if calculated_hash != block_hash:
                return False
                
        return True
    
    def print_status(self):
        """Zobrazí status blockchainu"""
        print("\n🚀 NOVÝ ZION BLOCKCHAIN STATUS")
        print("=" * 50)
        print(f"📊 Počet bloků: {len(self.blocks)}")
        print(f"💰 Celková nabídka: {self.get_total_supply():,.0f} ZION")
        print(f"⚖️  Validní řetězec: {'✅ ANO' if self.validate_chain() else '❌ NE'}")
        print(f"📋 Čekající transakce: {len(self.pending_transactions)}")
        
        print(f"\n🏦 PREMINE DISTRIBUCE:")
        total_premine = 0
        for address, info in self.premine_addresses.items():
            balance = self.get_balance(address)
            total_premine += balance
            print(f"   {info['purpose']}: {balance:,.0f} ZION")
            print(f"      └─ {address[:30]}...")
        
        print(f"\n💎 Total Premine: {total_premine:,.0f} ZION")
        print(f"🆔 Latest Block: {self.blocks[-1]['hash'][:32]}..." if self.blocks else "No blocks")

def main():
    """Spustí demo nového blockchainu"""
    print("🚀 Inicializuji nový ZION blockchain s novými adresami...")
    
    # Vytvoření nového blockchainu
    blockchain = NewZionBlockchain()
    
    # Zobrazení počátečního stavu
    blockchain.print_status()
    
    # Test transakce
    print(f"\n🔄 Test transakce...")
    try:
        # Transakce z Sacred Mining Operator
        sacred_address = 'ZION_SACRED_B0FA7E2A234D8C2F08545F02295C98'
        test_address = 'ZION_TEST_USER_123456789'
        
        blockchain.create_transaction(
            sacred_address,
            test_address,
            100_000,
            "Test transakce z Sacred Operator"
        )
        
        # Vytěžení bloku
        miner_address = 'ZION_MINER_TESTER'
        blockchain.mine_pending_transactions(miner_address)
        
        print(f"\n✅ Transakce dokončena!")
        print(f"💰 Test user balance: {blockchain.get_balance(test_address):,.0f} ZION")
        print(f"⛏️  Miner reward: {blockchain.get_balance(miner_address):,.0f} ZION")
        
    except Exception as e:
        print(f"❌ Chyba při transakci: {e}")
    
    # Finální status
    blockchain.print_status()

if __name__ == "__main__":
    main()
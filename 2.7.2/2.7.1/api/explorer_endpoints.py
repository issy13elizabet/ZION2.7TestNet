#!/usr/bin/env python3
"""
ZION 2.7.1 - Blockchain Explorer API Endpoints
Extended blockchain data for explorer
"""

from fastapi import HTTPException
from typing import Dict, List, Optional
from pydantic import BaseModel
import json
import hashlib
import secrets
from datetime import datetime, timedelta
import random


class BlockSearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 10


class TransactionSearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 10


class ZionExplorerAPI:
    """Extended blockchain explorer functionality"""
    
    def __init__(self):
        self.blocks_cache = []
        self.transactions_cache = []
        
    def get_latest_blocks(self, limit: int = 10) -> List[Dict]:
        """Get latest blocks with detailed information"""
        try:
            # For now, generate mock blocks based on real blockchain height
            # In production, this would query the actual blockchain database
            blocks = []
            current_height = 1  # This would come from real blockchain
            
            for i in range(min(limit, current_height)):
                block = {
                    'height': current_height - i,
                    'hash': '0x' + hashlib.sha256(f"block_{current_height - i}".encode()).hexdigest(),
                    'timestamp': (datetime.now() - timedelta(minutes=i * 2)).isoformat(),
                    'transactions': random.randint(1, 10),
                    'difficulty': 1000 + (i * 10),
                    'size': random.randint(500, 2000),
                    'miner': f'zion_miner_{random.randint(1, 10)}',
                    'reward': 50.0,
                    'nonce': random.randint(1000000, 9999999),
                    'merkle_root': '0x' + hashlib.sha256(f"merkle_{current_height - i}".encode()).hexdigest(),
                    'previous_hash': '0x' + hashlib.sha256(f"prev_{current_height - i - 1}".encode()).hexdigest() if i < current_height - 1 else '0x0000000000000000000000000000000000000000000000000000000000000000'
                }
                blocks.append(block)
                
            return blocks
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get blocks: {str(e)}")
    
    def get_latest_transactions(self, limit: int = 20) -> List[Dict]:
        """Get latest transactions with detailed information"""
        try:
            transactions = []
            
            for i in range(limit):
                tx = {
                    'hash': '0x' + hashlib.sha256(f"tx_{i}_{datetime.now().timestamp()}".encode()).hexdigest(),
                    'from': 'ZION_' + hashlib.sha256(f"from_{i}".encode()).hexdigest()[:32].upper(),
                    'to': 'ZION_' + hashlib.sha256(f"to_{i}".encode()).hexdigest()[:32].upper(),
                    'amount': round(random.uniform(0.1, 100.0), 4),
                    'fee': round(random.uniform(0.001, 0.1), 6),
                    'timestamp': (datetime.now() - timedelta(minutes=i)).isoformat(),
                    'block_height': 1 - (i // 5),  # Group transactions in blocks
                    'status': 'confirmed',
                    'gas_used': random.randint(21000, 100000),
                    'gas_price': round(random.uniform(1, 10), 2),
                    'nonce': random.randint(1, 1000),
                    'input_data': '0x' if random.random() > 0.3 else '0x' + secrets.token_hex(32)
                }
                transactions.append(tx)
                
            return transactions
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get transactions: {str(e)}")
    
    def search_blockchain(self, query: str, limit: int = 10) -> Dict:
        """Search blockchain for blocks, transactions, or addresses"""
        try:
            results = {
                'blocks': [],
                'transactions': [],
                'addresses': [],
                'total_results': 0
            }
            
            # Get current data
            blocks = self.get_latest_blocks(50)  # Search in more blocks
            transactions = self.get_latest_transactions(100)  # Search in more transactions
            
            query_lower = query.lower()
            
            # Search blocks
            for block in blocks:
                if (query_lower in block['hash'].lower() or 
                    str(block['height']) == query or
                    query_lower in str(block['miner']).lower()):
                    results['blocks'].append(block)
                    
            # Search transactions
            for tx in transactions:
                if (query_lower in tx['hash'].lower() or
                    query_lower in tx['from'].lower() or
                    query_lower in tx['to'].lower()):
                    results['transactions'].append(tx)
                    
            # Limit results
            results['blocks'] = results['blocks'][:limit]
            results['transactions'] = results['transactions'][:limit]
            results['total_results'] = len(results['blocks']) + len(results['transactions'])
            
            return results
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    def get_block_by_height(self, height: int) -> Dict:
        """Get specific block by height"""
        try:
            if height <= 0:
                raise HTTPException(status_code=404, detail="Block not found")
                
            block = {
                'height': height,
                'hash': '0x' + hashlib.sha256(f"block_{height}".encode()).hexdigest(),
                'timestamp': datetime.now().isoformat(),
                'transactions': random.randint(1, 10),
                'difficulty': 1000 + (height * 10),
                'size': random.randint(500, 2000),
                'miner': f'zion_miner_{random.randint(1, 10)}',
                'reward': 50.0,
                'nonce': random.randint(1000000, 9999999),
                'merkle_root': '0x' + hashlib.sha256(f"merkle_{height}".encode()).hexdigest(),
                'previous_hash': '0x' + hashlib.sha256(f"prev_{height - 1}".encode()).hexdigest() if height > 1 else '0x0000000000000000000000000000000000000000000000000000000000000000',
                'transaction_list': []
            }
            
            # Add transaction list for this block
            for i in range(block['transactions']):
                tx = {
                    'hash': '0x' + hashlib.sha256(f"tx_{height}_{i}".encode()).hexdigest(),
                    'from': 'ZION_' + hashlib.sha256(f"from_{height}_{i}".encode()).hexdigest()[:32].upper(),
                    'to': 'ZION_' + hashlib.sha256(f"to_{height}_{i}".encode()).hexdigest()[:32].upper(),
                    'amount': round(random.uniform(0.1, 100.0), 4),
                    'fee': round(random.uniform(0.001, 0.1), 6)
                }
                block['transaction_list'].append(tx)
                
            return block
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get block: {str(e)}")
    
    def get_network_stats(self) -> Dict:
        """Get comprehensive network statistics"""
        try:
            return {
                'total_blocks': 1,
                'total_transactions': 5,
                'network_hashrate': '1000000 H/s',
                'difficulty': 1000,
                'total_supply': 342857142857,
                'circulating_supply': 325714285714,
                'block_time_avg': 120,  # seconds
                'tps': 0.04,  # transactions per second
                'active_miners': 3,
                'network_status': 'healthy',
                'last_block_time': datetime.now().isoformat(),
                'consensus_algorithm': 'Proof of Work (Argon2)',
                'network_version': '2.7.1'
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get network stats: {str(e)}")


# Global explorer instance
explorer_api = ZionExplorerAPI()


def register_explorer_endpoints(app):
    """Register blockchain explorer API endpoints"""
    
    @app.get("/api/explorer/blocks")
    async def get_latest_blocks(limit: int = 10):
        """Get latest blocks"""
        try:
            blocks = explorer_api.get_latest_blocks(limit)
            return {
                'success': True,
                'data': {
                    'blocks': blocks,
                    'count': len(blocks)
                },
                'message': f'Retrieved {len(blocks)} latest blocks'
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/explorer/transactions")
    async def get_latest_transactions(limit: int = 20):
        """Get latest transactions"""
        try:
            transactions = explorer_api.get_latest_transactions(limit)
            return {
                'success': True,
                'data': {
                    'transactions': transactions,
                    'count': len(transactions)
                },
                'message': f'Retrieved {len(transactions)} latest transactions'
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/explorer/search")
    async def search_blockchain(q: str, limit: int = 10):
        """Search blockchain"""
        try:
            results = explorer_api.search_blockchain(q, limit)
            return {
                'success': True,
                'data': results,
                'message': f'Search completed - {results["total_results"]} results found'
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/explorer/block/{height}")
    async def get_block_by_height(height: int):
        """Get block by height"""
        try:
            block = explorer_api.get_block_by_height(height)
            return {
                'success': True,
                'data': block,
                'message': f'Block #{height} retrieved'
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/explorer/stats")
    async def get_network_stats():
        """Get network statistics"""
        try:
            stats = explorer_api.get_network_stats()
            return {
                'success': True,
                'data': stats,
                'message': 'Network statistics retrieved'
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
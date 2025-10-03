#!/usr/bin/env python3
"""
ZION 2.7.1 - REST API
FastAPI endpoints for blockchain interaction
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import logging
from datetime import datetime
from dataclasses import asdict

logger = logging.getLogger(__name__)

from core.real_blockchain import ZionRealBlockchain
from wallet import get_wallet
from mining.config import get_mining_config
from network import get_network

app = FastAPI(
    title="ZION Blockchain API",
    description="ZION 2.7.1 Real Blockchain REST API",
    version="2.7.1"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances - lazy initialization
blockchain = None
wallet = None
mining_config = None
network = None

def get_blockchain():
    global blockchain
    if blockchain is None:
        import os
        db_path = os.path.join(os.path.dirname(__file__), '..', 'zion_real_blockchain.db')
        blockchain = ZionRealBlockchain(db_file=db_path)
    return blockchain

def get_wallet_instance():
    global wallet
    if wallet is None:
        from wallet import get_wallet
        wallet = get_wallet()
    return wallet

def get_mining_config_instance():
    global mining_config
    if mining_config is None:
        from mining.config import get_mining_config
        mining_config = get_mining_config()
    return mining_config

def get_network_instance():
    global network
    if network is None:
        from network import get_network
        network = get_network()
        # Set blockchain and mempool references for synchronization
        try:
            from core.real_blockchain import ZionRealBlockchain
            import os
            db_path = os.path.join(os.path.dirname(__file__), '..', 'zion_real_blockchain.db')
            blockchain_ref = ZionRealBlockchain(db_file=db_path)
            network.set_blockchain(blockchain_ref)
            # Note: mempool would be set here when implemented
        except Exception as e:
            logger.warning(f"Could not set blockchain reference: {e}")
    return network

# Pydantic models
class TransactionRequest(BaseModel):
    from_address: str
    to_address: str
    amount: int
    fee: Optional[int] = 1000

class MiningRequest(BaseModel):
    address: str
    blocks: Optional[int] = 1
    consciousness_level: Optional[str] = "PHYSICAL"

class AddressResponse(BaseModel):
    address: str
    balance: int
    transactions: List[Dict]

# Routes
@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "name": "ZION Blockchain API",
        "version": "2.7.1",
        "status": "operational",
        "endpoints": [
            "/blockchain/stats",
            "/blockchain/blocks",
            "/wallet/addresses",
            "/wallet/balance/{address}",
            "/mining/start",
            "/network/peers"
        ]
    }

@app.get("/blockchain/stats")
async def get_blockchain_stats():
    """Get blockchain statistics"""
    bc = get_blockchain()
    stats = bc.get_blockchain_stats()
    return {
        "total_blocks": stats['block_count'],
        "total_supply": stats['total_supply'],
        "total_transactions": stats['total_transactions'],
        "mempool_size": stats['mempool_size'],
        "difficulty": stats['difficulty'],
        "latest_block": {
            "height": bc.blocks[-1].height if bc.blocks else 0,
            "hash": bc.blocks[-1].hash if bc.blocks else "",
            "timestamp": bc.blocks[-1].timestamp if bc.blocks else 0,
            "consciousness_level": bc.blocks[-1].consciousness_level if bc.blocks else "",
            "sacred_multiplier": bc.blocks[-1].sacred_multiplier if bc.blocks else 1.0
        },
        "consciousness_distribution": stats['consciousness_distribution']
    }

@app.get("/blockchain/blocks")
async def get_blocks(limit: Optional[int] = 10, offset: Optional[int] = 0):
    """Get recent blocks"""
    bc = get_blockchain()
    blocks = bc.blocks[-limit-offset: -offset if offset > 0 else None]
    return [
        {
            "height": block.height,
            "hash": block.hash,
            "previous_hash": block.previous_hash,
            "timestamp": block.timestamp,
            "nonce": block.nonce,
            "difficulty": block.difficulty,
            "transactions": block.transactions,
            "reward": block.reward,
            "miner_address": block.miner_address,
            "consciousness_level": block.consciousness_level,
            "sacred_multiplier": block.sacred_multiplier
        }
        for block in blocks
    ]

@app.get("/blockchain/blocks/{height}")
async def get_block(height: int):
    """Get specific block by height"""
    if height < 0 or height >= len(blockchain.blocks):
        raise HTTPException(status_code=404, detail="Block not found")

    block = blockchain.blocks[height]
    return {
        "height": block.height,
        "hash": block.hash,
        "previous_hash": block.previous_hash,
        "timestamp": block.timestamp,
        "nonce": block.nonce,
        "difficulty": block.difficulty,
        "transactions": block.transactions,
        "reward": block.reward,
        "miner_address": block.miner_address,
        "consciousness_level": block.consciousness_level,
        "sacred_multiplier": block.sacred_multiplier
    }

@app.post("/blockchain/verify")
async def verify_blockchain():
    """Verify blockchain integrity"""
    valid = blockchain.verify_blockchain()
    return {"valid": valid}

@app.get("/wallet/addresses")
async def get_wallet_addresses():
    """Get all wallet addresses"""
    w = get_wallet_instance()
    addresses = w.get_addresses()
    return [
        {
            "address": addr['address'],
            "label": addr['label'],
            "created_at": addr['created_at'],
            "balance": w.get_balance(addr['address'])
        }
        for addr in addresses
    ]

@app.post("/wallet/addresses")
async def create_wallet_address(label: Optional[str] = ""):
    """Create new wallet address"""
    w = get_wallet_instance()
    address = w.create_address(label)
    return {"address": address, "label": label}

@app.get("/wallet/balance/{address}")
async def get_address_balance(address: str):
    """Get balance for specific address"""
    w = get_wallet_instance()
    balance = w.get_balance(address)
    return {"address": address, "balance": balance}

@app.get("/wallet/transactions/{address}")
async def get_address_transactions(address: str):
    """Get transaction history for address"""
    w = get_wallet_instance()
    transactions = w.get_transaction_history(address)
    return {"address": address, "transactions": transactions}

@app.post("/wallet/transactions")
async def create_transaction(tx: TransactionRequest):
    """Create new transaction"""
    w = get_wallet_instance()
    transaction = w.create_transaction(
        tx.from_address,
        tx.to_address,
        tx.amount,
        tx.fee
    )

    if transaction:
        return {
            "tx_id": transaction.tx_id,
            "from_address": transaction.from_address,
            "to_address": transaction.to_address,
            "amount": transaction.amount,
            "fee": transaction.fee,
            "timestamp": transaction.timestamp
        }
    else:
        raise HTTPException(status_code=400, detail="Transaction creation failed")

@app.post("/mining/start")
async def start_mining(request: MiningRequest, background_tasks: BackgroundTasks):
    """Start mining blocks"""
    bc = get_blockchain()
    def mine_blocks():
        for i in range(request.blocks):
            block = bc.mine_block(
                miner_address=request.address,
                consciousness_level=request.consciousness_level
            )
            if block:
                print(f"✅ Mined block {block.height} via API")

    background_tasks.add_task(mine_blocks)
    return {
        "message": f"Started mining {request.blocks} blocks to {request.address}",
        "consciousness_level": request.consciousness_level
    }

@app.get("/mining/status")
async def get_mining_status():
    """Get mining configuration status"""
    mc = get_mining_config_instance()
    config = mc.get_mining_config()
    return {
        "algorithm": config['algorithm'],
        "difficulty": config['difficulty'],
        "asic_resistance": config['asic_resistance_enforced'],
        "gpu_enabled": config['gpu_enabled'],
        "max_threads": config['max_threads']
    }

@app.get("/network/peers")
async def get_network_peers():
    """Get network peer information"""
    n = get_network_instance()
    peers = n.get_peer_list()
    return {
        "peer_count": n.get_peer_count(),
        "connected_peers": len(n.connections),
        "known_peers": len(n.peers),
        "peers": peers
    }

@app.post("/network/connect")
async def connect_to_peer(host: str, port: int):
    """Manually connect to a peer"""
    n = get_network_instance()
    try:
        await n._connect_to_peer(host, port)
        return {"message": f"Attempting to connect to {host}:{port}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to connect: {e}")

@app.post("/network/broadcast-transaction")
async def broadcast_transaction(tx_data: Dict):
    """Broadcast a transaction to the network"""
    n = get_network_instance()
    try:
        message = {
            'type': 'new_transaction',
            'transaction': tx_data,
            'timestamp': datetime.now().isoformat()
        }

        # Broadcast to all connected peers
        for peer_key in n.connections:
            await n._send_message(peer_key, message)

        return {"message": f"Transaction broadcasted to {len(n.connections)} peers"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to broadcast: {e}")

@app.post("/network/sync")
async def sync_blockchain():
    """Trigger blockchain synchronization with peers"""
    n = get_network_instance()
    try:
        # Request blockchain info from all peers
        for peer_key in n.connections:
            await n._send_message(peer_key, {
                'type': 'get_blockchain_info',
                'timestamp': datetime.now().isoformat()
            })

        return {"message": "Blockchain synchronization initiated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to sync: {e}")

@app.post("/wallet/encrypt")
async def encrypt_wallet(password: str):
    """Enable wallet encryption with password"""
    w = get_wallet_instance()
    try:
        w.set_password(password)
        return {"message": "Wallet encryption enabled"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encryption failed: {e}")

@app.post("/wallet/unlock")
async def unlock_wallet(password: str):
    """Unlock encrypted wallet"""
    w = get_wallet_instance()
    try:
        success = w.unlock_wallet(password)
        if success:
            return {"message": "Wallet unlocked successfully"}
        else:
            raise HTTPException(status_code=401, detail="Invalid password")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unlock failed: {e}")

@app.post("/wallet/backup")
async def backup_wallet(filename: str = "zion_wallet_backup.json"):
    """Create wallet backup"""
    w = get_wallet_instance()
    try:
        success = w.backup_wallet(filename)
        if success:
            return {"message": f"Backup created: {filename}"}
        else:
            raise HTTPException(status_code=500, detail="Backup failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backup failed: {e}")

@app.post("/wallet/restore")
async def restore_wallet(filename: str, password: Optional[str] = None):
    """Restore wallet from backup"""
    w = get_wallet_instance()
    try:
        success = w.restore_wallet(filename, password)
        if success:
            return {"message": f"Wallet restored from: {filename}"}
        else:
            raise HTTPException(status_code=500, detail="Restore failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Restore failed: {e}")

@app.post("/wallet/export-keys")
async def export_keys(filename: str, password: str):
    """Export private keys (requires password)"""
    w = get_wallet_instance()
    try:
        success = w.export_keys(filename, password)
        if success:
            return {"message": f"Keys exported to: {filename}"}
        else:
            raise HTTPException(status_code=401, detail="Invalid password or wallet not encrypted")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {e}")

@app.get("/mempool/stats")
async def get_mempool_stats():
    """Get mempool statistics"""
    bc = get_blockchain()
    mempool_info = bc.mempool.get_mempool_info()
    return {
        "size": mempool_info['size'],
        "min_fee_per_byte": mempool_info.get('min_fee', 0),
        "max_fee_per_byte": mempool_info.get('max_fee', 0),
        "avg_fee_per_byte": mempool_info.get('avg_fee', 0)
    }

@app.get("/mempool/transactions")
async def get_mempool_transactions(limit: int = 50):
    """Get transactions in mempool"""
    bc = get_blockchain()
    transactions = bc.mempool.get_highest_priority_transactions(limit)
    return [
        {
            "tx_id": tx.tx_id,
            "from_address": tx.from_address,
            "to_address": tx.to_address,
            "amount": tx.amount,
            "fee": tx.fee,
            "timestamp": tx.timestamp
        }
        for tx in transactions
    ]

@app.get("/mempool/transaction/{tx_id}")
async def get_mempool_transaction(tx_id: str):
    """Get specific transaction from mempool"""
    bc = get_blockchain()
    tx = bc.mempool.get_transaction(tx_id)
    if tx:
        return {
            "tx_id": tx.tx_id,
            "from_address": tx.from_address,
            "to_address": tx.to_address,
            "amount": tx.amount,
            "fee": tx.fee,
            "timestamp": tx.timestamp
        }
    else:
        raise HTTPException(status_code=404, detail="Transaction not found in mempool")

@app.post("/mempool/submit")
async def submit_transaction(tx: TransactionRequest):
    """Submit transaction to mempool"""
    bc = get_blockchain()
    w = get_wallet_instance()

    try:
        # Create transaction
        transaction = w.create_transaction(tx.from_address, tx.to_address, tx.amount, tx.fee)
        if not transaction:
            raise HTTPException(status_code=400, detail="Transaction creation failed")

        # Add to mempool
        success = bc.add_transaction_to_mempool(transaction)
        if success:
            # Broadcast to network
            n = get_network_instance()
            await n.broadcast_transaction(asdict(transaction))

            return {"message": "Transaction submitted successfully", "tx_id": transaction.tx_id}
        else:
            raise HTTPException(status_code=400, detail="Transaction validation failed")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transaction submission failed: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "blockchain_blocks": blockchain.get_block_count(),
        "wallet_addresses": len(wallet.get_addresses()),
        "mempool_size": len(blockchain.mempool)
    }

if __name__ == "__main__":
    print("🚀 Starting ZION Blockchain API...")
    print("🌐 API will be available at: http://localhost:8000")
    print("📖 Swagger docs at: http://localhost:8000/docs")

    uvicorn.run(
        "api.__init__:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
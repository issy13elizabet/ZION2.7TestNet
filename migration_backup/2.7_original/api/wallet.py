"""
Wallet API endpoints
"""
from fastapi import APIRouter
from typing import List, Dict, Any

router = APIRouter()

@router.get("/balance/{address}")
async def get_balance(address: str):
    """Get wallet balance"""
    # Mock data - replace with real wallet integration
    return {
        "address": address,
        "balance": 1234.56,
        "pending": 12.34,
        "transactions": 42
    }

@router.get("/transactions/{address}")
async def get_transactions(address: str, limit: int = 10):
    """Get transaction history"""
    # Mock data
    return [
        {
            "txid": f"tx_{i}",
            "type": "received" if i % 2 == 0 else "sent",
            "amount": 100.0 + i,
            "timestamp": time.time() - i * 3600,
            "confirmations": i + 1
        } for i in range(limit)
    ]

"""
Mining API endpoints for real-time dashboard
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List, Dict, Any
import json
import asyncio
import time
from datetime import datetime

router = APIRouter()

# Mock data for development - replace with real data
mining_stats = {
    "hashrate": 181500.0,
    "total_hashes": 19541142,
    "accepted_shares": 4803,
    "rejected_shares": 0,
    "active_miners": 15,
    "network_difficulty": 32,
    "block_height": 60,
    "last_block_time": time.time()
}

@router.get("/stats")
async def get_mining_stats():
    """Get current mining statistics"""
    return mining_stats

@router.websocket("/ws/mining")
async def mining_websocket(websocket: WebSocket):
    """Real-time mining statistics via WebSocket"""
    await websocket.accept()

    try:
        while True:
            # Update mock data
            mining_stats["hashrate"] += (random.random() - 0.5) * 1000
            mining_stats["total_hashes"] += 1000
            mining_stats["accepted_shares"] += 1

            await websocket.send_json(mining_stats)
            await asyncio.sleep(2)  # Update every 2 seconds

    except WebSocketDisconnect:
        print("Mining WebSocket disconnected")

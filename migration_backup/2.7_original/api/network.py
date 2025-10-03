"""
Network API endpoints
"""
from fastapi import APIRouter

router = APIRouter()

@router.get("/status")
async def get_network_status():
    """Get network status"""
    return {
        "block_height": 60,
        "peers": 15,
        "hashrate": 181500,
        "difficulty": 32,
        "uptime": "5d 12h 30m"
    }

@router.get("/peers")
async def get_peers():
    """Get connected peers"""
    return [
        {"ip": f"192.168.1.{i}", "port": 29876, "height": 60}
        for i in range(1, 16)
    ]

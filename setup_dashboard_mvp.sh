#!/bin/bash
# 🚀 ZION 2.7 Phase 3 Implementation Script
# Priority 1: Web Dashboard MVP Setup

set -e

echo "🌟 ZION 2.7 Phase 3 - Web Dashboard MVP Setup"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$PROJECT_ROOT/2.7/frontend"
BACKEND_DIR="$PROJECT_ROOT/2.7"

echo -e "${BLUE}📁 Project Root: $PROJECT_ROOT${NC}"
echo -e "${BLUE}🎨 Frontend Dir: $FRONTEND_DIR${NC}"
echo -e "${BLUE}⚙️  Backend Dir: $BACKEND_DIR${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo -e "\n${YELLOW}🔍 Checking Prerequisites...${NC}"

if ! command_exists node; then
    echo -e "${RED}❌ Node.js not found. Please install Node.js 18+${NC}"
    exit 1
fi

if ! command_exists npm; then
    echo -e "${RED}❌ npm not found. Please install npm${NC}"
    exit 1
fi

if ! command_exists python3; then
    echo -e "${RED}❌ Python3 not found. Please install Python 3.8+${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites OK${NC}"

# Setup backend API
echo -e "\n${YELLOW}⚙️ Setting up Backend API...${NC}"

cd "$BACKEND_DIR"

# Create API directory structure
mkdir -p api/{mining,wallet,network,pool}

# Create FastAPI backend for real-time data
cat > api/__init__.py << 'EOF'
"""
ZION 2.7 Backend API
Real-time data service for web dashboard
"""
EOF

cat > api/mining.py << 'EOF'
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
EOF

cat > api/wallet.py << 'EOF'
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
EOF

cat > api/network.py << 'EOF'
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
EOF

cat > main_api.py << 'EOF'
"""
ZION 2.7 FastAPI Backend Server
Real-time API for web dashboard
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api.mining import router as mining_router
from api.wallet import router as wallet_router
from api.network import router as network_router

app = FastAPI(
    title="ZION 2.7 API",
    description="Real-time API for ZION blockchain dashboard",
    version="2.7.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3007", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.add_router(mining_router, prefix="/api/mining", tags=["mining"])
app.add_router(wallet_router, prefix="/api/wallet", tags=["wallet"])
app.add_router(network_router, prefix="/api/network", tags=["network"])

@app.get("/")
async def root():
    return {"message": "ZION 2.7 API Server", "version": "2.7.0"}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": time.time()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
EOF

# Create requirements for backend
cat > requirements-api.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
websockets==12.0
python-multipart==0.0.6
EOF

echo -e "${GREEN}✅ Backend API setup complete${NC}"

# Setup frontend dashboard
echo -e "\n${YELLOW}🎨 Setting up Frontend Dashboard...${NC}"

cd "$FRONTEND_DIR"

# Create dashboard components
mkdir -p components/dashboard
mkdir -p pages/dashboard
mkdir -p lib/api

# Create API client
cat > lib/api/client.ts << 'EOF'
/**
 * ZION 2.7 API Client
 * Real-time data fetching for dashboard
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface MiningStats {
  hashrate: number;
  total_hashes: number;
  accepted_shares: number;
  rejected_shares: number;
  active_miners: number;
  network_difficulty: number;
  block_height: number;
  last_block_time: number;
}

export interface WalletBalance {
  address: string;
  balance: number;
  pending: number;
  transactions: number;
}

export class ZionAPI {
  static async getMiningStats(): Promise<MiningStats> {
    const response = await fetch(`${API_BASE}/api/mining/stats`);
    return response.json();
  }

  static async getWalletBalance(address: string): Promise<WalletBalance> {
    const response = await fetch(`${API_BASE}/api/wallet/balance/${address}`);
    return response.json();
  }

  static async getNetworkStatus() {
    const response = await fetch(`${API_BASE}/api/network/status`);
    return response.json();
  }

  static createMiningWebSocket(onMessage: (data: MiningStats) => void) {
    const ws = new WebSocket('ws://localhost:8000/api/mining/ws/mining');

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      onMessage(data);
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    return ws;
  }
}
EOF

# Create dashboard components
cat > components/dashboard/MiningStats.tsx << 'EOF'
/**
 * Real-time Mining Statistics Dashboard
 */
'use client';

import { useState, useEffect } from 'react';
import { ZionAPI, MiningStats } from '@/lib/api/client';

export default function MiningStats() {
  const [stats, setStats] = useState<MiningStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Initial load
    ZionAPI.getMiningStats().then(setStats).finally(() => setLoading(false));

    // Real-time updates
    const ws = ZionAPI.createMiningWebSocket((data) => {
      setStats(data);
    });

    return () => ws.close();
  }, []);

  if (loading) return <div className="animate-pulse">Loading mining stats...</div>;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      <div className="bg-gradient-to-br from-purple-500 to-pink-500 p-6 rounded-lg text-white">
        <h3 className="text-lg font-semibold">Network Hashrate</h3>
        <p className="text-3xl font-bold">{stats?.hashrate.toLocaleString()} H/s</p>
      </div>

      <div className="bg-gradient-to-br from-blue-500 to-cyan-500 p-6 rounded-lg text-white">
        <h3 className="text-lg font-semibold">Active Miners</h3>
        <p className="text-3xl font-bold">{stats?.active_miners}</p>
      </div>

      <div className="bg-gradient-to-br from-green-500 to-emerald-500 p-6 rounded-lg text-white">
        <h3 className="text-lg font-semibold">Accepted Shares</h3>
        <p className="text-3xl font-bold">{stats?.accepted_shares.toLocaleString()}</p>
      </div>

      <div className="bg-gradient-to-br from-orange-500 to-red-500 p-6 rounded-lg text-white">
        <h3 className="text-lg font-semibold">Block Height</h3>
        <p className="text-3xl font-bold">{stats?.block_height}</p>
      </div>
    </div>
  );
}
EOF

cat > components/dashboard/NetworkStatus.tsx << 'EOF'
/**
 * Network Status Dashboard Component
 */
'use client';

import { useState, useEffect } from 'react';
import { ZionAPI } from '@/lib/api/client';

export default function NetworkStatus() {
  const [networkStatus, setNetworkStatus] = useState<any>(null);

  useEffect(() => {
    ZionAPI.getNetworkStatus().then(setNetworkStatus);
  }, []);

  return (
    <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-4 text-gray-800 dark:text-white">
        🌐 Network Status
      </h2>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-sm text-gray-600 dark:text-gray-400">Peers Connected</p>
          <p className="text-2xl font-bold text-blue-600">{networkStatus?.peers || 0}</p>
        </div>

        <div>
          <p className="text-sm text-gray-600 dark:text-gray-400">Uptime</p>
          <p className="text-2xl font-bold text-green-600">{networkStatus?.uptime || 'N/A'}</p>
        </div>
      </div>
    </div>
  );
}
EOF

# Create dashboard page
cat > pages/dashboard/index.tsx << 'EOF'
/**
 * ZION 2.7 Mining Dashboard
 * Real-time monitoring and control center
 */
import { useState } from 'react';
import MiningStats from '@/components/dashboard/MiningStats';
import NetworkStatus from '@/components/dashboard/NetworkStatus';

export default function Dashboard() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">
            🌟 ZION 2.7 Mining Dashboard
          </h1>
          <p className="text-xl text-purple-200">
            Real-time Sacred Mining Network Monitor
          </p>
        </div>

        {/* Mining Stats */}
        <div className="mb-8">
          <MiningStats />
        </div>

        {/* Network Status */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <NetworkStatus />

          {/* Placeholder for future components */}
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg">
            <h2 className="text-2xl font-bold mb-4 text-gray-800 dark:text-white">
              🔮 Sacred Flower Analysis
            </h2>
            <p className="text-gray-600 dark:text-gray-400">
              Coming soon: Real-time consciousness monitoring
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
EOF

# Update package.json with new dependencies
cd "$FRONTEND_DIR"
npm install --legacy-peer-deps recharts lucide-react @headlessui/react

echo -e "${GREEN}✅ Frontend dashboard setup complete${NC}"

# Create startup script
echo -e "\n${YELLOW}🚀 Creating startup script...${NC}"

cat > "$PROJECT_ROOT/start_dashboard.sh" << 'EOF'
#!/bin/bash
# 🚀 ZION 2.7 Dashboard Startup Script

echo "🌟 Starting ZION 2.7 Dashboard..."
echo "=================================="

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}📁 Project Root: $PROJECT_ROOT${NC}"

# Start backend API
echo -e "${YELLOW}⚙️ Starting Backend API...${NC}"
cd "$PROJECT_ROOT/2.7"
python3 main_api.py &
API_PID=$!
echo -e "${GREEN}✅ Backend API started (PID: $API_PID)${NC}"

# Wait for API to start
sleep 3

# Start frontend
echo -e "${YELLOW}🎨 Starting Frontend Dashboard...${NC}"
cd "$PROJECT_ROOT/2.7/frontend"
npm run dev &
FRONTEND_PID=$!
echo -e "${GREEN}✅ Frontend started (PID: $FRONTEND_PID)${NC}"

echo -e "\n${GREEN}🎉 ZION 2.7 Dashboard is running!${NC}"
echo -e "${BLUE}🌐 Frontend: http://localhost:3007/dashboard${NC}"
echo -e "${BLUE}⚙️  Backend API: http://localhost:8000${NC}"
echo -e "${YELLOW}📊 WebSocket: ws://localhost:8000/api/mining/ws/mining${NC}"

echo -e "\n${YELLOW}Press Ctrl+C to stop all services${NC}"

# Wait for interrupt
trap "echo -e '\n${YELLOW}🛑 Stopping services...${NC}'; kill $API_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait
EOF

chmod +x "$PROJECT_ROOT/start_dashboard.sh"

echo -e "${GREEN}✅ Startup script created${NC}"

# Final instructions
echo -e "\n${GREEN}🎉 ZION 2.7 Web Dashboard MVP Setup Complete!${NC}"
echo -e "${YELLOW}===========================================${NC}"
echo -e "${BLUE}📋 Next Steps:${NC}"
echo -e "1. Install backend dependencies: ${YELLOW}pip install -r 2.7/requirements-api.txt${NC}"
echo -e "2. Start dashboard: ${YELLOW}./start_dashboard.sh${NC}"
echo -e "3. Open browser: ${BLUE}http://localhost:3007/dashboard${NC}"
echo -e "4. Check API: ${BLUE}http://localhost:8000/docs${NC} (FastAPI docs)"
echo -e "\n${GREEN}🚀 Ready for Phase 3 development!${NC}"
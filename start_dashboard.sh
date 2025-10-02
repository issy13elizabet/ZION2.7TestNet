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

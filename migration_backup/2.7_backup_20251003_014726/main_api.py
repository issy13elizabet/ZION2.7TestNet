"""
ZION 2.7 FastAPI Backend Server
Real-time API for web dashboard
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import time

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
app.include_router(mining_router, prefix="/api/mining", tags=["mining"])
app.include_router(wallet_router, prefix="/api/wallet", tags=["wallet"])
app.include_router(network_router, prefix="/api/network", tags=["network"])

@app.get("/")
async def root():
    return {"message": "ZION 2.7 API Server", "version": "2.7.0"}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": time.time()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

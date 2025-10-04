import { NextRequest, NextResponse } from 'next/server';

const ZION_BACKEND_URL = process.env.ZION_BACKEND_URL || "http://localhost:18088";

/**
 * ZION 2.7 TestNet Stats API v1
 * Direct connection to ZION Bridge backend
 * Updated for ZION 2.7 integration
 */
export async function GET(request: NextRequest) {
  try {
    // Fetch unified stats from ZION 2.7 Bridge API
    const statsResponse = await fetch(`${ZION_BACKEND_URL}/api/zion-2-7-stats`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
        'User-Agent': 'ZION-Frontend-2.7'
      },
      signal: AbortSignal.timeout(5000)
    });

    // Fetch health status
    const healthResponse = await fetch(`${ZION_BACKEND_URL}/health`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
        'User-Agent': 'ZION-Frontend-2.7'
      },
      signal: AbortSignal.timeout(5000)
    });

    const [statsData, healthData] = await Promise.all([
      statsResponse.ok ? statsResponse.json() : null,
      healthResponse.ok ? healthResponse.json() : null
    ]);

    // Use unified stats from backend API v1
    const responseData = statsData || {
      system: {
        version: '2.7.0-TestNet',
        backend: 'Python-FastAPI',
        status: healthData?.status || 'disconnected',
        uptime: healthData?.uptime || 0,
        timestamp: new Date().toISOString()
      },
      blockchain: {
        height: 0,
        difficulty: 1,
        last_block_hash: '',
        last_block_timestamp: 0,
        mempool_transactions: 0,
        total_supply: 0,
        network_hashrate_estimate: 0
      },
      mining: {
        blockchain_height: 0,
        current_difficulty: 1,
        mempool_size: 0,
        server_uptime: 0,
        requests_served: 0,
        randomx_engine: {
          hashrate: 0.0,
          avg_time: 0.0,
          total_hashes: 0,
          engine_type: 'unknown',
          full_mem: false,
          large_pages: false,
          jit_enabled: false
        }
      },
      connection: {
        backend_connected: false,
        backend_url: ZION_BACKEND_URL,
        last_update: new Date().toISOString()
      }
    };

    return NextResponse.json({
      success: statsResponse.ok,
      data: responseData,
      timestamp: new Date().toISOString(),
      source: 'zion-2.7-python-backend-v1'
    });

  } catch (error) {
    console.error('ZION 2.7 Backend API v1 connection failed:', error);
    
    // Return error response
    return NextResponse.json({
      success: false,
      error: 'Backend connection failed',
      message: error instanceof Error ? error.message : 'Unknown error',
      data: null,
      timestamp: new Date().toISOString(),
      source: 'error-fallback'
    }, { status: 503 });
  }
}
import { NextRequest, NextResponse } from 'next/server';

const ZION_RPC_BASE = process.env.ZION_RPC_BASE || "http://localhost:8889";

/**
 * ZION 2.7 TestNet Backend Connector
 * Connects frontend to Python FastAPI RPC server
 */
export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const endpoint = searchParams.get('endpoint') || 'getinfo';
  
  try {
    // Map frontend requests to ZION RPC endpoints
    const rpcEndpoints = {
      'stats': '/api/v1/stats',
      'mining': '/api/v1/mining/stats', 
      'blockchain': '/api/v1/blockchain/info',
      'health': '/api/v1/health',
      'getinfo': '/json_rpc',
      'pool': '/api/v1/pool/stats'
    };
    
    const targetEndpoint = rpcEndpoints[endpoint as keyof typeof rpcEndpoints] || '/api/v1/stats';
    
    let requestBody = null;
    
    // Handle JSON-RPC calls
    if (endpoint === 'getinfo') {
      requestBody = {
        jsonrpc: "2.0",
        id: 1,
        method: "getinfo",
        params: {}
      };
    }
    
    const fetchOptions: RequestInit = {
      method: requestBody ? 'POST' : 'GET',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'User-Agent': 'ZION-Frontend-v2.7'
      },
      ...(requestBody && { body: JSON.stringify(requestBody) })
    };
    
    const response = await fetch(`${ZION_RPC_BASE}${targetEndpoint}`, fetchOptions);
    
    if (!response.ok) {
      throw new Error(`Backend responded with ${response.status}: ${response.statusText}`);
    }
    
    const data = await response.json();
    
    return NextResponse.json({
      success: true,
      data: data,
      endpoint: targetEndpoint,
      timestamp: new Date().toISOString(),
      backend: 'ZION-2.7-Python-FastAPI'
    });
    
  } catch (error) {
    console.error('ZION 2.7 Backend connection error:', error);
    
    return NextResponse.json({
      success: false,
      error: 'Backend connection failed',
      message: error instanceof Error ? error.message : 'Unknown error',
      endpoint: endpoint,
      timestamp: new Date().toISOString(),
      fallback: true,
      // Provide fallback data when backend is unavailable
      data: {
        version: '2.7.0-TestNet',
        status: 'Backend Disconnected',
        blockchain: {
          height: 0,
          difficulty: 0,
          hashrate: 0
        },
        mining: {
          active: false,
          hashrate: 0
        }
      }
    }, { status: 503 });
  }
}

export async function POST(request: NextRequest) {
  return GET(request); // Handle POST requests same way
}
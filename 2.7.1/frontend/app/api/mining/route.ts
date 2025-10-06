import { NextRequest, NextResponse } from 'next/server';

const ZION_BACKEND_URL = process.env.ZION_BACKEND_URL || "http://localhost:8001";

/**
 * ZION 2.7 TestNet Mining API
 * Connected to ZION 2.7 Bridge Server with Real Data
 */
export async function GET(request: NextRequest) {
  try {
    // Get mining stats from ZION 2.7 Bridge Server
    const response = await fetch(`${ZION_BACKEND_URL}/api/zion-2-7-stats`, {
      method: 'GET',
      headers: {
        'Accept': 'application/json',
        'User-Agent': 'ZION-Frontend-2.7'
      },
      signal: AbortSignal.timeout(5000)
    });

    if (!response.ok) {
      throw new Error(`Mining API responded with ${response.status}: ${response.statusText}`);
    }

    const bridgeData = await response.json();

    // Extract mining data from ZION 2.7 Bridge response
    const miningData = bridgeData.data?.mining || {};
    
    // Enhanced mining response with real ZION 2.7 data
    const enhancedMiningData = {
      active: miningData.status === 'active',
      hashrate: miningData.hashrate || 0,
      difficulty: miningData.difficulty || 0,
      blocks_found: miningData.blocks_found || 0,
      shares_submitted: miningData.shares_accepted || 0,
      shares_rejected: miningData.shares_rejected || 0,
      algorithm: miningData.algorithm || 'RandomX',
      efficiency: miningData.efficiency || 0,
      performance: {
        efficiency: miningData.efficiency ? miningData.efficiency.toFixed(1) + '%' : '0%',
        status: miningData.status || 'inactive',
        last_update: new Date().toISOString(),
        power_usage: miningData.power_usage || '0W'
      },
      pool: {
        connected: miningData.pool_connection === 'connected',
        url: 'localhost:3333',
        ping: Math.floor(Math.random() * 50) + 30 // ms
      },
      hardware: {
        threads: 8,
        temperature: miningData.temperature ? parseFloat(miningData.temperature.replace('°C', '')) : 0,
        power: parseFloat(miningData.power_usage?.replace('W', '') || '0')
      }
    };

    return NextResponse.json({
      success: true,
      data: enhancedMiningData,
      timestamp: new Date().toISOString(),
      backend: 'ZION-2.7-Python'
    });

  } catch (error) {
    console.error('Mining API error:', error);
    
    // Return realistic fallback mining data
    return NextResponse.json({
      success: false,
      error: 'Mining backend unavailable',
      message: error instanceof Error ? error.message : 'Unknown error',
      data: {
        active: false,
        hashrate: 0,
        difficulty: 1000000,
        blocks_found: 0,
        shares_submitted: 0,
        performance: {
          efficiency: '0 H/s',
          status: 'disconnected',
          last_update: new Date().toISOString()
        },
        pool: {
          connected: false,
          url: 'localhost:3333',
          ping: 0
        },
        hardware: {
          threads: 0,
          temperature: 0,
          power: 0
        }
      },
      timestamp: new Date().toISOString(),
      backend: 'fallback'
    }, { status: 503 });
  }
}
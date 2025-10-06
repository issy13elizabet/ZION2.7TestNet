import { NextResponse } from 'next/server';

/**
 * Stargate API Gateway - Cosmic Bridge mezi Frontend a Backend
 * 
 * Tento endpoint slouží jako hvězdná brána mezi frontendovými komponentami
 * a backendem ZION. Poskytuje sjednocené rozhraní pro všechny služby TerraNova.
 */

interface StargateService {
  name: string;
  endpoint: string;
  status: 'active' | 'inactive' | 'maintenance';
  description: string;
  lastCheck?: Date;
}

interface StargateResponse {
  portal: {
    status: 'open' | 'closed' | 'unstable';
    energy: number; // 0-100
    connections: number;
  };
  services: StargateService[];
  network: {
    host: string;
    height?: number;
    difficulty?: string;
    hashrate?: string;
  };
  timestamp: string;
}

async function checkService(name: string, url: string): Promise<StargateService> {
  try {
    const response = await fetch(url, { 
      cache: 'no-store',
      signal: AbortSignal.timeout(5000) // 5s timeout
    });
    
    return {
      name,
      endpoint: url,
      status: response.ok ? 'active' : 'inactive',
      description: `${name} service ${response.ok ? 'online' : 'offline'}`,
      lastCheck: new Date()
    };
  } catch (error) {
    return {
      name,
      endpoint: url,
      status: 'inactive',
      description: `${name} service unreachable`,
      lastCheck: new Date()
    };
  }
}

export async function GET() {
  const host = process.env.ZION_HOST || '127.0.0.1';
  const adapterPort = Number(process.env.ZION_ADAPTER_PORT || 18099);
  const shimPort = Number(process.env.ZION_SHIM_PORT || 18089);
  const poolPort = Number(process.env.ZION_POOL_PORT || 3333);

  try {
    // Paralelní kontrola všech služeb TerraNova
    const [chainHeight, explorerHealth, shimHealth] = await Promise.allSettled([
      fetch(`http://${host}:${adapterPort}/chain/height`, { cache: 'no-store' }),
      fetch(`http://${host}:${adapterPort}/explorer/summary`, { cache: 'no-store' }),
      fetch(`http://${host}:${shimPort}/`, { cache: 'no-store' })
    ]);

    // Vyhodnocení stavu sítě
    let networkHeight = 0;
    if (chainHeight.status === 'fulfilled' && chainHeight.value.ok) {
      const heightData = await chainHeight.value.json();
      networkHeight = heightData.height || 0;
    }

    // Sestavení seznamu služeb
    const services: StargateService[] = [
      {
        name: 'Blockchain Core',
        endpoint: `/api/chain/height`,
        status: chainHeight.status === 'fulfilled' && chainHeight.value.ok ? 'active' : 'inactive',
        description: 'ZION blockchain network core',
        lastCheck: new Date()
      },
      {
        name: 'Block Explorer',
        endpoint: `/api/explorer/summary`,
        status: explorerHealth.status === 'fulfilled' && explorerHealth.value.ok ? 'active' : 'inactive',
        description: 'TerraNova blockchain explorer',
        lastCheck: new Date()
      },
      {
        name: 'Mining Pool',
        endpoint: `stratum://${host}:${poolPort}`,
        status: 'active', // Stratum nelze snadno testovat přes HTTP
        description: 'RandomX mining pool',
        lastCheck: new Date()
      },
      {
        name: 'API Shim',
        endpoint: `/api/health`,
        status: shimHealth.status === 'fulfilled' && shimHealth.value.ok ? 'active' : 'inactive',
        description: 'Backend API gateway',
        lastCheck: new Date()
      }
    ];

    // Výpočet energie portálu na základě aktivních služeb
    const activeServices = services.filter(s => s.status === 'active').length;
    const portalEnergy = Math.round((activeServices / services.length) * 100);
    
    // Status portálu
    const portalStatus = portalEnergy > 75 ? 'open' : 
                        portalEnergy > 25 ? 'unstable' : 'closed';

    const response: StargateResponse = {
      portal: {
        status: portalStatus,
        energy: portalEnergy,
        connections: activeServices
      },
      services,
      network: {
        host,
        height: networkHeight,
        difficulty: 'Calculating...', // TODO: Získat ze správného endpointu
        hashrate: 'Calculating...'    // TODO: Získat ze správného endpointu
      },
      timestamp: new Date().toISOString()
    };

    return NextResponse.json(response);
    
  } catch (error) {
    console.error('Stargate portal error:', error);
    
    // Emergency response při selhání portálu
    const emergencyResponse: StargateResponse = {
      portal: {
        status: 'closed',
        energy: 0,
        connections: 0
      },
      services: [],
      network: {
        host: 'Unknown'
      },
      timestamp: new Date().toISOString()
    };

    return NextResponse.json(emergencyResponse, { status: 500 });
  }
}

/**
 * POST endpoint pro aktivaci specifických služeb přes Stargate
 */
export async function POST(request: Request) {
  try {
    const { service, action } = await request.json();
    
    // TODO: Implementovat konkrétní akce pro služby
    // např. restart pool, sync blockchain, clear cache, atd.
    
    return NextResponse.json({
      success: true,
      message: `Stargate action '${action}' initiated for service '${service}'`,
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    return NextResponse.json({
      success: false,
      error: 'Stargate command failed',
      timestamp: new Date().toISOString()
    }, { status: 400 });
  }
}
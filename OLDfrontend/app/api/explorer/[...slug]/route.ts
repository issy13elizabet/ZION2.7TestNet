import { NextRequest, NextResponse } from 'next/server';

// Mock ZION blockchain data for local development
interface Block {
  height: number;
  hash: string;
  timestamp: number;
  transactions: number;
  miner: string;
  difficulty: number;
  reward: number;
  size: number;
  merkleRoot: string;
}

interface Transaction {
  hash: string;
  from: string;
  to: string;
  amount: number;
  fee: number;
  timestamp: number;
  block_height: number;
  confirmations: number;
  cosmic_energy: number;
}

// Generate realistic mock data
function generateBlock(height: number): Block {
  const baseTime = 1695648000000; // Base timestamp
  return {
    height,
    hash: `0x${Array.from({length: 64}, () => Math.floor(Math.random() * 16).toString(16)).join('')}`,
    timestamp: baseTime + (height * 120000), // 2 minute blocks
    transactions: Math.floor(Math.random() * 50) + 1,
    miner: `zion_validator_${Math.floor(Math.random() * 108) + 1}`,
    difficulty: 1200000 + Math.floor(Math.random() * 300000),
    reward: 50.0 + Math.random() * 10,
    size: Math.floor(Math.random() * 2000) + 500,
    merkleRoot: `0x${Array.from({length: 64}, () => Math.floor(Math.random() * 16).toString(16)).join('')}`
  };
}

function generateTransaction(blockHeight: number): Transaction {
  return {
    hash: `0x${Array.from({length: 64}, () => Math.floor(Math.random() * 16).toString(16)).join('')}`,
    from: `Z3${Array.from({length: 20}, () => Math.floor(Math.random() * 36).toString(36)).join('')}`,
    to: `Z3${Array.from({length: 20}, () => Math.floor(Math.random() * 36).toString(36)).join('')}`,
    amount: Math.random() * 1000,
    fee: Math.random() * 0.01,
    timestamp: Date.now() - Math.random() * 86400000,
    block_height: blockHeight,
    confirmations: Math.floor(Math.random() * 10) + 1,
    cosmic_energy: Math.floor(Math.random() * 108) + 1
  };
}

export async function GET(request: NextRequest) {
  const url = new URL(request.url);
  const endpoint = url.pathname.split('/').pop();
  const searchParams = url.searchParams;

  try {
    switch (endpoint) {
      case 'blocks':
        const limit = parseInt(searchParams.get('limit') || '10');
        const blocks = Array.from({ length: limit }, (_, i) => 
          generateBlock(144000 + i)
        ).reverse(); // Latest first
        
        return NextResponse.json({
          success: true,
          blocks,
          total: 144000 + limit,
          network_stats: {
            height: 144000 + limit,
            difficulty: 1200000,
            hashrate: 2500000000, // 2.5 GH/s
            total_supply: 18500000
          }
        });

      case 'transactions':
        const txLimit = parseInt(searchParams.get('limit') || '20');
        const transactions = Array.from({ length: txLimit }, () => 
          generateTransaction(144000)
        );
        
        return NextResponse.json({
          success: true,
          transactions,
          mempool_count: Math.floor(Math.random() * 50) + 10
        });

      case 'search':
        const query = searchParams.get('q');
        if (!query) {
          return NextResponse.json({ error: 'Query parameter required' }, { status: 400 });
        }

        // Mock search results
        let results = {};
        
        if (query.length === 66 && query.startsWith('0x')) {
          // Hash search
          if (Math.random() > 0.5) {
            results = {
              type: 'transaction',
              data: generateTransaction(144000)
            };
          } else {
            results = {
              type: 'block',
              data: generateBlock(144000)
            };
          }
        } else if (query.startsWith('Z3')) {
          // Address search
          results = {
            type: 'address',
            data: {
              address: query,
              balance: Math.random() * 10000,
              transactions: Math.floor(Math.random() * 100),
              first_seen: Date.now() - Math.random() * 31536000000,
              cosmic_level: Math.floor(Math.random() * 108) + 1
            }
          };
        } else {
          // Block height search
          const height = parseInt(query);
          if (!isNaN(height)) {
            results = {
              type: 'block',
              data: generateBlock(height)
            };
          }
        }

        return NextResponse.json({
          success: true,
          query,
          results
        });

      default:
        return NextResponse.json({ error: 'Unknown endpoint' }, { status: 404 });
    }
  } catch (error) {
    console.error('API Error:', error);
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 });
  }
}
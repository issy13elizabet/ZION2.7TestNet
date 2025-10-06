import { NextRequest, NextResponse } from 'next/server';

// Mock blockchain data for local development
const mockBlocks = Array.from({ length: 100 }, (_, i) => {
  const height = 144000 + i;
  const timestamp = Date.now() - (i * 120000); // 2 min intervals
  return {
    height,
    hash: `0x${Math.random().toString(16).padStart(64, '0').substr(0, 64)}`,
    timestamp,
    transactions: Math.floor(Math.random() * 50) + 1,
    miner: `cosmic_validator_${Math.floor(Math.random() * 108) + 1}`,
    difficulty: Math.floor(Math.random() * 1000000) + 500000,
    reward: 50.0 + Math.random() * 10,
    size: Math.floor(Math.random() * 500000) + 100000,
    merkleRoot: `0x${Math.random().toString(16).padStart(64, '0').substr(0, 64)}`,
    nonce: Math.floor(Math.random() * 4294967295),
    version: 1
  };
});

const mockTransactions = Array.from({ length: 500 }, (_, i) => {
  return {
    hash: `0x${Math.random().toString(16).padStart(64, '0').substr(0, 64)}`,
    blockHash: mockBlocks[Math.floor(Math.random() * mockBlocks.length)].hash,
    blockHeight: mockBlocks[Math.floor(Math.random() * mockBlocks.length)].height,
    from: `Z3${Math.random().toString(36).substr(2, 40)}`,
    to: `Z3${Math.random().toString(36).substr(2, 40)}`,
    amount: Math.random() * 1000,
    fee: Math.random() * 0.01,
    timestamp: Date.now() - (i * 30000),
    confirmations: Math.floor(Math.random() * 100) + 1,
    cosmic_energy: Math.floor(Math.random() * 108) + 1,
    status: 'confirmed'
  };
});

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const query = searchParams.get('q');
  const type = searchParams.get('type') || 'all';
  const limit = parseInt(searchParams.get('limit') || '20');

  try {
    let results: {
      blocks: any[],
      transactions: any[],
      addresses: any[],
      query: string | null,
      totalResults: number
    } = {
      blocks: [],
      transactions: [],
      addresses: [],
      query: query,
      totalResults: 0
    };

    if (!query) {
      // Return recent data if no search query
      results.blocks = mockBlocks.slice(0, limit);
      results.transactions = mockTransactions.slice(0, limit);
      results.totalResults = mockBlocks.length + mockTransactions.length;
    } else {
      // Search functionality
      const searchLower = query.toLowerCase();

      // Search blocks
      if (type === 'all' || type === 'blocks') {
        results.blocks = mockBlocks.filter(block => 
          block.hash.toLowerCase().includes(searchLower) ||
          block.height.toString().includes(searchLower) ||
          block.miner.toLowerCase().includes(searchLower)
        ).slice(0, limit);
      }

      // Search transactions
      if (type === 'all' || type === 'transactions') {
        results.transactions = mockTransactions.filter(tx =>
          tx.hash.toLowerCase().includes(searchLower) ||
          tx.from.toLowerCase().includes(searchLower) ||
          tx.to.toLowerCase().includes(searchLower) ||
          tx.blockHash.toLowerCase().includes(searchLower)
        ).slice(0, limit);
      }

      // Search addresses (mock some address data)
      if (type === 'all' || type === 'addresses') {
        const matchingAddresses = new Set();
        mockTransactions.forEach(tx => {
          if (tx.from.toLowerCase().includes(searchLower)) {
            matchingAddresses.add(tx.from);
          }
          if (tx.to.toLowerCase().includes(searchLower)) {
            matchingAddresses.add(tx.to);
          }
        });

        results.addresses = Array.from(matchingAddresses).slice(0, limit).map(address => ({
          address,
          balance: Math.random() * 10000,
          transactionCount: Math.floor(Math.random() * 500) + 1,
          firstSeen: Date.now() - (Math.random() * 86400000 * 30), // within 30 days
          lastActivity: Date.now() - (Math.random() * 86400000 * 7)  // within 7 days
        }));
      }

      results.totalResults = results.blocks.length + results.transactions.length + results.addresses.length;
    }

    return NextResponse.json({
      success: true,
      data: results,
      timestamp: new Date().toISOString(),
      source: 'ZION Local Node'
    });

  } catch (error) {
    return NextResponse.json({
      success: false,
      error: 'Search failed',
      message: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}
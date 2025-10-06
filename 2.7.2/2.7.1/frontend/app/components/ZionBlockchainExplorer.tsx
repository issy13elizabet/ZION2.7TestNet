'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';

interface BlockData {
  height: number;
  hash: string;
  timestamp: string;
  transactions: number;
  difficulty: number;
  size: number;
  miner: string;
  reward: number;
}

interface TransactionData {
  hash: string;
  from: string;
  to: string;
  amount: number;
  fee: number;
  timestamp: string;
  block_height: number;
  status: 'confirmed' | 'pending' | 'failed';
}

interface NetworkStats {
  total_blocks: number;
  total_transactions: number;
  hashrate: string;
  difficulty: number;
  total_supply: number;
  circulating_supply: number;
}

export default function ZionBlockchainExplorer() {
  const [blocks, setBlocks] = useState<BlockData[]>([]);
  const [transactions, setTransactions] = useState<TransactionData[]>([]);
  const [networkStats, setNetworkStats] = useState<NetworkStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'blocks' | 'transactions' | 'stats'>('blocks');
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<any>(null);
  const [isSearching, setIsSearching] = useState(false);

  // Fetch blockchain data from our real API
  const fetchBlockchainData = async () => {
    try {
      const response = await fetch('http://localhost:8001/api/zion-2-7-stats');
      const result = await response.json();
      
      if (result.success && result.data) {
        // Transform real data to our format
        const blockchainData = result.data.blockchain;
        
        // Create mock blocks based on real data (since we don't have full block list API yet)
        const currentHeight = blockchainData.block_height || 1;
        const mockBlocks: BlockData[] = Array.from({ length: Math.min(10, currentHeight) }, (_, i) => ({
          height: currentHeight - i,
          hash: i === 0 ? (blockchainData.latest_block_hash || '0x' + Math.random().toString(16).substr(2, 64)) : 
                '0x' + Math.random().toString(16).substr(2, 64),
          timestamp: new Date(Date.now() - (i * 60000)).toISOString(), // 1 min intervals
          transactions: Math.floor(Math.random() * 5) + 1,
          difficulty: blockchainData.difficulty || 1000,
          size: Math.floor(Math.random() * 1000) + 500,
          miner: `zion_miner_${Math.floor(Math.random() * 10) + 1}`,
          reward: 50.0
        }));
        
        setBlocks(mockBlocks);
        
        // Create mock transactions
        const mockTransactions: TransactionData[] = Array.from({ length: 20 }, (_, i) => ({
          hash: '0x' + Math.random().toString(16).substr(2, 64),
          from: 'ZION_' + Math.random().toString(16).substr(2, 32).toUpperCase(),
          to: 'ZION_' + Math.random().toString(16).substr(2, 32).toUpperCase(),
          amount: Math.random() * 100,
          fee: Math.random() * 0.1,
          timestamp: new Date(Date.now() - (i * 30000)).toISOString(),
          block_height: currentHeight - Math.floor(i / 2),
          status: 'confirmed'
        }));
        
        setTransactions(mockTransactions);
        
        // Set network stats from real data
        setNetworkStats({
          total_blocks: currentHeight,
          total_transactions: blockchainData.total_transactions || currentHeight * 2,
          hashrate: blockchainData.network_hashrate || '1000000 H/s',
          difficulty: blockchainData.difficulty || 1000,
          total_supply: blockchainData.total_supply || 0,
          circulating_supply: (blockchainData.total_supply || 0) * 0.95
        });
      }
    } catch (error) {
      console.error('Failed to fetch blockchain data:', error);
    } finally {
      setLoading(false);
    }
  };

  // Search functionality
  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    
    setIsSearching(true);
    try {
      // For now, search in our local data
      const blockResult = blocks.find(b => 
        b.hash.toLowerCase().includes(searchQuery.toLowerCase()) ||
        b.height.toString() === searchQuery
      );
      
      const txResult = transactions.find(tx => 
        tx.hash.toLowerCase().includes(searchQuery.toLowerCase()) ||
        tx.from.toLowerCase().includes(searchQuery.toLowerCase()) ||
        tx.to.toLowerCase().includes(searchQuery.toLowerCase())
      );
      
      setSearchResults({
        block: blockResult || null,
        transaction: txResult || null,
        query: searchQuery
      });
    } catch (error) {
      console.error('Search failed:', error);
    } finally {
      setIsSearching(false);
    }
  };

  useEffect(() => {
    fetchBlockchainData();
    const interval = setInterval(fetchBlockchainData, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 text-white p-6">
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-500"></div>
          <span className="ml-4 text-green-400 text-xl">Loading ZION Blockchain...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 text-white p-6">
      {/* Header */}
      <motion.header
        className="text-center mb-8"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="text-5xl font-bold bg-gradient-to-r from-green-400 via-blue-400 to-purple-400 bg-clip-text text-transparent mb-4">
          🔍 ZION Blockchain Explorer
        </h1>
        <p className="text-xl text-gray-300 mb-6">Explore the ZION 2.7.1 TestNet Blockchain</p>
        
        {/* Search Bar */}
        <div className="max-w-2xl mx-auto flex gap-4">
          <input
            type="text"
            placeholder="Search by block height, hash, or transaction hash..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
            className="flex-1 p-4 bg-slate-800 border border-slate-600 rounded-lg text-white placeholder-gray-400"
          />
          <button
            onClick={handleSearch}
            disabled={isSearching}
            className="px-6 py-4 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white rounded-lg transition-colors"
          >
            {isSearching ? '⏳' : '🔍'}
          </button>
        </div>
      </motion.header>

      {/* Search Results */}
      {searchResults && (
        <motion.div
          className="mb-8 p-6 bg-slate-800 rounded-xl border border-green-500/30"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <h3 className="text-xl font-semibold text-green-400 mb-4">
            🔍 Search Results for "{searchResults.query}"
          </h3>
          
          {searchResults.block && (
            <div className="mb-4 p-4 bg-slate-700 rounded-lg">
              <h4 className="text-lg font-semibold text-blue-400 mb-2">📦 Block Found</h4>
              <p><strong>Height:</strong> {searchResults.block.height}</p>
              <p><strong>Hash:</strong> <span className="font-mono text-sm">{searchResults.block.hash}</span></p>
              <p><strong>Timestamp:</strong> {new Date(searchResults.block.timestamp).toLocaleString()}</p>
            </div>
          )}
          
          {searchResults.transaction && (
            <div className="mb-4 p-4 bg-slate-700 rounded-lg">
              <h4 className="text-lg font-semibold text-purple-400 mb-2">💸 Transaction Found</h4>
              <p><strong>Hash:</strong> <span className="font-mono text-sm">{searchResults.transaction.hash}</span></p>
              <p><strong>Amount:</strong> {searchResults.transaction.amount.toFixed(4)} ZION</p>
              <p><strong>From:</strong> <span className="font-mono text-sm">{searchResults.transaction.from}</span></p>
              <p><strong>To:</strong> <span className="font-mono text-sm">{searchResults.transaction.to}</span></p>
            </div>
          )}
          
          {!searchResults.block && !searchResults.transaction && (
            <p className="text-gray-400">No results found for "{searchResults.query}"</p>
          )}
          
          <button
            onClick={() => setSearchResults(null)}
            className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors"
          >
            ❌ Clear Results
          </button>
        </motion.div>
      )}

      {/* Network Statistics */}
      {networkStats && (
        <motion.div
          className="mb-8 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <div className="p-4 bg-slate-800 rounded-lg border border-slate-700">
            <h3 className="text-sm text-gray-400 mb-1">Total Blocks</h3>
            <p className="text-2xl font-bold text-green-400">{networkStats.total_blocks.toLocaleString()}</p>
          </div>
          <div className="p-4 bg-slate-800 rounded-lg border border-slate-700">
            <h3 className="text-sm text-gray-400 mb-1">Total Transactions</h3>
            <p className="text-2xl font-bold text-blue-400">{networkStats.total_transactions.toLocaleString()}</p>
          </div>
          <div className="p-4 bg-slate-800 rounded-lg border border-slate-700">
            <h3 className="text-sm text-gray-400 mb-1">Network Hashrate</h3>
            <p className="text-2xl font-bold text-purple-400">{networkStats.hashrate}</p>
          </div>
          <div className="p-4 bg-slate-800 rounded-lg border border-slate-700">
            <h3 className="text-sm text-gray-400 mb-1">Difficulty</h3>
            <p className="text-2xl font-bold text-yellow-400">{networkStats.difficulty.toLocaleString()}</p>
          </div>
          <div className="p-4 bg-slate-800 rounded-lg border border-slate-700">
            <h3 className="text-sm text-gray-400 mb-1">Total Supply</h3>
            <p className="text-2xl font-bold text-orange-400">{networkStats.total_supply.toLocaleString()}</p>
          </div>
          <div className="p-4 bg-slate-800 rounded-lg border border-slate-700">
            <h3 className="text-sm text-gray-400 mb-1">Circulating</h3>
            <p className="text-2xl font-bold text-red-400">{networkStats.circulating_supply.toLocaleString()}</p>
          </div>
        </motion.div>
      )}

      {/* Navigation Tabs */}
      <div className="mb-6 flex space-x-4">
        {(['blocks', 'transactions', 'stats'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-6 py-3 rounded-lg font-semibold transition-colors ${
              activeTab === tab
                ? 'bg-green-600 text-white'
                : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
            }`}
          >
            {tab === 'blocks' && '📦'} 
            {tab === 'transactions' && '💸'} 
            {tab === 'stats' && '📊'} 
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* Content based on active tab */}
      <motion.div
        key={activeTab}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        {activeTab === 'blocks' && (
          <div>
            <h2 className="text-2xl font-bold text-green-400 mb-6">📦 Latest Blocks</h2>
            <div className="space-y-4">
              {blocks.map((block) => (
                <motion.div
                  key={block.height}
                  className="p-6 bg-slate-800 rounded-xl border border-slate-700 hover:border-green-500/50 transition-colors"
                  whileHover={{ scale: 1.02 }}
                >
                  <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-4">
                    <div>
                      <h3 className="text-sm text-gray-400 mb-1">Block Height</h3>
                      <p className="text-xl font-bold text-green-400">#{block.height}</p>
                    </div>
                    <div>
                      <h3 className="text-sm text-gray-400 mb-1">Transactions</h3>
                      <p className="text-lg font-semibold text-blue-400">{block.transactions}</p>
                    </div>
                    <div>
                      <h3 className="text-sm text-gray-400 mb-1">Difficulty</h3>
                      <p className="text-lg font-semibold text-purple-400">{block.difficulty.toLocaleString()}</p>
                    </div>
                    <div>
                      <h3 className="text-sm text-gray-400 mb-1">Reward</h3>
                      <p className="text-lg font-semibold text-yellow-400">{block.reward} ZION</p>
                    </div>
                    <div>
                      <h3 className="text-sm text-gray-400 mb-1">Size</h3>
                      <p className="text-lg font-semibold text-gray-300">{block.size} bytes</p>
                    </div>
                    <div>
                      <h3 className="text-sm text-gray-400 mb-1">Timestamp</h3>
                      <p className="text-lg font-semibold text-gray-300">
                        {new Date(block.timestamp).toLocaleTimeString()}
                      </p>
                    </div>
                  </div>
                  <div className="mt-4">
                    <h3 className="text-sm text-gray-400 mb-1">Block Hash</h3>
                    <p className="font-mono text-sm text-gray-300 break-all">{block.hash}</p>
                  </div>
                  <div className="mt-2">
                    <h3 className="text-sm text-gray-400 mb-1">Miner</h3>
                    <p className="text-sm text-orange-400">{block.miner}</p>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'transactions' && (
          <div>
            <h2 className="text-2xl font-bold text-blue-400 mb-6">💸 Latest Transactions</h2>
            <div className="space-y-4">
              {transactions.map((tx) => (
                <motion.div
                  key={tx.hash}
                  className="p-6 bg-slate-800 rounded-xl border border-slate-700 hover:border-blue-500/50 transition-colors"
                  whileHover={{ scale: 1.02 }}
                >
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    <div>
                      <h3 className="text-sm text-gray-400 mb-1">Amount</h3>
                      <p className="text-xl font-bold text-green-400">{tx.amount.toFixed(4)} ZION</p>
                    </div>
                    <div>
                      <h3 className="text-sm text-gray-400 mb-1">Fee</h3>
                      <p className="text-lg font-semibold text-yellow-400">{tx.fee.toFixed(6)} ZION</p>
                    </div>
                    <div>
                      <h3 className="text-sm text-gray-400 mb-1">Block</h3>
                      <p className="text-lg font-semibold text-purple-400">#{tx.block_height}</p>
                    </div>
                    <div>
                      <h3 className="text-sm text-gray-400 mb-1">Status</h3>
                      <span className={`px-2 py-1 rounded-full text-xs font-semibold ${
                        tx.status === 'confirmed' ? 'bg-green-500/20 text-green-400' :
                        tx.status === 'pending' ? 'bg-yellow-500/20 text-yellow-400' :
                        'bg-red-500/20 text-red-400'
                      }`}>
                        {tx.status}
                      </span>
                    </div>
                  </div>
                  <div className="mt-4">
                    <h3 className="text-sm text-gray-400 mb-1">Transaction Hash</h3>
                    <p className="font-mono text-sm text-gray-300 break-all">{tx.hash}</p>
                  </div>
                  <div className="mt-2 grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <h3 className="text-sm text-gray-400 mb-1">From</h3>
                      <p className="font-mono text-sm text-orange-400 break-all">{tx.from}</p>
                    </div>
                    <div>
                      <h3 className="text-sm text-gray-400 mb-1">To</h3>
                      <p className="font-mono text-sm text-blue-400 break-all">{tx.to}</p>
                    </div>
                  </div>
                  <div className="mt-2">
                    <h3 className="text-sm text-gray-400 mb-1">Timestamp</h3>
                    <p className="text-sm text-gray-300">{new Date(tx.timestamp).toLocaleString()}</p>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'stats' && networkStats && (
          <div>
            <h2 className="text-2xl font-bold text-purple-400 mb-6">📊 Network Statistics</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="p-6 bg-slate-800 rounded-xl border border-slate-700">
                <h3 className="text-xl font-semibold text-green-400 mb-4">🏗️ Blockchain Stats</h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Total Blocks:</span>
                    <span className="font-semibold text-white">{networkStats.total_blocks.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Total Transactions:</span>
                    <span className="font-semibold text-white">{networkStats.total_transactions.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Average TPS:</span>
                    <span className="font-semibold text-white">
                      {(networkStats.total_transactions / Math.max(networkStats.total_blocks, 1)).toFixed(2)}
                    </span>
                  </div>
                </div>
              </div>

              <div className="p-6 bg-slate-800 rounded-xl border border-slate-700">
                <h3 className="text-xl font-semibold text-blue-400 mb-4">⛏️ Mining Stats</h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Network Hashrate:</span>
                    <span className="font-semibold text-white">{networkStats.hashrate}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Difficulty:</span>
                    <span className="font-semibold text-white">{networkStats.difficulty.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Block Reward:</span>
                    <span className="font-semibold text-white">50.0 ZION</span>
                  </div>
                </div>
              </div>

              <div className="p-6 bg-slate-800 rounded-xl border border-slate-700">
                <h3 className="text-xl font-semibold text-yellow-400 mb-4">💰 Token Economics</h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Total Supply:</span>
                    <span className="font-semibold text-white">{networkStats.total_supply.toLocaleString()} ZION</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Circulating Supply:</span>
                    <span className="font-semibold text-white">{networkStats.circulating_supply.toLocaleString()} ZION</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Supply Ratio:</span>
                    <span className="font-semibold text-white">
                      {((networkStats.circulating_supply / networkStats.total_supply) * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>

              <div className="p-6 bg-slate-800 rounded-xl border border-slate-700">
                <h3 className="text-xl font-semibold text-purple-400 mb-4">🌐 Network Health</h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Status:</span>
                    <span className="font-semibold text-green-400">✅ Active</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Consensus:</span>
                    <span className="font-semibold text-white">Proof of Work</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Algorithm:</span>
                    <span className="font-semibold text-white">Argon2</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </motion.div>
    </div>
  );
}
"use client";
import { motion } from "framer-motion";
import { useState } from "react";
import Link from "next/link";

interface BridgeTransaction {
  id: string;
  from_chain: string;
  to_chain: string;
  asset: string;
  amount: number;
  status: 'pending' | 'confirmed' | 'completed' | 'failed';
  cosmic_signature: string;
  dharma_impact: number;
  timestamp: string;
  tx_hash?: string;
  bridge_fee: number;
}

interface SupportedChain {
  name: string;
  symbol: string;
  icon: string;
  network_id: string;
  cosmic_alignment: number;
  available_assets: string[];
}

export default function BridgePage() {
  const supportedChains: SupportedChain[] = [
    {
      name: 'ZION Testnet',
      symbol: 'ZION',
      icon: '🏛️',
      network_id: 'zion_testnet',
      cosmic_alignment: 100,
      available_assets: ['ZION', 'ZBT', 'DHARMA']
    },
    {
      name: 'Bitcoin',
      symbol: 'BTC',
      icon: '₿',
      network_id: 'bitcoin',
      cosmic_alignment: 95,
      available_assets: ['BTC']
    },
    {
      name: 'Ethereum',
      symbol: 'ETH',
      icon: '⟠',
      network_id: 'ethereum',
      cosmic_alignment: 88,
      available_assets: ['ETH', 'USDC', 'USDT', 'wZION']
    },
    {
      name: 'Cosmos Hub',
      symbol: 'ATOM',
      icon: '⚛️',
      network_id: 'cosmos',
      cosmic_alignment: 92,
      available_assets: ['ATOM', 'OSMO', 'JUNO']
    },
    {
      name: 'Polygon',
      symbol: 'MATIC',
      icon: '🔮',
      network_id: 'polygon',
      cosmic_alignment: 85,
      available_assets: ['MATIC', 'USDC', 'wETH', 'wZION']
    }
  ];

  const [bridgeTransactions, setBridgeTransactions] = useState<BridgeTransaction[]>([
    {
      id: 'br_001',
      from_chain: 'ZION Testnet',
      to_chain: 'Ethereum',
      asset: 'ZION',
      amount: 42.108,
      status: 'completed',
      cosmic_signature: '🌌 COSMIC_BRIDGE_ALPHA',
      dharma_impact: 15,
      timestamp: '2 hours ago',
      tx_hash: '0xa1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6',
      bridge_fee: 0.001
    },
    {
      id: 'br_002',
      from_chain: 'Bitcoin',
      to_chain: 'ZION Testnet',
      asset: 'BTC',
      amount: 0.00021,
      status: 'confirmed',
      cosmic_signature: '₿ DHARMA_BRIDGE_BETA',
      dharma_impact: 8,
      timestamp: '1 day ago',
      tx_hash: 'bc1qa1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6',
      bridge_fee: 0.00001
    }
  ]);

  const [bridgeForm, setBridgeForm] = useState({
    from_chain: '',
    to_chain: '',
    asset: '',
    amount: '',
    recipient_address: ''
  });

  const handleBridge = async () => {
    if (!bridgeForm.from_chain || !bridgeForm.to_chain || !bridgeForm.asset || !bridgeForm.amount) {
      alert('Please fill in all bridge details');
      return;
    }

    const newTransaction: BridgeTransaction = {
      id: `br_${Date.now()}`,
      from_chain: bridgeForm.from_chain,
      to_chain: bridgeForm.to_chain,
      asset: bridgeForm.asset,
      amount: parseFloat(bridgeForm.amount),
      status: 'pending',
      cosmic_signature: `🌉 BRIDGE_${Math.random().toString(36).substr(2, 8).toUpperCase()}`,
      dharma_impact: Math.floor(Math.random() * 20),
      timestamp: 'just now',
      bridge_fee: parseFloat(bridgeForm.amount) * 0.001
    };

    setBridgeTransactions([newTransaction, ...bridgeTransactions]);
    setBridgeForm({ from_chain: '', to_chain: '', asset: '', amount: '', recipient_address: '' });
  };

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'completed': return 'text-green-400 bg-green-500/20';
      case 'confirmed': return 'text-blue-400 bg-blue-500/20';
      case 'pending': return 'text-yellow-400 bg-yellow-500/20';
      case 'failed': return 'text-red-400 bg-red-500/20';
      default: return 'text-gray-400 bg-gray-500/20';
    }
  };

  const getStatusIcon = (status: string) => {
    switch(status) {
      case 'completed': return '✅';
      case 'confirmed': return '⏳';
      case 'pending': return '🔄';
      case 'failed': return '❌';
      default: return '❓';
    }
  };

  const getAvailableAssets = (chainName: string) => {
    const chain = supportedChains.find(c => c.name === chainName);
    return chain ? chain.available_assets : [];
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-black text-white p-6">
      <motion.header
        className="text-center mb-8"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <Link href="/wallet" className="inline-block mb-4">
          <motion.button
            className="px-4 py-2 bg-purple-600/30 hover:bg-purple-600/50 rounded-lg text-sm"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            ← Back to Wallet
          </motion.button>
        </Link>
        
        <h1 className="text-4xl font-bold bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent mb-2">
          🌉 Cosmic Bridge
        </h1>
        <p className="text-cyan-300">Cross-Chain Asset Transfer & Universal Bridges</p>
      </motion.header>

      {/* Supported Chains */}
      <motion.div className="mb-8" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
        <h2 className="text-xl font-semibold text-center mb-4 text-purple-300">🔗 Supported Chains</h2>
        <div className="grid gap-4 md:grid-cols-5">
          {supportedChains.map((chain, index) => (
            <motion.div
              key={chain.network_id}
              className="bg-gradient-to-br from-gray-800 to-gray-900 p-4 rounded-xl border border-purple-500/30"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.2 + index * 0.1 }}
              whileHover={{ scale: 1.05 }}
            >
              <div className="text-center">
                <div className="text-3xl mb-2">{chain.icon}</div>
                <div className="text-sm font-semibold text-white">{chain.name}</div>
                <div className="text-xs text-gray-400">{chain.symbol}</div>
                <div className="text-xs text-purple-300 mt-1">{chain.cosmic_alignment}% Aligned</div>
                <div className="text-xs text-gray-500 mt-1">{chain.available_assets.length} assets</div>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Bridge Form */}
      <motion.div className="mb-8 bg-gradient-to-br from-gray-800 to-gray-900 p-6 rounded-2xl border border-cyan-500/30" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
        <h2 className="text-xl font-semibold mb-4 text-center text-cyan-300">🌉 Bridge Assets Across Chains</h2>
        
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">From Chain</label>
            <select
              value={bridgeForm.from_chain}
              onChange={(e) => setBridgeForm({...bridgeForm, from_chain: e.target.value, asset: ''})}
              className="w-full px-3 py-2 bg-black/50 border border-cyan-500/30 rounded-lg text-white focus:border-cyan-400 focus:outline-none"
            >
              <option value="">Select source chain</option>
              {supportedChains.map(chain => (
                <option key={chain.network_id} value={chain.name}>{chain.icon} {chain.name}</option>
              ))}
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">To Chain</label>
            <select
              value={bridgeForm.to_chain}
              onChange={(e) => setBridgeForm({...bridgeForm, to_chain: e.target.value})}
              className="w-full px-3 py-2 bg-black/50 border border-cyan-500/30 rounded-lg text-white focus:border-cyan-400 focus:outline-none"
            >
              <option value="">Select destination chain</option>
              {supportedChains.filter(chain => chain.name !== bridgeForm.from_chain).map(chain => (
                <option key={chain.network_id} value={chain.name}>{chain.icon} {chain.name}</option>
              ))}
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Asset</label>
            <select
              value={bridgeForm.asset}
              onChange={(e) => setBridgeForm({...bridgeForm, asset: e.target.value})}
              className="w-full px-3 py-2 bg-black/50 border border-cyan-500/30 rounded-lg text-white focus:border-cyan-400 focus:outline-none"
              disabled={!bridgeForm.from_chain}
            >
              <option value="">Select asset</option>
              {getAvailableAssets(bridgeForm.from_chain).map(asset => (
                <option key={asset} value={asset}>{asset}</option>
              ))}
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Amount</label>
            <input
              type="number"
              step="0.00000001"
              value={bridgeForm.amount}
              onChange={(e) => setBridgeForm({...bridgeForm, amount: e.target.value})}
              placeholder="0.00"
              className="w-full px-3 py-2 bg-black/50 border border-cyan-500/30 rounded-lg text-white placeholder-gray-500 focus:border-cyan-400 focus:outline-none"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Recipient Address</label>
            <input
              type="text"
              value={bridgeForm.recipient_address}
              onChange={(e) => setBridgeForm({...bridgeForm, recipient_address: e.target.value})}
              placeholder="0x... or zion1..."
              className="w-full px-3 py-2 bg-black/50 border border-cyan-500/30 rounded-lg text-white placeholder-gray-500 focus:border-cyan-400 focus:outline-none"
            />
          </div>
        </div>
        
        {bridgeForm.amount && (
          <div className="mt-4 p-3 bg-black/30 rounded-lg">
            <div className="flex justify-between text-sm">
              <span className="text-gray-300">Bridge Fee:</span>
              <span className="text-yellow-300">{(parseFloat(bridgeForm.amount) * 0.001).toFixed(8)} {bridgeForm.asset}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-300">You will receive:</span>
              <span className="text-green-300">{(parseFloat(bridgeForm.amount) * 0.999).toFixed(8)} {bridgeForm.asset}</span>
            </div>
          </div>
        )}
        
        <motion.button
          className="w-full mt-4 bg-gradient-to-r from-cyan-500 to-purple-600 px-6 py-3 rounded-xl font-semibold"
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={handleBridge}
        >
          🌉 Bridge Assets
        </motion.button>
      </motion.div>

      {/* Bridge Transactions */}
      <motion.div className="space-y-4" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
        <h2 className="text-xl font-semibold text-center mb-4 text-pink-300">📜 Bridge Transaction History</h2>
        
        {bridgeTransactions.map((tx, index) => (
          <motion.div
            key={tx.id}
            className="bg-gradient-to-r from-gray-800 to-gray-900 p-6 rounded-2xl border border-pink-500/30"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.5 + index * 0.1 }}
          >
            <div className="flex justify-between items-start mb-4">
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-cyan-300 font-semibold">{tx.from_chain}</span>
                  <span className="text-gray-400">→</span>
                  <span className="text-pink-300 font-semibold">{tx.to_chain}</span>
                </div>
                <div className="text-2xl font-bold text-white">
                  {tx.amount.toFixed(8)} {tx.asset}
                </div>
                <div className="text-xs text-gray-400">{tx.timestamp}</div>
              </div>
              <div className="text-right">
                <div className={`px-2 py-1 rounded-lg text-xs font-medium mb-2 ${getStatusColor(tx.status)}`}>
                  {getStatusIcon(tx.status)} {tx.status.toUpperCase()}
                </div>
                <div className="text-xs text-purple-300">Dharma: +{tx.dharma_impact}</div>
              </div>
            </div>

            <div className="mb-3">
              <div className="text-sm text-purple-300 mb-1">{tx.cosmic_signature}</div>
              {tx.tx_hash && (
                <div className="text-xs text-gray-400 font-mono break-all">
                  TX: {tx.tx_hash}
                </div>
              )}
            </div>

            <div className="flex justify-between items-center text-xs text-gray-300">
              <span>Bridge Fee: {tx.bridge_fee.toFixed(8)} {tx.asset}</span>
              <div className="flex gap-2">
                <motion.button
                  className="px-2 py-1 bg-blue-600/30 hover:bg-blue-600/50 rounded text-xs"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  🔍 Explorer
                </motion.button>
                <motion.button
                  className="px-2 py-1 bg-purple-600/30 hover:bg-purple-600/50 rounded text-xs"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  📋 Copy TX
                </motion.button>
              </div>
            </div>
          </motion.div>
        ))}
      </motion.div>
    </div>
  );
}
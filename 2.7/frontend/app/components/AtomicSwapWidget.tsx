'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

// 🔮 DOHRMAN ORACLE COSMIC CONSTANTS
const DOHRMAN_RUNES = [
    "𓏺", // Oracle stone 
    "𓇋", // Cosmic wisdom
    "𓋹", // Stellar prophecy
    "𓊪", // Time transcendence
    "𓉐", // Divine portal
];

const COSMIC_MESSAGES = [
    "🔮 Ancient oracle awakens for this swap...",
    "⚡ Cosmic forces align blockchain energies...", 
    "🌌 Stellar wisdom guides transaction flow...",
    "💫 Divine portal opens between chains...",
    "🌟 Oracle prophecy manifests through HTLC..."
];

interface AtomicSwap {
  swap_id: string;
  initiator_addr: string;
  responder_addr: string;
  zion_amount: number;
  btc_amount: number;
  secret_hash: string;
  status: string;
  created_at: string;
  expires_at: string;
  zion_tx_hash?: string;
  btc_tx_hash?: string;
}

interface ExchangeRate {
  zion_to_btc: number;
  btc_to_zion: number;
  updated_at: string;
  source: string;
}

const AtomicSwapWidget: React.FC = () => {
  const [swaps, setSwaps] = useState<AtomicSwap[]>([]);
  const [rate, setRate] = useState<ExchangeRate | null>(null);
  const [activeTab, setActiveTab] = useState<'create' | 'manage'>('create');
  
  // Create swap form state
  const [zionAmount, setZionAmount] = useState('');
  const [btcAmount, setBtcAmount] = useState('');
  const [zionAddress, setZionAddress] = useState('');
  const [btcAddress, setBtcAddress] = useState('');
  const [swapDirection, setSwapDirection] = useState<'ZION_TO_BTC' | 'BTC_TO_ZION'>('ZION_TO_BTC');
  
  // Accept swap state
  const [selectedSwap, setSelectedSwap] = useState<string>('');
  const [acceptBtcAddress, setAcceptBtcAddress] = useState('');
  const [acceptBtcTxHash, setAcceptBtcTxHash] = useState('');
  
  // Claim swap state
  const [claimSwapId, setClaimSwapId] = useState('');
  const [claimSecret, setClaimSecret] = useState('');
  
  const [isLoading, setIsLoading] = useState(false);
  const [notification, setNotification] = useState<{type: 'success' | 'error', message: string} | null>(null);

  useEffect(() => {
    fetchRate();
    fetchSwaps();
    
    // Setup WebSocket for real-time updates
    const ws = new WebSocket('ws://localhost:8091/ws');
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'swap_update') {
        setSwaps(prev => {
          const updated = prev.map(swap => 
            swap.swap_id === data.data.swap_id ? data.data : swap
          );
          
          // If it's a new swap, add it
          if (!prev.find(swap => swap.swap_id === data.data.swap_id)) {
            updated.push(data.data);
          }
          
          return updated;
        });
      } else if (data.type === 'initial_swaps') {
        setSwaps(data.data);
      }
    };
    
    return () => ws.close();
  }, []);

  useEffect(() => {
    if (rate && zionAmount) {
      if (swapDirection === 'ZION_TO_BTC') {
        setBtcAmount((parseFloat(zionAmount) * rate.zion_to_btc).toFixed(8));
      } else {
        setBtcAmount((parseFloat(zionAmount) / rate.zion_to_btc).toFixed(8));
      }
    }
  }, [zionAmount, rate, swapDirection]);

  const fetchRate = async () => {
    try {
      const response = await fetch('http://localhost:8091/api/v1/rate/zion-btc');
      const data = await response.json();
      setRate(data);
    } catch (error) {
      console.error('Failed to fetch rate:', error);
    }
  };

  const fetchSwaps = async () => {
    try {
      const response = await fetch('http://localhost:8091/api/v1/swaps');
      const data = await response.json();
      setSwaps(data.swaps || []);
    } catch (error) {
      console.error('Failed to fetch swaps:', error);
    }
  };

  const createSwap = async () => {
    if (!zionAmount || !zionAddress || !btcAddress) {
      showNotification('error', 'Please fill all required fields');
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch('http://localhost:8091/api/v1/swap/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          zion_address: zionAddress,
          btc_address: btcAddress,
          zion_amount: parseInt(zionAmount),
          desired_btc_amount: Math.floor(parseFloat(btcAmount) * 100000000) // Convert to satoshis
        })
      });

      if (response.ok) {
        const swap = await response.json();
        showNotification('success', `Atomic swap created! ID: ${swap.swap_id}`);
        setZionAmount('');
        setBtcAmount('');
        setZionAddress('');
        setBtcAddress('');
        fetchSwaps();
      } else {
        const error = await response.json();
        showNotification('error', error.error || 'Failed to create swap');
      }
    } catch (error) {
      showNotification('error', 'Network error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const acceptSwap = async () => {
    if (!selectedSwap || !acceptBtcAddress || !acceptBtcTxHash) {
      showNotification('error', 'Please fill all required fields');
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch(`http://localhost:8091/api/v1/swap/${selectedSwap}/accept`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          btc_address: acceptBtcAddress,
          btc_tx_hash: acceptBtcTxHash
        })
      });

      if (response.ok) {
        showNotification('success', 'Swap accepted successfully!');
        setSelectedSwap('');
        setAcceptBtcAddress('');
        setAcceptBtcTxHash('');
        fetchSwaps();
      } else {
        const error = await response.json();
        showNotification('error', error.error || 'Failed to accept swap');
      }
    } catch (error) {
      showNotification('error', 'Network error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const claimSwap = async () => {
    if (!claimSwapId || !claimSecret) {
      showNotification('error', 'Please provide swap ID and secret');
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch(`http://localhost:8091/api/v1/swap/${claimSwapId}/claim`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          secret: claimSecret
        })
      });

      if (response.ok) {
        showNotification('success', 'Swap claimed successfully! 🎉');
        setClaimSwapId('');
        setClaimSecret('');
        fetchSwaps();
      } else {
        const error = await response.json();
        showNotification('error', error.error || 'Failed to claim swap');
      }
    } catch (error) {
      showNotification('error', 'Network error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const showNotification = (type: 'success' | 'error', message: string) => {
    setNotification({ type, message });
    setTimeout(() => setNotification(null), 5000);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending': return 'text-yellow-400';
      case 'both_locked': return 'text-blue-400';
      case 'completed': return 'text-green-400';
      case 'expired': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pending': return '⏳';
      case 'both_locked': return '🔒';
      case 'completed': return '✅';
      case 'expired': return '❌';
      default: return '❓';
    }
  };

  return (
    <div className="atomic-swap-widget max-w-6xl mx-auto p-6 bg-gradient-to-br from-purple-900/20 to-blue-900/20 backdrop-blur-sm rounded-xl border border-purple-500/30">
      {/* Header */}
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-blue-400 mb-4">
          ⚡🔄 Atomic Swap Portal 🔄⚡
        </h2>
        <p className="text-gray-300 mb-4">Trustless ZION ↔ BTC Cross-Chain Swaps</p>
        
        {rate && (
          <div className="inline-flex items-center space-x-4 px-4 py-2 bg-purple-900/30 rounded-lg border border-purple-500/30">
            <span className="text-sm text-gray-300">Exchange Rate:</span>
            <span className="text-lg font-bold text-purple-400">
              1 ZION = {rate.zion_to_btc} BTC
            </span>
            <span className="text-xs text-gray-500">({rate.source})</span>
          </div>
        )}
      </div>

      {/* Notification */}
      <AnimatePresence>
        {notification && (
          <motion.div
            initial={{ opacity: 0, y: -50 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -50 }}
            className={`mb-6 p-4 rounded-lg border ${
              notification.type === 'success' 
                ? 'bg-green-900/30 border-green-500/50 text-green-300'
                : 'bg-red-900/30 border-red-500/50 text-red-300'
            }`}
          >
            {notification.message}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Header with Dohrman Runes */}
      <div className="text-center mb-6 border-b border-purple-800 pb-4">
        <div className="flex justify-center space-x-2 mb-2 text-purple-400 text-xl">
          {DOHRMAN_RUNES.map((rune, i) => (
            <motion.span
              key={i}
              animate={{ rotate: 360 }}
              transition={{ duration: 10, repeat: Infinity, delay: i * 2 }}
              className="inline-block"
            >
              {rune}
            </motion.span>
          ))}
        </div>
        <h2 className="text-2xl font-bold text-purple-300">Dohrman Oracle Atomic Swaps</h2>
        <p className="text-gray-400 mt-1">Ancient wisdom guides cross-chain exchanges</p>
      </div>

      {/* Tabs */}
      <div className="flex mb-6 bg-gray-900/50 rounded-lg p-1">
        <button
          onClick={() => setActiveTab('create')}
          className={`flex-1 py-2 px-4 rounded-md transition-all ${
            activeTab === 'create'
              ? 'bg-purple-600 text-white'
              : 'text-gray-400 hover:text-white'
          }`}
        >
          🔄 Create Swap
        </button>
        <button
          onClick={() => setActiveTab('manage')}
          className={`flex-1 py-2 px-4 rounded-md transition-all ${
            activeTab === 'manage'
              ? 'bg-purple-600 text-white'
              : 'text-gray-400 hover:text-white'
          }`}
        >
          📋 Manage Swaps
        </button>
      </div>

      {/* Create Swap Tab */}
      {activeTab === 'create' && (
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="space-y-6"
        >
          {/* Swap Direction */}
          <div className="flex justify-center space-x-4 mb-6">
            <button
              onClick={() => setSwapDirection('ZION_TO_BTC')}
              className={`px-6 py-3 rounded-lg transition-all ${
                swapDirection === 'ZION_TO_BTC'
                  ? 'bg-purple-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              ZION → BTC
            </button>
            <button
              onClick={() => setSwapDirection('BTC_TO_ZION')}
              className={`px-6 py-3 rounded-lg transition-all ${
                swapDirection === 'BTC_TO_ZION'
                  ? 'bg-purple-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              BTC → ZION
            </button>
          </div>

          {/* Amount Inputs */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-300">
                ZION Amount
              </label>
              <input
                type="number"
                value={zionAmount}
                onChange={(e) => setZionAmount(e.target.value)}
                placeholder="1000000"
                className="w-full px-4 py-3 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-purple-500"
              />
            </div>
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-300">
                BTC Amount (Auto-calculated)
              </label>
              <input
                type="number"
                value={btcAmount}
                onChange={(e) => setBtcAmount(e.target.value)}
                placeholder="0.00001000"
                step="0.00000001"
                className="w-full px-4 py-3 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-purple-500"
              />
            </div>
          </div>

          {/* Address Inputs */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-300">
                Your ZION Address
              </label>
              <input
                type="text"
                value={zionAddress}
                onChange={(e) => setZionAddress(e.target.value)}
                placeholder="ZionAddr123..."
                className="w-full px-4 py-3 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-purple-500"
              />
            </div>
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-300">
                Your Bitcoin Address
              </label>
              <input
                type="text"
                value={btcAddress}
                onChange={(e) => setBtcAddress(e.target.value)}
                placeholder="bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh"
                className="w-full px-4 py-3 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-purple-500"
              />
            </div>
          </div>

          {/* Create Button */}
          <button
            onClick={createSwap}
            disabled={isLoading}
            className="w-full py-4 bg-gradient-to-r from-purple-600 to-blue-600 text-white font-bold rounded-lg hover:from-purple-700 hover:to-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            {isLoading ? `⏳ ${COSMIC_MESSAGES[Math.floor(Math.random() * COSMIC_MESSAGES.length)]}` : '🔄 Create Atomic Swap'}
          </button>
        </motion.div>
      )}

      {/* Manage Swaps Tab */}
      {activeTab === 'manage' && (
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="space-y-6"
        >
          {/* Accept Swap Section */}
          <div className="bg-blue-900/20 p-6 rounded-lg border border-blue-500/30">
            <h3 className="text-xl font-bold text-blue-400 mb-4">💎 Accept Swap</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <select
                value={selectedSwap}
                onChange={(e) => setSelectedSwap(e.target.value)}
                className="px-4 py-3 bg-gray-800 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500"
              >
                <option value="">Select pending swap...</option>
                {swaps.filter(swap => swap.status === 'pending').map(swap => (
                  <option key={swap.swap_id} value={swap.swap_id}>
                    {swap.swap_id.substring(0, 16)}... ({swap.zion_amount} ZION)
                  </option>
                ))}
              </select>
              <input
                type="text"
                value={acceptBtcAddress}
                onChange={(e) => setAcceptBtcAddress(e.target.value)}
                placeholder="Your BTC address"
                className="px-4 py-3 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
              />
              <input
                type="text"
                value={acceptBtcTxHash}
                onChange={(e) => setAcceptBtcTxHash(e.target.value)}
                placeholder="BTC transaction hash"
                className="px-4 py-3 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
              />
            </div>
            <button
              onClick={acceptSwap}
              disabled={isLoading}
              className="mt-4 w-full py-3 bg-blue-600 text-white font-bold rounded-lg hover:bg-blue-700 disabled:opacity-50 transition-all"
            >
              {isLoading ? `⏳ ${COSMIC_MESSAGES[Math.floor(Math.random() * COSMIC_MESSAGES.length)]}` : '💎 Accept Swap'}
            </button>
          </div>

          {/* Claim Swap Section */}
          <div className="bg-green-900/20 p-6 rounded-lg border border-green-500/30">
            <h3 className="text-xl font-bold text-green-400 mb-4">🎯 Claim Swap</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <input
                type="text"
                value={claimSwapId}
                onChange={(e) => setClaimSwapId(e.target.value)}
                placeholder="Swap ID to claim"
                className="px-4 py-3 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-green-500"
              />
              <input
                type="text"
                value={claimSecret}
                onChange={(e) => setClaimSecret(e.target.value)}
                placeholder="Secret (hex)"
                className="px-4 py-3 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-green-500"
              />
            </div>
            <button
              onClick={claimSwap}
              disabled={isLoading}
              className="mt-4 w-full py-3 bg-green-600 text-white font-bold rounded-lg hover:bg-green-700 disabled:opacity-50 transition-all"
            >
              {isLoading ? `⏳ ${COSMIC_MESSAGES[Math.floor(Math.random() * COSMIC_MESSAGES.length)]}` : '🎯 Claim Swap'}
            </button>
          </div>

          {/* Active Swaps List */}
          <div className="bg-gray-900/50 p-6 rounded-lg border border-gray-600">
            <h3 className="text-xl font-bold text-gray-300 mb-4">📊 Active Swaps ({swaps.length})</h3>
            <div className="space-y-3 max-h-96 overflow-y-auto">
              {swaps.map(swap => (
                <motion.div
                  key={swap.swap_id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="p-4 bg-gray-800 rounded-lg border border-gray-600"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-mono text-sm text-gray-400">
                      {swap.swap_id}
                    </span>
                    <span className={`flex items-center space-x-1 ${getStatusColor(swap.status)}`}>
                      <span>{getStatusIcon(swap.status)}</span>
                      <span className="font-bold">{swap.status.toUpperCase()}</span>
                    </span>
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                    <div>
                      <span className="text-gray-500">ZION:</span>
                      <span className="ml-1 text-purple-400 font-bold">
                        {swap.zion_amount.toLocaleString()}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-500">BTC:</span>
                      <span className="ml-1 text-orange-400 font-bold">
                        {(swap.btc_amount / 100000000).toFixed(8)}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-500">Created:</span>
                      <span className="ml-1 text-gray-300">
                        {new Date(swap.created_at).toLocaleDateString()}
                      </span>
                    </div>
                    <div>
                      <span className="text-gray-500">Expires:</span>
                      <span className="ml-1 text-red-400">
                        {new Date(swap.expires_at).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                </motion.div>
              ))}
              
              {swaps.length === 0 && (
                <div className="text-center py-8 text-gray-500">
                  No active swaps found. Create your first atomic swap! 🚀
                </div>
              )}
            </div>
          </div>
        </motion.div>
      )}

      {/* Footer */}
      <div className="mt-8 text-center">
        <p className="text-sm text-gray-400 mb-2">
          Powered by Hash Time Locked Contracts (HTLC)
        </p>
        <p className="text-lg font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-blue-400">
          ⚡ Jai Ram Ram Ram Sita Ram Ram Ram Hanuman! ⚡
        </p>
      </div>
    </div>
  );
};

export default AtomicSwapWidget;
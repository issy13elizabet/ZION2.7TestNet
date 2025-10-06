'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

interface WalletData {
  name: string;
  label: string;
  created_at: string;
  address_count: number;
  total_balance: number;
  unlocked: boolean;
}

interface WalletAddress {
  address: string;
  public_key: string;
  label: string;
  created_at: string;
  balance: number;
}

interface WalletInfo {
  name: string;
  label: string;
  addresses: WalletAddress[];
  total_balance: number;
  address_count: number;
  transaction_count: number;
  created_at: string;
}

export default function ZionWalletWidget() {
  const [wallets, setWallets] = useState<WalletData[]>([]);
  const [selectedWallet, setSelectedWallet] = useState<WalletInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [unlocking, setUnlocking] = useState(false);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [showUnlockForm, setShowUnlockForm] = useState<string | null>(null);
  
  // Form states
  const [newWalletName, setNewWalletName] = useState('');
  const [newWalletLabel, setNewWalletLabel] = useState('');
  const [newWalletPassword, setNewWalletPassword] = useState('');
  const [unlockPassword, setUnlockPassword] = useState('');

  // Fetch wallet list
  const fetchWallets = async () => {
    try {
      const response = await fetch('http://localhost:8001/api/wallet/list');
      const result = await response.json();
      if (result.success) {
        setWallets(result.data.wallets);
      }
    } catch (error) {
      console.error('Failed to fetch wallets:', error);
    } finally {
      setLoading(false);
    }
  };

  // Create new wallet
  const createWallet = async () => {
    if (!newWalletName || !newWalletPassword) return;
    
    setCreating(true);
    try {
      const response = await fetch('http://localhost:8001/api/wallet/create', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: newWalletName,
          password: newWalletPassword,
          label: newWalletLabel || newWalletName
        }),
      });
      
      const result = await response.json();
      if (result.success) {
        setNewWalletName('');
        setNewWalletLabel('');
        setNewWalletPassword('');
        setShowCreateForm(false);
        fetchWallets();
        alert(`✅ Wallet "${newWalletName}" created successfully!\\nAddress: ${result.data.primary_address}\\nBalance: ${result.data.balance} ZION`);
      } else {
        alert(`❌ Failed to create wallet: ${result.error || 'Unknown error'}`);
      }
    } catch (error) {
      alert(`❌ Network error: ${error}`);
    } finally {
      setCreating(false);
    }
  };

  // Unlock wallet
  const unlockWallet = async (walletName: string) => {
    if (!unlockPassword) return;
    
    setUnlocking(true);
    try {
      const response = await fetch('http://localhost:8001/api/wallet/unlock', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          name: walletName,
          password: unlockPassword
        }),
      });
      
      const result = await response.json();
      if (result.success) {
        setUnlockPassword('');
        setShowUnlockForm(null);
        fetchWallets();
        // Get wallet details
        getWalletInfo(walletName);
        alert(`🔓 Wallet "${walletName}" unlocked successfully!`);
      } else {
        alert(`❌ Failed to unlock wallet: ${result.error || 'Invalid password'}`);
      }
    } catch (error) {
      alert(`❌ Network error: ${error}`);
    } finally {
      setUnlocking(false);
    }
  };

  // Get wallet details
  const getWalletInfo = async (walletName: string) => {
    try {
      const response = await fetch(`http://localhost:8001/api/wallet/${walletName}`);
      const result = await response.json();
      if (result.success) {
        setSelectedWallet(result.data);
      }
    } catch (error) {
      console.error('Failed to get wallet info:', error);
    }
  };

  useEffect(() => {
    fetchWallets();
    const interval = setInterval(fetchWallets, 10000); // Update every 10 seconds
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="bg-gradient-to-br from-slate-900 to-slate-800 rounded-xl p-6 border border-green-500/30">
        <div className="flex items-center justify-center h-32">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-green-500"></div>
          <span className="ml-3 text-green-400">Loading wallets...</span>
        </div>
      </div>
    );
  }

  return (
    <motion.div
      className="bg-gradient-to-br from-slate-900 to-slate-800 rounded-xl p-6 border border-green-500/30"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-xl font-bold text-green-400">💰 ZION Wallets</h3>
        <button
          onClick={() => setShowCreateForm(true)}
          className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
        >
          🆕 Create Wallet
        </button>
      </div>

      {/* Create Wallet Form */}
      {showCreateForm && (
        <motion.div
          className="mb-6 p-4 bg-slate-800 rounded-lg border border-green-500/20"
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
        >
          <h4 className="text-lg font-semibold text-green-400 mb-4">Create New Wallet</h4>
          <div className="space-y-3">
            <input
              type="text"
              placeholder="Wallet Name"
              value={newWalletName}
              onChange={(e) => setNewWalletName(e.target.value)}
              className="w-full p-3 bg-slate-700 border border-slate-600 rounded-lg text-white"
            />
            <input
              type="text"
              placeholder="Wallet Label (optional)"
              value={newWalletLabel}
              onChange={(e) => setNewWalletLabel(e.target.value)}
              className="w-full p-3 bg-slate-700 border border-slate-600 rounded-lg text-white"
            />
            <input
              type="password"
              placeholder="Password"
              value={newWalletPassword}
              onChange={(e) => setNewWalletPassword(e.target.value)}
              className="w-full p-3 bg-slate-700 border border-slate-600 rounded-lg text-white"
            />
            <div className="flex space-x-3">
              <button
                onClick={createWallet}
                disabled={creating || !newWalletName || !newWalletPassword}
                className="px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white rounded-lg transition-colors"
              >
                {creating ? '⏳ Creating...' : '✅ Create'}
              </button>
              <button
                onClick={() => {
                  setShowCreateForm(false);
                  setNewWalletName('');
                  setNewWalletLabel('');
                  setNewWalletPassword('');
                }}
                className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors"
              >
                ❌ Cancel
              </button>
            </div>
          </div>
        </motion.div>
      )}

      {/* Wallet List */}
      <div className="space-y-3">
        {wallets.length === 0 ? (
          <div className="text-center py-8 text-gray-400">
            <p>No wallets found</p>
            <p className="text-sm">Create your first ZION wallet to get started</p>
          </div>
        ) : (
          wallets.map((wallet) => (
            <motion.div
              key={wallet.name}
              className="p-4 bg-slate-800 rounded-lg border border-slate-700 hover:border-green-500/50 transition-colors"
              whileHover={{ scale: 1.02 }}
            >
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="font-semibold text-white">{wallet.label || wallet.name}</h4>
                  <p className="text-sm text-gray-400">
                    {wallet.address_count} address{wallet.address_count !== 1 ? 'es' : ''} • 
                    {wallet.total_balance.toLocaleString()} ZION
                  </p>
                  <p className="text-xs text-gray-500">
                    Created: {new Date(wallet.created_at).toLocaleDateString()}
                  </p>
                </div>
                <div className="flex items-center space-x-2">
                  <span className={`px-2 py-1 rounded-full text-xs ${
                    wallet.unlocked 
                      ? 'bg-green-500/20 text-green-400' 
                      : 'bg-red-500/20 text-red-400'
                  }`}>
                    {wallet.unlocked ? '🔓 Unlocked' : '🔒 Locked'}
                  </span>
                  {!wallet.unlocked ? (
                    <button
                      onClick={() => setShowUnlockForm(wallet.name)}
                      className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm transition-colors"
                    >
                      🔓 Unlock
                    </button>
                  ) : (
                    <button
                      onClick={() => getWalletInfo(wallet.name)}
                      className="px-3 py-1 bg-green-600 hover:bg-green-700 text-white rounded text-sm transition-colors"
                    >
                      📄 Details
                    </button>
                  )}
                </div>
              </div>

              {/* Unlock Form */}
              {showUnlockForm === wallet.name && (
                <motion.div
                  className="mt-3 pt-3 border-t border-slate-700"
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                >
                  <div className="flex space-x-3">
                    <input
                      type="password"
                      placeholder="Enter wallet password"
                      value={unlockPassword}
                      onChange={(e) => setUnlockPassword(e.target.value)}
                      className="flex-1 p-2 bg-slate-700 border border-slate-600 rounded text-white"
                      onKeyPress={(e) => e.key === 'Enter' && unlockWallet(wallet.name)}
                    />
                    <button
                      onClick={() => unlockWallet(wallet.name)}
                      disabled={unlocking || !unlockPassword}
                      className="px-3 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded transition-colors"
                    >
                      {unlocking ? '⏳' : '🔓'}
                    </button>
                    <button
                      onClick={() => {
                        setShowUnlockForm(null);
                        setUnlockPassword('');
                      }}
                      className="px-3 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded transition-colors"
                    >
                      ❌
                    </button>
                  </div>
                </motion.div>
              )}
            </motion.div>
          ))
        )}
      </div>

      {/* Wallet Details */}
      {selectedWallet && (
        <motion.div
          className="mt-6 p-4 bg-slate-800 rounded-lg border border-green-500/30"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-lg font-semibold text-green-400">
              💼 {selectedWallet.label || selectedWallet.name}
            </h4>
            <button
              onClick={() => setSelectedWallet(null)}
              className="text-gray-400 hover:text-white"
            >
              ❌
            </button>
          </div>
          
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <p className="text-sm text-gray-400">Total Balance</p>
              <p className="text-xl font-bold text-green-400">
                {selectedWallet.total_balance.toLocaleString()} ZION
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-400">Addresses</p>
              <p className="text-xl font-bold text-white">{selectedWallet.address_count}</p>
            </div>
          </div>

          <div className="space-y-2">
            <h5 className="font-semibold text-white">Addresses:</h5>
            {selectedWallet.addresses.map((addr, index) => (
              <div key={addr.address} className="p-3 bg-slate-700 rounded border border-slate-600">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-mono text-sm text-green-400">{addr.address}</p>
                    <p className="text-xs text-gray-400">{addr.label}</p>
                  </div>
                  <div className="text-right">
                    <p className="font-semibold text-white">{addr.balance.toLocaleString()} ZION</p>
                    <p className="text-xs text-gray-400">Balance</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}
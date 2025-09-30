"use client";
import { motion } from "framer-motion";
import { useState } from "react";
import Link from "next/link";

interface LightningChannel {
  id: string;
  name: string;
  capacity: number;
  local_balance: number;
  remote_balance: number;
  status: 'active' | 'inactive' | 'pending';
  peer_pubkey: string;
  cosmic_enhancement: number;
}

export default function LightningPage() {
  const [channels, setChannels] = useState<LightningChannel[]>([
    {
      id: 'ch_001',
      name: 'Cosmic Lightning Node 1',
      capacity: 1000000,
      local_balance: 650000,
      remote_balance: 350000,
      status: 'active',
      peer_pubkey: '02a1b2c3d4e5f6...cosmic_peer_42',
      cosmic_enhancement: 85
    },
    {
      id: 'ch_002', 
      name: 'Dharma Channel Alpha',
      capacity: 500000,
      local_balance: 200000,
      remote_balance: 300000,
      status: 'active',
      peer_pubkey: '03f1e2d3c4b5a6...dharma_node_108',
      cosmic_enhancement: 92
    },
    {
      id: 'ch_003',
      name: 'Universal Bridge Beta',
      capacity: 2000000,
      local_balance: 0,
      remote_balance: 0,
      status: 'pending',
      peer_pubkey: '04c5d6e7f8g9h0...bridge_cosmic_21',
      cosmic_enhancement: 96
    }
  ]);

  const [newChannelForm, setNewChannelForm] = useState({
    peer_pubkey: '',
    capacity: '',
    push_sat: ''
  });

  const handleOpenChannel = async () => {
    if (!newChannelForm.peer_pubkey || !newChannelForm.capacity) {
      alert('Please fill in peer pubkey and capacity');
      return;
    }

    const newChannel: LightningChannel = {
      id: `ch_${Date.now()}`,
      name: `Channel ${channels.length + 1}`,
      capacity: parseInt(newChannelForm.capacity),
      local_balance: parseInt(newChannelForm.push_sat) || 0,
      remote_balance: 0,
      status: 'pending',
      peer_pubkey: newChannelForm.peer_pubkey,
      cosmic_enhancement: Math.floor(Math.random() * 30) + 70
    };

    setChannels([...channels, newChannel]);
    setNewChannelForm({ peer_pubkey: '', capacity: '', push_sat: '' });
  };

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'active': return 'text-green-400 bg-green-500/20';
      case 'inactive': return 'text-red-400 bg-red-500/20';
      case 'pending': return 'text-yellow-400 bg-yellow-500/20';
      default: return 'text-gray-400 bg-gray-500/20';
    }
  };

  const getStatusIcon = (status: string) => {
    switch(status) {
      case 'active': return '🟢';
      case 'inactive': return '🔴';
      case 'pending': return '🟡';
      default: return '⚪';
    }
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
        
        <h1 className="text-4xl font-bold bg-gradient-to-r from-yellow-400 via-blue-400 to-purple-400 bg-clip-text text-transparent mb-2">
          ⚡ Lightning Network
        </h1>
        <p className="text-blue-300">Instant Cosmic Payments & Channel Management</p>
      </motion.header>

      {/* Channel Statistics */}
      <div className="grid gap-6 md:grid-cols-4 mb-8">
        <motion.div className="bg-gradient-to-br from-blue-500 to-cyan-600 p-1 rounded-2xl" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.1 }}>
          <div className="bg-black/70 rounded-xl p-6 text-center">
            <div className="text-3xl mb-2">🌐</div>
            <div className="text-2xl font-bold text-cyan-300">{channels.length}</div>
            <div className="text-sm text-cyan-200">Active Channels</div>
          </div>
        </motion.div>

        <motion.div className="bg-gradient-to-br from-green-500 to-emerald-600 p-1 rounded-2xl" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.2 }}>
          <div className="bg-black/70 rounded-xl p-6 text-center">
            <div className="text-3xl mb-2">💰</div>
            <div className="text-lg font-bold text-emerald-300 break-all">{channels.reduce((sum, ch) => sum + ch.local_balance, 0).toLocaleString()}</div>
            <div className="text-sm text-emerald-200">Local Balance</div>
          </div>
        </motion.div>

        <motion.div className="bg-gradient-to-br from-orange-500 to-red-600 p-1 rounded-2xl" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.3 }}>
          <div className="bg-black/70 rounded-xl p-6 text-center">
            <div className="text-3xl mb-2">🔗</div>
            <div className="text-lg font-bold text-orange-300 break-all">{channels.reduce((sum, ch) => sum + ch.capacity, 0).toLocaleString()}</div>
            <div className="text-sm text-orange-200">Total Capacity</div>
          </div>
        </motion.div>

        <motion.div className="bg-gradient-to-br from-purple-500 to-pink-600 p-1 rounded-2xl" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.4 }}>
          <div className="bg-black/70 rounded-xl p-6 text-center">
            <div className="text-3xl mb-2">✨</div>
            <div className="text-2xl font-bold text-pink-300">{Math.floor(channels.reduce((sum, ch) => sum + ch.cosmic_enhancement, 0) / channels.length)}%</div>
            <div className="text-sm text-pink-200">Cosmic Enhancement</div>
          </div>
        </motion.div>
      </div>

      {/* New Channel Form */}
      <motion.div className="mb-8 bg-gradient-to-br from-gray-800 to-gray-900 p-6 rounded-2xl border border-purple-500/30" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }}>
        <h2 className="text-xl font-semibold mb-4 text-center text-purple-300">⚡ Open New Lightning Channel</h2>
        
        <div className="grid gap-4 md:grid-cols-3">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Peer Public Key</label>
            <input
              type="text"
              value={newChannelForm.peer_pubkey}
              onChange={(e) => setNewChannelForm({...newChannelForm, peer_pubkey: e.target.value})}
              placeholder="02a1b2c3d4e5f6..."
              className="w-full px-3 py-2 bg-black/50 border border-purple-500/30 rounded-lg text-white placeholder-gray-500 focus:border-purple-400 focus:outline-none"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Channel Capacity (sats)</label>
            <input
              type="number"
              value={newChannelForm.capacity}
              onChange={(e) => setNewChannelForm({...newChannelForm, capacity: e.target.value})}
              placeholder="1000000"
              className="w-full px-3 py-2 bg-black/50 border border-purple-500/30 rounded-lg text-white placeholder-gray-500 focus:border-purple-400 focus:outline-none"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Push Amount (sats)</label>
            <input
              type="number"
              value={newChannelForm.push_sat}
              onChange={(e) => setNewChannelForm({...newChannelForm, push_sat: e.target.value})}
              placeholder="500000"
              className="w-full px-3 py-2 bg-black/50 border border-purple-500/30 rounded-lg text-white placeholder-gray-500 focus:border-purple-400 focus:outline-none"
            />
          </div>
        </div>
        
        <motion.button
          className="w-full mt-4 bg-gradient-to-r from-blue-500 to-purple-600 px-6 py-3 rounded-xl font-semibold"
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={handleOpenChannel}
        >
          ⚡ Open Lightning Channel
        </motion.button>
      </motion.div>

      {/* Channels List */}
      <motion.div className="space-y-4" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.6 }}>
        <h2 className="text-xl font-semibold text-center mb-4 text-cyan-300">🌐 Lightning Channels</h2>
        
        {channels.map((channel, index) => (
          <motion.div
            key={channel.id}
            className="bg-gradient-to-r from-gray-800 to-gray-900 p-6 rounded-2xl border border-cyan-500/30"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.7 + index * 0.1 }}
          >
            <div className="flex justify-between items-start mb-4">
              <div>
                <h3 className="text-lg font-semibold text-cyan-300">{channel.name}</h3>
                <p className="text-xs text-gray-400 font-mono break-all">{channel.peer_pubkey}</p>
              </div>
              <div className="flex items-center gap-2">
                <span className={`px-2 py-1 rounded-lg text-xs font-medium ${getStatusColor(channel.status)}`}>
                  {getStatusIcon(channel.status)} {channel.status.toUpperCase()}
                </span>
                <div className="text-right">
                  <div className="text-sm text-purple-300">{channel.cosmic_enhancement}%</div>
                  <div className="text-xs text-purple-200">Cosmic</div>
                </div>
              </div>
            </div>

            {/* Balance Bar */}
            <div className="mb-4">
              <div className="flex justify-between text-xs text-gray-300 mb-1">
                <span>Local: {channel.local_balance.toLocaleString()} sats</span>
                <span>Remote: {channel.remote_balance.toLocaleString()} sats</span>
              </div>
              <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-gradient-to-r from-green-500 to-emerald-400"
                  style={{ width: `${(channel.local_balance / channel.capacity) * 100}%` }}
                ></div>
              </div>
              <div className="text-center text-xs text-gray-400 mt-1">
                Capacity: {channel.capacity.toLocaleString()} sats
              </div>
            </div>

            {/* Channel Actions */}
            <div className="flex gap-2">
              <motion.button
                className="flex-1 px-3 py-2 bg-blue-600/30 hover:bg-blue-600/50 rounded-lg text-sm"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                💸 Send Payment
              </motion.button>
              <motion.button
                className="flex-1 px-3 py-2 bg-green-600/30 hover:bg-green-600/50 rounded-lg text-sm"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                📥 Create Invoice
              </motion.button>
              <motion.button
                className="flex-1 px-3 py-2 bg-red-600/30 hover:bg-red-600/50 rounded-lg text-sm"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                ❌ Close Channel
              </motion.button>
            </div>
          </motion.div>
        ))}
      </motion.div>
    </div>
  );
}
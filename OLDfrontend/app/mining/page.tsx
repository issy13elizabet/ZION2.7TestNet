"use client";
import { motion } from "framer-motion";
import { useState } from "react";
import Link from "next/link";

interface MiningRig {
  id: string;
  name: string;
  hashrate: number;
  power_consumption: number;
  temperature: number;
  status: 'mining' | 'idle' | 'maintenance' | 'offline';
  cosmic_efficiency: number;
  rewards_24h: number;
}

interface MiningPool {
  name: string;
  url: string;
  fee: number;
  hashrate: number;
  miners: number;
  cosmic_alignment: number;
}

export default function MiningPage() {
  const [miningRigs, setMiningRigs] = useState<MiningRig[]>([
    {
      id: 'rig_001',
      name: 'Cosmic Dharma Miner Alpha',
      hashrate: 420.69,
      power_consumption: 1337,
      temperature: 68,
      status: 'mining',
      cosmic_efficiency: 96.3,
      rewards_24h: 42.108
    },
    {
      id: 'rig_002',
      name: 'Universal Hash Beast',
      hashrate: 888.21,
      power_consumption: 2100,
      temperature: 72,
      status: 'mining',
      cosmic_efficiency: 94.7,
      rewards_24h: 88.216
    },
    {
      id: 'rig_003',
      name: 'Zen Mining Node',
      hashrate: 0,
      power_consumption: 0,
      temperature: 25,
      status: 'maintenance',
      cosmic_efficiency: 0,
      rewards_24h: 0
    }
  ]);

  const miningPools: MiningPool[] = [
    {
      name: 'Cosmic ZION Pool',
      url: 'stratum+tcp://cosmic-pool.zion:4242',
      fee: 1.0,
      hashrate: 42108.21,
      miners: 2108,
      cosmic_alignment: 100
    },
    {
      name: 'Dharma Mining Collective',
      url: 'stratum+tcp://dharma.pool:8080',
      fee: 0.5,
      hashrate: 108216.42,
      miners: 4216,
      cosmic_alignment: 98
    },
    {
      name: 'Universal Hash Network',
      url: 'stratum+tcp://universal.mining:3333',
      fee: 1.5,
      hashrate: 216432.84,
      miners: 8432,
      cosmic_alignment: 95
    }
  ];

  const [newRigForm, setNewRigForm] = useState({
    name: '',
    hashrate: '',
    power_consumption: ''
  });

  const handleAddRig = () => {
    if (!newRigForm.name || !newRigForm.hashrate || !newRigForm.power_consumption) {
      alert('Please fill in all rig details');
      return;
    }

    const newRig: MiningRig = {
      id: `rig_${Date.now()}`,
      name: newRigForm.name,
      hashrate: parseFloat(newRigForm.hashrate),
      power_consumption: parseFloat(newRigForm.power_consumption),
      temperature: Math.floor(Math.random() * 20) + 50,
      status: 'idle',
      cosmic_efficiency: Math.floor(Math.random() * 20) + 80,
      rewards_24h: 0
    };

    setMiningRigs([...miningRigs, newRig]);
    setNewRigForm({ name: '', hashrate: '', power_consumption: '' });
  };

  const toggleRigStatus = (rigId: string) => {
    setMiningRigs(rigs => rigs.map(rig => {
      if (rig.id === rigId) {
        const newStatus = rig.status === 'mining' ? 'idle' : 'mining';
        return {
          ...rig,
          status: newStatus,
          rewards_24h: newStatus === 'mining' ? rig.hashrate * 0.1 : 0
        };
      }
      return rig;
    }));
  };

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'mining': return 'text-green-400 bg-green-500/20';
      case 'idle': return 'text-yellow-400 bg-yellow-500/20';
      case 'maintenance': return 'text-blue-400 bg-blue-500/20';
      case 'offline': return 'text-red-400 bg-red-500/20';
      default: return 'text-gray-400 bg-gray-500/20';
    }
  };

  const getStatusIcon = (status: string) => {
    switch(status) {
      case 'mining': return '⛏️';
      case 'idle': return '⏸️';
      case 'maintenance': return '🔧';
      case 'offline': return '❌';
      default: return '❓';
    }
  };

  const getTempColor = (temp: number) => {
    if (temp < 60) return 'text-blue-400';
    if (temp < 75) return 'text-green-400';
    if (temp < 85) return 'text-yellow-400';
    return 'text-red-400';
  };

  const totalHashrate = miningRigs.filter(r => r.status === 'mining').reduce((sum, rig) => sum + rig.hashrate, 0);
  const totalRewards = miningRigs.reduce((sum, rig) => sum + rig.rewards_24h, 0);
  const avgEfficiency = miningRigs.length > 0 ? miningRigs.reduce((sum, rig) => sum + rig.cosmic_efficiency, 0) / miningRigs.length : 0;

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
        
        <h1 className="text-4xl font-bold bg-gradient-to-r from-orange-400 via-red-400 to-yellow-400 bg-clip-text text-transparent mb-2">
          ⛏️ Cosmic Mining
        </h1>
        <p className="text-orange-300">Universal Hash Mining & Dharma Generation</p>
      </motion.header>

      {/* Mining Stats */}
      <div className="grid gap-6 md:grid-cols-4 mb-8">
        <motion.div className="bg-gradient-to-br from-green-500 to-emerald-600 p-1 rounded-2xl" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.1 }}>
          <div className="bg-black/70 rounded-xl p-6 text-center">
            <div className="text-3xl mb-2">⚡</div>
            <div className="text-lg font-bold text-emerald-300 break-all">{totalHashrate.toFixed(2)}</div>
            <div className="text-sm text-emerald-200">MH/s Total</div>
          </div>
        </motion.div>

        <motion.div className="bg-gradient-to-br from-blue-500 to-cyan-600 p-1 rounded-2xl" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.2 }}>
          <div className="bg-black/70 rounded-xl p-6 text-center">
            <div className="text-3xl mb-2">💎</div>
            <div className="text-lg font-bold text-cyan-300 break-all">{totalRewards.toFixed(3)}</div>
            <div className="text-sm text-cyan-200">ZION/24h</div>
          </div>
        </motion.div>

        <motion.div className="bg-gradient-to-br from-purple-500 to-pink-600 p-1 rounded-2xl" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.3 }}>
          <div className="bg-black/70 rounded-xl p-6 text-center">
            <div className="text-3xl mb-2">🌟</div>
            <div className="text-2xl font-bold text-pink-300">{avgEfficiency.toFixed(1)}%</div>
            <div className="text-sm text-pink-200">Cosmic Efficiency</div>
          </div>
        </motion.div>

        <motion.div className="bg-gradient-to-br from-orange-500 to-red-600 p-1 rounded-2xl" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.4 }}>
          <div className="bg-black/70 rounded-xl p-6 text-center">
            <div className="text-3xl mb-2">⛏️</div>
            <div className="text-2xl font-bold text-orange-300">{miningRigs.filter(r => r.status === 'mining').length}</div>
            <div className="text-sm text-orange-200">Active Rigs</div>
          </div>
        </motion.div>
      </div>

      {/* Add New Rig */}
      <motion.div className="mb-8 bg-gradient-to-br from-gray-800 to-gray-900 p-6 rounded-2xl border border-orange-500/30" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }}>
        <h2 className="text-xl font-semibold mb-4 text-center text-orange-300">⚒️ Add New Mining Rig</h2>
        
        <div className="grid gap-4 md:grid-cols-3">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Rig Name</label>
            <input
              type="text"
              value={newRigForm.name}
              onChange={(e) => setNewRigForm({...newRigForm, name: e.target.value})}
              placeholder="Cosmic Mining Beast"
              className="w-full px-3 py-2 bg-black/50 border border-orange-500/30 rounded-lg text-white placeholder-gray-500 focus:border-orange-400 focus:outline-none"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Hashrate (MH/s)</label>
            <input
              type="number"
              step="0.01"
              value={newRigForm.hashrate}
              onChange={(e) => setNewRigForm({...newRigForm, hashrate: e.target.value})}
              placeholder="420.69"
              className="w-full px-3 py-2 bg-black/50 border border-orange-500/30 rounded-lg text-white placeholder-gray-500 focus:border-orange-400 focus:outline-none"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Power (W)</label>
            <input
              type="number"
              value={newRigForm.power_consumption}
              onChange={(e) => setNewRigForm({...newRigForm, power_consumption: e.target.value})}
              placeholder="1337"
              className="w-full px-3 py-2 bg-black/50 border border-orange-500/30 rounded-lg text-white placeholder-gray-500 focus:border-orange-400 focus:outline-none"
            />
          </div>
        </div>
        
        <motion.button
          className="w-full mt-4 bg-gradient-to-r from-orange-500 to-red-600 px-6 py-3 rounded-xl font-semibold"
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={handleAddRig}
        >
          ⚒️ Add Mining Rig
        </motion.button>
      </motion.div>

      {/* Mining Rigs */}
      <motion.div className="mb-8" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.6 }}>
        <h2 className="text-xl font-semibold text-center mb-4 text-cyan-300">⛏️ Mining Rigs</h2>
        
        <div className="grid gap-4">
          {miningRigs.map((rig, index) => (
            <motion.div
              key={rig.id}
              className="bg-gradient-to-r from-gray-800 to-gray-900 p-6 rounded-2xl border border-cyan-500/30"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.7 + index * 0.1 }}
            >
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h3 className="text-lg font-semibold text-cyan-300">{rig.name}</h3>
                  <div className="flex items-center gap-4 mt-2 text-sm">
                    <span className="text-green-300">⚡ {rig.hashrate} MH/s</span>
                    <span className="text-red-300">🔌 {rig.power_consumption}W</span>
                    <span className={getTempColor(rig.temperature)}>🌡️ {rig.temperature}°C</span>
                  </div>
                </div>
                <div className="text-right">
                  <div className={`px-2 py-1 rounded-lg text-xs font-medium mb-2 ${getStatusColor(rig.status)}`}>
                    {getStatusIcon(rig.status)} {rig.status.toUpperCase()}
                  </div>
                  <div className="text-sm text-purple-300">{rig.cosmic_efficiency}%</div>
                  <div className="text-xs text-purple-200">Efficiency</div>
                </div>
              </div>

              <div className="grid gap-2 md:grid-cols-2 mb-4">
                <div className="bg-black/30 p-3 rounded-lg">
                  <div className="text-xs text-gray-400">24h Rewards</div>
                  <div className="text-lg font-bold text-yellow-300">{rig.rewards_24h.toFixed(3)} ZION</div>
                </div>
                <div className="bg-black/30 p-3 rounded-lg">
                  <div className="text-xs text-gray-400">Power Efficiency</div>
                  <div className="text-lg font-bold text-green-300">
                    {rig.power_consumption > 0 ? (rig.hashrate / rig.power_consumption * 1000).toFixed(2) : '0'} MH/kW
                  </div>
                </div>
              </div>

              <div className="flex gap-2">
                <motion.button
                  className={`flex-1 px-3 py-2 rounded-lg text-sm ${
                    rig.status === 'mining' 
                      ? 'bg-red-600/30 hover:bg-red-600/50' 
                      : 'bg-green-600/30 hover:bg-green-600/50'
                  }`}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => toggleRigStatus(rig.id)}
                  disabled={rig.status === 'maintenance'}
                >
                  {rig.status === 'mining' ? '⏹️ Stop Mining' : '▶️ Start Mining'}
                </motion.button>
                <motion.button
                  className="flex-1 px-3 py-2 bg-blue-600/30 hover:bg-blue-600/50 rounded-lg text-sm"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  🔧 Configure
                </motion.button>
                <motion.button
                  className="flex-1 px-3 py-2 bg-purple-600/30 hover:bg-purple-600/50 rounded-lg text-sm"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  📊 Stats
                </motion.button>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Mining Pools */}
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.8 }}>
        <h2 className="text-xl font-semibold text-center mb-4 text-pink-300">🏊 Mining Pools</h2>
        
        <div className="grid gap-4">
          {miningPools.map((pool, index) => (
            <motion.div
              key={pool.name}
              className="bg-gradient-to-r from-gray-800 to-gray-900 p-6 rounded-2xl border border-pink-500/30"
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.9 + index * 0.1 }}
            >
              <div className="flex justify-between items-start mb-3">
                <div>
                  <h3 className="text-lg font-semibold text-pink-300">{pool.name}</h3>
                  <div className="text-xs text-gray-400 font-mono">{pool.url}</div>
                </div>
                <div className="text-right">
                  <div className="text-sm text-purple-300">{pool.cosmic_alignment}%</div>
                  <div className="text-xs text-purple-200">Aligned</div>
                </div>
              </div>

              <div className="grid gap-2 md:grid-cols-4 mb-4">
                <div className="bg-black/30 p-2 rounded text-center">
                  <div className="text-xs text-gray-400">Pool Fee</div>
                  <div className="text-sm font-bold text-red-300">{pool.fee}%</div>
                </div>
                <div className="bg-black/30 p-2 rounded text-center">
                  <div className="text-xs text-gray-400">Hashrate</div>
                  <div className="text-sm font-bold text-green-300">{pool.hashrate.toLocaleString()}</div>
                </div>
                <div className="bg-black/30 p-2 rounded text-center">
                  <div className="text-xs text-gray-400">Miners</div>
                  <div className="text-sm font-bold text-blue-300">{pool.miners.toLocaleString()}</div>
                </div>
                <div className="bg-black/30 p-2 rounded text-center">
                  <div className="text-xs text-gray-400">Action</div>
                  <motion.button
                    className="text-xs px-2 py-1 bg-pink-600/30 hover:bg-pink-600/50 rounded"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    🔗 Connect
                  </motion.button>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>
    </div>
  );
}
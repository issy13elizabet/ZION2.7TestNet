"use client";
import { motion } from "framer-motion";
import { useState, useEffect } from "react";

interface MiningStats {
  hash_rate: number;
  blocks_mined: number;
  cosmic_efficiency: number;
  dharma_bonus: number;
  stellar_alignment: number;
}

interface CosmicConditions {
  moon_phase: string;
  planetary_alignment: number;
  solar_activity: 'low' | 'medium' | 'high';
  merkaba_rotation: number;
  quantum_field_strength: number;
}

interface MiningPool {
  name: string;
  url: string;
  cosmic_rating: number;
  dharma_aligned: boolean;
  users_online: number;
  network_hashrate: string;
}

export default function MinerPage() {
  const [miningStats, setMiningStats] = useState<MiningStats | null>(null);
  const [cosmicConditions, setCosmicConditions] = useState<CosmicConditions | null>(null);
  const [availablePools, setAvailablePools] = useState<MiningPool[]>([]);
  const [isMining, setIsMining] = useState(false);
  const [selectedPool, setSelectedPool] = useState<string>("");
  const [walletAddress, setWalletAddress] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setTimeout(() => {
      setMiningStats({
        hash_rate: 4269.42,
        blocks_mined: 108,
        cosmic_efficiency: 94.7,
        dharma_bonus: 21.5,
        stellar_alignment: 88.8
      });

      setCosmicConditions({
        moon_phase: "Waxing Gibbous 🌔",
        planetary_alignment: 73,
        solar_activity: 'medium',
        merkaba_rotation: 1337,
        quantum_field_strength: 96
      });

      setAvailablePools([
        {
          name: "🌟 COSMIC_MINING_NEXUS",
          url: "cosmic.zion.pool",
          cosmic_rating: 97,
          dharma_aligned: true,
          users_online: 420,
          network_hashrate: "21.08 EH/s"
        },
        {
          name: "⚡ LIGHTNING_DHARMA_POOL",
          url: "lightning.dharma.zion",
          cosmic_rating: 94,
          dharma_aligned: true,
          users_online: 333,
          network_hashrate: "18.42 EH/s"
        },
        {
          name: "🔮 QUANTUM_MINING_ORACLE",
          url: "quantum.oracle.zion",
          cosmic_rating: 91,
          dharma_aligned: true,
          users_online: 666,
          network_hashrate: "25.21 EH/s"
        }
      ]);

      setLoading(false);
    }, 2000);
  }, []);

  const toggleMining = () => {
    setIsMining(!isMining);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-black text-white p-6 flex items-center justify-center">
        <motion.div className="text-center">
          <motion.div 
            className="text-8xl mb-4"
            animate={{ 
              rotate: [0, 360],
              scale: [1, 1.3, 1],
            }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            ⛏️
          </motion.div>
          <h2 className="text-2xl font-semibold mb-2">Initializing Cosmic Mining Rig...</h2>
          <p className="text-orange-300">Calibrating quantum hash algorithms...</p>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-black text-white p-6">
      <motion.header
        className="text-center mb-8"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="text-5xl font-bold bg-gradient-to-r from-orange-400 via-red-400 to-yellow-400 bg-clip-text text-transparent mb-2">
          ⛏️ Cosmic Mining Station
        </h1>
        <p className="text-orange-300 text-lg">Quantum-Enhanced Proof of Work with Dharma Optimization</p>
      </motion.header>

      {/* Mining Status & Controls */}
      <motion.div 
        className="mb-8 bg-black/30 border border-orange-500/30 rounded-2xl p-6 backdrop-blur-sm"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <div className="flex flex-col lg:flex-row gap-6 items-center justify-between">
          <div className="text-center lg:text-left">
            <h2 className="text-2xl font-semibold text-orange-200 mb-2">Mining Control Center</h2>
            <p className="text-gray-300">Status: <span className={isMining ? 'text-green-400' : 'text-red-400'}>
              {isMining ? '🟢 MINING ACTIVE' : '🔴 MINING STOPPED'}
            </span></p>
          </div>
          <motion.button
            className={`px-8 py-4 rounded-2xl font-bold text-xl ${
              isMining 
                ? 'bg-gradient-to-r from-red-500 to-pink-600 hover:from-red-600 hover:to-pink-700' 
                : 'bg-gradient-to-r from-green-500 to-blue-600 hover:from-green-600 hover:to-blue-700'
            } transition-all duration-300`}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={toggleMining}
          >
            {isMining ? '⏹️ Stop Mining' : '▶️ Start Mining'}
          </motion.button>
        </div>
      </motion.div>

      {/* Mining Stats */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-5 mb-8">
        <motion.div className="bg-gradient-to-br from-orange-500 to-red-600 p-1 rounded-2xl" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.2 }}>
          <div className="bg-black/70 rounded-xl p-6 text-center">
            <div className="text-3xl mb-2">⚡</div>
            <div className="text-xl font-bold text-orange-300">{miningStats?.hash_rate}</div>
            <div className="text-sm text-orange-200">H/s</div>
            <div className="text-xs text-orange-100 mt-1">Hash Rate</div>
          </div>
        </motion.div>

        <motion.div className="bg-gradient-to-br from-blue-500 to-purple-600 p-1 rounded-2xl" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.3 }}>
          <div className="bg-black/70 rounded-xl p-6 text-center">
            <div className="text-3xl mb-2">🧊</div>
            <div className="text-xl font-bold text-blue-300">{miningStats?.blocks_mined}</div>
            <div className="text-sm text-blue-200">Blocks</div>
            <div className="text-xs text-blue-100 mt-1">Mined Today</div>
          </div>
        </motion.div>

        <motion.div className="bg-gradient-to-br from-green-500 to-teal-600 p-1 rounded-2xl" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.4 }}>
          <div className="bg-black/70 rounded-xl p-6 text-center">
            <div className="text-3xl mb-2">🎯</div>
            <div className="text-xl font-bold text-green-300">{miningStats?.cosmic_efficiency}%</div>
            <div className="text-sm text-green-200">Efficiency</div>
            <div className="text-xs text-green-100 mt-1">Cosmic Aligned</div>
          </div>
        </motion.div>

        <motion.div className="bg-gradient-to-br from-purple-500 to-pink-600 p-1 rounded-2xl" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.5 }}>
          <div className="bg-black/70 rounded-xl p-6 text-center">
            <div className="text-3xl mb-2">☸️</div>
            <div className="text-xl font-bold text-purple-300">{miningStats?.dharma_bonus}%</div>
            <div className="text-sm text-purple-200">Dharma Bonus</div>
            <div className="text-xs text-purple-100 mt-1">Karma Reward</div>
          </div>
        </motion.div>

        <motion.div className="bg-gradient-to-br from-yellow-500 to-orange-600 p-1 rounded-2xl" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.6 }}>
          <div className="bg-black/70 rounded-xl p-6 text-center">
            <div className="text-3xl mb-2">🌟</div>
            <div className="text-xl font-bold text-yellow-300">{miningStats?.stellar_alignment}%</div>
            <div className="text-sm text-yellow-200">Stellar Align</div>
            <div className="text-xs text-yellow-100 mt-1">Cosmic Sync</div>
          </div>
        </motion.div>
      </div>

      {/* Cosmic Mining Conditions */}
      <motion.div 
        className="mb-8 bg-black/30 border border-purple-500/30 rounded-2xl p-6 backdrop-blur-sm"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7 }}
      >
        <h2 className="text-xl font-semibold text-purple-200 mb-4 text-center">🌌 Cosmic Mining Conditions</h2>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
          <div className="text-center p-4 bg-purple-900/30 rounded-xl border border-purple-500/30">
            <div className="text-lg font-bold text-purple-300">{cosmicConditions?.moon_phase}</div>
            <div className="text-sm text-purple-200">Moon Phase</div>
          </div>
          <div className="text-center p-4 bg-blue-900/30 rounded-xl border border-blue-500/30">
            <div className="text-lg font-bold text-blue-300">{cosmicConditions?.planetary_alignment}%</div>
            <div className="text-sm text-blue-200">Planetary Align</div>
          </div>
          <div className="text-center p-4 bg-yellow-900/30 rounded-xl border border-yellow-500/30">
            <div className={`text-lg font-bold capitalize ${
              cosmicConditions?.solar_activity === 'high' ? 'text-red-400' :
              cosmicConditions?.solar_activity === 'medium' ? 'text-yellow-400' : 'text-green-400'
            }`}>
              {cosmicConditions?.solar_activity}
            </div>
            <div className="text-sm text-yellow-200">Solar Activity</div>
          </div>
          <div className="text-center p-4 bg-green-900/30 rounded-xl border border-green-500/30">
            <div className="text-lg font-bold text-green-300">{cosmicConditions?.merkaba_rotation}</div>
            <div className="text-sm text-green-200">Merkaba RPM</div>
          </div>
          <div className="text-center p-4 bg-pink-900/30 rounded-xl border border-pink-500/30">
            <div className="text-lg font-bold text-pink-300">{cosmicConditions?.quantum_field_strength}%</div>
            <div className="text-sm text-pink-200">Quantum Field</div>
          </div>
        </div>
      </motion.div>

      {/* Mining Pool Selection */}
      <motion.div 
        className="mb-8 bg-black/30 border border-blue-500/30 rounded-2xl p-6 backdrop-blur-sm"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
      >
        <h2 className="text-xl font-semibold text-blue-200 mb-4 text-center">🏊 Cosmic Mining Pools</h2>
        <div className="grid gap-4">
          {availablePools.map((pool, i) => (
            <motion.div
              key={pool.name}
              className={`p-4 rounded-xl border cursor-pointer transition-all ${
                selectedPool === pool.url
                  ? 'bg-blue-900/50 border-blue-400/50 scale-105'
                  : 'bg-black/40 border-gray-600/30 hover:border-blue-500/50'
              }`}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.9 + i * 0.1 }}
              onClick={() => setSelectedPool(pool.url)}
            >
              <div className="flex justify-between items-center">
                <div>
                  <h3 className="font-semibold text-white mb-1">{pool.name}</h3>
                  <p className="text-sm text-gray-400">{pool.url}</p>
                </div>
                <div className="text-right">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-sm">Cosmic Rating:</span>
                    <div className="flex items-center gap-1">
                      <div className="w-16 bg-gray-700 rounded-full h-1">
                        <div className="bg-green-400 h-1 rounded-full" style={{ width: `${pool.cosmic_rating}%` }} />
                      </div>
                      <span className="text-green-400 text-xs">{pool.cosmic_rating}%</span>
                    </div>
                  </div>
                  <div className="text-xs text-gray-400">
                    👥 {pool.users_online} miners | ⚡ {pool.network_hashrate}
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Configuration */}
      <motion.div 
        className="bg-black/30 border border-green-500/30 rounded-2xl p-6 backdrop-blur-sm"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 1.1 }}
      >
        <h2 className="text-xl font-semibold text-green-200 mb-4 text-center">⚙️ Quantum Configuration</h2>
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              💰 ZION Wallet Address (Cosmic Receiver)
            </label>
            <input
              type="text"
              value={walletAddress}
              onChange={(e) => setWalletAddress(e.target.value)}
              placeholder="zion1cosmic42dharma108enlightenment21m..."
              className="w-full p-3 bg-black/50 border border-gray-600 rounded-xl text-white placeholder-gray-400 focus:border-green-500 focus:outline-none"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              🧵 Mining Threads (Quantum Cores)
            </label>
            <select className="w-full p-3 bg-black/50 border border-gray-600 rounded-xl text-white focus:border-green-500 focus:outline-none">
              <option>Auto-detect (Recommended)</option>
              <option>1 Thread</option>
              <option>2 Threads</option>
              <option>4 Threads</option>
              <option>8 Threads</option>
            </select>
          </div>
        </div>
        
        <div className="mt-6 p-4 bg-green-900/30 rounded-xl border border-green-500/30">
          <h3 className="font-semibold text-green-200 mb-2">🔮 Cosmic Mining Prophecy</h3>
          <p className="text-sm text-green-100">
            Current conditions are {cosmicConditions?.quantum_field_strength && cosmicConditions.quantum_field_strength > 95 ? 'OPTIMAL' : 'GOOD'} for mining. 
            {cosmicConditions?.moon_phase?.includes('Waxing') ? ' Waxing moon phase increases dharma bonus by 21%.' : ''}
            {cosmicConditions?.planetary_alignment && cosmicConditions.planetary_alignment > 70 ? ' Planetary alignment enhances quantum field resonance.' : ''}
          </p>
        </div>
      </motion.div>
    </div>
  );
}
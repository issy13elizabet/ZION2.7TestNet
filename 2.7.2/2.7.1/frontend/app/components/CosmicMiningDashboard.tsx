"use client";

import { motion } from "framer-motion";
import { useState, useEffect } from "react";
import { useCosmicSounds } from "./CosmicSounds";

interface MiningStats {
  hashrate: number;
  difficulty: number;
  blockHeight: number;
  totalShares: number;
  acceptedShares: number;
  rejectedShares: number;
  miners: number;
  temperature: number;
  power: number;
  efficiency: number;
}

interface PoolStats {
  connected: boolean;
  ping: number;
  luck: number;
  effort: number;
  rewards: number;
  lastBlockTime: number;
}

const cosmicMiningMantras = [
  "🕉️ Sacred hashrate flows through cosmic circuits! 🕉️",
  "⚡ Divine algorithms unlock universal truth! ⚡", 
  "🌟 Mining sacred blocks in the dharmic blockchain! 🌟",
  "🔮 Computational consciousness evolves through proof-of-work! 🔮",
  "⚛️ From quantum fluctuations to golden hashes! ⚛️"
];

export default function CosmicMiningDashboard() {
  const [stats, setStats] = useState<MiningStats>({
    hashrate: 0,
    difficulty: 0,
    blockHeight: 0,
    totalShares: 0,
    acceptedShares: 0,
    rejectedShares: 0,
    miners: 0,
    temperature: 0,
    power: 0,
    efficiency: 0
  });

  const [poolStats, setPoolStats] = useState<PoolStats>({
    connected: false,
    ping: 0,
    luck: 0,
    effort: 0,
    rewards: 0,
    lastBlockTime: 0
  });

  const [currentMantra, setCurrentMantra] = useState(0);
  const [cosmicEnergy, setCosmicEnergy] = useState(108);
  const [soundEnabled, setSoundEnabled] = useState(false);

  const sounds = useCosmicSounds();

  // Simulate mining data
  useEffect(() => {
    const updateStats = () => {
      setStats(prev => ({
        ...prev,
        hashrate: 2500000 + (Math.random() * 500000), // 2.5 MH/s base
        difficulty: 1200000 + (Math.random() * 300000),
        blockHeight: 144000 + Math.floor(Date.now() / 120000),
        totalShares: prev.totalShares + Math.floor(Math.random() * 10) + 1,
        acceptedShares: prev.acceptedShares + Math.floor(Math.random() * 8) + 1,
        rejectedShares: prev.rejectedShares + (Math.random() > 0.95 ? 1 : 0),
        miners: 3 + Math.floor(Math.random() * 2),
        temperature: 65 + Math.random() * 10,
        power: 150 + Math.random() * 50,
        efficiency: 15 + Math.random() * 5
      }));

      setPoolStats({
        connected: Math.random() > 0.1, // 90% uptime
        ping: 45 + Math.random() * 20,
        luck: 80 + Math.random() * 40,
        effort: 95 + Math.random() * 10,
        rewards: Math.random() * 50 + 25,
        lastBlockTime: Date.now() - (Math.random() * 600000) // within 10 min
      });
    };

    updateStats();
    const interval = setInterval(updateStats, 5000);
    return () => clearInterval(interval);
  }, []);

  // Rotate mantras
  useEffect(() => {
    const mantraInterval = setInterval(() => {
      setCurrentMantra((prev) => (prev + 1) % cosmicMiningMantras.length);
      setCosmicEnergy(prev => (prev % 108) + 1);
    }, 4000);

    return () => clearInterval(mantraInterval);
  }, []);

  const formatHashrate = (hashrate: number) => {
    if (hashrate >= 1e9) return `${(hashrate / 1e9).toFixed(2)} GH/s`;
    if (hashrate >= 1e6) return `${(hashrate / 1e6).toFixed(2)} MH/s`;
    if (hashrate >= 1e3) return `${(hashrate / 1e3).toFixed(2)} KH/s`;
    return `${hashrate.toFixed(2)} H/s`;
  };

  const getStatusColor = (connected: boolean) => connected ? "text-green-400" : "text-red-400";
  const getEfficiencyColor = (efficiency: number) => {
    if (efficiency > 18) return "text-green-400";
    if (efficiency > 15) return "text-yellow-400";
    return "text-red-400";
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-900">
      {/* Cosmic Mining Particles */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        {Array.from({ length: 30 }).map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-2 h-2 bg-yellow-400 rounded-full opacity-70"
            animate={{
              x: [0, window.innerWidth || 1920],
              y: [Math.random() * (window.innerHeight || 1080), Math.random() * (window.innerHeight || 1080)],
              rotate: [0, 360],
            }}
            transition={{
              duration: Math.random() * 15 + 10,
              repeat: Infinity,
              ease: "linear",
            }}
            style={{
              left: Math.random() * 100 + "%",
              top: Math.random() * 100 + "%",
            }}
          />
        ))}
      </div>

      <div className="relative z-10 container mx-auto px-6 py-8">
        {/* Header */}
        <motion.header
          className="text-center mb-12"
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <motion.h1 
            className="text-6xl font-bold bg-gradient-to-r from-orange-400 via-yellow-400 to-red-400 bg-clip-text text-transparent mb-4"
            animate={{ 
              backgroundPosition: ["0%", "100%", "0%"],
              scale: [1, 1.02, 1]
            }}
            transition={{ duration: 3, repeat: Infinity }}
          >
            ⛏️ COSMIC MINING COMMAND CENTER ⛏️
          </motion.h1>
          
          <motion.div
            className="text-xl text-cyan-300 mb-6 font-mono"
            animate={{ opacity: [0.7, 1, 0.7] }}
            transition={{ duration: 3, repeat: Infinity }}
          >
            {cosmicMiningMantras[currentMantra]}
          </motion.div>

          <div className="flex justify-center items-center space-x-8 text-lg">
            <motion.div 
              className="flex items-center space-x-2"
              animate={{ scale: [1, 1.05, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              <span className="text-yellow-400">🔥 Mining Power:</span>
              <span className={getStatusColor(poolStats.connected)}>{formatHashrate(stats.hashrate)}</span>
            </motion.div>
            <div className="flex items-center space-x-2">
              <span className="text-purple-400">🌌 Cosmic Energy:</span>
              <span className="text-green-400 font-bold">{cosmicEnergy}/108</span>
            </div>
            <motion.button
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg border transition-all ${
                soundEnabled 
                  ? 'bg-orange-500/20 border-orange-400 text-orange-300' 
                  : 'bg-gray-500/20 border-gray-600 text-gray-400'
              }`}
              onClick={() => {
                setSoundEnabled(!soundEnabled);
                if (!soundEnabled) {
                  sounds.playCosmicAmbient();
                  sounds.playHashrateBoost();
                } else {
                  sounds.stopAmbient();
                }
              }}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <span>{soundEnabled ? '🎵' : '🔇'}</span>
              <span className="text-sm">Mining Audio</span>
            </motion.button>
          </div>
        </motion.header>

        {/* Main Stats Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-8 mb-8">
          
          {/* Pool Connection Status */}
          <motion.div
            className="bg-black/40 backdrop-blur-md rounded-xl p-6 border border-cyan-500/30"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.1 }}
          >
            <h3 className="text-xl font-bold text-cyan-300 mb-4 flex items-center">
              🌐 Pool Connection
              <motion.div
                className={`ml-3 w-3 h-3 rounded-full ${poolStats.connected ? 'bg-green-400' : 'bg-red-400'}`}
                animate={{ opacity: [0.3, 1, 0.3] }}
                transition={{ duration: 2, repeat: Infinity }}
              />
            </h3>
            
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-300">Status:</span>
                <span className={getStatusColor(poolStats.connected)}>
                  {poolStats.connected ? "🟢 CONNECTED" : "🔴 DISCONNECTED"}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Ping:</span>
                <span className="text-yellow-400">{poolStats.ping.toFixed(0)}ms</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Pool Luck:</span>
                <span className="text-purple-400">{poolStats.luck.toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Effort:</span>
                <span className="text-green-400">{poolStats.effort.toFixed(1)}%</span>
              </div>
            </div>
          </motion.div>

          {/* Hashrate & Performance */}
          <motion.div
            className="bg-black/40 backdrop-blur-md rounded-xl p-6 border border-yellow-500/30"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2 }}
          >
            <h3 className="text-xl font-bold text-yellow-300 mb-4">⚡ Performance Metrics</h3>
            
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-300">Hashrate:</span>
                <motion.span 
                  className="text-orange-400 font-bold"
                  animate={{ scale: [1, 1.05, 1] }}
                  transition={{ duration: 2, repeat: Infinity }}
                >
                  {formatHashrate(stats.hashrate)}
                </motion.span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Temperature:</span>
                <span className="text-red-400">{stats.temperature.toFixed(1)}°C</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Power Draw:</span>
                <span className="text-pink-400">{stats.power.toFixed(0)}W</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Efficiency:</span>
                <span className={getEfficiencyColor(stats.efficiency)}>
                  {stats.efficiency.toFixed(1)} MH/J
                </span>
              </div>
            </div>
          </motion.div>

          {/* Shares & Blocks */}
          <motion.div
            className="bg-black/40 backdrop-blur-md rounded-xl p-6 border border-green-500/30"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.3 }}
          >
            <h3 className="text-xl font-bold text-green-300 mb-4">🎯 Shares & Results</h3>
            
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-300">Accepted:</span>
                <span className="text-green-400">{stats.acceptedShares}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Rejected:</span>
                <span className="text-red-400">{stats.rejectedShares}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Success Rate:</span>
                <span className="text-cyan-400">
                  {stats.totalShares > 0 ? ((stats.acceptedShares / stats.totalShares) * 100).toFixed(1) : 0}%
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-300">Block Height:</span>
                <span className="text-purple-400">#{stats.blockHeight}</span>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Live Mining Visualization */}
        <motion.div
          className="bg-black/40 backdrop-blur-md rounded-xl p-6 border border-purple-500/30 mb-8"
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <h3 className="text-2xl font-bold text-purple-300 mb-6">🔮 Live Mining Visualization</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <h4 className="text-lg text-cyan-300">⚙️ System Status</h4>
              <div className="bg-purple-900/20 rounded-lg p-4">
                {Array.from({ length: 8 }).map((_, i) => (
                  <motion.div
                    key={i}
                    className="flex items-center space-x-2 mb-2"
                    animate={{ opacity: [0.5, 1, 0.5] }}
                    transition={{ duration: 2 + i * 0.2, repeat: Infinity }}
                  >
                    <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                    <span className="text-sm text-gray-300">Mining Thread {i + 1}</span>
                    <span className="text-xs text-yellow-400 ml-auto">
                      {(stats.hashrate / 8 / 1000).toFixed(1)} KH/s
                    </span>
                  </motion.div>
                ))}
              </div>
            </div>
            
            <div className="space-y-4">
              <h4 className="text-lg text-orange-300">🎲 Recent Activity</h4>
              <div className="bg-orange-900/20 rounded-lg p-4 max-h-48 overflow-y-auto">
                {Array.from({ length: 6 }).map((_, i) => (
                  <motion.div
                    key={i}
                    className="text-xs text-gray-300 mb-1 opacity-70"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 0.7, x: 0 }}
                    transition={{ delay: i * 0.1 }}
                  >
                    [{new Date(Date.now() - i * 30000).toLocaleTimeString()}] Share submitted: 
                    <span className="text-green-400 ml-1">ACCEPTED</span>
                  </motion.div>
                ))}
              </div>
            </div>
          </div>
        </motion.div>

        {/* Cosmic Footer */}
        <motion.footer
          className="text-center mt-12"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
        >
          <motion.div
            className="text-lg text-yellow-400 mb-2"
            animate={{ y: [0, -3, 0] }}
            transition={{ duration: 3, repeat: Infinity }}
          >
            ⛏️ "Mining sacred blocks in the cosmic dharmic ledger!" ⛏️
          </motion.div>
          <p className="text-sm text-gray-500">
            ZION Mining Pool • Powered by Cosmic Hash Power • Jai Ram Ram Ram! 🕉️
          </p>
        </motion.footer>
      </div>
    </div>
  );
}
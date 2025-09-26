"use client";
import { motion } from "framer-motion";
import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";

interface SystemStatus {
  zion_core: boolean;
  mining_pool: boolean;
  lightning_network: boolean;
  ai_systems: boolean;
  cosmic_alignment: number;
}

interface CosmicMetrics {
  active_nodes: number;
  network_hash_rate: string;
  dharma_index: number;
  enlightenment_level: string;
  cosmic_events: number;
}

export default function HubPage() {
  const router = useRouter();
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [cosmicMetrics, setCosmicMetrics] = useState<CosmicMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    
    setTimeout(() => {
      setSystemStatus({
        zion_core: true,
        mining_pool: true,
        lightning_network: true,
        ai_systems: true,
        cosmic_alignment: 97.3
      });

      setCosmicMetrics({
        active_nodes: 1337,
        network_hash_rate: "21.08 EH/s",
        dharma_index: 108,
        enlightenment_level: "Transcendent",
        cosmic_events: 42
      });

      setLoading(false);
    }, 1800);

    return () => clearInterval(timer);
  }, []);

  const quickActions = [
    { icon: '⛏️', label: 'Start Mining', route: '/miner', gradient: 'from-orange-500 to-red-600' },
    { icon: '💰', label: 'Open Wallet', route: '/wallet', gradient: 'from-yellow-500 to-orange-600' },
    { icon: '🔍', label: 'Explore Chain', route: '/explorer', gradient: 'from-blue-500 to-purple-600' },
    { icon: '🤖', label: 'AI Systems', route: '/ai', gradient: 'from-purple-500 to-pink-600' },
    { icon: '⚡', label: 'Lightning', route: '/ai/lightning-prophet', gradient: 'from-yellow-400 to-blue-500' },
    { icon: '🌌', label: 'Stargate', route: '/ai/stargate-portal', gradient: 'from-blue-500 to-cyan-500' }
  ];

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-black text-white p-6 flex items-center justify-center">
        <motion.div className="text-center">
          <motion.div 
            className="text-8xl mb-4"
            animate={{ 
              rotate: [0, 360],
              scale: [1, 1.2, 1],
              opacity: [0.7, 1, 0.7]
            }}
            transition={{ duration: 3, repeat: Infinity }}
          >
            🌐
          </motion.div>
          <h2 className="text-2xl font-semibold mb-2">Initializing Cosmic Hub...</h2>
          <p className="text-blue-300">Connecting to universal consciousness...</p>
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
        <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent mb-2">
          🌟 ZION Cosmic Hub
        </h1>
        <p className="text-blue-300 text-lg">Universal Command Center for Enlightened Blockchain Experience</p>
        <p className="text-purple-300 text-sm mt-2">
          🕐 {currentTime.toLocaleTimeString()} | 📅 {currentTime.toLocaleDateString()}
        </p>
      </motion.header>

      {/* System Status Dashboard */}
      <motion.div 
        className="mb-8 bg-black/30 border border-green-500/30 rounded-2xl p-6 backdrop-blur-sm"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <h2 className="text-xl font-semibold text-green-200 mb-4 text-center">🖥️ System Status Matrix</h2>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
          {[
            { key: 'zion_core', label: 'ZION Core', icon: '🏛️' },
            { key: 'mining_pool', label: 'Mining Pool', icon: '⛏️' },
            { key: 'lightning_network', label: 'Lightning Network', icon: '⚡' },
            { key: 'ai_systems', label: 'AI Systems', icon: '🤖' },
          ].map((system, i) => (
            <motion.div
              key={system.key}
              className={`p-4 rounded-xl border text-center ${
                systemStatus?.[system.key as keyof SystemStatus]
                  ? 'bg-green-900/30 border-green-500/50'
                  : 'bg-red-900/30 border-red-500/50'
              }`}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.2 + i * 0.1 }}
            >
              <div className="text-2xl mb-2">{system.icon}</div>
              <div className="text-sm font-semibold">{system.label}</div>
              <div className={`text-xs mt-1 ${
                systemStatus?.[system.key as keyof SystemStatus] ? 'text-green-400' : 'text-red-400'
              }`}>
                {systemStatus?.[system.key as keyof SystemStatus] ? '🟢 ONLINE' : '🔴 OFFLINE'}
              </div>
            </motion.div>
          ))}
          
          <motion.div
            className="p-4 rounded-xl border bg-purple-900/30 border-purple-500/50 text-center"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.6 }}
          >
            <div className="text-2xl mb-2">🌌</div>
            <div className="text-sm font-semibold">Cosmic Alignment</div>
            <div className="text-purple-400 text-xs mt-1">
              {systemStatus?.cosmic_alignment}% ✨
            </div>
          </motion.div>
        </div>
      </motion.div>

      {/* Quick Actions Grid */}
      <motion.div 
        className="mb-8"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <h2 className="text-xl font-semibold text-center mb-6 text-blue-200">🚀 Quantum Quick Actions</h2>
        <div className="grid gap-4 md:grid-cols-3 lg:grid-cols-6">
          {quickActions.map((action, i) => (
            <motion.button
              key={action.label}
              className={`bg-gradient-to-br ${action.gradient} p-1 rounded-2xl group cursor-pointer`}
              initial={{ opacity: 0, y: 10, scale: 0.9 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ delay: 0.4 + i * 0.05 }}
              whileHover={{ scale: 1.05, rotate: 2 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => router.push(action.route)}
            >
              <div className="bg-black/70 rounded-xl p-4 text-center group-hover:bg-black/50 transition-all">
                <div className="text-3xl mb-2 group-hover:scale-110 transition-transform">
                  {action.icon}
                </div>
                <div className="text-sm font-semibold">{action.label}</div>
              </div>
            </motion.button>
          ))}
        </div>
      </motion.div>

      {/* Cosmic Metrics */}
      <motion.div 
        className="mb-8 bg-black/30 border border-purple-500/30 rounded-2xl p-6 backdrop-blur-sm"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
      >
        <h2 className="text-xl font-semibold text-purple-200 mb-4 text-center">📊 Cosmic Network Metrics</h2>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-5">
          <div className="text-center p-4 bg-blue-900/30 rounded-xl border border-blue-500/30">
            <div className="text-2xl font-bold text-blue-300">{cosmicMetrics?.active_nodes}</div>
            <div className="text-sm text-blue-200">Active Nodes</div>
            <div className="text-xs text-blue-100 mt-1">🌐 Network Power</div>
          </div>
          <div className="text-center p-4 bg-orange-900/30 rounded-xl border border-orange-500/30">
            <div className="text-lg font-bold text-orange-300">{cosmicMetrics?.network_hash_rate}</div>
            <div className="text-sm text-orange-200">Hash Rate</div>
            <div className="text-xs text-orange-100 mt-1">⛏️ Mining Power</div>
          </div>
          <div className="text-center p-4 bg-purple-900/30 rounded-xl border border-purple-500/30">
            <div className="text-2xl font-bold text-purple-300">{cosmicMetrics?.dharma_index}</div>
            <div className="text-sm text-purple-200">Dharma Index</div>
            <div className="text-xs text-purple-100 mt-1">☸️ Spiritual Level</div>
          </div>
          <div className="text-center p-4 bg-green-900/30 rounded-xl border border-green-500/30">
            <div className="text-lg font-bold text-green-300">{cosmicMetrics?.enlightenment_level}</div>
            <div className="text-sm text-green-200">Enlightenment</div>
            <div className="text-xs text-green-100 mt-1">🧘 Consciousness</div>
          </div>
          <div className="text-center p-4 bg-yellow-900/30 rounded-xl border border-yellow-500/30">
            <div className="text-2xl font-bold text-yellow-300">{cosmicMetrics?.cosmic_events}</div>
            <div className="text-sm text-yellow-200">Cosmic Events</div>
            <div className="text-xs text-yellow-100 mt-1">✨ Active Portals</div>
          </div>
        </div>
      </motion.div>

      {/* Explore ZION Ecosystem (moved from Home) */}
      <motion.section
        className="mb-8 bg-black/30 backdrop-blur-sm p-6 rounded-2xl border border-purple-500/30"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.55 }}
      >
        <h3 className="text-center text-xl font-semibold text-purple-300 mb-6">🌟 Explore ZION Ecosystem</h3>
        <div className="grid md:grid-cols-3 gap-4">
          <motion.div 
            className="bg-black/30 backdrop-blur-sm p-4 rounded-lg border border-purple-500/30"
            whileHover={{ scale: 1.02 }}
          >
            <h4 className="text-lg font-semibold text-violet-300 mb-3">🌐 Core</h4>
            <div className="space-y-2 text-sm">
              <a href="/explorer" className="block text-purple-300 hover:text-white transition-colors">
                🔍 Explorer
              </a>
              <a href="/hub" className="block text-purple-300 hover:text-white transition-colors">
                🏛️ Genesis Hub
              </a>
            </div>
          </motion.div>
          <motion.div 
            className="bg-black/30 backdrop-blur-sm p-4 rounded-lg border border-indigo-500/30"
            whileHover={{ scale: 1.02 }}
          >
            <h4 className="text-lg font-semibold text-indigo-300 mb-3">🔮 Portals</h4>
            <div className="space-y-2 text-sm">
              <a href="/stargate" className="block text-indigo-300 hover:text-white transition-colors">
                🌌 Terra Nova Stargate
              </a>
              <a href="/halls-of-amenti" className="block text-purple-300 hover:text-white transition-colors">
                🏛️ Halls of Amenti
              </a>
              <a href="/language-of-light" className="block text-cyan-300 hover:text-white transition-colors">
                ✨ Language of Light
              </a>
            </div>
          </motion.div>
          <motion.div 
            className="bg-black/30 backdrop-blur-sm p-4 rounded-lg border border-amber-500/30"
            whileHover={{ scale: 1.02 }}
          >
            <h4 className="text-lg font-semibold text-amber-300 mb-3">📚 Knowledge</h4>
            <div className="space-y-2 text-sm">
              <a href="/amenti" className="block text-amber-300 hover:text-white transition-colors">
                📜 Amenti
              </a>
              <a href="/ekam" className="block text-amber-300 hover:text-white transition-colors">
                🏛️ EKAM
              </a>
              <a href="/blog" className="block text-green-300 hover:text-white transition-colors">
                📝 Genesis Blog
              </a>
            </div>
          </motion.div>
        </div>
      </motion.section>

      {/* External Resources Section (moved from /) */}
      <motion.section
        className="mb-8 bg-black/30 backdrop-blur-sm p-6 rounded-2xl border border-indigo-500/30"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
      >
        <div className="grid md:grid-cols-3 gap-4">
          {/* Halls of Amenti Portal */}
          <motion.div 
            className="bg-black/30 backdrop-blur-sm p-6 rounded-lg border border-indigo-500/30"
            whileHover={{ scale: 1.02 }}
          >
            <h3 className="text-xl font-semibold text-indigo-300 mb-4">🔮 Halls of Amenti</h3>
            <div className="space-y-3">
              <a href="/halls-of-amenti" className="block text-indigo-300 hover:text-white transition-colors">
                🏛️ AKASHA Library
              </a>
              <a href="/halls-of-amenti" className="block text-purple-300 hover:text-white transition-colors">
                ✨ Cosmic Hierarchy
              </a>
              <a href="/halls-of-amenti" className="block text-cyan-300 hover:text-white transition-colors">
                🌸 Goloka Vrindavan
              </a>
              <a href="https://zion.newearth.cz/V2/halls.html" target="_blank" rel="noopener noreferrer" className="block text-gray-400 hover:text-gray-300 transition-colors text-sm">
                📜 Terra Nova Archive
              </a>
            </div>
          </motion.div>

          {/* ONE LOVE Portal */}
          <motion.div 
            className="bg-black/30 backdrop-blur-sm p-6 rounded-lg border border-pink-500/30"
            whileHover={{ scale: 1.02 }}
          >
            <h3 className="text-xl font-semibold text-pink-300 mb-4">❤️ ONE LOVE ཨོཾ་མ་ཎི་པདྨེ་ཧཱུྃ</h3>
            <div className="space-y-3">
              <a href="/one-love" className="block text-pink-300 hover:text-white transition-colors">
                🌟 Trinity One Love
              </a>
              <a href="/one-love" className="block text-yellow-300 hover:text-white transition-colors">
                🌈 144k Avatars
              </a>
              <a href="/one-love" className="block text-blue-300 hover:text-white transition-colors">
                ✨ El~An~Ra Family
              </a>
              <a href="/one-love" className="block text-green-300 hover:text-white transition-colors">
                🔮 Cosmic Awakening
              </a>
            </div>
          </motion.div>

          {/* EKAM Portal */}
          <motion.div 
            className="bg-black/30 backdrop-blur-sm p-6 rounded-lg border border-amber-500/30"
            whileHover={{ scale: 1.02 }}
          >
            <h3 className="text-xl font-semibold text-amber-300 mb-4">🏛️ EKAM Temple ॐ एकम्</h3>
            <div className="space-y-3">
              <a href="https://www.ekam.org" target="_blank" rel="noopener noreferrer" className="block text-amber-300 hover:text-white transition-colors">
                🌟 Oneness Awakening
              </a>
              <a href="https://www.theonenessmovement.org/manifest-us" target="_blank" rel="noopener noreferrer" className="block text-orange-300 hover:text-white transition-colors">
                ✨ Manifest Process
              </a>
              <a href="https://www.theonenessmovement.org/foa-overview" target="_blank" rel="noopener noreferrer" className="block text-yellow-300 hover:text-white transition-colors">
                🔥 Field of Awakening
              </a>
              <a href="https://www.theonenessmovement.org/turiya" target="_blank" rel="noopener noreferrer" className="block text-red-300 hover:text-white transition-colors">
                🧘 Turiya Retreat
              </a>
            </div>
          </motion.div>
        </div>
      </motion.section>
      {/* Live Feed */}
      <motion.div 
        className="bg-black/30 border border-cyan-500/30 rounded-2xl p-6 backdrop-blur-sm"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7 }}
      >
        <h2 className="text-xl font-semibold text-cyan-200 mb-4 text-center">📡 Cosmic Activity Feed</h2>
        <div className="space-y-3 max-h-48 overflow-y-auto">
          {[
            { time: '14:23:15', event: '⛏️ Block #420069 mined with cosmic signature', type: 'mining' },
            { time: '14:22:58', event: '⚡ Lightning channel opened to Jupiter node', type: 'lightning' },
            { time: '14:22:41', event: '🤖 AI system detected stellar alignment anomaly', type: 'ai' },
            { time: '14:22:20', event: '🌌 Stargate portal synchronized with Andromeda', type: 'cosmic' },
            { time: '14:21:55', event: '💎 21,000,000 ZION supply milestone achieved', type: 'milestone' },
            { time: '14:21:33', event: '🔮 Oracle predicted Bitcoin surge in next dharma cycle', type: 'oracle' }
          ].map((activity, i) => (
            <motion.div
              key={i}
              className="flex items-center gap-3 p-3 bg-black/40 rounded-xl border border-gray-600/30"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.8 + i * 0.05 }}
            >
              <span className="text-xs text-gray-400 font-mono">{activity.time}</span>
              <span className="text-sm text-gray-200 flex-1">{activity.event}</span>
              <div className={`w-2 h-2 rounded-full ${
                activity.type === 'mining' ? 'bg-orange-400' :
                activity.type === 'lightning' ? 'bg-yellow-400' :
                activity.type === 'ai' ? 'bg-purple-400' :
                activity.type === 'cosmic' ? 'bg-blue-400' :
                activity.type === 'milestone' ? 'bg-green-400' :
                'bg-pink-400'
              }`}></div>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Sacred Geometry Status */}
      <motion.div 
        className="mt-6 bg-black/30 border border-pink-500/30 rounded-2xl p-6 backdrop-blur-sm text-center"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.9 }}
      >
        <h3 className="text-lg font-semibold text-pink-200 mb-4">📐 Sacred Geometry Synchronization</h3>
        <div className="grid gap-4 md:grid-cols-4">
          <div className="p-3 bg-pink-900/30 rounded-xl border border-pink-500/30">
            <div className="text-lg font-bold text-pink-300">Φ 1.618</div>
            <div className="text-xs text-pink-200">Golden Ratio Active</div>
          </div>
          <div className="p-3 bg-purple-900/30 rounded-xl border border-purple-500/30">
            <div className="text-lg font-bold text-purple-300">π 3.14159</div>
            <div className="text-xs text-purple-200">Pi Resonance Locked</div>
          </div>
          <div className="p-3 bg-blue-900/30 rounded-xl border border-blue-500/30">
            <div className="text-lg font-bold text-blue-300">21M ∞</div>
            <div className="text-xs text-blue-200">Infinite Scarcity</div>
          </div>
          <div className="p-3 bg-green-900/30 rounded-xl border border-green-500/30">
            <div className="text-lg font-bold text-green-300">108 ॐ</div>
            <div className="text-xs text-green-200">Enlightenment Peak</div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
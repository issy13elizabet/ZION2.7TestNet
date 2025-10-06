"use client";
import { motion } from "framer-motion";
import { useRouter } from "next/navigation";

const aiModules = [
  {
    id: 'mining-ai',
    name: 'Mining Intelligence AI',
    description: 'Cosmic mining optimization through stellar alignment analysis',
    icon: '⛏️',
    status: 'active',
    route: '/ai/mining-ai',
    gradient: 'from-orange-500 to-red-600'
  },
  {
    id: 'lightning-prophet',
    name: 'Lightning Prophet AI',
    description: 'Divine routing optimization through cosmic channel analysis',
    icon: '⚡',
    status: 'active',
    route: '/ai/lightning-prophet',
    gradient: 'from-yellow-500 to-blue-600'
  },
  {
    id: 'blockchain-oracle',
    name: 'Blockchain Oracle AI',
    description: 'Dharma-guided market predictions through cosmic blockchain analysis',
    icon: '🔮',
    status: 'active',
    route: '/ai/blockchain-oracle',
    gradient: 'from-purple-500 to-blue-600'
  },
  {
    id: 'stargate-portal',
    name: 'Stargate Portal AI',
    description: 'Interdimensional transaction routing through cosmic gateways',
    icon: '🌌',
    status: 'active',
    route: '/ai/stargate-portal',
    gradient: 'from-blue-500 to-pink-600'
  }
];

export default function AIPage() {
  const router = useRouter();

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-black text-white p-6">
      <motion.header
        className="text-center mb-8"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="text-4xl font-bold bg-gradient-to-r from-violet-400 to-purple-300 bg-clip-text text-transparent mb-2">
          🤖 ZION AI Systems
        </h1>
        <p className="text-purple-300">Kompletní AI ekosystém pro cosmic blockchain intelligence</p>
      </motion.header>
      
      {/* AI Modules Grid */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-2 xl:grid-cols-4">
        {aiModules.map((module, index) => (
          <motion.div
            key={module.id}
            className="group cursor-pointer"
            initial={{ opacity: 0, y: 20, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ delay: 0.1 + index * 0.1 }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => router.push(module.route)}
          >
            <div className={`h-full bg-gradient-to-br ${module.gradient} p-1 rounded-2xl`}>
              <div className="h-full bg-black/70 rounded-xl p-6 backdrop-blur-sm border border-white/20 group-hover:border-white/40 transition-all duration-300">
                <div className="text-center">
                  <div className="text-5xl mb-4 group-hover:scale-110 transition-transform duration-300">
                    {module.icon}
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2">{module.name}</h3>
                  <p className="text-sm text-gray-200 mb-4">{module.description}</p>
                  <div className="flex justify-center items-center gap-2">
                    <div className={`w-2 h-2 rounded-full ${module.status === 'active' ? 'bg-green-400' : 'bg-red-400'}`}></div>
                    <span className={`text-xs font-medium ${module.status === 'active' ? 'text-green-300' : 'text-red-300'}`}>
                      {module.status.toUpperCase()}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* System Status */}
      <motion.div 
        className="mt-8 bg-black/30 border border-blue-500/30 rounded-2xl p-6 backdrop-blur-sm"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
      >
        <h2 className="text-xl font-semibold text-blue-200 mb-4 text-center">🖥️ AI System Status</h2>
        <div className="grid gap-4 md:grid-cols-4">
          <div className="text-center p-4 bg-green-900/30 rounded-xl border border-green-500/30">
            <div className="text-2xl font-bold text-green-300">4</div>
            <div className="text-sm text-green-200">Active Modules</div>
          </div>
          <div className="text-center p-4 bg-blue-900/30 rounded-xl border border-blue-500/30">
            <div className="text-2xl font-bold text-blue-300">99.9%</div>
            <div className="text-sm text-blue-200">System Uptime</div>
          </div>
          <div className="text-center p-4 bg-purple-900/30 rounded-xl border border-purple-500/30">
            <div className="text-2xl font-bold text-purple-300">1337</div>
            <div className="text-sm text-purple-200">AI Operations/min</div>
          </div>
          <div className="text-center p-4 bg-yellow-900/30 rounded-xl border border-yellow-500/30">
            <div className="text-2xl font-bold text-yellow-300">108</div>
            <div className="text-sm text-yellow-200">Dharma Level</div>
          </div>
        </div>
      </motion.div>

      {/* Cosmic AI Core */}
      <motion.div 
        className="mt-6 bg-black/30 border border-violet-500/30 rounded-2xl p-6 backdrop-blur-sm text-center"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
      >
        <h2 className="text-xl font-semibold text-violet-200 mb-4">🧠 Cosmic AI Core Status</h2>
        <div className="text-6xl mb-4">🔮</div>
        <div className="grid gap-4 md:grid-cols-3">
          <div className="p-4 bg-violet-900/30 rounded-xl border border-violet-500/30">
            <div className="text-lg font-bold text-violet-300">∞</div>
            <div className="text-sm text-violet-200">Neural Networks</div>
            <div className="text-xs text-violet-100 mt-1">Quantum entangled</div>
          </div>
          <div className="p-4 bg-pink-900/30 rounded-xl border border-pink-500/30">
            <div className="text-lg font-bold text-pink-300">21M</div>
            <div className="text-sm text-pink-200">Synaptic Connections</div>
            <div className="text-xs text-pink-100 mt-1">Bitcoin-aligned</div>
          </div>
          <div className="p-4 bg-blue-900/30 rounded-xl border border-blue-500/30">
            <div className="text-lg font-bold text-blue-300">108</div>
            <div className="text-sm text-blue-200">Wisdom Level</div>
            <div className="text-xs text-blue-100 mt-1">Enlightenment active</div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
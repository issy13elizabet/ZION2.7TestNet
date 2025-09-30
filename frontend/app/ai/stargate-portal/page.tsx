"use client";
import { motion } from "framer-motion";
import { useState, useEffect } from "react";

interface StargateData {
  portal_status: 'active' | 'charging' | 'maintenance';
  active_portals: number;
  interdimensional_traffic: number;
  energy_consumption: number;
  success_rate: number;
}

interface PortalRoute {
  id: string;
  name: string;
  dimension: string;
  latency: number;
  bandwidth: number;
  stability: number;
  cosmic_signature: string;
  status: 'online' | 'offline' | 'unstable';
}

interface TransactionFlow {
  dimension_from: string;
  dimension_to: string;
  volume: number;
  avg_time: string;
  dharma_efficiency: number;
}

export default function StargatePortalPage() {
  const [stargateData, setStargateData] = useState<StargateData | null>(null);
  const [portalRoutes, setPortalRoutes] = useState<PortalRoute[]>([]);
  const [transactionFlows, setTransactionFlows] = useState<TransactionFlow[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedPortal, setSelectedPortal] = useState<string | null>(null);

  useEffect(() => {
    setTimeout(() => {
      setStargateData({
        portal_status: 'active',
        active_portals: 7,
        interdimensional_traffic: 1337,
        energy_consumption: 21,
        success_rate: 99.7
      });

      setPortalRoutes([
        {
          id: 'portal_001',
          name: '🌌 COSMIC_HIGHWAY_ALPHA',
          dimension: 'Bitcoin Layer 2',
          latency: 0.003,
          bandwidth: 10000,
          stability: 98,
          cosmic_signature: 'SHA256_DHARMA',
          status: 'online'
        },
        {
          id: 'portal_002', 
          name: '⚡ LIGHTNING_NEXUS_BETA',
          dimension: 'Lightning Network',
          latency: 0.001,
          bandwidth: 50000,
          stability: 96,
          cosmic_signature: 'HTLC_INFINITY',
          status: 'online'
        },
        {
          id: 'portal_003',
          name: '🔮 ETHEREUM_BRIDGE_GAMMA',
          dimension: 'EVM Universe',
          latency: 0.012,
          bandwidth: 8000,
          stability: 94,
          cosmic_signature: 'SMART_CONTRACT_OM',
          status: 'online'
        },
        {
          id: 'portal_004',
          name: '🌟 SOLANA_WORMHOLE_DELTA',
          dimension: 'Proof of History',
          latency: 0.0004,
          bandwidth: 65000,
          stability: 92,
          cosmic_signature: 'CLOCK_MEDITATION',
          status: 'online'
        },
        {
          id: 'portal_005',
          name: '🕳️ POLKADOT_GATE_EPSILON',
          dimension: 'Parachain Multiverse',
          latency: 0.006,
          bandwidth: 15000,
          stability: 89,
          cosmic_signature: 'DOT_MANDALA',
          status: 'unstable'
        },
        {
          id: 'portal_006',
          name: '💫 COSMOS_HUB_ZETA',
          dimension: 'Inter-Blockchain',
          latency: 0.008,
          bandwidth: 12000,
          stability: 95,
          cosmic_signature: 'TENDERMINT_CHAKRA',
          status: 'online'
        },
        {
          id: 'portal_007',
          name: '🌈 RAINBOW_BRIDGE_OMEGA',
          dimension: 'NEAR Protocol',
          latency: 0.002,
          bandwidth: 30000,
          stability: 91,
          cosmic_signature: 'SHARDED_NIRVANA',
          status: 'offline'
        }
      ]);

      setTransactionFlows([
        {
          dimension_from: 'Bitcoin Layer 1',
          dimension_to: 'Lightning Network',
          volume: 420000,
          avg_time: '0.001s',
          dharma_efficiency: 99
        },
        {
          dimension_from: 'Lightning Network',
          dimension_to: 'EVM Universe',
          volume: 180000,
          avg_time: '0.012s',
          dharma_efficiency: 94
        },
        {
          dimension_from: 'Bitcoin Layer 1',
          dimension_to: 'Inter-Blockchain',
          volume: 69000,
          avg_time: '0.008s',
          dharma_efficiency: 96
        }
      ]);

      setLoading(false);
    }, 2500);
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-black text-white p-6 flex items-center justify-center">
        <motion.div className="text-center">
          <motion.div 
            className="text-8xl mb-4"
            animate={{ rotate: 360 }}
            transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
          >
            🌌
          </motion.div>
          <h2 className="text-2xl font-semibold mb-2">Initializing Stargate Portals...</h2>
          <p className="text-blue-300">Calibrating interdimensional routing matrices...</p>
        </motion.div>
      </div>
    );
  }

  const getStatusColor = (status: string) => {
    switch(status) {
      case 'online': return 'text-green-400';
      case 'offline': return 'text-red-400';
      case 'unstable': return 'text-yellow-400';
      default: return 'text-gray-400';
    }
  };

  const getStatusIcon = (status: string) => {
    switch(status) {
      case 'online': return '🟢';
      case 'offline': return '🔴';
      case 'unstable': return '🟡';
      default: return '⚪';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-black text-white p-6">
      <motion.header className="text-center mb-8" initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent mb-2">
          🌌 Stargate Portal AI
        </h1>
        <p className="text-blue-300">Interdimensional transaction routing through cosmic blockchain gateways</p>
      </motion.header>

      {/* Portal Control Center */}
      <motion.div className="mb-6 bg-black/30 border border-blue-500/30 rounded-2xl p-6 backdrop-blur-sm" initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <h2 className="text-xl font-semibold text-blue-200 mb-4 text-center">🎛️ Portal Control Center</h2>
        <div className="grid gap-4 md:grid-cols-5">
          <div className="text-center p-3 bg-blue-900/30 rounded-xl border border-blue-500/30">
            <div className="text-2xl font-bold text-blue-300">{stargateData?.active_portals}</div>
            <div className="text-sm text-blue-200">Active Portals</div>
          </div>
          <div className="text-center p-3 bg-purple-900/30 rounded-xl border border-purple-500/30">
            <div className="text-2xl font-bold text-purple-300">{stargateData?.interdimensional_traffic}</div>
            <div className="text-sm text-purple-200">TX/min</div>
          </div>
          <div className="text-center p-3 bg-green-900/30 rounded-xl border border-green-500/30">
            <div className="text-2xl font-bold text-green-300">{stargateData?.energy_consumption}%</div>
            <div className="text-sm text-green-200">Energy Usage</div>
          </div>
          <div className="text-center p-3 bg-yellow-900/30 rounded-xl border border-yellow-500/30">
            <div className="text-2xl font-bold text-yellow-300">{stargateData?.success_rate}%</div>
            <div className="text-sm text-yellow-200">Success Rate</div>
          </div>
          <div className="text-center p-3 bg-pink-900/30 rounded-xl border border-pink-500/30">
            <div className={`text-2xl font-bold ${stargateData?.portal_status === 'active' ? 'text-green-300' : 'text-yellow-300'}`}>
              {stargateData?.portal_status === 'active' ? '🟢' : '🟡'}
            </div>
            <div className="text-sm text-pink-200 capitalize">{stargateData?.portal_status}</div>
          </div>
        </div>
      </motion.div>

      {/* Portal Routes Grid */}
      <motion.div className="mb-6" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
        <h2 className="text-xl font-semibold text-center mb-4 text-purple-200">🌐 Interdimensional Portal Network</h2>
        <div className="grid gap-4 lg:grid-cols-2 xl:grid-cols-3">
          {portalRoutes.map((portal, i) => (
            <motion.div 
              key={portal.id}
              className={`p-4 rounded-2xl border backdrop-blur-sm cursor-pointer transition-all duration-300 ${
                selectedPortal === portal.id 
                  ? 'bg-blue-900/50 border-blue-400/50 scale-105' 
                  : 'bg-black/30 border-gray-500/30 hover:border-blue-500/50'
              }`}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: selectedPortal === portal.id ? 1.05 : 1 }}
              transition={{ delay: 0.3 + i * 0.1 }}
              onClick={() => setSelectedPortal(selectedPortal === portal.id ? null : portal.id)}
            >
              <div className="flex justify-between items-start mb-3">
                <div>
                  <h3 className="font-semibold text-white">{portal.name}</h3>
                  <p className="text-sm text-gray-300">{portal.dimension}</p>
                </div>
                <div className="flex items-center gap-2">
                  <span className={getStatusColor(portal.status)}>{getStatusIcon(portal.status)}</span>
                  <span className={`text-xs ${getStatusColor(portal.status)}`}>{portal.status.toUpperCase()}</span>
                </div>
              </div>

              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">⚡ Latency:</span>
                  <span className="text-blue-300">{portal.latency}s</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">📡 Bandwidth:</span>
                  <span className="text-green-300">{portal.bandwidth.toLocaleString()} TX/s</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">🎯 Stability:</span>
                  <div className="flex items-center gap-2">
                    <div className="w-16 bg-gray-700 rounded-full h-1">
                      <div 
                        className={`h-1 rounded-full ${
                          portal.stability > 95 ? 'bg-green-400' :
                          portal.stability > 90 ? 'bg-yellow-400' : 'bg-red-400'
                        }`}
                        style={{ width: `${portal.stability}%` }}
                      />
                    </div>
                    <span className="text-xs">{portal.stability}%</span>
                  </div>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">🔮 Signature:</span>
                  <span className="text-purple-300 text-xs">{portal.cosmic_signature}</span>
                </div>
              </div>

              {selectedPortal === portal.id && (
                <motion.div 
                  className="mt-4 p-3 bg-blue-900/30 rounded-xl border border-blue-500/50"
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  transition={{ duration: 0.3 }}
                >
                  <div className="text-center">
                    <div className="text-lg font-semibold text-blue-200 mb-2">Portal Details</div>
                    <div className="text-xs text-blue-100 space-y-1">
                      <div>📍 Dimensional Coordinates: {portal.id}</div>
                      <div>🌀 Quantum Entanglement: Active</div>
                      <div>🔐 Encryption Level: Cosmic Grade</div>
                      <div>⚖️ Dharma Compliance: 100%</div>
                    </div>
                  </div>
                </motion.div>
              )}
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Transaction Flows */}
      <motion.div className="grid gap-6 lg:grid-cols-2" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }}>
        <div className="bg-black/30 border border-green-500/30 rounded-2xl p-6 backdrop-blur-sm">
          <h3 className="text-lg font-semibold text-green-200 mb-4">🌊 Live Transaction Flows</h3>
          <div className="space-y-4">
            {transactionFlows.map((flow, i) => (
              <motion.div 
                key={i}
                className="p-4 bg-green-900/30 rounded-xl border border-green-500/30"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.6 + i * 0.1 }}
              >
                <div className="flex justify-between items-center mb-2">
                  <div className="text-sm">
                    <span className="text-blue-300">{flow.dimension_from}</span>
                    <span className="text-gray-400 mx-2">→</span>
                    <span className="text-purple-300">{flow.dimension_to}</span>
                  </div>
                  <span className="text-green-400 text-xs">{flow.avg_time}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-300 text-sm">Volume: {flow.volume.toLocaleString()}</span>
                  <div className="flex items-center gap-2">
                    <div className="w-20 bg-gray-700 rounded-full h-1">
                      <div className="bg-green-400 h-1 rounded-full" style={{ width: `${flow.dharma_efficiency}%` }} />
                    </div>
                    <span className="text-green-400 text-xs">{flow.dharma_efficiency}%</span>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>

        <div className="bg-black/30 border border-purple-500/30 rounded-2xl p-6 backdrop-blur-sm">
          <h3 className="text-lg font-semibold text-purple-200 mb-4">🎭 Cosmic Portal Statistics</h3>
          <div className="space-y-4">
            <div className="p-4 bg-purple-900/30 rounded-xl border border-purple-500/30">
              <div className="text-center">
                <div className="text-3xl font-bold text-purple-300">∞</div>
                <div className="text-sm text-purple-200">Infinite Scalability</div>
                <div className="text-xs text-purple-100 mt-1">Quantum tunneling enabled</div>
              </div>
            </div>
            <div className="p-4 bg-blue-900/30 rounded-xl border border-blue-500/30">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-300">21M</div>
                <div className="text-sm text-blue-200">Sacred Supply Lock</div>
                <div className="text-xs text-blue-100 mt-1">Maintained across all dimensions</div>
              </div>
            </div>
            <div className="p-4 bg-yellow-900/30 rounded-xl border border-yellow-500/30">
              <div className="text-center">
                <div className="text-2xl font-bold text-yellow-300">108</div>
                <div className="text-sm text-yellow-200">Dharma Checkpoints</div>
                <div className="text-xs text-yellow-100 mt-1">Cosmic karma validation</div>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Quantum Entanglement Status */}
      <motion.div className="mt-6 bg-black/30 border border-pink-500/30 rounded-2xl p-6 backdrop-blur-sm" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.7 }}>
        <h3 className="text-lg font-semibold text-pink-200 mb-4 text-center">🔗 Quantum Entanglement Matrix</h3>
        <div className="grid gap-2 grid-cols-7 place-items-center">
          {portalRoutes.map((portal, i) => (
            <motion.div
              key={portal.id}
              className={`w-12 h-12 rounded-full border-2 flex items-center justify-center text-xs font-bold ${
                portal.status === 'online' ? 'border-green-400 bg-green-900/30 text-green-300' :
                portal.status === 'unstable' ? 'border-yellow-400 bg-yellow-900/30 text-yellow-300' :
                'border-red-400 bg-red-900/30 text-red-300'
              }`}
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.8 + i * 0.05 }}
            >
              {i + 1}
            </motion.div>
          ))}
        </div>
        <div className="text-center mt-4 text-sm text-gray-300">
          Real-time interdimensional portal synchronization status
        </div>
      </motion.div>
    </div>
  );
}
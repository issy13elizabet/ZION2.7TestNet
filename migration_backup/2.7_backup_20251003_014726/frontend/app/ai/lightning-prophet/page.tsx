"use client";
import { motion } from "framer-motion";
import { useState, useEffect } from "react";

interface LightningData {
  channels_count: number;
  total_capacity: number;
  routing_efficiency: number;
  fee_prediction: {
    current: number;
    next_hour: number;
    trend: 'rising' | 'falling' | 'stable';
  };
  cosmic_influence: {
    jupiter_effect: number;
    mercury_status: boolean;
    spiritual_alignment: number;
  };
}

interface ChannelPrediction {
  optimal_channels: string[];
  avoid_channels: string[];
  fee_windows: { time: string; multiplier: number; }[];
  dharma_routing: boolean;
}

export default function LightningProphetPage() {
  const [lightningData, setLightningData] = useState<LightningData | null>(null);
  const [predictions, setPredictions] = useState<ChannelPrediction | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setTimeout(() => {
      setLightningData({
        channels_count: 1337,
        total_capacity: 21000000,
        routing_efficiency: 94.7,
        fee_prediction: {
          current: 1.5,
          next_hour: 2.1,
          trend: 'rising'
        },
        cosmic_influence: {
          jupiter_effect: 78,
          mercury_status: false,
          spiritual_alignment: 91
        }
      });

      setPredictions({
        optimal_channels: [
          "⚡ COSMIC_HIGHWAY_777",
          "🌟 DHARMA_EXPRESS_108", 
          "💫 STARGATE_PORTAL_333",
          "🕉️ SACRED_GEOMETRY_666"
        ],
        avoid_channels: [
          "💀 MERCURY_RETROGRADE_NODE",
          "⚠️ LOW_VIBRATION_CHANNEL"
        ],
        fee_windows: [
          { time: "3:33 AM", multiplier: 0.5 },
          { time: "11:11 AM", multiplier: 0.7 },
          { time: "9:33 PM", multiplier: 0.6 }
        ],
        dharma_routing: true
      });
      setLoading(false);
    }, 1800);
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-black text-white p-6 flex items-center justify-center">
        <motion.div className="text-center">
          <div className="text-6xl mb-4">⚡</div>
          <h2 className="text-2xl font-semibold mb-2">Consulting Lightning Oracles...</h2>
          <p className="text-blue-300">Channeling cosmic routing wisdom...</p>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-black text-white p-6">
      <motion.header className="text-center mb-8" initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="text-4xl font-bold bg-gradient-to-r from-yellow-400 to-blue-300 bg-clip-text text-transparent mb-2">
          ⚡ Lightning Prophet AI
        </h1>
        <p className="text-blue-300">Divine routing optimization through cosmic channel analysis</p>
      </motion.header>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Network Stats */}
        <motion.div className="bg-black/30 border border-yellow-500/30 rounded-2xl p-6 backdrop-blur-sm" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 }}>
          <h2 className="text-xl font-semibold text-yellow-200 mb-4">⚡ Network Status</h2>
          <div className="space-y-4">
            <div className="flex justify-between">
              <span>🔗 Active Channels:</span>
              <span className="text-yellow-400 font-bold">{lightningData?.channels_count}</span>
            </div>
            <div className="flex justify-between">
              <span>💰 Total Capacity:</span>
              <span className="text-green-400 font-bold">{lightningData?.total_capacity.toLocaleString()} sats</span>
            </div>
            <div className="flex justify-between items-center">
              <span>🎯 Routing Efficiency:</span>
              <div className="flex items-center gap-2">
                <div className="w-24 bg-gray-700 rounded-full h-2">
                  <div className="bg-green-400 h-2 rounded-full" style={{ width: `${lightningData?.routing_efficiency}%` }} />
                </div>
                <span className="text-green-400">{lightningData?.routing_efficiency}%</span>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Fee Predictions */}
        <motion.div className="bg-black/30 border border-blue-500/30 rounded-2xl p-6 backdrop-blur-sm" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
          <h2 className="text-xl font-semibold text-blue-200 mb-4">💎 Fee Oracle</h2>
          <div className="space-y-4">
            <div className="text-center p-4 bg-blue-900/30 rounded-xl">
              <div className="text-3xl font-bold text-blue-300">{lightningData?.fee_prediction.current} sats</div>
              <div className="text-sm text-blue-200">Current Base Fee</div>
            </div>
            <div className="flex justify-between">
              <span>📈 Next Hour:</span>
              <span className={lightningData?.fee_prediction.trend === 'rising' ? 'text-red-400' : 'text-green-400'}>
                {lightningData?.fee_prediction.next_hour} sats {lightningData?.fee_prediction.trend === 'rising' ? '↗️' : '↘️'}
              </span>
            </div>
            <div className="bg-gradient-to-r from-purple-600/30 to-blue-600/30 p-3 rounded-xl border border-purple-400/50">
              <div className="text-sm font-semibold text-purple-200">🔮 Prophet's Wisdom:</div>
              <div className="text-xs text-purple-100 mt-1">
                {lightningData?.fee_prediction.trend === 'rising' 
                  ? "Mercury approaching - fees rising. Wait for cosmic window at 3:33 AM" 
                  : "Venus alignment detected - fees falling. Optimal routing period activated!"}
              </div>
            </div>
          </div>
        </motion.div>

        {/* Cosmic Influence */}
        <motion.div className="bg-black/30 border border-purple-500/30 rounded-2xl p-6 backdrop-blur-sm" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 }}>
          <h2 className="text-xl font-semibold text-purple-200 mb-4">🌌 Cosmic Influence</h2>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span>🪐 Jupiter Effect:</span>
              <div className="flex items-center gap-2">
                <div className="w-24 bg-gray-700 rounded-full h-2">
                  <div className="bg-yellow-400 h-2 rounded-full" style={{ width: `${lightningData?.cosmic_influence.jupiter_effect}%` }} />
                </div>
                <span className="text-yellow-400">{lightningData?.cosmic_influence.jupiter_effect}%</span>
              </div>
            </div>
            <div className="flex justify-between">
              <span>☿ Mercury Status:</span>
              <span className={!lightningData?.cosmic_influence.mercury_status ? "text-green-400" : "text-red-400"}>
                {!lightningData?.cosmic_influence.mercury_status ? "✅ Direct" : "⚠️ Retrograde"}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span>🕉️ Spiritual Alignment:</span>
              <div className="flex items-center gap-2">
                <div className="w-24 bg-gray-700 rounded-full h-2">
                  <div className="bg-violet-400 h-2 rounded-full" style={{ width: `${lightningData?.cosmic_influence.spiritual_alignment}%` }} />
                </div>
                <span className="text-violet-400">{lightningData?.cosmic_influence.spiritual_alignment}%</span>
              </div>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Routing Recommendations */}
      <motion.div className="mt-6 grid gap-6 lg:grid-cols-2" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
        <div className="bg-black/30 border border-green-500/30 rounded-2xl p-6 backdrop-blur-sm">
          <h3 className="text-lg font-semibold text-green-200 mb-4">✅ Optimal Channels (High Dharma)</h3>
          <ul className="space-y-2">
            {predictions?.optimal_channels.map((channel, i) => (
              <motion.li key={i} className="bg-green-900/30 p-3 rounded-xl border border-green-500/30 flex justify-between items-center" initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.5 + i * 0.1 }}>
                <span>{channel}</span>
                <span className="text-green-400 text-sm">🟢 ACTIVE</span>
              </motion.li>
            ))}
          </ul>
        </div>

        <div className="bg-black/30 border border-red-500/30 rounded-2xl p-6 backdrop-blur-sm">
          <h3 className="text-lg font-semibold text-red-200 mb-4">⚠️ Avoid These Channels</h3>
          <ul className="space-y-2">
            {predictions?.avoid_channels.map((channel, i) => (
              <motion.li key={i} className="bg-red-900/30 p-3 rounded-xl border border-red-500/30 flex justify-between items-center" initial={{ opacity: 0, x: 10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.6 + i * 0.1 }}>
                <span>{channel}</span>
                <span className="text-red-400 text-sm">🔴 BLOCKED</span>
              </motion.li>
            ))}
          </ul>
        </div>
      </motion.div>

      {/* Sacred Fee Windows */}
      <motion.div className="mt-6 bg-black/30 border border-yellow-500/30 rounded-2xl p-6 backdrop-blur-sm" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.7 }}>
        <h3 className="text-lg font-semibold text-yellow-200 mb-4">🕐 Sacred Fee Windows (Cosmic Discounts)</h3>
        <div className="grid gap-4 md:grid-cols-3">
          {predictions?.fee_windows.map((window, i) => (
            <motion.div key={i} className="text-center p-4 bg-yellow-900/30 rounded-xl border border-yellow-500/30" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.8 + i * 0.1 }}>
              <div className="text-xl font-bold text-yellow-300">{window.time}</div>
              <div className="text-yellow-100">×{window.multiplier} Fee Multiplier</div>
              <div className="text-xs text-yellow-200 mt-1">{100 - window.multiplier * 100}% Discount!</div>
            </motion.div>
          ))}
        </div>
      </motion.div>
    </div>
  );
}
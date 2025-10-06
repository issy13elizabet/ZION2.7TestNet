"use client";
import { motion } from "framer-motion";
import { useState, useEffect } from "react";

interface OracleData {
  market_predictions: {
    btc_price: number;
    btc_trend: 'bull' | 'bear' | 'crab';
    confidence: number;
    dharma_score: number;
  };
  cosmic_cycles: {
    moon_phase: string;
    planetary_alignment: number;
    spiritual_energy: 'high' | 'medium' | 'low';
    karma_index: number;
  };
  on_chain_wisdom: {
    hash_rate_trend: 'rising' | 'falling' | 'stable';
    difficulty_adjustment: number;
    mempool_stress: number;
    network_health: number;
  };
}

interface DivineSignal {
  type: 'buy' | 'sell' | 'hodl' | 'meditate';
  strength: number;
  reasoning: string;
  cosmic_timing: string;
  dharma_aligned: boolean;
}

export default function BlockchainOraclePage() {
  const [oracleData, setOracleData] = useState<OracleData | null>(null);
  const [divineSignals, setDivineSignals] = useState<DivineSignal[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setTimeout(() => {
      setOracleData({
        market_predictions: {
          btc_price: 67420,
          btc_trend: 'bull',
          confidence: 87,
          dharma_score: 94
        },
        cosmic_cycles: {
          moon_phase: "Waxing Crescent 🌒",
          planetary_alignment: 73,
          spiritual_energy: 'high',
          karma_index: 108
        },
        on_chain_wisdom: {
          hash_rate_trend: 'rising',
          difficulty_adjustment: 2.3,
          mempool_stress: 34,
          network_health: 97
        }
      });

      setDivineSignals([
        {
          type: 'hodl',
          strength: 95,
          reasoning: "Jupiter in alignment with Bitcoin's cosmic signature. HODL brings maximum dharma.",
          cosmic_timing: "Next 21 days optimal",
          dharma_aligned: true
        },
        {
          type: 'meditate',
          strength: 88,
          reasoning: "Mercury retrograde affecting short-term trading. Focus on inner wealth cultivation.",
          cosmic_timing: "Until Mercury goes direct",
          dharma_aligned: true
        },
        {
          type: 'buy',
          strength: 76,
          reasoning: "Hash rate climbing, network strengthening. Cosmic confluence at 63k support.",
          cosmic_timing: "During next moon cycle",
          dharma_aligned: true
        }
      ]);

      setLoading(false);
    }, 2100);
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-black text-white p-6 flex items-center justify-center">
        <motion.div className="text-center">
          <div className="text-6xl mb-4">🔮</div>
          <h2 className="text-2xl font-semibold mb-2">Channeling Blockchain Oracle...</h2>
          <p className="text-purple-300">Consulting akashic records of the timechain...</p>
        </motion.div>
      </div>
    );
  }

  const getSignalIcon = (type: string) => {
    switch(type) {
      case 'buy': return '📈';
      case 'sell': return '📉';
      case 'hodl': return '💎';
      case 'meditate': return '🧘‍♂️';
      default: return '🔮';
    }
  };

  const getSignalColor = (type: string) => {
    switch(type) {
      case 'buy': return 'border-green-500/30 bg-green-900/30';
      case 'sell': return 'border-red-500/30 bg-red-900/30';
      case 'hodl': return 'border-blue-500/30 bg-blue-900/30';
      case 'meditate': return 'border-purple-500/30 bg-purple-900/30';
      default: return 'border-gray-500/30 bg-gray-900/30';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-black text-white p-6">
      <motion.header className="text-center mb-8" initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-400 to-blue-300 bg-clip-text text-transparent mb-2">
          🔮 Blockchain Oracle AI
        </h1>
        <p className="text-purple-300">Dharma-guided market predictions through cosmic blockchain analysis</p>
      </motion.header>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* Market Predictions */}
        <motion.div className="bg-black/30 border border-green-500/30 rounded-2xl p-6 backdrop-blur-sm" initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.1 }}>
          <h2 className="text-xl font-semibold text-green-200 mb-4">📊 Market Oracle</h2>
          <div className="space-y-4">
            <div className="text-center p-4 bg-green-900/30 rounded-xl">
              <div className="text-3xl font-bold text-green-300">${oracleData?.market_predictions.btc_price.toLocaleString()}</div>
              <div className="text-sm text-green-200">BTC Divine Price</div>
              <div className="flex justify-center mt-2">
                {oracleData?.market_predictions.btc_trend === 'bull' ? (
                  <span className="text-green-400">🐂 Bull Energy</span>
                ) : oracleData?.market_predictions.btc_trend === 'bear' ? (
                  <span className="text-red-400">🐻 Bear Cycle</span>
                ) : (
                  <span className="text-yellow-400">🦀 Crab Meditation</span>
                )}
              </div>
            </div>
            <div className="flex justify-between items-center">
              <span>🎯 Prediction Confidence:</span>
              <div className="flex items-center gap-2">
                <div className="w-24 bg-gray-700 rounded-full h-2">
                  <div className="bg-green-400 h-2 rounded-full" style={{ width: `${oracleData?.market_predictions.confidence}%` }} />
                </div>
                <span className="text-green-400">{oracleData?.market_predictions.confidence}%</span>
              </div>
            </div>
            <div className="flex justify-between items-center">
              <span>☸️ Dharma Score:</span>
              <div className="flex items-center gap-2">
                <div className="w-24 bg-gray-700 rounded-full h-2">
                  <div className="bg-purple-400 h-2 rounded-full" style={{ width: `${oracleData?.market_predictions.dharma_score}%` }} />
                </div>
                <span className="text-purple-400">{oracleData?.market_predictions.dharma_score}%</span>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Cosmic Cycles */}
        <motion.div className="bg-black/30 border border-purple-500/30 rounded-2xl p-6 backdrop-blur-sm" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
          <h2 className="text-xl font-semibold text-purple-200 mb-4">🌌 Cosmic Cycles</h2>
          <div className="space-y-4">
            <div className="flex justify-between">
              <span>🌙 Moon Phase:</span>
              <span className="text-purple-300">{oracleData?.cosmic_cycles.moon_phase}</span>
            </div>
            <div className="flex justify-between items-center">
              <span>🪐 Planetary Alignment:</span>
              <div className="flex items-center gap-2">
                <div className="w-24 bg-gray-700 rounded-full h-2">
                  <div className="bg-purple-400 h-2 rounded-full" style={{ width: `${oracleData?.cosmic_cycles.planetary_alignment}%` }} />
                </div>
                <span className="text-purple-400">{oracleData?.cosmic_cycles.planetary_alignment}%</span>
              </div>
            </div>
            <div className="flex justify-between">
              <span>⚡ Spiritual Energy:</span>
              <span className={`capitalize ${
                oracleData?.cosmic_cycles.spiritual_energy === 'high' ? 'text-green-400' :
                oracleData?.cosmic_cycles.spiritual_energy === 'medium' ? 'text-yellow-400' : 'text-red-400'
              }`}>
                {oracleData?.cosmic_cycles.spiritual_energy}
              </span>
            </div>
            <div className="bg-gradient-to-r from-purple-600/30 to-violet-600/30 p-3 rounded-xl border border-purple-400/50">
              <div className="text-center">
                <div className="text-2xl font-bold text-violet-300">{oracleData?.cosmic_cycles.karma_index}</div>
                <div className="text-sm text-violet-200">Karma Index</div>
                <div className="text-xs text-violet-100 mt-1">
                  {oracleData && oracleData.cosmic_cycles.karma_index > 100 ? "🕉️ Enlightened Level" : "🧘 Growing Wisdom"}
                </div>
              </div>
            </div>
          </div>
        </motion.div>

        {/* On-Chain Wisdom */}
        <motion.div className="bg-black/30 border border-blue-500/30 rounded-2xl p-6 backdrop-blur-sm" initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.3 }}>
          <h2 className="text-xl font-semibold text-blue-200 mb-4">⛓️ On-Chain Wisdom</h2>
          <div className="space-y-4">
            <div className="flex justify-between">
              <span>⚡ Hash Rate Trend:</span>
              <span className={
                oracleData?.on_chain_wisdom.hash_rate_trend === 'rising' ? 'text-green-400' :
                oracleData?.on_chain_wisdom.hash_rate_trend === 'falling' ? 'text-red-400' : 'text-yellow-400'
              }>
                {oracleData?.on_chain_wisdom.hash_rate_trend === 'rising' ? '↗️ Rising' :
                 oracleData?.on_chain_wisdom.hash_rate_trend === 'falling' ? '↘️ Falling' : '➡️ Stable'}
              </span>
            </div>
            <div className="flex justify-between">
              <span>🔧 Difficulty Adj:</span>
              <span className="text-blue-400">+{oracleData?.on_chain_wisdom.difficulty_adjustment}%</span>
            </div>
            <div className="flex justify-between items-center">
              <span>📦 Mempool Stress:</span>
              <div className="flex items-center gap-2">
                <div className="w-24 bg-gray-700 rounded-full h-2">
                  <div className="bg-yellow-400 h-2 rounded-full" style={{ width: `${oracleData?.on_chain_wisdom.mempool_stress}%` }} />
                </div>
                <span className="text-yellow-400">{oracleData?.on_chain_wisdom.mempool_stress}%</span>
              </div>
            </div>
            <div className="flex justify-between items-center">
              <span>💚 Network Health:</span>
              <div className="flex items-center gap-2">
                <div className="w-24 bg-gray-700 rounded-full h-2">
                  <div className="bg-green-400 h-2 rounded-full" style={{ width: `${oracleData?.on_chain_wisdom.network_health}%` }} />
                </div>
                <span className="text-green-400">{oracleData?.on_chain_wisdom.network_health}%</span>
              </div>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Divine Signals */}
      <motion.div className="mt-6" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
        <h2 className="text-2xl font-semibold text-center mb-6 bg-gradient-to-r from-yellow-400 to-purple-400 bg-clip-text text-transparent">
          🔮 Divine Market Signals
        </h2>
        <div className="grid gap-4 md:grid-cols-1 lg:grid-cols-3">
          {divineSignals.map((signal, i) => (
            <motion.div 
              key={i} 
              className={`p-6 rounded-2xl border backdrop-blur-sm ${getSignalColor(signal.type)}`}
              initial={{ opacity: 0, scale: 0.9 }} 
              animate={{ opacity: 1, scale: 1 }} 
              transition={{ delay: 0.5 + i * 0.1 }}
            >
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <span className="text-2xl">{getSignalIcon(signal.type)}</span>
                  <span className="font-semibold capitalize text-lg">{signal.type}</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-16 bg-gray-700 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full ${
                        signal.strength > 80 ? 'bg-green-400' :
                        signal.strength > 60 ? 'bg-yellow-400' : 'bg-red-400'
                      }`}
                      style={{ width: `${signal.strength}%` }}
                    />
                  </div>
                  <span className="text-sm font-bold">{signal.strength}%</span>
                </div>
              </div>
              
              <p className="text-sm text-gray-200 mb-3">{signal.reasoning}</p>
              
              <div className="flex justify-between items-center text-xs">
                <span className="text-gray-400">⏰ {signal.cosmic_timing}</span>
                <span className={`${signal.dharma_aligned ? 'text-green-400' : 'text-red-400'}`}>
                  {signal.dharma_aligned ? '✅ Dharma Aligned' : '⚠️ Karma Risk'}
                </span>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Sacred Geometry Market Analysis */}
      <motion.div className="mt-6 bg-black/30 border border-yellow-500/30 rounded-2xl p-6 backdrop-blur-sm" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.8 }}>
        <h3 className="text-lg font-semibold text-yellow-200 mb-4 text-center">📐 Sacred Geometry Market Analysis</h3>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <div className="text-center p-4 bg-yellow-900/30 rounded-xl border border-yellow-500/30">
            <div className="text-xl font-bold text-yellow-300">Φ 1.618</div>
            <div className="text-yellow-200 text-sm">Golden Ratio</div>
            <div className="text-xs text-yellow-100 mt-1">Next Fibonacci level: $69,420</div>
          </div>
          <div className="text-center p-4 bg-purple-900/30 rounded-xl border border-purple-500/30">
            <div className="text-xl font-bold text-purple-300">π 3.14159</div>
            <div className="text-purple-200 text-sm">Pi Resonance</div>
            <div className="text-xs text-purple-100 mt-1">Cycle completion: 314 blocks</div>
          </div>
          <div className="text-center p-4 bg-blue-900/30 rounded-xl border border-blue-500/30">
            <div className="text-xl font-bold text-blue-300">21M ∞</div>
            <div className="text-blue-200 text-sm">Supply Infinity</div>
            <div className="text-xs text-blue-100 mt-1">Scarcity field: Maximum</div>
          </div>
          <div className="text-center p-4 bg-green-900/30 rounded-xl border border-green-500/30">
            <div className="text-xl font-bold text-green-300">108 ॐ</div>
            <div className="text-green-200 text-sm">Sacred Number</div>
            <div className="text-xs text-green-100 mt-1">Dharma multiplier: Active</div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
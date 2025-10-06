"use client";
import { motion } from "framer-motion";
import { useState, useEffect } from "react";

interface CosmicConditions {
  solar_activity: number;
  lunar_phase: string;
  mercury_retrograde: boolean;
  crystal_grid_active: boolean;
  dharma_alignment: number;
}

interface MiningPrediction {
  optimal_time: string;
  expected_hashrate_boost: number;
  cosmic_multiplier: number;
  recommendations: string[];
}

export default function MiningAIPage() {
  const [cosmicData, setCosmicData] = useState<CosmicConditions | null>(null);
  const [prediction, setPrediction] = useState<MiningPrediction | null>(null);
  const [realMiningData, setRealMiningData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  // Fetch enhanced AI mining data from API
  const fetchRealMiningData = async () => {
    try {
      const response = await fetch('/api/ai/mining');
      const data = await response.json();
      if (data.success) {
        setRealMiningData(data.raw_data);
        
        // Use enhanced AI predictions from API
        setCosmicData(data.cosmic_conditions);
        setPrediction({
          optimal_time: data.ai_predictions.optimal_time,
          expected_hashrate_boost: data.ai_predictions.expected_hashrate_boost,
          cosmic_multiplier: data.ai_predictions.cosmic_multiplier,
          recommendations: data.ai_predictions.recommendations
        });
      }
    } catch (error) {
      console.error('Failed to fetch real mining data:', error);
      // Fallback to mock data
      setCosmicData({
        solar_activity: 87,
        lunar_phase: "Connecting... 🌔",
        mercury_retrograde: false,
        crystal_grid_active: true,
        dharma_alignment: 94
      });

      setPrediction({
        optimal_time: "Connecting to ZION 2.7 Bridge...",
        expected_hashrate_boost: 0,
        cosmic_multiplier: 1.0,
        recommendations: [
          "🔌 Connecting to ZION 2.7 backend...",
          "⚡ Please ensure bridge server is running on port 18088",
          "🌟 Fallback cosmic data active",
        ]
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRealMiningData();
    // Update every 10 seconds
    const interval = setInterval(fetchRealMiningData, 10000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-black text-white p-6">
        <motion.div 
          className="flex items-center justify-center min-h-screen"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <div className="text-center">
            <div className="text-6xl mb-4">🔮</div>
            <h2 className="text-2xl font-semibold mb-2">Channeling Cosmic Data...</h2>
            <p className="text-purple-300">Consulting Akashic Records for optimal mining conditions...</p>
          </div>
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
        <h1 className="text-4xl font-bold bg-gradient-to-r from-violet-400 to-purple-300 bg-clip-text text-transparent mb-2">
          🧠 Mining Intelligence AI
        </h1>
        <p className="text-purple-300">Cosmic-powered mining optimization using sacred geometry & astral data</p>
      </motion.header>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Cosmic Conditions */}
        <motion.div 
          className="bg-black/30 border border-purple-500/30 rounded-2xl p-6 backdrop-blur-sm"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
        >
          <h2 className="text-xl font-semibold text-purple-200 mb-4">🌌 Current Cosmic Conditions</h2>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span>☀️ Solar Activity:</span>
              <div className="flex items-center gap-2">
                <div className="w-32 bg-gray-700 rounded-full h-2">
                  <div 
                    className="bg-yellow-400 h-2 rounded-full transition-all duration-1000"
                    style={{ width: `${cosmicData?.solar_activity}%` }}
                  />
                </div>
                <span className="text-yellow-400 font-semibold">{cosmicData?.solar_activity}%</span>
              </div>
            </div>

            <div className="flex justify-between">
              <span>🌙 Lunar Phase:</span>
              <span className="text-blue-300">{cosmicData?.lunar_phase}</span>
            </div>

            <div className="flex justify-between">
              <span>☿ Mercury Retrograde:</span>
              <span className={cosmicData?.mercury_retrograde ? "text-red-400" : "text-green-400"}>
                {cosmicData?.mercury_retrograde ? "⚠️ Active" : "✅ Clear"}
              </span>
            </div>

            <div className="flex justify-between">
              <span>💎 Crystal Grid:</span>
              <span className={cosmicData?.crystal_grid_active ? "text-green-400" : "text-gray-400"}>
                {cosmicData?.crystal_grid_active ? "🟢 ACTIVE" : "⚫ Dormant"}
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span>🕉️ Dharma Alignment:</span>
              <div className="flex items-center gap-2">
                <div className="w-32 bg-gray-700 rounded-full h-2">
                  <div 
                    className="bg-violet-400 h-2 rounded-full transition-all duration-1000"
                    style={{ width: `${cosmicData?.dharma_alignment}%` }}
                  />
                </div>
                <span className="text-violet-400 font-semibold">{cosmicData?.dharma_alignment}%</span>
              </div>
            </div>
          </div>
        </motion.div>

        {/* AI Predictions */}
        <motion.div 
          className="bg-black/30 border border-green-500/30 rounded-2xl p-6 backdrop-blur-sm"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
        >
          <h2 className="text-xl font-semibold text-green-200 mb-4">🔮 AI Predictions & Recommendations</h2>
          
          <div className="space-y-4">
            <div>
              <h3 className="font-semibold text-green-300 mb-2">⏰ Optimal Mining Window:</h3>
              <p className="text-green-100 bg-green-900/30 p-3 rounded-xl border border-green-500/30">
                {prediction?.optimal_time}
              </p>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="text-center p-3 bg-blue-900/30 rounded-xl border border-blue-500/30">
                <div className="text-2xl font-bold text-blue-300">+{prediction?.expected_hashrate_boost}%</div>
                <div className="text-sm text-blue-200">Expected Boost</div>
              </div>
              <div className="text-center p-3 bg-purple-900/30 rounded-xl border border-purple-500/30">
                <div className="text-2xl font-bold text-purple-300">×{prediction?.cosmic_multiplier}</div>
                <div className="text-sm text-purple-200">Cosmic Multiplier</div>
              </div>
            </div>

            <div>
              <h3 className="font-semibold text-green-300 mb-2">📋 AI Recommendations:</h3>
              <ul className="space-y-2">
                {prediction?.recommendations.map((rec, i) => (
                  <motion.li 
                    key={i}
                    className="text-sm text-gray-300 bg-gray-800/50 p-2 rounded-lg border-l-4 border-green-400"
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.3 + i * 0.1 }}
                  >
                    {rec}
                  </motion.li>
                ))}
              </ul>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Action Buttons */}
      <motion.div 
        className="mt-8 text-center space-x-4"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
      >
        <button className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white px-6 py-3 rounded-2xl font-semibold transition-all transform hover:scale-105">
          🚀 Activate Cosmic Mining Mode
        </button>
        <button className="bg-gradient-to-r from-green-600 to-teal-600 hover:from-green-700 hover:to-teal-700 text-white px-6 py-3 rounded-2xl font-semibold transition-all transform hover:scale-105">
          📊 View Detailed Analytics  
        </button>
      </motion.div>

      {/* Fun Footer */}
      <motion.div 
        className="mt-8 text-center text-sm text-gray-400"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
      >
        <p>✨ Powered by Sacred Geometry, Quantum Entanglement & Pure Dharma Energy ✨</p>
        <p className="mt-1">🕉️ "When the cosmic conditions align, the hashrate shall follow" - Ancient ZION Wisdom 🕉️</p>
      </motion.div>
    </div>
  );
}
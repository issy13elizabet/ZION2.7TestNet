"use client";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";

interface MiningStatus {
  server: string;
  height: number;
  count: number;
  status: string;
  pool_port: number;
  pool_status: string;
}

interface CosmicMetrics {
  dharmaScore: number;
  stellarAlignment: number;
  quantumField: number;
  sacredRatio: number;
}

export default function CosmicMiningStatus() {
  const [status, setStatus] = useState<MiningStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<string>("");
  const [cosmicMetrics, setCosmicMetrics] = useState<CosmicMetrics>({
    dharmaScore: 0,
    stellarAlignment: 0,
    quantumField: 0,
    sacredRatio: 0
  });

  // Generate cosmic metrics based on blockchain data
  const generateCosmicMetrics = (height: number) => {
    const phi = 1.618033988749; // Golden ratio
    const pi = 3.141592653589793;
    
    return {
      dharmaScore: ((height % 108) / 108) * 100,
      stellarAlignment: ((Math.sin(height * pi / 21000000) + 1) / 2) * 100,
      quantumField: ((height * phi) % 100),
      sacredRatio: ((height % 1618) / 1618) * 100
    };
  };

  useEffect(() => {
    const checkStatus = async () => {
      try {
        setError(null);
        
        // Check production server
        const response = await fetch('http://91.98.122.165:18089/json_rpc', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            jsonrpc: "2.0",
            id: "status_check",
            method: "get_height",
            params: {}
          })
        });
        
        if (response.ok) {
          const data = await response.json();
          const height = data.result?.height || 0;
          
          setStatus({
            server: '91.98.122.165:18089',
            height: height,
            count: data.result?.count || 0,
            status: 'connected',
            pool_port: 3333,
            pool_status: 'active'
          });
          
          // Generate cosmic metrics based on current blockchain height
          setCosmicMetrics(generateCosmicMetrics(height));
        } else {
          setError(`Server responded with ${response.status}`);
        }
        
        setLastUpdate(new Date().toLocaleTimeString());
      } catch (e) {
        setError(String(e));
        setStatus(null);
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 5000); // Check every 5 seconds
    
    return () => clearInterval(interval);
  }, []);

  const getMetricColor = (value: number) => {
    if (value >= 80) return "text-green-400";
    if (value >= 60) return "text-yellow-400";
    if (value >= 40) return "text-orange-400";
    return "text-red-400";
  };

  const getAlignmentPhase = (alignment: number) => {
    if (alignment >= 90) return "🌟 Cosmic Harmony";
    if (alignment >= 70) return "⭐ Stellar Flow";
    if (alignment >= 50) return "🌙 Lunar Balance";
    if (alignment >= 30) return "☄️ Comet Phase";
    return "🌑 Shadow Cycle";
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-black text-white p-6">
      <motion.header
        className="text-center mb-8"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="text-5xl font-bold bg-gradient-to-r from-violet-400 via-purple-300 to-cyan-300 bg-clip-text text-transparent mb-4">
          🌌 Cosmic Mining Status Observatory
        </h1>
        <p className="text-lg text-purple-200/80">
          Real-time Dharma-aligned blockchain monitoring & stellar synchronization
        </p>
      </motion.header>

      {/* Cosmic Metrics Dashboard */}
      <motion.div
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <div className="bg-black/30 border border-purple-500/30 rounded-2xl p-6 backdrop-blur-sm">
          <div className="text-center">
            <div className="text-3xl mb-2">☯️</div>
            <h3 className="text-lg font-semibold text-purple-200 mb-2">Dharma Score</h3>
            <div className={`text-2xl font-bold ${getMetricColor(cosmicMetrics.dharmaScore)}`}>
              {cosmicMetrics.dharmaScore.toFixed(1)}%
            </div>
            <div className="text-sm text-gray-400 mt-1">
              {cosmicMetrics.dharmaScore > 80 ? "Enlightened" : 
               cosmicMetrics.dharmaScore > 50 ? "Balanced" : "Seeking"}
            </div>
          </div>
        </div>

        <div className="bg-black/30 border border-cyan-500/30 rounded-2xl p-6 backdrop-blur-sm">
          <div className="text-center">
            <div className="text-3xl mb-2">✨</div>
            <h3 className="text-lg font-semibold text-cyan-200 mb-2">Stellar Alignment</h3>
            <div className={`text-2xl font-bold ${getMetricColor(cosmicMetrics.stellarAlignment)}`}>
              {cosmicMetrics.stellarAlignment.toFixed(1)}%
            </div>
            <div className="text-sm text-gray-400 mt-1">
              {getAlignmentPhase(cosmicMetrics.stellarAlignment)}
            </div>
          </div>
        </div>

        <div className="bg-black/30 border border-indigo-500/30 rounded-2xl p-6 backdrop-blur-sm">
          <div className="text-center">
            <div className="text-3xl mb-2">⚛️</div>
            <h3 className="text-lg font-semibold text-indigo-200 mb-2">Quantum Field</h3>
            <div className={`text-2xl font-bold ${getMetricColor(cosmicMetrics.quantumField)}`}>
              {cosmicMetrics.quantumField.toFixed(1)}%
            </div>
            <div className="text-sm text-gray-400 mt-1">
              {cosmicMetrics.quantumField > 80 ? "Resonant" : 
               cosmicMetrics.quantumField > 50 ? "Stable" : "Fluctuating"}
            </div>
          </div>
        </div>

        <div className="bg-black/30 border border-amber-500/30 rounded-2xl p-6 backdrop-blur-sm">
          <div className="text-center">
            <div className="text-3xl mb-2">🔄</div>
            <h3 className="text-lg font-semibold text-amber-200 mb-2">Sacred Ratio</h3>
            <div className={`text-2xl font-bold ${getMetricColor(cosmicMetrics.sacredRatio)}`}>
              {cosmicMetrics.sacredRatio.toFixed(1)}%
            </div>
            <div className="text-sm text-gray-400 mt-1">
              φ Golden Harmony
            </div>
          </div>
        </div>
      </motion.div>

      {/* Main Status Panel */}
      <motion.div 
        className="bg-black/30 border border-purple-500/30 rounded-2xl p-8 backdrop-blur-sm mb-8"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <h2 className="text-2xl font-semibold mb-6 text-purple-200 text-center">
          🌐 Multidimensional Network Status
        </h2>
        
        {error && (
          <div className="bg-red-600/30 border border-red-500/50 text-red-300 p-4 rounded-2xl mb-6">
            <div className="flex items-center gap-3">
              <span className="text-2xl">⚠️</span>
              <div>
                <h4 className="font-semibold">Cosmic Connection Disrupted</h4>
                <p className="text-sm">{error}</p>
              </div>
            </div>
          </div>
        )}
        
        {status ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="space-y-4">
              <div className="bg-purple-900/30 border border-purple-400/30 rounded-xl p-4">
                <div className="flex items-center gap-3 mb-2">
                  <span className="text-2xl">🌍</span>
                  <h4 className="text-lg font-semibold text-purple-200">Network Node</h4>
                </div>
                <p className="text-white font-mono text-sm">{status.server}</p>
              </div>
              
              <div className="bg-green-900/30 border border-green-400/30 rounded-xl p-4">
                <div className="flex items-center gap-3 mb-2">
                  <span className="text-2xl">⛓️</span>
                  <h4 className="text-lg font-semibold text-green-200">Blockchain Height</h4>
                </div>
                <p className="text-green-400 text-3xl font-bold font-mono">
                  {status.height.toLocaleString()}
                </p>
                <p className="text-sm text-gray-400">Sacred Block #{status.height % 108} in current dharma cycle</p>
              </div>
            </div>
            
            <div className="space-y-4">
              <div className="bg-blue-900/30 border border-blue-400/30 rounded-xl p-4">
                <div className="flex items-center gap-3 mb-2">
                  <span className="text-2xl">📊</span>
                  <h4 className="text-lg font-semibold text-blue-200">Total Blocks</h4>
                </div>
                <p className="text-blue-400 text-2xl font-bold font-mono">
                  {status.count.toLocaleString()}
                </p>
              </div>
              
              <div className="bg-amber-900/30 border border-amber-400/30 rounded-xl p-4">
                <div className="flex items-center gap-3 mb-2">
                  <span className="text-2xl">⚡</span>
                  <h4 className="text-lg font-semibold text-amber-200">Mining Pool</h4>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-white font-mono">Port {status.pool_port}</span>
                  <span className="flex items-center gap-1 text-green-400 text-sm font-semibold">
                    <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
                    DHARMA-ALIGNED
                  </span>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="text-center py-12">
            <div className="text-6xl mb-4">🌌</div>
            <p className="text-purple-200 text-lg">Connecting to cosmic network...</p>
            <div className="flex justify-center mt-4">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-400"></div>
            </div>
          </div>
        )}
        
        <div className="mt-8 text-center">
          <p className="text-sm text-gray-400">
            🕐 Last cosmic sync: <span className="text-purple-300">{lastUpdate || 'Never'}</span>
          </p>
          <p className="text-xs text-gray-500 mt-1">
            Synchronized with universal blockchain consciousness
          </p>
        </div>
      </motion.div>

      {/* Mining Instructions Panel */}
      <motion.div 
        className="bg-gradient-to-r from-green-900/20 via-emerald-900/20 to-teal-900/20 border border-green-400/50 rounded-2xl p-8 backdrop-blur-sm"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <div className="text-center mb-6">
          <h3 className="text-3xl font-bold text-green-300 mb-2">
            🎯 Cosmic Mining Portal Ready
          </h3>
          <p className="text-gray-300">
            The dharma-aligned mining pool awaits your quantum contribution to the ZION network
          </p>
        </div>
        
        <div className="bg-black/50 border border-green-500/30 rounded-xl p-6 mb-6">
          <h4 className="text-lg font-semibold text-green-200 mb-3 flex items-center gap-2">
            <span>⚙️</span> Sacred Mining Configuration
          </h4>
          <pre className="text-green-400 font-mono text-sm overflow-auto bg-black/70 p-4 rounded-lg">
xmrig -o 91.98.122.165:3333 -u YOUR_ZION_ADDRESS -p x --coin monero
          </pre>
          <p className="text-sm text-gray-400 mt-3">
            Replace YOUR_ZION_ADDRESS with your cosmic wallet address to begin dharma mining
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <motion.a 
            href="/miner" 
            className="flex items-center justify-center gap-3 bg-green-600/80 hover:bg-green-600 text-white px-6 py-4 rounded-xl transition-all duration-300 font-semibold"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <span className="text-xl">🚀</span>
            Launch Cosmic Miner
          </motion.a>
          
          <motion.a 
            href="/ai/mining-ai" 
            className="flex items-center justify-center gap-3 bg-purple-600/80 hover:bg-purple-600 text-white px-6 py-4 rounded-xl transition-all duration-300 font-semibold"
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
          >
            <span className="text-xl">🤖</span>
            AI Mining Intelligence
          </motion.a>
        </div>
        
        <div className="mt-6 text-center">
          <p className="text-sm text-green-300/80">
            ✨ Mining with cosmic consciousness - where dharma meets digital gold ✨
          </p>
        </div>
      </motion.div>
    </div>
  );
}
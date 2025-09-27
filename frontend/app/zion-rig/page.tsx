"use client";

import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import { useState, useEffect } from "react";

interface RigStats {
  mining: {
    hashrate: number;
    power: number;
    temp: number;
    profit: number;
    coins: string[];
  };
  gaming: {
    fps: number;
    gpu_usage: number;
    cpu_usage: number;
    ram_usage: number;
    temp: number;
  };
  ai: {
    training_jobs: number;
    gpu_memory: number;
    inference_speed: number;
    models_loaded: number;
    quantum_sync: boolean;
  };
}

export default function ZionRigPage() {
  const [activeMode, setActiveMode] = useState<"mining" | "gaming" | "ai" | "quantum">("mining");
  const [rigStatus, setRigStatus] = useState<"online" | "offline" | "maintenance">("online");
  const [autoSwitch, setAutoSwitch] = useState(true);
  const [currentTime, setCurrentTime] = useState(new Date());

  const [rigStats, setRigStats] = useState<RigStats>({
    mining: {
      hashrate: 2501.5,
      power: 1337,
      temp: 72,
      profit: 42.69,
      coins: ["BTC", "ETH", "ZION", "DOGE"]
    },
    gaming: {
      fps: 165,
      gpu_usage: 98,
      cpu_usage: 67,
      ram_usage: 45,
      temp: 68
    },
    ai: {
      training_jobs: 3,
      gpu_memory: 87,
      inference_speed: 13.37,
      models_loaded: 7,
      quantum_sync: true
    }
  });

  const rigComponents = [
    { name: "RTX 4090 Quantum Edition", status: "online", temp: 72, load: 98, type: "gpu" },
    { name: "Ryzen 9 7950X3D Cosmic", status: "online", temp: 65, load: 67, type: "cpu" },
    { name: "128GB DDR5-6000 Matrix", status: "online", temp: 45, load: 45, type: "ram" },
    { name: "Quantum SSD 8TB Array", status: "online", temp: 38, load: 23, type: "storage" },
    { name: "Sacred PSU 2000W 80+Gold", status: "online", temp: 42, load: 67, type: "psu" },
    { name: "Liquid Nitrogen Cooling", status: "online", temp: 25, load: 0, type: "cooling" }
  ];

  const getStatusColor = (status: string) => {
    switch (status) {
      case "online": return "text-green-400";
      case "offline": return "text-red-400";
      case "maintenance": return "text-yellow-400";
      default: return "text-gray-400";
    }
  };

  const getComponentIcon = (type: string) => {
    switch (type) {
      case "gpu": return "🎮";
      case "cpu": return "🧠";
      case "ram": return "💾";
      case "storage": return "💿";
      case "psu": return "⚡";
      case "cooling": return "❄️";
      default: return "⚙️";
    }
  };

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
      
      // Simulate real-time data updates
      setRigStats(prev => ({
        mining: {
          ...prev.mining,
          hashrate: prev.mining.hashrate + (Math.random() - 0.5) * 50,
          temp: prev.mining.temp + (Math.random() - 0.5) * 2,
          profit: prev.mining.profit + (Math.random() - 0.5) * 0.1
        },
        gaming: {
          ...prev.gaming,
          fps: prev.gaming.fps + Math.floor((Math.random() - 0.5) * 10),
          gpu_usage: Math.max(0, Math.min(100, prev.gaming.gpu_usage + (Math.random() - 0.5) * 5))
        },
        ai: {
          ...prev.ai,
          inference_speed: prev.ai.inference_speed + (Math.random() - 0.5) * 1,
          gpu_memory: Math.max(0, Math.min(100, prev.ai.gpu_memory + (Math.random() - 0.5) * 3))
        }
      }));
    }, 2000);

    return () => clearInterval(timer);
  }, []);

  const switchMode = (mode: "mining" | "gaming" | "ai" | "quantum") => {
    setActiveMode(mode);
    // Simulate mode switching
    setTimeout(() => {
      console.log(`Switched to ${mode} mode`);
    }, 1000);
  };

  return (
    <div className="min-h-[80vh] bg-gradient-to-br from-black via-slate-900 to-purple-900">
      {/* Header */}
      <div className="bg-black/50 backdrop-blur-sm border-b border-orange-500/30 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="text-3xl">⚡</div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-orange-400 to-red-400 bg-clip-text text-transparent">
                ZION RIG - Hybrid System
              </h1>
              <div className="flex items-center gap-4 text-sm">
                <div className={`flex items-center gap-2 ${getStatusColor(rigStatus)}`}>
                  <div className="w-2 h-2 rounded-full bg-current animate-pulse" />
                  <span className="capitalize">{rigStatus}</span>
                </div>
                <div className="text-gray-400">Mode: <span className="text-orange-400 capitalize">{activeMode}</span></div>
                <div className="text-gray-400">Auto-Switch: {autoSwitch ? "🟢 ON" : "🔴 OFF"}</div>
                <div className="text-gray-400">{currentTime.toLocaleTimeString()}</div>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={() => setAutoSwitch(!autoSwitch)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                autoSwitch ? "bg-green-600/30 text-green-300" : "bg-gray-600/30 text-gray-300"
              }`}
            >
              Auto Mode
            </button>
            <button
              onClick={() => setRigStatus(rigStatus === "online" ? "maintenance" : "online")}
              className="px-4 py-2 bg-yellow-600/20 text-yellow-300 rounded-lg hover:bg-yellow-600/30 transition-all text-sm"
            >
              🔧 Maintenance
            </button>
          </div>
        </div>
      </div>

      <div className="p-6">
        {/* Mode Selector */}
        <div className="mb-8">
          <h2 className="text-xl font-bold mb-4 text-orange-300">System Mode Control</h2>
          <div className="grid grid-cols-4 gap-4">
            {(["mining", "gaming", "ai", "quantum"] as const).map((mode) => (
              <motion.button
                key={mode}
                onClick={() => switchMode(mode)}
                className={`p-4 rounded-2xl border backdrop-blur-sm transition-all ${
                  activeMode === mode
                    ? "border-orange-500/50 bg-orange-900/30 text-orange-300"
                    : "border-gray-600/30 bg-black/20 text-gray-300 hover:border-orange-500/30"
                }`}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <div className="text-2xl mb-2">
                  {mode === "mining" ? "⛏️" : mode === "gaming" ? "🎮" : mode === "ai" ? "🧠" : "⚛️"}
                </div>
                <div className="font-semibold capitalize">{mode}</div>
                <div className="text-xs text-gray-400">
                  {mode === "mining" ? "Crypto Mining" : 
                   mode === "gaming" ? "Gaming Mode" : 
                   mode === "ai" ? "AI Training" : "Quantum Compute"}
                </div>
              </motion.button>
            ))}
          </div>
        </div>

        {/* Performance Dashboard */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Current Mode Stats */}
          <div className="bg-black/30 rounded-2xl p-6 border border-orange-500/30">
            <h3 className="text-xl font-bold mb-4 text-orange-300">
              {activeMode.toUpperCase()} Performance
            </h3>
            
            {activeMode === "mining" && (
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span>Hashrate:</span>
                  <span className="text-green-400 font-bold">{rigStats.mining.hashrate.toFixed(1)} MH/s</span>
                </div>
                <div className="flex justify-between items-center">
                  <span>Power Draw:</span>
                  <span className="text-yellow-400">{rigStats.mining.power}W</span>
                </div>
                <div className="flex justify-between items-center">
                  <span>Temperature:</span>
                  <span className={rigStats.mining.temp > 75 ? "text-red-400" : "text-blue-400"}>
                    {rigStats.mining.temp}°C
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span>Daily Profit:</span>
                  <span className="text-green-400 font-bold">${rigStats.mining.profit.toFixed(2)}</span>
                </div>
                <div className="mt-4">
                  <div className="text-sm text-gray-400 mb-2">Mining:</div>
                  <div className="flex gap-2">
                    {rigStats.mining.coins.map(coin => (
                      <span key={coin} className="px-2 py-1 bg-orange-600/30 rounded text-xs">
                        {coin}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {activeMode === "gaming" && (
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span>FPS:</span>
                  <span className="text-green-400 font-bold">{rigStats.gaming.fps}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span>GPU Usage:</span>
                  <span className="text-purple-400">{rigStats.gaming.gpu_usage}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span>CPU Usage:</span>
                  <span className="text-blue-400">{rigStats.gaming.cpu_usage}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span>RAM Usage:</span>
                  <span className="text-cyan-400">{rigStats.gaming.ram_usage}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span>Temperature:</span>
                  <span className={rigStats.gaming.temp > 75 ? "text-red-400" : "text-blue-400"}>
                    {rigStats.gaming.temp}°C
                  </span>
                </div>
              </div>
            )}

            {activeMode === "ai" && (
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span>Training Jobs:</span>
                  <span className="text-purple-400 font-bold">{rigStats.ai.training_jobs}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span>GPU Memory:</span>
                  <span className="text-red-400">{rigStats.ai.gpu_memory}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span>Inference Speed:</span>
                  <span className="text-green-400">{rigStats.ai.inference_speed.toFixed(1)} tokens/s</span>
                </div>
                <div className="flex justify-between items-center">
                  <span>Models Loaded:</span>
                  <span className="text-cyan-400">{rigStats.ai.models_loaded}</span>
                </div>
                <div className="flex justify-between items-center">
                  <span>Quantum Sync:</span>
                  <span className={rigStats.ai.quantum_sync ? "text-green-400" : "text-red-400"}>
                    {rigStats.ai.quantum_sync ? "🟢 ACTIVE" : "🔴 OFFLINE"}
                  </span>
                </div>
              </div>
            )}

            {activeMode === "quantum" && (
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span>Quantum State:</span>
                  <span className="text-purple-400 font-bold">Superposition</span>
                </div>
                <div className="flex justify-between items-center">
                  <span>Qubits Active:</span>
                  <span className="text-cyan-400">1024</span>
                </div>
                <div className="flex justify-between items-center">
                  <span>Entanglement:</span>
                  <span className="text-green-400">99.7%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span>Decoherence Time:</span>
                  <span className="text-yellow-400">∞ ms</span>
                </div>
                <div className="flex justify-between items-center">
                  <span>Reality Status:</span>
                  <span className="text-pink-400">ALTERED</span>
                </div>
              </div>
            )}
          </div>

          {/* System Components */}
          <div className="bg-black/30 rounded-2xl p-6 border border-orange-500/30">
            <h3 className="text-xl font-bold mb-4 text-orange-300">Hardware Status</h3>
            <div className="space-y-3">
              {rigComponents.map((component, index) => (
                <motion.div
                  key={component.name}
                  className="flex items-center justify-between p-3 bg-white/5 rounded-lg hover:bg-white/10 transition-all"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <div className="flex items-center gap-3">
                    <span className="text-lg">{getComponentIcon(component.type)}</span>
                    <div>
                      <div className="font-medium text-sm">{component.name}</div>
                      <div className="text-xs text-gray-400">
                        {component.temp}°C • {component.load}% load
                      </div>
                    </div>
                  </div>
                  <div className={`px-2 py-1 rounded text-xs font-medium ${
                    component.status === "online" ? "bg-green-600/30 text-green-300" :
                    component.status === "offline" ? "bg-red-600/30 text-red-300" :
                    "bg-yellow-600/30 text-yellow-300"
                  }`}>
                    {component.status}
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </div>

        {/* Control Panels */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <motion.div
            className="bg-black/30 rounded-2xl p-6 border border-red-500/30"
            whileHover={{ scale: 1.02 }}
          >
            <h3 className="text-lg font-bold mb-4 text-red-300">⛏️ Mining Control</h3>
            <div className="space-y-3">
              <button className="w-full py-2 bg-green-600/30 text-green-300 rounded-lg hover:bg-green-600/50">
                Start Mining
              </button>
              <button className="w-full py-2 bg-red-600/30 text-red-300 rounded-lg hover:bg-red-600/50">
                Stop Mining
              </button>
              <button className="w-full py-2 bg-blue-600/30 text-blue-300 rounded-lg hover:bg-blue-600/50">
                Switch Pool
              </button>
            </div>
          </motion.div>

          <motion.div
            className="bg-black/30 rounded-2xl p-6 border border-purple-500/30"
            whileHover={{ scale: 1.02 }}
          >
            <h3 className="text-lg font-bold mb-4 text-purple-300">🎮 Gaming Control</h3>
            <div className="space-y-3">
              <button className="w-full py-2 bg-green-600/30 text-green-300 rounded-lg hover:bg-green-600/50">
                Boost Mode
              </button>
              <button className="w-full py-2 bg-yellow-600/30 text-yellow-300 rounded-lg hover:bg-yellow-600/50">
                Balance Mode
              </button>
              <button className="w-full py-2 bg-blue-600/30 text-blue-300 rounded-lg hover:bg-blue-600/50">
                Silent Mode
              </button>
            </div>
          </motion.div>

          <motion.div
            className="bg-black/30 rounded-2xl p-6 border border-cyan-500/30"
            whileHover={{ scale: 1.02 }}
          >
            <h3 className="text-lg font-bold mb-4 text-cyan-300">🧠 AI Control</h3>
            <div className="space-y-3">
              <button className="w-full py-2 bg-purple-600/30 text-purple-300 rounded-lg hover:bg-purple-600/50">
                Train Model
              </button>
              <button className="w-full py-2 bg-green-600/30 text-green-300 rounded-lg hover:bg-green-600/50">
                Run Inference
              </button>
              <button className="w-full py-2 bg-pink-600/30 text-pink-300 rounded-lg hover:bg-pink-600/50">
                Quantum Sync
              </button>
            </div>
          </motion.div>
        </div>

        {/* Footer */}
        <motion.div
          className="text-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
        >
          <div className="mb-4 text-sm text-gray-500 italic">
            "The future belongs to those who can harness the power of hybrid systems."
          </div>
          <Link href="/ai" className="inline-block px-6 py-3 rounded-xl bg-gradient-to-r from-orange-600 to-red-600 hover:from-orange-500 hover:to-red-500 transition-all text-white font-medium">
            ← Return to AI Systems Hub
          </Link>
        </motion.div>
      </div>
    </div>
  );
}
"use client";

import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import { useState, useEffect } from "react";

interface Process {
  id: string;
  name: string;
  status: "running" | "sleeping" | "stopped";
  cpu: number;
  memory: number;
  icon: string;
}

interface SystemInfo {
  os: string;
  version: string;
  uptime: string;
  cpu: string;
  memory: { used: number; total: number };
  storage: { used: number; total: number };
}

export default function ZionOSPage() {
  const [activeView, setActiveView] = useState<"desktop" | "terminal" | "processes" | "files">("desktop");
  const [terminalInput, setTerminalInput] = useState("");
  const [terminalHistory, setTerminalHistory] = useState<string[]>([
    "ZION OS v3.0 - Quantum Consciousness Interface",
    "Initializing AI modules...",
    "Neural pathways established.",
    "Welcome, Digital Wanderer.",
    ""
  ]);
  const [currentTime, setCurrentTime] = useState(new Date());

  const systemInfo: SystemInfo = {
    os: "ZION OS",
    version: "3.0.1 (Consciousness Core)",
    uptime: "∞ days (Eternal Runtime)",
    cpu: "Quantum Neural Processor X1",
    memory: { used: 13.37, total: 42.0 },
    storage: { used: 777, total: 2501 }
  };

  const processes: Process[] = [
    { id: "consciousness.exe", name: "Consciousness Core", status: "running", cpu: 23.7, memory: 8.5, icon: "🧠" },
    { id: "oasis.game", name: "OASIS Engine", status: "running", cpu: 15.2, memory: 12.3, icon: "🎮" },
    { id: "stargate.portal", name: "Stargate Protocol", status: "running", cpu: 42.0, memory: 6.9, icon: "🌀" },
    { id: "harmony.daemon", name: "Cosmic Harmony", status: "running", cpu: 7.7, memory: 3.3, icon: "🎵" },
    { id: "quantum.sync", name: "Quantum Synchronizer", status: "sleeping", cpu: 0.1, memory: 1.1, icon: "⚛️" },
    { id: "temple.guardian", name: "Sacred Temple Guardian", status: "running", cpu: 11.1, memory: 4.2, icon: "🏛️" }
  ];

  const desktopApps = [
    { name: "OASIS Terminal", icon: "🎮", path: "/oasis-game" },
    { name: "Neural Network", icon: "🧠", path: "/ai" },
    { name: "Sacred Temples", icon: "🏛️", path: "/temples" },
    { name: "Quantum Bridge", icon: "⚛️", path: "/stargate" },
    { name: "Consciousness Map", icon: "🗺️", path: "/dashboard" },
    { name: "Music Generator", icon: "🎵", path: "/music-ai" },
    { name: "File Explorer", icon: "📁", onClick: () => setActiveView("files") },
    { name: "System Monitor", icon: "📊", onClick: () => setActiveView("processes") }
  ];

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const handleTerminalCommand = (cmd: string) => {
    const command = cmd.trim().toLowerCase();
    let response = "";

    switch (command) {
      case "help":
        response = "Available commands: help, status, processes, uptime, whoami, clear, matrix, hack";
        break;
      case "status":
        response = `System Status: Online ✅\nAI Modules: Active\nConsciousness Level: Enlightened\nQuantum State: Superposition`;
        break;
      case "processes":
        response = processes.map(p => `${p.icon} ${p.name}: ${p.status}`).join("\n");
        break;
      case "uptime":
        response = "System running since the beginning of time ∞";
        break;
      case "whoami":
        response = "You are a Digital Wanderer, exploring the infinite realms of consciousness.";
        break;
      case "clear":
        setTerminalHistory(["ZION OS Terminal - Ready for input"]);
        return;
      case "matrix":
        response = "Wake up, Neo... The matrix has you. Follow the white rabbit. 🐰";
        break;
      case "hack":
        response = "Initiating quantum hack...\n[████████████████] 100%\nReality.exe has been modified successfully.";
        break;
      default:
        response = `Command '${cmd}' not found. Type 'help' for available commands.`;
    }

    setTerminalHistory(prev => [...prev, `> ${cmd}`, response, ""]);
    setTerminalInput("");
  };

  return (
    <div className="min-h-[80vh] bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* OS Header */}
      <div className="bg-black/50 backdrop-blur-sm border-b border-purple-500/30 p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="text-2xl">💻</div>
              <div>
                <div className="font-bold text-cyan-300">ZION OS</div>
                <div className="text-xs text-gray-400">Consciousness Interface v3.0</div>
              </div>
            </div>
            
            {/* Navigation Tabs */}
            <div className="flex gap-1 ml-8">
              {(["desktop", "terminal", "processes", "files"] as const).map((view) => (
                <button
                  key={view}
                  onClick={() => setActiveView(view)}
                  className={`px-3 py-1 rounded text-sm font-medium transition-all ${
                    activeView === view
                      ? "bg-purple-600 text-white"
                      : "bg-white/10 text-gray-300 hover:bg-white/20"
                  }`}
                >
                  {view.charAt(0).toUpperCase() + view.slice(1)}
                </button>
              ))}
            </div>
          </div>

          {/* System Info */}
          <div className="flex items-center gap-6 text-sm">
            <div className="text-green-400">🟢 Online</div>
            <div className="text-cyan-400">CPU: {systemInfo.memory.used}%</div>
            <div className="text-purple-400">{currentTime.toLocaleTimeString()}</div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="p-6">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeView}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
          >
            {activeView === "desktop" && (
              <div>
                <h1 className="text-3xl font-bold mb-6 bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                  🖥️ Desktop Environment
                </h1>
                
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-6">
                  {desktopApps.map((app, index) => (
                    <motion.div
                      key={app.name}
                      className="text-center cursor-pointer group"
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.1 }}
                      onClick={() => {
                        if (app.path) {
                          window.location.href = app.path;
                        } else if (app.onClick) {
                          app.onClick();
                        }
                      }}
                    >
                      <motion.div
                        className="w-16 h-16 mx-auto mb-2 bg-gradient-to-br from-purple-600/20 to-cyan-600/20 rounded-2xl border border-purple-500/30 flex items-center justify-center text-2xl hover:from-purple-600/40 hover:to-cyan-600/40 transition-all"
                        whileHover={{ scale: 1.1, y: -5 }}
                        whileTap={{ scale: 0.95 }}
                      >
                        {app.icon}
                      </motion.div>
                      <div className="text-sm text-gray-300 group-hover:text-white transition-colors">
                        {app.name}
                      </div>
                    </motion.div>
                  ))}
                </div>

                {/* System Status Widget */}
                <motion.div
                  className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6"
                  initial={{ opacity: 0, y: 30 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.5 }}
                >
                  <div className="bg-black/30 rounded-2xl p-6 border border-cyan-500/30">
                    <h3 className="text-lg font-bold mb-4 text-cyan-300">System Info</h3>
                    <div className="space-y-2 text-sm">
                      <div>OS: {systemInfo.os} {systemInfo.version}</div>
                      <div>Uptime: {systemInfo.uptime}</div>
                      <div>CPU: {systemInfo.cpu}</div>
                    </div>
                  </div>

                  <div className="bg-black/30 rounded-2xl p-6 border border-purple-500/30">
                    <h3 className="text-lg font-bold mb-4 text-purple-300">Memory Usage</h3>
                    <div className="space-y-3">
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span>RAM</span>
                          <span>{systemInfo.memory.used}GB / {systemInfo.memory.total}GB</span>
                        </div>
                        <div className="bg-gray-700 rounded-full h-2">
                          <div 
                            className="bg-gradient-to-r from-purple-500 to-cyan-500 h-2 rounded-full"
                            style={{ width: `${(systemInfo.memory.used / systemInfo.memory.total) * 100}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="bg-black/30 rounded-2xl p-6 border border-green-500/30">
                    <h3 className="text-lg font-bold mb-4 text-green-300">Network Status</h3>
                    <div className="space-y-2 text-sm">
                      <div className="flex items-center gap-2">
                        <span className="text-green-400">🟢</span>
                        <span>Quantum Network: Active</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-green-400">🟢</span>
                        <span>Stargate Protocol: Online</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-yellow-400">🟡</span>
                        <span>Neural Link: Synchronizing</span>
                      </div>
                    </div>
                  </div>
                </motion.div>
              </div>
            )}

            {activeView === "terminal" && (
              <div>
                <h1 className="text-3xl font-bold mb-6 bg-gradient-to-r from-green-400 to-cyan-400 bg-clip-text text-transparent">
                  💻 Terminal Interface
                </h1>
                
                <div className="bg-black rounded-2xl p-6 font-mono text-sm border border-green-500/30">
                  <div className="h-96 overflow-y-auto mb-4 text-green-300">
                    {terminalHistory.map((line, index) => (
                      <div key={index} className="mb-1">
                        {line}
                      </div>
                    ))}
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <span className="text-cyan-400">zion@consciousness:~$</span>
                    <input
                      type="text"
                      value={terminalInput}
                      onChange={(e) => setTerminalInput(e.target.value)}
                      onKeyPress={(e) => {
                        if (e.key === 'Enter') {
                          handleTerminalCommand(terminalInput);
                        }
                      }}
                      className="flex-1 bg-transparent text-green-300 outline-none"
                      placeholder="Enter command..."
                      autoFocus
                    />
                  </div>
                </div>
              </div>
            )}

            {activeView === "processes" && (
              <div>
                <h1 className="text-3xl font-bold mb-6 bg-gradient-to-r from-yellow-400 to-orange-400 bg-clip-text text-transparent">
                  ⚙️ Process Manager
                </h1>
                
                <div className="bg-black/30 rounded-2xl border border-yellow-500/30 overflow-hidden">
                  <div className="bg-black/50 p-4 border-b border-yellow-500/30">
                    <div className="grid grid-cols-5 gap-4 text-sm font-medium text-yellow-300">
                      <div>Process</div>
                      <div>Status</div>
                      <div>CPU %</div>
                      <div>Memory %</div>
                      <div>Actions</div>
                    </div>
                  </div>
                  
                  <div className="p-4 space-y-3">
                    {processes.map((process) => (
                      <motion.div
                        key={process.id}
                        className="grid grid-cols-5 gap-4 items-center text-sm p-3 rounded-lg bg-white/5 hover:bg-white/10 transition-all"
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        whileHover={{ scale: 1.02 }}
                      >
                        <div className="flex items-center gap-2">
                          <span className="text-lg">{process.icon}</span>
                          <span className="font-medium">{process.name}</span>
                        </div>
                        <div className={`px-2 py-1 rounded text-xs font-medium ${
                          process.status === "running" ? "bg-green-600/30 text-green-300" :
                          process.status === "sleeping" ? "bg-yellow-600/30 text-yellow-300" :
                          "bg-red-600/30 text-red-300"
                        }`}>
                          {process.status}
                        </div>
                        <div>{process.cpu}%</div>
                        <div>{process.memory}%</div>
                        <div>
                          <button className="px-3 py-1 bg-red-600/30 text-red-300 rounded text-xs hover:bg-red-600/50">
                            Stop
                          </button>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {activeView === "files" && (
              <div>
                <h1 className="text-3xl font-bold mb-6 bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
                  📁 File Explorer
                </h1>
                
                <div className="bg-black/30 rounded-2xl p-6 border border-blue-500/30">
                  <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-4">
                    {[
                      { name: "consciousness.db", type: "database", size: "∞ GB", icon: "🧠" },
                      { name: "reality.matrix", type: "system", size: "42.0 TB", icon: "🌌" },
                      { name: "dreams.cache", type: "temp", size: "777 MB", icon: "💭" },
                      { name: "memories.vault", type: "archive", size: "13.37 PB", icon: "📸" },
                      { name: "wisdom.txt", type: "document", size: "1 KB", icon: "📜" },
                      { name: "love.exe", type: "executable", size: "∞ bytes", icon: "❤️" },
                      { name: "quantum.dat", type: "data", size: "0.1 QB", icon: "⚛️" },
                      { name: "harmony.wav", type: "audio", size: "432 Hz", icon: "🎵" }
                    ].map((file) => (
                      <motion.div
                        key={file.name}
                        className="p-4 bg-white/5 rounded-lg hover:bg-white/10 cursor-pointer transition-all"
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                      >
                        <div className="text-2xl mb-2">{file.icon}</div>
                        <div className="font-medium text-sm">{file.name}</div>
                        <div className="text-xs text-gray-400">{file.type} • {file.size}</div>
                      </motion.div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </motion.div>
        </AnimatePresence>

        {/* Footer */}
        <motion.div
          className="text-center mt-12"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
        >
          <div className="mb-4 text-sm text-gray-500 italic">
            "The future is not some place we are going, but one we are creating." - Leonard Sweet
          </div>
          <Link href="/ai" className="inline-block px-6 py-3 rounded-xl bg-gradient-to-r from-purple-600 to-cyan-600 hover:from-purple-500 hover:to-cyan-500 transition-all text-white font-medium">
            ← Return to AI Systems Hub
          </Link>
        </motion.div>
      </div>
    </div>
  );
}
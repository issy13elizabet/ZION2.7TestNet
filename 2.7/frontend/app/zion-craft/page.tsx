"use client";

import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import { useState, useEffect } from "react";

interface GameWorld {
  id: string;
  name: string;
  creator: string;
  players: number;
  maxPlayers: number;
  gameMode: "survival" | "creative" | "adventure" | "scripted";
  thumbnail: string;
  rating: number;
  featured: boolean;
}

interface BuildingBlock {
  id: string;
  name: string;
  type: "basic" | "decorative" | "functional" | "scripted";
  icon: string;
  rarity: "common" | "rare" | "epic" | "legendary";
}

export default function ZionCraftPage() {
  const [activeTab, setActiveTab] = useState<"play" | "create" | "worlds" | "scripts">("play");
  const [selectedWorld, setSelectedWorld] = useState<GameWorld | null>(null);
  const [isBuilding, setIsBuilding] = useState(false);
  const [playerStats, setPlayerStats] = useState({
    level: 42,
    blocksPlaced: 13337,
    worldsCreated: 7,
    scriptsWritten: 23,
    friends: 156
  });

  const gameWorlds: GameWorld[] = [
    {
      id: "zion-city",
      name: "New Zion City",
      creator: "DigitalArchitect",
      players: 847,
      maxPlayers: 1000,
      gameMode: "creative",
      thumbnail: "🏙️",
      rating: 4.9,
      featured: true
    },
    {
      id: "quantum-mines",
      name: "Quantum Crystal Mines",
      creator: "CryptoMiner42",
      players: 234,
      maxPlayers: 500,
      gameMode: "survival",
      thumbnail: "⛏️",
      rating: 4.7,
      featured: true
    },
    {
      id: "sky-kingdom",
      name: "Floating Sky Kingdom",
      creator: "CloudBuilder",
      players: 456,
      maxPlayers: 300,
      gameMode: "adventure",
      thumbnail: "☁️",
      rating: 4.8,
      featured: false
    },
    {
      id: "code-world",
      name: "Programming Playground",
      creator: "ScriptMaster",
      players: 123,
      maxPlayers: 200,
      gameMode: "scripted",
      thumbnail: "💻",
      rating: 4.6,
      featured: true
    }
  ];

  const buildingBlocks: BuildingBlock[] = [
    { id: "stone", name: "Stone Block", type: "basic", icon: "🪨", rarity: "common" },
    { id: "wood", name: "Wood Plank", type: "basic", icon: "🪵", rarity: "common" },
    { id: "glass", name: "Crystal Glass", type: "decorative", icon: "💎", rarity: "rare" },
    { id: "redstone", name: "Quantum Redstone", type: "functional", icon: "⚡", rarity: "epic" },
    { id: "portal", name: "Portal Block", type: "scripted", icon: "🌀", rarity: "legendary" },
    { id: "light", name: "Sacred Light", type: "decorative", icon: "✨", rarity: "epic" },
    { id: "water", name: "Liquid Consciousness", type: "functional", icon: "🌊", rarity: "rare" },
    { id: "fire", name: "Divine Flame", type: "functional", icon: "🔥", rarity: "epic" }
  ];

  const getRarityColor = (rarity: string) => {
    switch (rarity) {
      case "common": return "text-gray-400 border-gray-500/30";
      case "rare": return "text-blue-400 border-blue-500/30";
      case "epic": return "text-purple-400 border-purple-500/30";
      case "legendary": return "text-yellow-400 border-yellow-500/30";
      default: return "text-gray-400 border-gray-500/30";
    }
  };

  const getGameModeColor = (mode: string) => {
    switch (mode) {
      case "survival": return "text-red-400";
      case "creative": return "text-green-400";
      case "adventure": return "text-blue-400";
      case "scripted": return "text-purple-400";
      default: return "text-gray-400";
    }
  };

  const joinWorld = (world: GameWorld) => {
    setSelectedWorld(world);
    setTimeout(() => {
      console.log(`Joining ${world.name}...`);
      setSelectedWorld(null);
    }, 2000);
  };

  return (
    <div className="min-h-[80vh] bg-gradient-to-br from-green-900/20 via-blue-900/20 to-purple-900/20">
      {/* Header */}
      <div className="bg-black/50 backdrop-blur-sm border-b border-green-500/30 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="text-3xl">🧱</div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-green-400 to-blue-400 bg-clip-text text-transparent">
                ZION CRAFT UNIVERSE
              </h1>
              <div className="text-sm text-gray-400">
                Minecraft × Roblox Fusion • Infinite Possibilities
              </div>
            </div>
          </div>

          <div className="flex items-center gap-4 text-sm">
            <div className="text-green-400">🟢 Online Players: 2,501</div>
            <div className="text-blue-400">🌍 Active Worlds: 847</div>
            <div className="text-purple-400">⚡ Level {playerStats.level}</div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="p-6 pb-0">
        <div className="flex justify-center mb-8">
          <div className="flex gap-1 bg-black/40 rounded-xl p-1">
            {(["play", "create", "worlds", "scripts"] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-6 py-2 rounded-lg text-sm font-medium transition-all ${
                  activeTab === tab
                    ? "bg-gradient-to-r from-green-600 to-blue-600 text-white"
                    : "text-gray-400 hover:text-white"
                }`}
              >
                {tab === "play" ? "🎮 Play" :
                 tab === "create" ? "🔨 Create" :
                 tab === "worlds" ? "🌍 Worlds" : "💻 Scripts"}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="px-6">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
          >
            {activeTab === "play" && (
              <div>
                <h2 className="text-2xl font-bold mb-6 text-green-300">🎮 Featured Worlds</h2>
                
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {gameWorlds.filter(w => w.featured).map((world) => (
                    <motion.div
                      key={world.id}
                      className="bg-black/30 rounded-2xl border border-green-500/30 overflow-hidden hover:border-green-400/50 transition-all cursor-pointer"
                      whileHover={{ scale: 1.05, y: -5 }}
                      onClick={() => joinWorld(world)}
                    >
                      <div className="aspect-video bg-gradient-to-br from-green-600/20 to-blue-600/20 flex items-center justify-center text-6xl">
                        {world.thumbnail}
                      </div>
                      
                      <div className="p-4">
                        <div className="flex items-center justify-between mb-2">
                          <h3 className="font-bold text-lg">{world.name}</h3>
                          <div className="flex items-center gap-1">
                            <span className="text-yellow-400">⭐</span>
                            <span className="text-sm">{world.rating}</span>
                          </div>
                        </div>
                        
                        <div className="text-sm text-gray-400 mb-3">
                          by <span className="text-blue-400">{world.creator}</span>
                        </div>
                        
                        <div className="flex items-center justify-between text-sm">
                          <div className={`font-medium ${getGameModeColor(world.gameMode)}`}>
                            {world.gameMode.toUpperCase()}
                          </div>
                          <div className="text-gray-400">
                            {world.players}/{world.maxPlayers} players
                          </div>
                        </div>
                        
                        <button className="w-full mt-3 py-2 bg-gradient-to-r from-green-600 to-blue-600 rounded-lg font-medium hover:from-green-500 hover:to-blue-500 transition-all">
                          Join World
                        </button>
                      </div>
                    </motion.div>
                  ))}
                </div>

                {/* Player Stats */}
                <motion.div
                  className="mt-12 bg-black/30 rounded-2xl p-6 border border-green-500/30"
                  initial={{ opacity: 0, y: 30 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 }}
                >
                  <h3 className="text-xl font-bold mb-4 text-green-300">Your Stats</h3>
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-400">{playerStats.level}</div>
                      <div className="text-sm text-gray-400">Level</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-400">{playerStats.blocksPlaced.toLocaleString()}</div>
                      <div className="text-sm text-gray-400">Blocks Placed</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-purple-400">{playerStats.worldsCreated}</div>
                      <div className="text-sm text-gray-400">Worlds Created</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-cyan-400">{playerStats.scriptsWritten}</div>
                      <div className="text-sm text-gray-400">Scripts Written</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-pink-400">{playerStats.friends}</div>
                      <div className="text-sm text-gray-400">Friends</div>
                    </div>
                  </div>
                </motion.div>
              </div>
            )}

            {activeTab === "create" && (
              <div>
                <h2 className="text-2xl font-bold mb-6 text-green-300">🔨 Building Tools</h2>
                
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  {/* Block Palette */}
                  <div className="lg:col-span-2">
                    <h3 className="text-lg font-bold mb-4 text-blue-300">Block Palette</h3>
                    <div className="grid grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-3">
                      {buildingBlocks.map((block) => (
                        <motion.div
                          key={block.id}
                          className={`aspect-square bg-black/40 rounded-lg border-2 cursor-pointer flex flex-col items-center justify-center hover:scale-110 transition-all ${getRarityColor(block.rarity)}`}
                          whileHover={{ y: -5 }}
                          whileTap={{ scale: 0.95 }}
                        >
                          <div className="text-2xl mb-1">{block.icon}</div>
                          <div className="text-xs text-center font-medium">{block.name.split(' ')[0]}</div>
                        </motion.div>
                      ))}
                    </div>
                  </div>

                  {/* Tools Panel */}
                  <div>
                    <h3 className="text-lg font-bold mb-4 text-blue-300">Creator Tools</h3>
                    <div className="space-y-3">
                      <button 
                        onClick={() => setIsBuilding(!isBuilding)}
                        className={`w-full py-3 rounded-lg font-medium transition-all ${
                          isBuilding 
                            ? "bg-red-600/30 text-red-300 hover:bg-red-600/50"
                            : "bg-green-600/30 text-green-300 hover:bg-green-600/50"
                        }`}
                      >
                        {isBuilding ? "🛑 Stop Building" : "🔨 Start Building"}
                      </button>
                      
                      <button className="w-full py-3 bg-blue-600/30 text-blue-300 rounded-lg hover:bg-blue-600/50">
                        📐 Grid Tools
                      </button>
                      
                      <button className="w-full py-3 bg-purple-600/30 text-purple-300 rounded-lg hover:bg-purple-600/50">
                        🎨 Paint Mode
                      </button>
                      
                      <button className="w-full py-3 bg-yellow-600/30 text-yellow-300 rounded-lg hover:bg-yellow-600/50">
                        ⚡ Script Block
                      </button>
                      
                      <button className="w-full py-3 bg-pink-600/30 text-pink-300 rounded-lg hover:bg-pink-600/50">
                        💾 Save World
                      </button>
                    </div>
                  </div>
                </div>

                {/* 3D Preview Area */}
                <motion.div
                  className="mt-8 bg-gradient-to-br from-black/40 to-gray-900/40 rounded-2xl p-8 border border-green-500/30 min-h-[300px] flex items-center justify-center"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.2 }}
                >
                  <div className="text-center">
                    <div className="text-6xl mb-4">🏗️</div>
                    <h3 className="text-xl font-bold text-green-300 mb-2">3D Building Canvas</h3>
                    <p className="text-gray-400">Your creative workspace • Drag blocks to build</p>
                    {isBuilding && (
                      <motion.div
                        className="mt-4 text-green-400"
                        animate={{ opacity: [0.5, 1, 0.5] }}
                        transition={{ duration: 2, repeat: Infinity }}
                      >
                        ⚡ Building Mode Active ⚡
                      </motion.div>
                    )}
                  </div>
                </motion.div>
              </div>
            )}

            {activeTab === "worlds" && (
              <div>
                <h2 className="text-2xl font-bold mb-6 text-green-300">🌍 All Worlds</h2>
                
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  {gameWorlds.map((world) => (
                    <motion.div
                      key={world.id}
                      className="bg-black/30 rounded-xl border border-gray-600/30 p-4 hover:border-green-500/50 transition-all cursor-pointer"
                      whileHover={{ scale: 1.02 }}
                      onClick={() => joinWorld(world)}
                    >
                      <div className="text-3xl mb-2">{world.thumbnail}</div>
                      <h3 className="font-bold mb-1">{world.name}</h3>
                      <div className="text-sm text-gray-400 mb-2">by {world.creator}</div>
                      <div className="flex items-center justify-between text-sm">
                        <div className={getGameModeColor(world.gameMode)}>
                          {world.gameMode}
                        </div>
                        <div className="text-gray-400">
                          {world.players} players
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>
            )}

            {activeTab === "scripts" && (
              <div>
                <h2 className="text-2xl font-bold mb-6 text-green-300">💻 Script Editor</h2>
                
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <div>
                    <h3 className="text-lg font-bold mb-4 text-blue-300">Block Behaviors</h3>
                    <div className="bg-black/50 rounded-xl p-4 font-mono text-sm border border-purple-500/30">
                      <div className="text-green-400 mb-2">// ZION Script Example</div>
                      <div className="text-blue-400">function</div> <span className="text-yellow-400">onBlockPlace</span>() {"{"}
                      <div className="ml-4 text-gray-300">
                        <div><span className="text-cyan-400">player</span>.addXP(<span className="text-orange-400">10</span>);</div>
                        <div><span className="text-cyan-400">world</span>.spawnParticles(<span className="text-green-400">"✨"</span>);</div>
                        <div><span className="text-cyan-400">console</span>.log(<span className="text-green-400">"Block placed!"</span>);</div>
                      </div>
                      {"}"}
                    </div>
                  </div>

                  <div>
                    <h3 className="text-lg font-bold mb-4 text-blue-300">Available APIs</h3>
                    <div className="space-y-2 text-sm">
                      <div className="bg-black/30 p-3 rounded-lg">
                        <span className="text-purple-400">player</span> - Player interactions & stats
                      </div>
                      <div className="bg-black/30 p-3 rounded-lg">
                        <span className="text-cyan-400">world</span> - World manipulation & physics  
                      </div>
                      <div className="bg-black/30 p-3 rounded-lg">
                        <span className="text-green-400">blocks</span> - Block creation & destruction
                      </div>
                      <div className="bg-black/30 p-3 rounded-lg">
                        <span className="text-yellow-400">events</span> - Game events & triggers
                      </div>
                      <div className="bg-black/30 p-3 rounded-lg">
                        <span className="text-pink-400">ui</span> - Custom user interfaces
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </motion.div>
        </AnimatePresence>

        {/* Join World Modal */}
        <AnimatePresence>
          {selectedWorld && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/80 flex items-center justify-center z-50"
            >
              <motion.div
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.8, opacity: 0 }}
                className="bg-gradient-to-br from-green-900/90 to-blue-900/90 rounded-2xl p-8 border border-green-500/30 text-center"
              >
                <div className="text-6xl mb-4">{selectedWorld.thumbnail}</div>
                <h3 className="text-2xl font-bold mb-2">Joining {selectedWorld.name}</h3>
                <p className="text-gray-300 mb-6">Preparing your adventure...</p>
                <motion.div
                  className="w-64 h-2 bg-black/50 rounded-full overflow-hidden mx-auto"
                >
                  <motion.div
                    className="h-full bg-gradient-to-r from-green-500 to-blue-500"
                    initial={{ width: 0 }}
                    animate={{ width: "100%" }}
                    transition={{ duration: 2 }}
                  />
                </motion.div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Footer */}
        <motion.div
          className="text-center mt-12 pb-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
        >
          <div className="mb-4 text-sm text-gray-500 italic">
            "Build worlds, script adventures, create infinite possibilities."
          </div>
          <Link href="/ai" className="inline-block px-6 py-3 rounded-xl bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-500 hover:to-blue-500 transition-all text-white font-medium">
            ← Return to AI Systems Hub
          </Link>
        </motion.div>
      </div>
    </div>
  );
}
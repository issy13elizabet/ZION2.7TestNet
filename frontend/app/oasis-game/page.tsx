"use client";

import { motion, AnimatePresence } from "framer-motion";
import Link from "next/link";
import { useState, useEffect } from "react";

interface Player {
  id: string;
  name: string;
  level: number;
  xp: number;
  coins: number;
  avatar: string;
}

interface Quest {
  id: string;
  title: string;
  description: string;
  xpReward: number;
  coinReward: number;
  completed: boolean;
  icon: string;
}

export default function OasisGamePage() {
  const [player, setPlayer] = useState<Player>({
    id: "parzival_zion",
    name: "Digital Wanderer",
    level: 42,
    xp: 13337,
    coins: 2501,
    avatar: "🌟"
  });

  const [activeTab, setActiveTab] = useState<"quests" | "leaderboard" | "worlds">("quests");
  const [showAchievement, setShowAchievement] = useState<string | null>(null);

  const dailyQuests: Quest[] = [
    {
      id: "meditation",
      title: "Sacred Breath Alignment",
      description: "Complete 10 minutes of conscious breathing",
      xpReward: 150,
      coinReward: 25,
      completed: false,
      icon: "🧘‍♂️"
    },
    {
      id: "gratitude",
      title: "Trinity of Gratitude", 
      description: "Express gratitude for 3 aspects of your reality",
      xpReward: 100,
      coinReward: 15,
      completed: true,
      icon: "🙏"
    },
    {
      id: "kindness",
      title: "Random Act of Light",
      description: "Perform one unexpected act of kindness",
      xpReward: 200,
      coinReward: 30,
      completed: false,
      icon: "✨"
    },
    {
      id: "learning",
      title: "Wisdom Download",
      description: "Learn something new that expands your consciousness",
      xpReward: 180,
      coinReward: 20,
      completed: false,
      icon: "📚"
    }
  ];

  const worlds = [
    { name: "New Jerusalem", status: "Active", players: 1337, icon: "🏛️" },
    { name: "Crystal Realms", status: "Beta", players: 888, icon: "💎" },
    { name: "Quantum Fields", status: "Coming Soon", players: 0, icon: "⚛️" },
    { name: "Stargate Network", status: "Active", players: 2501, icon: "🌀" }
  ];

  const completeQuest = (questId: string) => {
    const quest = dailyQuests.find(q => q.id === questId);
    if (quest && !quest.completed) {
      setPlayer(prev => ({
        ...prev,
        xp: prev.xp + quest.xpReward,
        coins: prev.coins + quest.coinReward
      }));
      setShowAchievement(`+${quest.xpReward} XP, +${quest.coinReward} coins`);
      setTimeout(() => setShowAchievement(null), 3000);
    }
  };

  return (
    <div className="min-h-[80vh] relative">
      {/* Achievement Popup */}
      <AnimatePresence>
        {showAchievement && (
          <motion.div
            initial={{ opacity: 0, y: -50, scale: 0.8 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -50, scale: 0.8 }}
            className="fixed top-32 left-1/2 transform -translate-x-1/2 z-50 bg-gradient-to-r from-yellow-500 to-orange-500 text-black px-6 py-3 rounded-xl font-bold shadow-2xl"
          >
            🏆 Quest Complete! {showAchievement}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Header */}
      <motion.header
        className="text-center mb-8"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="text-5xl font-extrabold bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent mb-2">
          🎮 THE OASIS
        </h1>
        <p className="text-lg text-gray-300 mb-1">
          Ready Player One • Consciousness Edition
        </p>
        <p className="text-sm text-gray-500">
          "The OASIS is the only place that feels like I mean anything."
        </p>
      </motion.header>

      {/* Player HUD */}
      <motion.div
        className="mb-8 p-6 rounded-2xl border border-cyan-500/30 bg-gradient-to-r from-blue-900/20 to-purple-900/20 backdrop-blur-sm"
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.1 }}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="text-4xl">{player.avatar}</div>
            <div>
              <h2 className="text-xl font-bold text-cyan-300">{player.name}</h2>
              <div className="text-sm text-gray-400">Level {player.level} • ID: {player.id}</div>
            </div>
          </div>
          <div className="text-right">
            <div className="text-lg font-bold text-yellow-400">💰 {player.coins.toLocaleString()}</div>
            <div className="text-sm text-purple-400">⚡ {player.xp.toLocaleString()} XP</div>
          </div>
        </div>
        
        {/* XP Bar */}
        <div className="mt-4 bg-gray-800 rounded-full h-3 overflow-hidden">
          <motion.div
            className="h-full bg-gradient-to-r from-purple-500 to-cyan-500"
            initial={{ width: 0 }}
            animate={{ width: `${(player.xp % 1000) / 10}%` }}
            transition={{ duration: 1, delay: 0.5 }}
          />
        </div>
        <div className="text-xs text-gray-500 mt-1">
          {player.xp % 1000}/1000 XP to Level {player.level + 1}
        </div>
      </motion.div>

      {/* Navigation Tabs */}
      <div className="flex justify-center mb-8">
        <div className="flex gap-1 bg-black/40 rounded-xl p-1">
          {(["quests", "worlds", "leaderboard"] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-6 py-2 rounded-lg text-sm font-medium transition-all ${
                activeTab === tab
                  ? "bg-gradient-to-r from-purple-600 to-cyan-600 text-white"
                  : "text-gray-400 hover:text-white"
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Content Area */}
      <AnimatePresence mode="wait">
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          transition={{ duration: 0.2 }}
        >
          {activeTab === "quests" && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {dailyQuests.map((quest, index) => (
                <motion.div
                  key={quest.id}
                  className={`rounded-2xl border p-6 backdrop-blur-sm cursor-pointer transition-all hover:scale-105 ${
                    quest.completed
                      ? "border-green-500/30 bg-green-900/20"
                      : "border-purple-500/30 bg-black/30 hover:border-purple-400/50"
                  }`}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  onClick={() => !quest.completed && completeQuest(quest.id)}
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="text-2xl">{quest.icon}</div>
                    <div className="text-right text-sm">
                      <div className="text-yellow-400">+{quest.xpReward} XP</div>
                      <div className="text-green-400">+{quest.coinReward} coins</div>
                    </div>
                  </div>
                  
                  <h3 className="text-lg font-semibold mb-2 text-white">
                    {quest.title}
                  </h3>
                  <p className="text-gray-400 text-sm mb-4">
                    {quest.description}
                  </p>
                  
                  <div className={`text-center py-2 rounded-lg text-sm font-medium ${
                    quest.completed
                      ? "bg-green-600/30 text-green-300"
                      : "bg-purple-600/30 text-purple-300 hover:bg-purple-500/30"
                  }`}>
                    {quest.completed ? "✓ Complete" : "Click to Complete"}
                  </div>
                </motion.div>
              ))}
            </div>
          )}

          {activeTab === "worlds" && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {worlds.map((world, index) => (
                <motion.div
                  key={world.name}
                  className="rounded-2xl border border-cyan-500/30 bg-gradient-to-br from-blue-900/20 to-purple-900/20 backdrop-blur-sm p-6 hover:scale-105 transition-all cursor-pointer"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.15 }}
                >
                  <div className="text-3xl mb-3">{world.icon}</div>
                  <h3 className="text-xl font-bold mb-2 text-cyan-300">{world.name}</h3>
                  <div className="flex justify-between items-center mb-4">
                    <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                      world.status === "Active" ? "bg-green-600/30 text-green-300" :
                      world.status === "Beta" ? "bg-yellow-600/30 text-yellow-300" :
                      "bg-gray-600/30 text-gray-300"
                    }`}>
                      {world.status}
                    </span>
                    <span className="text-sm text-gray-400">{world.players} players</span>
                  </div>
                  <button className="w-full py-2 bg-gradient-to-r from-cyan-600 to-purple-600 rounded-lg font-medium hover:from-cyan-500 hover:to-purple-500 transition-all">
                    {world.status === "Coming Soon" ? "Coming Soon" : "Enter World"}
                  </button>
                </motion.div>
              ))}
            </div>
          )}

          {activeTab === "leaderboard" && (
            <div className="rounded-2xl border border-yellow-500/30 bg-gradient-to-br from-yellow-900/10 to-orange-900/10 backdrop-blur-sm p-8">
              <h2 className="text-2xl font-bold mb-6 text-center text-yellow-300">🏆 Global Leaderboard</h2>
              <div className="space-y-4">
                {[
                  { rank: 1, name: "Parzival", level: 99, score: 999999, avatar: "👑" },
                  { rank: 2, name: "Art3mis", level: 97, score: 888888, avatar: "🦋" },
                  { rank: 3, name: "Aech", level: 95, score: 777777, avatar: "⚡" },
                  { rank: 42, name: "Digital Wanderer", level: 42, score: 13337, avatar: "🌟" }
                ].map((entry) => (
                  <div
                    key={entry.rank}
                    className={`flex items-center justify-between p-4 rounded-xl ${
                      entry.name === "Digital Wanderer" 
                        ? "bg-purple-600/20 border border-purple-500/30" 
                        : "bg-black/20"
                    }`}
                  >
                    <div className="flex items-center gap-4">
                      <div className="text-2xl font-bold text-yellow-400">#{entry.rank}</div>
                      <div className="text-2xl">{entry.avatar}</div>
                      <div>
                        <div className="font-semibold">{entry.name}</div>
                        <div className="text-sm text-gray-400">Level {entry.level}</div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold text-cyan-400">{entry.score.toLocaleString()}</div>
                      <div className="text-sm text-gray-400">Total Score</div>
                    </div>
                  </div>
                ))}
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
        transition={{ delay: 0.5 }}
      >
        <div className="mb-4 text-sm text-gray-500 italic">
          "People come to the OASIS for all the things they can do, but they stay for all the things they can be."
        </div>
        <Link href="/ai" className="inline-block px-6 py-3 rounded-xl bg-gradient-to-r from-purple-600 to-cyan-600 hover:from-purple-500 hover:to-cyan-500 transition-all text-white font-medium">
          ← Return to AI Systems Hub
        </Link>
      </motion.div>
    </div>
  );
}

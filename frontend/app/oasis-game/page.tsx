"use client";

import { motion } from "framer-motion";
import Link from "next/link";

export default function OasisGamePage() {
  return (
    <div className="min-h-[60vh]">
      <motion.header
        className="text-center mb-8"
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="text-4xl font-extrabold bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 bg-clip-text text-transparent">
          🎮 OASIS GAME
        </h1>
        <p className="text-sm text-gray-400 mt-2">
          Playful path to mastery • Level up your consciousness
        </p>
      </motion.header>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <motion.div
          className="rounded-2xl border border-purple-500/30 bg-black/30 backdrop-blur-sm p-6"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.05 }}
        >
          <h2 className="text-xl font-semibold mb-2">Daily Quests</h2>
          <p className="text-gray-400 text-sm mb-4">
            Simple rituals to build momentum and unlock rewards.
          </p>
          <ul className="list-disc list-inside space-y-1 text-sm text-gray-300">
            <li>5m Breath Alignment</li>
            <li>Gratitude x3</li>
            <li>One Loving Action</li>
          </ul>
        </motion.div>

        <motion.div
          className="rounded-2xl border border-purple-500/30 bg-black/30 backdrop-blur-sm p-6"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <h2 className="text-xl font-semibold mb-2">Power Ups</h2>
          <p className="text-gray-400 text-sm mb-4">
            Collect sigils and mantras to boost your journey.
          </p>
          <div className="flex gap-2 text-sm">
            <span className="px-3 py-1 rounded-full bg-purple-700/40 border border-purple-500/40">⚡ Focus</span>
            <span className="px-3 py-1 rounded-full bg-blue-700/40 border border-blue-500/40">💧 Calm</span>
            <span className="px-3 py-1 rounded-full bg-pink-700/40 border border-pink-500/40">❤️ Love</span>
          </div>
        </motion.div>
      </div>

      <motion.div
        className="text-center mt-8"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.15 }}
      >
        <Link href="/ai" className="inline-block px-4 py-2 rounded-lg bg-white/10 hover:bg-white/20 border border-white/20 text-sm">
          ← Back to AI Systems
        </Link>
      </motion.div>
    </div>
  );
}

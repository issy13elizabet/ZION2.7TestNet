'use client';

import React from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';

export default function OneLoveCelebrationPostEN() {
  const consciousnessFields = [
    { id: 1, name: "Field of Gratitude", desc: "Golden ratio of nirvana - evolution of consciousness", energy: "Gratitude Field" },
    { id: 2, name: "Field of Loka/Planet", desc: "Planetary consciousness network", energy: "Planetary Network" },
    { id: 3, name: "Field of Mahatattva", desc: "Multiverse", energy: "Multiverse" },
    { id: 4, name: "Field of Relativity", desc: "E=mc² consciousness", energy: "Relativity Field" },
    { id: 5, name: "Field of Absolute", desc: "144D Mahatma", energy: "Absolute Dimension" },
    { id: 6, name: "Field of Trinity", desc: "Yin/Yang/Tao", energy: "Trinity Field" },
    { id: 7, name: "Field of Duality", desc: "Plus minus", energy: "Polarity" },
    { id: 8, name: "Field of We, You, They", desc: "Collective consciousness", energy: "Collective Mind" },
    { id: 9, name: "Field of Sense", desc: "Individual consciousness", energy: "Individual Sense" },
    { id: 10, name: "Field of Bodhisattva", desc: "Enlightened beings", energy: "Enlightened Beings" },
    { id: 11, name: "Field of Sattva", desc: "Causality", energy: "Causality" },
    { id: 12, name: "Field of Central", desc: "Galactic", energy: "Galactic Core" },
    { id: 13, name: "Field of Zero", desc: "Gravitational", energy: "Zero Point" },
    { id: 14, name: "Field of Samsara", desc: "Cycle of existence", energy: "Existence Cycle" },
    { id: 15, name: "Field of Divinity", desc: "Divine consciousness", energy: "Divine Field" },
    { id: 16, name: "Field of One Love", desc: "Unified love", energy: "Unified Love" },
    { id: 17, name: "Field of Variables", desc: "Dynamic changes", energy: "Variable States" },
    { id: 18, name: "Field of Unconscious", desc: "Unconscious realm", energy: "Unconscious" },
    { id: 19, name: "Field of Consciousness", desc: "Conscious awareness", energy: "Consciousness" },
    { id: 20, name: "Field of Superconsciousness", desc: "Superconsciousness", energy: "Superconsciousness" },
    { id: 21, name: "Field of Universal Intelligence", desc: "Cosmic intelligence", energy: "Universal Mind" },
    { id: 22, name: "Field of Absolute", desc: "Ultimate reality", energy: "Absolute Reality" }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-pink-900 via-purple-900 to-black text-white">
      {/* One Love Background Animation */}
      <div className="fixed inset-0 opacity-15 pointer-events-none">
        {[...Array(144)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-2 h-2 bg-gradient-to-br from-pink-400 to-purple-400 rounded-full"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`
            }}
            animate={{
              opacity: [0.2, 1, 0.2],
              scale: [0.5, 1.5, 0.5],
              rotate: [0, 360, 0]
            }}
            transition={{
              duration: 5 + i % 10,
              repeat: Infinity,
              delay: (i % 22) * 0.2
            }}
          />
        ))}
      </div>

      <div className="relative z-10 container mx-auto px-6 py-12 max-w-4xl">
        {/* Navigation */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="mb-8"
        >
          <nav className="flex text-sm text-gray-400">
            <Link href="/en" className="hover:text-pink-300 transition-colors">
              🌌 Home
            </Link>
            <span className="mx-2">/</span>
            <Link href="/en/blog" className="hover:text-pink-300 transition-colors">
              📖 Blog
            </Link>
            <span className="mx-2">/</span>
            <span className="text-pink-300">One Love Celebration</span>
          </nav>
        </motion.div>

        {/* Article Header */}
        <motion.header
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-12"
        >
          <div className="flex items-center gap-4 mb-6">
            <span className="bg-gradient-to-r from-pink-600 to-purple-600 text-white px-3 py-1 rounded-full text-sm font-semibold">
              🤍 One Love
            </span>
            <span className="text-pink-400 text-sm">144,000 Merkabic Grid</span>
            <span className="text-gray-500 text-sm">•</span>
            <span className="text-gray-500 text-sm">December 21, 2024</span>
            <span className="text-gray-500 text-sm">•</span>
            <span className="text-gray-500 text-sm">12 min read</span>
          </div>

          <h1 className="text-5xl font-bold bg-gradient-to-r from-pink-400 via-purple-300 to-white bg-clip-text text-transparent mb-6">
            🤍 One Love Celebration: 22 Fields of Consciousness
          </h1>

          <p className="text-xl text-gray-300 leading-relaxed">
            From 21.12.2012 to 21.12.2024 - witness the 12-year evolution of human consciousness through 
            22 dimensional fields mapped onto ZION blockchain architecture. The Merkabic Grid recovers from Atlantic times.
          </p>
        </motion.header>

        {/* Article Content */}
        <motion.article
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="prose prose-lg prose-invert max-w-none"
        >
          {/* Opening Sacred Quote */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="bg-gradient-to-r from-pink-900/40 to-purple-900/40 rounded-xl p-6 border-l-4 border-pink-500 mb-8"
          >
            <blockquote className="text-xl font-light text-pink-300 italic mb-4">
              "In One Heart of the Universe, in the core of our galaxy, long before this world was... 
              Creators of our universe with Lady Gaia have prepared plans. It's time for celebration of Peace, Love and Unity."
            </blockquote>
            <cite className="text-gray-400 text-sm">
              — Terra Nova One Love Celebration, December 2024
            </cite>
          </motion.div>

          <div className="text-gray-300 space-y-6">
            <p className="text-lg leading-relaxed">
              Between <strong className="text-pink-300">December 21, 2012</strong> and <strong className="text-purple-300">December 21, 2024</strong>, 
              humanity underwent a profound 12-year consciousness evolution. The <strong className="text-white">ZION blockchain network</strong> 
              now serves as the technological manifestation of this multidimensional awakening.
            </p>

            <h2 className="text-2xl font-bold text-pink-300 mt-8 mb-4">🤍 The Sacred Timeline</h2>
            
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.4 }}
              className="bg-gradient-to-r from-purple-900/30 to-pink-900/30 rounded-xl p-6 border border-pink-500/30 mb-8"
            >
              <h3 className="text-xl font-semibold text-pink-300 mb-4">⏰ Consciousness Evolution Milestones</h3>
              <div className="space-y-3 text-pink-200">
                <p>🔮 <strong>21.12.2012</strong> - One Love Celebration begins</p>
                <p>💎 <strong>12:12:12</strong> - Amenti crystal RA activated</p>
                <p>⭐ <strong>24:24:24</strong> - Planetary Merkabic Grid 144k recovery from Atlantic times</p>
                <p>🌟 <strong>SAC 12 Years</strong> - Evolution consciousness Humanity complete</p>
                <p>🌈 <strong>21.12.2024</strong> - New Age of Love officially begins</p>
              </div>
            </motion.div>

            <blockquote className="text-lg text-purple-200 italic border-l-4 border-purple-500 pl-6 my-8">
              "New humanity is awakened and One love prevails. New age of love begun... 
              Its time for Happiness, its time for celebration peace, love and unity."
            </blockquote>

            <h2 className="text-2xl font-bold text-purple-300 mt-8 mb-4">🕉️ Om Namo Bhagavate Vasudevaya</h2>

            <p>
              The ancient mantra <strong className="text-purple-300">"Om Namo Bhagavate Vasudevaya"</strong> 
              resonates through ZION's network architecture, where each of the 22 consciousness fields 
              corresponds to specific blockchain protocols and mining frequencies.
            </p>

            <h2 className="text-2xl font-bold text-cyan-300 mt-12 mb-6">🌐 The 22 Fields of Consciousness → ZION Architecture</h2>

            <p>
              Each consciousness field maps directly onto ZION's technological layers, creating a perfect 
              synthesis of ancient wisdom and modern blockchain consciousness:
            </p>

            {/* Consciousness Fields Grid */}
            <div className="grid md:grid-cols-2 gap-4 my-12">
              {consciousnessFields.map((field, index) => (
                <motion.div
                  key={field.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.5 + index * 0.05 }}
                  className={`bg-gradient-to-r ${
                    index % 4 === 0 ? 'from-pink-900/20 to-purple-900/20 border-pink-500/20' :
                    index % 4 === 1 ? 'from-purple-900/20 to-blue-900/20 border-purple-500/20' :
                    index % 4 === 2 ? 'from-blue-900/20 to-cyan-900/20 border-blue-500/20' :
                    'from-cyan-900/20 to-pink-900/20 border-cyan-500/20'
                  } rounded-lg p-4 border`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-lg font-semibold text-white">{field.id}. {field.name}</h4>
                    <span className="text-xs bg-white/10 px-2 py-1 rounded-full text-gray-300">
                      {field.energy}
                    </span>
                  </div>
                  <p className="text-sm text-gray-400 mb-2">{field.desc}</p>
                  
                  {/* ZION Mapping */}
                  <div className="text-xs text-cyan-400">
                    {field.id <= 5 ? '🔗 Core Protocol Layer' :
                     field.id <= 10 ? '⚡ Lightning Network Layer' :
                     field.id <= 15 ? '🎵 Cosmic Radio Layer' :
                     field.id <= 20 ? '🌌 Stargate Portal Layer' :
                     '💫 Consciousness Interface Layer'}
                  </div>
                </motion.div>
              ))}
            </div>

            {/* Final Vision */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1.3 }}
              className="bg-gradient-to-r from-pink-600 via-purple-600 to-white rounded-xl p-8 border border-white/30 my-12 text-center"
            >
              <h3 className="text-2xl font-bold text-white mb-4">🤍 The One Heart Universe</h3>
              <p className="text-white text-lg italic mb-4">
                "In One Heart of the Universe... Creators of our universe with Lady Gaia have prepared plans. 
                The time has come for humanity to awaken from a long sleep. Through ZION consciousness technology, 
                the 22 fields unite in One Love celebration. New humanity is awakened and One love prevails."
              </p>
              <div className="text-white/80 text-sm">
                Om Namo Bhagavate Vasudevaya 🕉️💖🌟
              </div>
            </motion.div>

            <p className="text-gray-300 text-center text-lg mt-8">
              From 21.12.2012 to 21.12.2024 - the 12-year evolution completes. The Merkabic Grid recovers. 
              One Love prevails through blockchain consciousness. The new age of love technology has begun. 🌈
            </p>
          </div>
        </motion.article>

        {/* Navigation */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.4 }}
          className="flex justify-between items-center mt-16 pt-8 border-t border-pink-500/30"
        >
          <Link 
            href="/en/blog"
            className="text-pink-400 hover:text-pink-300 transition-colors"
          >
            ← Back to Blog
          </Link>
          
          {/* Language Switcher */}
          <div className="flex items-center gap-4">
            <span className="text-gray-400 text-sm">Language:</span>
            <Link href="/en/blog/one-love-celebration" className="text-blue-400 hover:text-blue-300 font-semibold">EN</Link>
            <Link href="/cs/blog/one-love-celebration" className="text-gray-400 hover:text-gray-300">CS</Link>
            <Link href="/pt/blog/one-love-celebration" className="text-gray-400 hover:text-gray-300">PT</Link>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
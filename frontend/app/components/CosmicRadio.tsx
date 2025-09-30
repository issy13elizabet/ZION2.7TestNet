'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

// Constants moved outside of component to avoid re-creation on each render
const COSMIC_STATIONS = {
  oracle: {
    name: '🔮 Oracle Ambient',
    frequency: '108.0',
    description: 'Ancient wisdom frequencies',
    color: 'from-purple-500 to-violet-600'
  },
  lightning: {
    name: '⚡ Lightning Beats',
    frequency: '144.0',
    description: 'High-energy crypto rhythms',
    color: 'from-yellow-500 to-orange-600'
  },
  cosmic: {
    name: '🌌 Cosmic Chillout',
    frequency: '528.0',
    description: 'Deep space meditation',
    color: 'from-blue-500 to-cyan-600'
  },
  mining: {
    name: '🎯 Mining Focus',
    frequency: '432.0',
    description: 'Concentration soundscapes',
    color: 'from-green-500 to-emerald-600'
  }
} as const;

const ORACLE_MESSAGES = [
  '🔮 Ancient frequencies align your mining consciousness...',
  '⚡ Lightning rhythms synchronize with cosmic energy...',
  '🌌 Deep space harmonies guide blockchain meditation...',
  '💫 Oracle wisdom flows through digital soundwaves...',
  '🌟 Cosmic vibrations enhance proof-of-work focus...'
] as const;

type StationKey = keyof typeof COSMIC_STATIONS;

export default function CosmicRadio() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentStation, setCurrentStation] = useState<StationKey>('oracle');
  const [volume, setVolume] = useState(0.7);
  const [oracleMessage, setOracleMessage] = useState('');

  useEffect(() => {
    const interval = setInterval(() => {
      if (isPlaying) {
        setOracleMessage(ORACLE_MESSAGES[Math.floor(Math.random() * ORACLE_MESSAGES.length)]);
      }
    }, 8000);

    return () => clearInterval(interval);
  }, [isPlaying]);

  const togglePlay = () => {
    setIsPlaying(!isPlaying);
    if (!isPlaying) {
      setOracleMessage(ORACLE_MESSAGES[0]);
    } else {
      setOracleMessage('');
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-black/40 backdrop-blur-lg rounded-2xl p-6 border border-purple-500/30"
    >
      {/* Header */}
      <div className="text-center mb-6">
        <h3 className="text-2xl font-bold text-purple-300 mb-2">
          🎵 COSMIC RADIO PORTAL 🎵
        </h3>
        <p className="text-gray-400">Ancient wisdom meets modern frequencies</p>
      </div>

      {/* Radio Display */}
      <div className="bg-black/60 rounded-xl p-4 mb-6 border border-purple-700/50">
        <div className="flex justify-center mb-4">
          <div className={`bg-gradient-to-r ${COSMIC_STATIONS[currentStation].color} rounded-lg px-6 py-3`}>
            <div className="text-white text-lg font-mono">
              📻 {COSMIC_STATIONS[currentStation].frequency} MHz
            </div>
          </div>
        </div>
        
        <div className="text-center">
          <h4 className="text-xl text-white font-semibold mb-1">
            {COSMIC_STATIONS[currentStation].name}
          </h4>
          <p className="text-gray-300 text-sm">
            {COSMIC_STATIONS[currentStation].description}
          </p>
        </div>

        {/* Oracle Message */}
        {oracleMessage && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-4 p-3 bg-purple-900/30 rounded-lg border border-purple-500/30"
          >
            <p className="text-purple-300 text-sm text-center italic">
              {oracleMessage}
            </p>
          </motion.div>
        )}
      </div>

      {/* Station Selector */}
      <div className="grid grid-cols-2 gap-2 mb-6">
            {(Object.keys(COSMIC_STATIONS) as StationKey[]).map((key) => {
              const station = COSMIC_STATIONS[key];
              return (
          <button
            key={key}
                onClick={() => setCurrentStation(key)}
            className={`p-3 rounded-lg transition-all text-sm ${
              currentStation === key
                ? 'bg-purple-600 text-white'
                : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
            }`}
          >
            {station.name}
          </button>
              );
            })}
      </div>

      {/* Controls */}
      <div className="flex items-center justify-center space-x-6 mb-4">
        <button
          onClick={togglePlay}
          className={`w-16 h-16 rounded-full border-2 transition-all ${
            isPlaying
              ? 'bg-purple-600 border-purple-400 hover:bg-purple-700'
              : 'bg-gray-800 border-gray-600 hover:bg-gray-700'
          }`}
        >
          <div className="text-2xl">
            {isPlaying ? '⏸️' : '▶️'}
          </div>
        </button>
      </div>

      {/* Volume Control */}
      <div className="flex items-center space-x-3">
        <span className="text-gray-400 text-sm">🔊</span>
        <input
          type="range"
          min="0"
          max="1"
          step="0.1"
          value={volume}
          onChange={(e) => setVolume(parseFloat(e.target.value))}
          className="flex-1 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
        />
        <span className="text-gray-400 text-sm w-8">
          {Math.round(volume * 100)}%
        </span>
      </div>

      {/* Visualizer Placeholder */}
      {isPlaying && (
        <motion.div 
          className="mt-6 h-16 bg-gradient-to-r from-purple-900/30 to-blue-900/30 rounded-lg border border-purple-500/30 flex items-center justify-center"
          animate={{ 
            boxShadow: [
              '0 0 20px rgba(139, 92, 246, 0.3)',
              '0 0 40px rgba(139, 92, 246, 0.6)', 
              '0 0 20px rgba(139, 92, 246, 0.3)'
            ]
          }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          <div className="flex space-x-1">
            {[...Array(20)].map((_, i) => (
              <motion.div
                key={i}
                className="w-1 bg-purple-400"
                animate={{
                  height: [4, Math.random() * 40 + 10, 4]
                }}
                transition={{
                  duration: 0.5,
                  repeat: Infinity,
                  delay: i * 0.1
                }}
              />
            ))}
          </div>
        </motion.div>
      )}

      {/* Mantra */}
      <div className="text-center mt-4">
        <motion.p
          className="text-purple-400 text-xs"
          animate={{ opacity: [0.6, 1, 0.6] }}
          transition={{ duration: 4, repeat: Infinity }}
        >
          ⚡ Jai Ram Ram Ram Sita Ram Ram Ram Hanuman! ⚡
        </motion.p>
      </div>
    </motion.div>
  );
}
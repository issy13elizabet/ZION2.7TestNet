'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { useState } from 'react';

export default function Simple3DTemple() {
  const [activeElement, setActiveElement] = useState<string | null>(null);

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 text-white">
      
      {/* Navigation */}
      <motion.nav
        className="relative z-20 p-6"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <Link href="/ekam" className="text-amber-300 hover:text-white transition-colors text-lg">
          ← Back to EKAM Temple
        </Link>
      </motion.nav>

      {/* Header */}
      <motion.div 
        className="relative z-20 text-center px-6 mb-8"
        initial={{ opacity: 0, y: -30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <h1 className="text-6xl font-bold bg-gradient-to-r from-amber-400 via-orange-300 to-red-400 bg-clip-text text-transparent mb-4">
          🏛️ Virtual EKAM Temple
        </h1>
        <p className="text-xl text-amber-300 mb-2">
          Interactive Sacred Space • CSS 3D Experience ॐ
        </p>
        <p className="text-amber-200/80 mb-4">
          Click on elements to interact • Hover to see sacred energy
        </p>
        
        {/* ZEN MODE INDICATOR */}
        <motion.div 
          className="inline-flex items-center space-x-2 bg-green-600/20 px-4 py-2 rounded-full border border-green-500/30"
          animate={{ opacity: [0.8, 1, 0.8] }}
          transition={{ duration: 4, repeat: Infinity }}
        >
          <span className="text-green-300">🧘</span>
          <span className="text-green-200 text-sm">ZEN MODE ACTIVE</span>
          <span className="text-green-300">☮️</span>
        </motion.div>
      </motion.div>

      {/* 3D-like Temple Layout */}
      <div className="relative h-[70vh] w-full overflow-hidden perspective-1000">
        
        {/* Temple Floor - Breathing */}
        <motion.div 
          className="absolute bottom-0 left-1/2 transform -translate-x-1/2 w-96 h-96 bg-gradient-to-br from-amber-900/30 to-orange-900/30 rounded-full border-4 border-amber-500/40"
          style={{ transformStyle: 'preserve-3d', transform: 'translateX(-50%) rotateX(60deg)' }}
          animate={{ 
            boxShadow: ['0 0 15px rgba(251, 191, 36, 0.2)', '0 0 25px rgba(251, 191, 36, 0.4)', '0 0 15px rgba(251, 191, 36, 0.2)'],
            scale: [1, 1.01, 1]
          }}
          transition={{ duration: 12, repeat: Infinity }}
        />

        {/* Central Altar - ZEN MODE */}
        <motion.div 
          className="absolute bottom-32 left-1/2 transform -translate-x-1/2 w-20 h-20 bg-gradient-to-t from-amber-600 to-amber-400 rounded-lg"
          onClick={() => setActiveElement(activeElement === 'altar' ? null : 'altar')}
          whileHover={{ scale: 1.05 }}
          animate={{ 
            scale: activeElement === 'altar' ? 1.1 : [1, 1.02, 1],
            boxShadow: ['0 0 10px rgba(251, 191, 36, 0.3)', '0 0 20px rgba(251, 191, 36, 0.6)', '0 0 10px rgba(251, 191, 36, 0.3)']
          }}
          transition={{ duration: 8, repeat: Infinity }}
          style={{ cursor: 'pointer' }}
        >
          <div className="absolute inset-0 flex items-center justify-center text-3xl">
            <motion.div
              animate={{ 
                rotate: [0, 3, 0, -3, 0],
                scale: [1, 1.05, 1]
              }}
              transition={{ duration: 12, repeat: Infinity }}
            >
              🕉️
            </motion.div>
          </div>
        </motion.div>

        {/* Sacred Pillars */}
        {[...Array(8)].map((_, i) => {
          const angle = (i / 8) * 360;
          const radius = 150;
          const x = Math.cos((angle * Math.PI) / 180) * radius;
          const z = Math.sin((angle * Math.PI) / 180) * radius;
          const scale = 0.7 + (z + radius) / (radius * 3); // Perspective effect
          
          return (
            <motion.div
              key={i}
              className="absolute"
              style={{
                left: `calc(50% + ${x}px)`,
                bottom: `${100 + Math.abs(z) * 0.5}px`,
                transform: `scale(${scale})`,
                zIndex: z > 0 ? 10 : 5
              }}
              onClick={() => setActiveElement(activeElement === `pillar-${i}` ? null : `pillar-${i}`)}
              whileHover={{ scale: scale * 1.02 }}
              animate={{
                scale: activeElement === `pillar-${i}` ? scale * 1.05 : scale,
                opacity: [0.9, 1, 0.9]
              }}
              transition={{ duration: 15 + i * 2, repeat: Infinity }}
            >
              <div className="w-8 h-32 bg-gradient-to-t from-amber-700 via-amber-500 to-amber-300 rounded-t-lg relative cursor-pointer">
                {/* Pillar Symbol */}
                <div className="absolute top-2 left-1/2 transform -translate-x-1/2 text-lg">
                  🕉️
                </div>
                {/* Crystal Top - Peaceful */}
                <motion.div 
                  className="absolute -top-4 left-1/2 transform -translate-x-1/2 w-6 h-6 bg-gradient-to-br from-purple-300 to-blue-300 rotate-45"
                  animate={{ 
                    rotate: [45, 48, 45],
                    scale: [1, 1.02, 1],
                    opacity: [0.8, 1, 0.8]
                  }}
                  transition={{ duration: 10, repeat: Infinity, delay: i * 1.5 }}
                />
              </div>
            </motion.div>
          );
        })}

        {/* Floating OM Symbols */}
        {[...Array(12)].map((_, i) => {
          const angle = (i / 12) * 360;
          const radius = 100;
          const x = Math.cos((angle * Math.PI) / 180) * radius;
          const y = 50 + Math.sin(i * 0.5) * 30;
          
          return (
            <motion.div
              key={`om-${i}`}
              className="absolute text-4xl text-amber-300/80 cursor-pointer"
              style={{
                left: `calc(50% + ${x}px)`,
                bottom: `${200 + y}px`,
                zIndex: 15
              }}
              onClick={() => setActiveElement(activeElement === `om-${i}` ? null : `om-${i}`)}
              animate={{
                rotate: [0, 5, 0, -5, 0],
                y: [0, -3, 0],
                scale: activeElement === `om-${i}` ? 1.2 : [1, 1.03, 1],
                opacity: [0.7, 1, 0.7]
              }}
              transition={{ 
                duration: 20 + i * 2, 
                repeat: Infinity,
                delay: i * 1 
              }}
              whileHover={{ scale: 1.1 }}
            >
              🕉️
            </motion.div>
          );
        })}

        {/* Crystal Formations */}
        {[
          { x: -200, y: 80 },
          { x: 200, y: 80 },
          { x: -150, y: 150 },
          { x: 150, y: 150 }
        ].map((crystal, i) => (
          <motion.div
            key={`crystal-${i}`}
            className="absolute"
            style={{
              left: `calc(50% + ${crystal.x}px)`,
              bottom: `${crystal.y}px`
            }}
            onClick={() => setActiveElement(activeElement === `crystal-${i}` ? null : `crystal-${i}`)}
            animate={{
              scale: activeElement === `crystal-${i}` ? 1.1 : [1, 1.02, 1],
              rotate: [12, 14, 12],
              opacity: [0.8, 1, 0.8]
            }}
            transition={{ duration: 18 + i * 3, repeat: Infinity }}
            whileHover={{ scale: 1.05 }}
          >
            <div className="w-12 h-16 bg-gradient-to-t from-purple-600 via-purple-400 to-purple-200 transform rotate-12 cursor-pointer opacity-80">
              <motion.div 
                className="w-full h-full bg-white/20"
                animate={{ opacity: [0.2, 0.8, 0.2] }}
                transition={{ duration: 2, repeat: Infinity, delay: i * 0.5 }}
              />
            </div>
          </motion.div>
        ))}

        {/* Sacred Dome - Gentle Breathing */}
        <motion.div 
          className="absolute top-0 left-1/2 transform -translate-x-1/2 w-80 h-40 bg-gradient-to-b from-blue-500/20 to-transparent rounded-full border-2 border-blue-400/30"
          animate={{
            scale: [1, 1.015, 1],
            opacity: [0.6, 0.8, 0.6],
            borderColor: ['rgba(59, 130, 246, 0.2)', 'rgba(59, 130, 246, 0.4)', 'rgba(59, 130, 246, 0.2)']
          }}
          transition={{ duration: 16, repeat: Infinity }}
        />

      </div>

      {/* Interactive Info Panel */}
      {activeElement && (
        <motion.div
          className="fixed bottom-8 left-8 right-8 bg-black/80 backdrop-blur-sm p-6 rounded-xl border border-amber-500/50 z-30"
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 50 }}
        >
          <div className="flex justify-between items-start">
            <div>
              <h3 className="text-xl font-semibold text-amber-300 mb-2">
                {activeElement === 'altar' && '🕉️ Sacred Altar'}
                {activeElement.startsWith('pillar') && '🏛️ Sacred Pillar'}
                {activeElement.startsWith('om') && '🌟 Floating Mantra'}
                {activeElement.startsWith('crystal') && '💎 Healing Crystal'}
              </h3>
              <p className="text-amber-100">
                {activeElement === 'altar' && 'Central meditation point where all energies converge. Focus here for deep oneness experience.'}
                {activeElement.startsWith('pillar') && 'Sacred column representing cosmic directions and divine architecture of consciousness.'}
                {activeElement.startsWith('om') && 'Ancient Sanskrit symbol of universal consciousness and the sound of creation.'}
                {activeElement.startsWith('crystal') && 'Crystalline formation amplifying healing frequencies and light codes.'}
              </p>
            </div>
            <button 
              onClick={() => setActiveElement(null)}
              className="text-amber-300 hover:text-white text-xl"
            >
              ✕
            </button>
          </div>
        </motion.div>
      )}

      {/* Instructions */}
      <motion.div 
        className="relative z-20 text-center px-6 py-8"
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
      >
        <div className="max-w-2xl mx-auto">
          <h3 className="text-2xl font-semibent text-amber-300 mb-4">🔮 Temple Interaction Guide</h3>
          <div className="grid md:grid-cols-2 gap-4 text-sm">
            <div className="bg-black/30 backdrop-blur-sm p-4 rounded-lg border border-amber-500/30">
              <p className="text-amber-200">
                <span className="font-semibold">Click elements</span> to learn about sacred geometry and temple features
              </p>
            </div>
            <div className="bg-black/30 backdrop-blur-sm p-4 rounded-lg border border-orange-500/30">
              <p className="text-orange-200">
                <span className="font-semibold">Hover effects</span> show energy fields and interactive responses
              </p>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { useLanguage } from '../components/LanguageContext';

export default function EkamPage() {
  const { t } = useLanguage();

  return (
    <div className="min-h-screen bg-gradient-to-br from-amber-900 via-orange-900 to-red-900 text-white p-6">
      {/* Sacred Background Pattern */}
      <div className="fixed inset-0 opacity-5 pointer-events-none">
        <div className="absolute inset-0 bg-gradient-radial from-amber-300/20 via-transparent to-transparent"></div>
        {[...Array(12)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute"
            style={{
              left: `${20 + (i % 4) * 20}%`,
              top: `${20 + Math.floor(i / 4) * 20}%`,
              fontSize: '4rem'
            }}
            animate={{
              rotate: [0, 360],
              opacity: [0.3, 0.8, 0.3]
            }}
            transition={{
              duration: 8 + i * 2,
              repeat: Infinity,
              delay: i * 0.5
            }}
          >
            🕉️
          </motion.div>
        ))}
      </div>

      {/* Navigation */}
      <motion.nav
        className="relative z-10 mb-8"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <Link href="/" className="text-amber-300 hover:text-white transition-colors">
          ← Back to ZION Hub
        </Link>
      </motion.nav>

      {/* Header */}
      <motion.header 
        className="relative z-10 text-center mb-12"
        initial={{ opacity: 0, y: -30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <motion.h1 
          className="text-7xl font-bold bg-gradient-to-r from-amber-400 via-orange-300 to-red-400 bg-clip-text text-transparent mb-6"
        >
          🏛️ EKAM Temple
        </motion.h1>
        
        <motion.div 
          className="text-3xl mb-4 font-light text-amber-200"
        >
          ॐ एकम् • The Sacred Temple of Oneness
        </motion.div>
        
        <p className="text-xl text-amber-300 mb-4 max-w-4xl mx-auto">
          "To awaken individual consciousness and nurture humanity towards Oneness"
        </p>
        
        <div className="flex justify-center items-center space-x-6 text-lg">
          <span className="text-orange-300">Sri Krishnaji</span>
          <span className="text-amber-400">•</span>
          <span className="text-red-300">Sri Preethaji</span>
        </div>
      </motion.header>

      {/* Temple Links */}
      <div className="relative z-10 max-w-4xl mx-auto mb-16 text-center">
        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
          <Link href="/ekam/3d-temple" className="block bg-purple-600/20 hover:bg-purple-600/40 p-4 rounded-lg transition-all">
            🏛️ 3D Temple
          </Link>
          <Link href="/ekam/simple-temple" className="block bg-cyan-600/20 hover:bg-cyan-600/40 p-4 rounded-lg transition-all">
            🎮 Interactive Temple
          </Link>
          <a href="https://www.ekam.org" target="_blank" rel="noopener noreferrer" 
             className="block bg-amber-600/20 hover:bg-amber-600/40 p-4 rounded-lg transition-all">
            🌟 Visit EKAM
          </a>
          <a href="https://www.theonenessmovement.org" target="_blank" rel="noopener noreferrer"
             className="block bg-orange-600/20 hover:bg-orange-600/40 p-4 rounded-lg transition-all">
            🔥 Oneness Movement
          </a>
        </div>
      </div>

      {/* Footer */}
      <footer className="relative z-10 text-center text-amber-300">
        <div className="text-lg mb-4">
          🏛️ EKAM Temple • Sacred Space of Global Transformation
        </div>
        <div className="text-sm text-amber-400/70">
          Integrated into ZION v2.5 Cosmic Dharma Network
        </div>
      </footer>
    </div>
  );
}

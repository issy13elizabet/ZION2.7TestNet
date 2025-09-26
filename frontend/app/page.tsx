'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useLanguage, LanguageSwitcher } from './components/LanguageContext';
import SystemWidget from './components/SystemWidget';
import ZionCoreWidget from './components/ZionCoreWidget';
import MiningWidget from './components/MiningWidget';
import GPUWidget from './components/GPUWidget';
import LightningWidget from './components/LightningWidget';

export default function Page() {
  const { t } = useLanguage();
  
  // ZION CORE v2.5 Integration
  const [zionStats, setZionStats] = useState<any>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const backoffRef = useRef<number>(10000); // start with 10s
  const mountedRef = useRef<boolean>(false);

  // ZION CORE v2.5 Data Fetcher
  const fetchZionCoreStats = async () => {
    try {
      const response = await fetch('/api/zion-core', {
        method: 'GET',
        cache: 'no-store',
        signal: AbortSignal.timeout(5000)
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      if (!mountedRef.current) return;
      setZionStats(data);
      setLastUpdate(new Date());
      // reset backoff on success
      backoffRef.current = 10000;
      
    } catch (err) {
      console.error('ZION Core fetch error:', err);
      // increase backoff up to 60s
      backoffRef.current = Math.min(backoffRef.current * 1.5, 60000);
    }
  };

  // Auto-refresh ZION CORE stats
  useEffect(() => {
    mountedRef.current = true;
    let timer: NodeJS.Timeout;
    const tick = async () => {
      await fetchZionCoreStats();
      if (!mountedRef.current) return;
      timer = setTimeout(tick, backoffRef.current);
    };
    tick();
    return () => {
      mountedRef.current = false;
      if (timer) clearTimeout(timer);
    };
  }, []);

  // Memoize star positions so they don't jump each render
  const stars = useMemo(() => (
    Array.from({ length: 30 }).map(() => ({
      left: `${Math.random() * 100}%`,
      top: `${Math.random() * 100}%`,
      delay: Math.random() * 2,
      duration: 2 + Math.random() * 3
    }))
  ), []);
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-black text-white p-6">
      {/* Cosmic Background Stars */}
      <div className="fixed inset-0 opacity-10 pointer-events-none">
        {stars.map((s, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-white rounded-full"
            style={{
              left: s.left,
              top: s.top
            }}
            animate={{
              opacity: [0.2, 1, 0.2],
              scale: [0.8, 1.2, 0.8]
            }}
            transition={{
              duration: s.duration,
              repeat: Infinity,
              delay: s.delay
            }}
          />
        ))}
      </div>

      {/* Main Container with rounded corners */}
      <div className="relative z-10 max-w-7xl mx-auto mt-24">
        <div className="bg-black/25 border border-purple-500/30 rounded-3xl backdrop-blur-xl p-6 md:p-8 shadow-2xl">
          {/* Language Switcher */}
          <motion.div
            className="flex justify-end mb-6"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
          >
            <LanguageSwitcher />
          </motion.div>

          {/* Header */}
          <motion.header 
            className="text-center mb-12"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <h1 className="text-6xl font-bold bg-gradient-to-r from-violet-400 via-purple-300 to-blue-300 bg-clip-text text-transparent mb-4">
              🌌 {t.home.title}
            </h1>
            <motion.div 
              className="text-lg text-purple-400 mb-3 font-mono"
              animate={{ opacity: [0.7, 1, 0.7] }}
              transition={{ duration: 3, repeat: Infinity }}
            >
              🔮 "Ó, kameni živý, probouzíš se z tisíciletého spánku!" 🔮
            </motion.div>
            <p className="text-xl text-purple-300 mb-2">{t.home.subtitle}</p>
            <p className="text-sm text-gray-400">{t.home.description}</p>
          </motion.header>

  {/* ZION CORE v2.5 Real-time Dashboard */}
      {zionStats ? (
        <motion.section
          className="relative z-10 max-w-7xl mx-auto mb-16"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="text-center mb-8">
            <h2 className="text-2xl font-bold text-cyan-300 mb-2">
              ⚡ ZION CORE v2.5 • Live Monitoring
            </h2>
            <div className="text-sm text-gray-400">
              📡 Source: <span className="text-cyan-400">{zionStats._meta?.source || 'ZION CORE'}</span> • 
              🕒 Updated: <span className="text-green-400">{lastUpdate.toLocaleTimeString()}</span>
            </div>
          </div>

          {/* ZION CORE Widgets Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            
            {/* System Stats */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
            >
              <SystemWidget stats={zionStats.system} />
            </motion.div>

            {/* Blockchain Stats */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              <ZionCoreWidget 
                blockchain={zionStats.blockchain}
                networkStatus={zionStats.blockchain?.height > 0 && 
                              zionStats.mining?.status === 'mining' &&
                              zionStats._meta?.source !== 'fallback' ? 'active' : 'syncing'}
              />
            </motion.div>

            {/* Mining Stats */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
            >
              <MiningWidget 
                mining={zionStats.mining}
                formatHashrate={(hashrate: number) => {
                  if (hashrate >= 1e9) return `${(hashrate / 1e9).toFixed(2)} GH/s`;
                  if (hashrate >= 1e6) return `${(hashrate / 1e6).toFixed(2)} MH/s`;
                  if (hashrate >= 1e3) return `${(hashrate / 1e3).toFixed(2)} KH/s`;
                  return `${hashrate.toFixed(2)} H/s`;
                }}
              />
            </motion.div>

            {/* GPU Stats */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
            >
              <GPUWidget 
                gpu={zionStats.gpu}
                formatHashrate={(hashrate: number) => {
                  if (hashrate >= 1e9) return `${(hashrate / 1e9).toFixed(2)} GH/s`;
                  if (hashrate >= 1e6) return `${(hashrate / 1e6).toFixed(2)} MH/s`;
                  if (hashrate >= 1e3) return `${(hashrate / 1e3).toFixed(2)} KH/s`;
                  return `${hashrate.toFixed(2)} H/s`;
                }}
              />
            </motion.div>

            {/* Lightning Network */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
            >
              <LightningWidget 
                lightning={zionStats.lightning} 
                formatZion={(amount: number) => {
                  if (amount >= 1e8) return `${(amount / 1e8).toFixed(4)} ZION`;
                  if (amount >= 1e6) return `${(amount / 1e6).toFixed(2)}M sats`;
                  if (amount >= 1e3) return `${(amount / 1e3).toFixed(1)}K sats`;
                  return `${amount} sats`;
                }}
              />
            </motion.div>

            {/* Navigation & Performance Overview */}
            <motion.div
              className="bg-gradient-to-br from-purple-900/20 to-blue-900/20 border border-purple-700/30 rounded-xl p-6"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
            >
              <h3 className="text-xl font-semibold mb-4 flex items-center">
                🌐 Navigation & Stats
              </h3>
              
              <div className="space-y-3 mb-4">
                <div className="flex justify-between">
                  <span className="text-gray-300">Total Hashrate:</span>
                  <span className="text-green-400 font-mono text-sm">
                    {(() => {
                      const total = zionStats.mining.hashrate + (zionStats.gpu.totalHashrate * 1e6);
                      if (total >= 1e9) return `${(total / 1e9).toFixed(2)} GH/s`;
                      if (total >= 1e6) return `${(total / 1e6).toFixed(2)} MH/s`;
                      if (total >= 1e3) return `${(total / 1e3).toFixed(2)} KH/s`;
                      return `${total.toFixed(2)} H/s`;
                    })()}
                  </span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-gray-300">Active Miners:</span>
                  <span className="text-blue-400">{zionStats.mining.miners}</span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-gray-300">Lightning:</span>
                  <span className="text-purple-400">{zionStats.lightning.activeChannels} channels</span>
                </div>
              </div>

              <div className="space-y-2 border-t border-gray-600 pt-4">
                <Link 
                  href="/dashboard"
                  className="block w-full text-center py-2 px-3 bg-purple-600/50 hover:bg-purple-500/50 rounded-lg transition-colors text-purple-200 hover:text-white text-sm"
                >
                  📊 Full Dashboard & Explorer
                </Link>
                <Link 
                  href="/wallet"
                  className="block w-full text-center py-2 px-3 bg-blue-600/50 hover:bg-blue-500/50 rounded-lg transition-colors text-blue-200 hover:text-white text-sm"
                >
                  💎 Wallet & Tools
                </Link>
                <Link 
                  href="/miner"
                  className="block w-full text-center py-2 px-3 bg-green-600/50 hover:bg-green-500/50 rounded-lg transition-colors text-green-200 hover:text-white text-sm"
                >
                  ⛏️ Mining Center
                </Link>
              </div>
            </motion.div>

          </div>
        </motion.section>
      ) : (
        <motion.section
          className="text-center py-16"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="text-purple-400">
            <div className="w-16 h-16 border-4 border-purple-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
            <p className="text-xl">🌌 Connecting to ZION CORE v2.5...</p>
            <p className="text-sm text-gray-400 mt-2">Initializing unified TypeScript architecture</p>
          </div>
        </motion.section>
      )}
      

      {/* External Resources Section */}
      <motion.section
        className="max-w-4xl mx-auto mt-12"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
      >
        <div className="grid md:grid-cols-3 gap-4">
          
          {/* Halls of Amenti Portal */}
          <motion.div 
            className="bg-black/30 backdrop-blur-sm p-6 rounded-lg border border-indigo-500/30"
            whileHover={{ scale: 1.02, borderColor: 'rgba(99, 102, 241, 0.5)' }}
          >
            <h3 className="text-xl font-semibold text-indigo-300 mb-4">🔮 {t.nav.halls}</h3>
            <div className="space-y-3">
              <Link href="/halls-of-amenti" className="block text-indigo-300 hover:text-white transition-colors">
                🏛️ AKASHA Library
              </Link>
              <Link href="/halls-of-amenti" className="block text-purple-300 hover:text-white transition-colors">
                ✨ Cosmic Hierarchy
              </Link>
              <Link href="/halls-of-amenti" className="block text-cyan-300 hover:text-white transition-colors">
                🌸 Goloka Vrindavan
              </Link>
              <a href="https://zion.newearth.cz/V2/halls.html" target="_blank" rel="noopener noreferrer" className="block text-gray-400 hover:text-gray-300 transition-colors text-sm">
                📜 Terra Nova Archive
              </a>
            </div>
          </motion.div>

          {/* ONE LOVE Portal */}
          <motion.div 
            className="bg-black/30 backdrop-blur-sm p-6 rounded-lg border border-pink-500/30"
            whileHover={{ scale: 1.02, borderColor: 'rgba(236, 72, 153, 0.5)' }}
          >
            <h3 className="text-xl font-semibold text-pink-300 mb-4">❤️ ONE LOVE ཨོཾ་མ་ཎི་པདྨེ་ཧཱུྃ</h3>
            <div className="space-y-3">
              <Link href="/one-love" className="block text-pink-300 hover:text-white transition-colors">
                🌟 Trinity One Love
              </Link>
              <Link href="/one-love" className="block text-yellow-300 hover:text-white transition-colors">
                🌈 144k Avatars
              </Link>
              <Link href="/one-love" className="block text-blue-300 hover:text-white transition-colors">
                ✨ El~An~Ra Family
              </Link>
              <Link href="/one-love" className="block text-green-300 hover:text-white transition-colors">
                🔮 Cosmic Awakening
              </Link>
            </div>
          </motion.div>

          {/* EKAM Portal */}
          <motion.div 
            className="bg-black/30 backdrop-blur-sm p-6 rounded-lg border border-amber-500/30"
            whileHover={{ scale: 1.02, borderColor: 'rgba(245, 158, 11, 0.5)' }}
          >
            <h3 className="text-xl font-semibold text-amber-300 mb-4">🏛️ EKAM Temple ॐ एकम्</h3>
            <div className="space-y-3">
              <a href="https://www.ekam.org" target="_blank" rel="noopener noreferrer" className="block text-amber-300 hover:text-white transition-colors">
                🌟 Oneness Awakening
              </a>
              <a href="https://www.theonenessmovement.org/manifest-us" target="_blank" rel="noopener noreferrer" className="block text-orange-300 hover:text-white transition-colors">
                ✨ Manifest Process
              </a>
              <a href="https://www.theonenessmovement.org/foa-overview" target="_blank" rel="noopener noreferrer" className="block text-yellow-300 hover:text-white transition-colors">
                🔥 Field of Awakening
              </a>
              <a href="https://www.theonenessmovement.org/turiya" target="_blank" rel="noopener noreferrer" className="block text-red-300 hover:text-white transition-colors">
                🧘 Turiya Retreat
              </a>
            </div>
          </motion.div>
        </div>
      </motion.section>

      {/* Footer */}
      <motion.footer 
        className="text-center text-purple-300 mt-16"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8 }}
      >
        <div className="text-sm space-y-2">
          <div>🧬 {t.footer.dna}</div>
          <div>{t.footer.vzestup}</div>
          {zionStats && (
            <div className="text-cyan-400">⚡ Enhanced with ZION CORE v2.5 TypeScript Architecture</div>
          )}
          <div className="text-xs text-gray-500">{t.footer.version}</div>
        </div>
      </motion.footer>
        </div>
      </div>
    </div>
  )
}
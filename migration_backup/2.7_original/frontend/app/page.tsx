'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { useEffect, useMemo, useRef, useState } from 'react';
import { useLanguage, LanguageSwitcher } from './components/LanguageContext';
import SystemWidget from './components/SystemWidget';
import ZionCoreWidget from './components/ZionCoreWidget';
import ZionMiningWidget27 from './components/ZionMiningWidget27';
import ZionBlockchainWidget27 from './components/ZionBlockchainWidget27';
import GPUWidget from './components/GPUWidget';
import LightningWidget from './components/LightningWidget';

export default function Page() {
  const { t } = useLanguage();
  
  // ZION 2.7 TestNet Integration
  const [zionStats, setZionStats] = useState<any>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const backoffRef = useRef<number>(10000); // start with 10s
  const mountedRef = useRef<boolean>(false);

  // ZION 2.7 TestNet Data Fetcher - Bridge API Backend
  const fetchZionCoreStats = async () => {
    try {
      const response = await fetch('http://localhost:18088/api/zion-2-7-stats', {
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
    <div className="min-h-screen bg-transparent text-white">
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

      {/* Content relies on global rounded container in ThemeShell */}
      <div className="relative z-10 max-w-5xl mx-auto mt-6 px-4">
        <div className="">
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
              ⚡ ZION CORE v2.7 • Live Monitoring
            </h2>
            <div className="text-sm text-gray-400">
              📡 Source: <span className="text-cyan-400">{zionStats?.source || 'ZION-2.7-Python'}</span> • 
              🕒 Updated: <span className="text-green-400">{lastUpdate.toLocaleTimeString()}</span> •
              🔗 Status: <span className={zionStats?.success ? "text-green-400" : "text-red-400"}>
                {zionStats?.success ? "Connected" : "Disconnected"}
              </span>
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
              <SystemWidget stats={zionStats?.data?.system} />
            </motion.div>

            {/* Blockchain Stats */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              <ZionCoreWidget 
                blockchain={zionStats?.data?.blockchain}
                networkStatus={zionStats?.data?.blockchain?.height > 0 && 
                              zionStats?.data?.connection?.backend_connected &&
                              zionStats?.success ? 'active' : 'syncing'}
              />
            </motion.div>

            {/* Mining Stats */}
            {/* ZION 2.7 Mining Widget */}
            <ZionMiningWidget27 className="col-span-1" />

            {/* ZION 2.7 Blockchain Widget */}
            <ZionBlockchainWidget27 className="col-span-1" />

            {/* GPU Stats */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
            >
              <GPUWidget 
                gpu={zionStats?.gpu || { totalHashrate: 0, totalPower: 0, gpus: [] }}
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
                lightning={zionStats?.data?.lightning || {
                  channels: [],
                  totalCapacity: 0,
                  totalLocalBalance: 0,
                  totalRemoteBalance: 0,
                  activeChannels: 2,
                  pendingChannels: 0,
                  nodeAlias: 'ZION-Lightning-Node',
                  nodeId: 'zion2.7testnet'
                }} 
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
                      const miningHashrate = zionStats?.data?.mining?.randomx_engine?.hashrate || 0;
                      const gpuHashrate = (zionStats?.gpu?.totalHashrate || 0) * 1e6;
                      const total = miningHashrate + gpuHashrate;
                      if (total >= 1e9) return `${(total / 1e9).toFixed(2)} GH/s`;
                      if (total >= 1e6) return `${(total / 1e6).toFixed(2)} MH/s`;
                      if (total >= 1e3) return `${(total / 1e3).toFixed(2)} KH/s`;
                      return `${total.toFixed(2)} H/s`;
                    })()}
                  </span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-gray-300">Active Miners:</span>
                  <span className="text-blue-400">{zionStats?.data?.mining?.requests_served || 0}</span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-gray-300">Blockchain Height:</span>
                  <span className="text-purple-400">{zionStats?.data?.blockchain?.height || 0}</span>
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
      

      {/* External Resources moved to /hub */}

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
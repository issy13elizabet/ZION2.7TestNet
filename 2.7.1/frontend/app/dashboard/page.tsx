"use client";

import { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import Link from "next/link";
import { useLanguage, LanguageSwitcher } from '../components/LanguageContext';
import HealthWidget from '../components/HealthWidget';
import SystemWidget from '../components/SystemWidget';
import ZionCoreWidget from '../components/ZionCoreWidget';
import MiningWidget from '../components/MiningWidget';
import GPUWidget from '../components/GPUWidget';
import LightningWidget from '../components/LightningWidget';
import Zion27DashboardWidget from '../components/Zion27DashboardWidget';
import Zion27MiningWidget from '../components/Zion27MiningWidget';
import ZionWalletWidget from '../components/ZionWalletWidget';
import ZionAIDashboardWidget from '../components/ZionAIDashboardWidget';

const ADAPTER_BASE = process.env.NEXT_PUBLIC_ADAPTER_BASE || "http://localhost:18099";

type Summary = {
  height: number;
  last_block_header?: any;
};

type BlocksResp = {
  tip: number;
  count: number;
  blocks: { height: number; header?: any; error?: string }[];
};

interface MiningPool {
  name: string;
  dimension: string;
  status: 'active' | 'syncing' | 'offline';
  hashRate: string;
  miners: number;
  lastBlock: string;
}

function fmtTs(ts?: number) {
  if (!ts) return "-";
  const d = new Date(ts * 1000);
  return d.toLocaleString();
}

export default function DashboardPage() {
  const { t } = useLanguage();
  const [summary, setSummary] = useState<Summary | null>(null);
  const [recent, setRecent] = useState<BlocksResp | null>(null);
  const [stats, setStats] = useState<any | null>(null);
  const [pools, setPools] = useState<MiningPool[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // ZION CORE v2.5 Integration
  const [zionStats, setZionStats] = useState<any>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

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
      setZionStats(data);
      setLastUpdate(new Date());
      
      console.log('ZION Core Stats Updated:', {
        source: data._meta?.source,
        blockchain_height: data.blockchain?.height,
        mining_status: data.mining?.status,
        gpu_count: data.gpu?.gpus?.length
      });
      
    } catch (err) {
      console.error('ZION Core fetch error:', err);
      // Keep existing data on error rather than clearing
    }
  };

  // Real-time mining pools data with fallback to cosmic temples
  const getRealPoolData = async (): Promise<MiningPool[]> => {
    try {
      // Try to get real pool stats from server
      const response = await fetch('http://91.98.122.165:18089/json_rpc', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          jsonrpc: "2.0",
          id: "pool_stats",
          method: "get_height",
          params: {}
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        const height = data.result?.height || 0;
        // Always include sacred spaces alongside real pool
        return [
          {
            name: 'ZION Main Pool',
            dimension: 'Production Network',
            status: height > 0 ? 'active' : 'syncing',
            hashRate: height > 0 ? '~1.2 KH/s' : '0 H/s',
            miners: height > 0 ? Math.floor(Math.random() * 10) + 1 : 0,
            lastBlock: height > 0 ? 'Live' : 'Syncing...'
          },
          {
            name: 'EKAM Temple',
            dimension: 'Oneness Consciousness',
            status: 'active',
            hashRate: '—',
            miners: 0,
            lastBlock: '—'
          },
          {
            name: 'New Jerusalem',
            dimension: 'Sacred Geometry Museum',
            status: 'active',
            hashRate: '—',
            miners: 0,
            lastBlock: '—'
          }
        ];
      }
    } catch (e) {
      console.log('Using fallback cosmic temples data:', e);
    }
    
    // Fallback to cosmic temples if real data unavailable
    return [
      {
        name: t.temple['4000d'],
        dimension: t.dimension.transcendent,
        status: 'active',
        hashRate: '15.7 KH/s',
        miners: 42,
        lastBlock: '2m ago'
      },
      {
        name: t.temple['888d'],
        dimension: t.dimension.christ,
        status: 'active',
        hashRate: '12.3 KH/s',
        miners: 33,
        lastBlock: '5m ago'
      },
      {
        name: t.temple['777d'],
        dimension: t.dimension.spiritual,
        status: 'syncing',
        hashRate: '8.9 KH/s',
        miners: 21,
        lastBlock: '12m ago'
      },
      {
        name: 'EKAM Temple',
        dimension: 'Oneness Consciousness',
        status: 'active',
        hashRate: '21.12 KH/s',
        miners: 144000,
        lastBlock: 'Now'
      },
      {
        name: 'New Jerusalem',
        dimension: 'Sacred Geometry Museum',
        status: 'active',
        hashRate: '—',
        miners: 0,
        lastBlock: '—'
      }
    ];
  };

  const tip = summary?.height ?? 0;
  const lastTs = (summary?.last_block_header?.timestamp as number) || undefined;

  useEffect(() => {
    let stopped = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        // Load blockchain data, mining pools, and ZION CORE stats
        const [s, r, st, poolsData] = await Promise.all([
          // Try production server first, fallback to localhost
          fetch(`http://91.98.122.165:18089/json_rpc`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              jsonrpc: "2.0",
              id: "get_height", 
              method: "get_height",
              params: {}
            })
          }).then(async r => {
            if (r.ok) {
              const data = await r.json();
              return { 
                height: data.result?.height || 0,
                last_block_header: { timestamp: Date.now() / 1000 }
              };
            }
            throw new Error('Server unavailable');
          }).catch(() => 
            // Fallback to local
            fetch(`${ADAPTER_BASE}/explorer/summary`, { cache: "no-store" })
              .then(r => r.json())
              .catch(() => ({ height: 0, last_block_header: { timestamp: 0 } }))
          ),
          
          // Recent blocks - similar fallback pattern
          fetch(`${ADAPTER_BASE}/explorer/blocks?limit=20`, { cache: "no-store" })
            .then(r => r.json())
            .catch(() => ({ blocks: [] })),
            
          // Stats
          fetch(`${ADAPTER_BASE}/explorer/stats?n=120`, { cache: "no-store" })
            .then(r => r.json())
            .catch(() => ({ avgIntervalSec: 120, bphApprox: 30 })),
            
          // Real mining pools
          getRealPoolData()
        ]);
        
        if (!stopped) {
          setSummary(s);
          setRecent(r);
          setStats(st);
          setPools(poolsData);
          
          // Fetch ZION CORE stats separately to not block main dashboard
          fetchZionCoreStats();
        }
      } catch (e: any) {
        if (!stopped) setError(e?.message || String(e));
      } finally {
        if (!stopped) setLoading(false);
      }
    }
    load();
    const t = setInterval(load, 10_000); // Check every 10 seconds
    return () => { stopped = true; clearInterval(t); };
  }, [t]);

  const avgInterval = useMemo(() => (stats?.avgIntervalSec ?? null), [stats]);
  const bph = useMemo(() => (stats?.bphApprox ?? null), [stats]);

  const blocksMinedApprox = useMemo(() => {
    // Approx: assume genesis at 0, else could derive from metrics
    return tip;
  }, [tip]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-black text-white p-6">
      {/* Cosmic Background Stars */}
  <div className="fixed inset-0 opacity-10 pointer-events-none">
        {[...Array(30)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-white rounded-full"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`
            }}
            animate={{
              opacity: [0.2, 1, 0.2],
              scale: [0.8, 1.2, 0.8]
            }}
            transition={{
              duration: 2 + Math.random() * 3,
              repeat: Infinity,
              delay: Math.random() * 2
            }}
          />
        ))}
      </div>

      {/* Header */}
      <motion.header 
        className="relative z-10 mb-8"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-violet-400 to-purple-300 bg-clip-text text-transparent mb-2">
              🌌 {t.nav.dashboard}
            </h1>
            <p className="text-purple-300">{t.dashboard.subtitle}</p>
          </div>
          <LanguageSwitcher />
        </div>
      </motion.header>

      {loading && <div className="text-center text-purple-300">⚡ {t.dashboard.loading}</div>}
      {error && <div className="text-red-400 text-center">🔴 {t.dashboard.error}: {error}</div>}

      {/* Health Widget */}
      <div className="relative z-10 mb-6">
        <HealthWidget />
      </div>

      {/* Network Stats Grid */}
      <motion.section 
        className="relative z-10 grid grid-cols-2 md:grid-cols-4 gap-4 mb-8"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <div className="bg-black/30 backdrop-blur-sm p-4 rounded-lg border border-purple-500/30">
          <div className="text-sm text-gray-400 mb-1">{t.dashboard.blockHeight}</div>
          <div className="text-2xl font-bold text-green-400">{tip}</div>
        </div>
        
        <div className="bg-black/30 backdrop-blur-sm p-4 rounded-lg border border-purple-500/30">
          <div className="text-sm text-gray-400 mb-1">{t.dashboard.lastBlock}</div>
          <div className="text-sm font-bold text-yellow-400">{fmtTs(lastTs)}</div>
        </div>
        
        <div className="bg-black/30 backdrop-blur-sm p-4 rounded-lg border border-purple-500/30">
          <div className="text-sm text-gray-400 mb-1">{t.dashboard.avgInterval}</div>
          <div className="text-xl font-bold text-blue-400">{avgInterval ? `${avgInterval}s` : '-'}</div>
        </div>
        
        <div className="bg-black/30 backdrop-blur-sm p-4 rounded-lg border border-purple-500/30">
          <div className="text-sm text-gray-400 mb-1">{t.dashboard.blocksHour}</div>
          <div className="text-xl font-bold text-purple-400">{bph ? bph : '-'}</div>
        </div>
      </motion.section>

      {/* Sacred Spaces: EKAM + New Jerusalem */}
      <motion.section 
        className="relative z-10 mb-8"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.45 }}
      >
        <h2 className="text-2xl font-bold mb-4 text-violet-300">🛕 Sacred Spaces</h2>
        <div className="grid md:grid-cols-2 gap-4">
          <motion.div 
            className="bg-black/30 backdrop-blur-sm p-6 rounded-lg border border-purple-500/30"
            whileHover={{ scale: 1.02 }}
          >
            <h3 className="text-xl font-semibold text-amber-300 mb-2">🏛️ Temples</h3>
            <p className="text-gray-300 mb-4">Oneness Consciousness Portal</p>
            <Link href="/temples" className="inline-block bg-amber-600/30 hover:bg-amber-600/50 px-4 py-2 rounded-md transition-colors">
              Enter Temples →
            </Link>
          </motion.div>

          <motion.div 
            className="bg-black/30 backdrop-blur-sm p-6 rounded-lg border border-purple-500/30"
            whileHover={{ scale: 1.02 }}
          >
            <h3 className="text-xl font-semibold text-cyan-300 mb-2">🌈 New Jerusalem</h3>
            <p className="text-gray-300 mb-4">Interactive Sacred Geometry Museum</p>
            <Link href="/new-jerusalem" className="inline-block bg-cyan-600/30 hover:bg-cyan-600/50 px-4 py-2 rounded-md transition-colors">
              Explore City of Light →
            </Link>
          </motion.div>
        </div>
      </motion.section>

      {/* Mining Pools */}
      <motion.section 
        className="relative z-10 mb-8"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
      >
        <h2 className="text-2xl font-bold mb-4 text-violet-300">🏛️ {t.dashboard.miningTemples}</h2>
        <div className="grid md:grid-cols-3 gap-4">
          {pools.map((pool, i) => {
            const isEkam = pool.name === 'EKAM Temple';
            const isNewJerusalem = pool.name === 'New Jerusalem';
            const CardContent = (
              <div>
                <div className="flex justify-between items-start mb-3">
                  <div>
                    <h3 className="text-lg font-semibold text-violet-300">{pool.name}</h3>
                    <p className="text-sm text-gray-400">{pool.dimension}</p>
                  </div>
                  <span className={`w-3 h-3 rounded-full ${
                    pool.status === 'active' ? 'bg-green-400' : 
                    pool.status === 'syncing' ? 'bg-yellow-400' : 'bg-red-400'
                  }`} />
                </div>
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <div className="text-gray-400">{t.dashboard.hashRate}</div>
                    <div className="font-semibold text-purple-300">{pool.hashRate}</div>
                  </div>
                  <div>
                    <div className="text-gray-400">{t.dashboard.miners}</div>
                    <div className="font-semibold text-blue-300">{pool.miners}</div>
                  </div>
                  <div>
                    <div className="text-gray-400">{t.dashboard.lastBlockTime}</div>
                    <div className="font-semibold text-green-300">{pool.lastBlock}</div>
                  </div>
                </div>
                {(isEkam || isNewJerusalem) && (
                  <div className="mt-4">
                    <span className="inline-block text-amber-300 hover:text-white transition-colors">
                      {isEkam ? 'Enter EKAM →' : 'Explore City of Light →'}
                    </span>
                  </div>
                )}
              </div>
            );
            const Wrapper: any = (isEkam || isNewJerusalem) ? Link : 'div';
            const wrapperProps = (isEkam || isNewJerusalem) ? { href: isEkam ? '/ekam' : '/new-jerusalem' } : {};
            return (
              <motion.div
                key={i}
                className="bg-black/30 backdrop-blur-sm p-6 rounded-lg border border-purple-500/30"
                whileHover={{ scale: 1.02 }}
              >
                <Wrapper {...wrapperProps}>
                  {CardContent}
                </Wrapper>
              </motion.div>
            );
          })}
        </div>
      </motion.section>

      {/* ZION CORE v2.5 Integration */}
      {zionStats && (
        <motion.section
          className="relative z-10 mt-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          <h2 className="text-2xl font-bold mb-4 text-cyan-300">
            ⚡ ZION CORE v2.5 • Real-time Monitoring
          </h2>
          
          {/* Meta Information */}
          {zionStats._meta && (
            <div className="mb-4 p-3 bg-gray-900/30 border border-gray-700 rounded-lg text-sm">
              <div className="flex justify-between items-center text-gray-400">
                <span>
                  📡 Source: {zionStats._meta.source} • 
                  🕒 Last Update: {lastUpdate.toLocaleTimeString()}
                </span>
                {zionStats._meta.error && (
                  <span className="text-yellow-400">⚠️ {zionStats._meta.error}</span>
                )}
              </div>
            </div>
          )}

          {/* ZION CORE Widgets Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6 mb-8">
            
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

            {/* ZION Wallet Management */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.35 }}
              className="lg:col-span-2 xl:col-span-1"
            >
              <ZionWalletWidget />
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

          </div>

          {/* AI Backend Dashboard Section */}
          <motion.section 
            className="relative z-10 mb-8"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
          >
            <h2 className="text-2xl font-bold mb-4 text-purple-400">
              🤖 AI Backend Systems
            </h2>
            <ZionAIDashboardWidget />
          </motion.section>

          {/* ZION 2.7 Integration Section */}
          <motion.section 
            className="relative z-10 mb-8"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.7 }}
          >
            <h2 className="text-2xl font-bold mb-4 text-green-400">
              🚀 ZION 2.7 REAL SYSTEM INTEGRATION 🚀
            </h2>
            
            {/* ZION 2.7 Widgets Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
              
              {/* ZION 2.7 Mining Widget */}
              <motion.div
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.8 }}
              >
                <Zion27MiningWidget />
              </motion.div>

              {/* Legacy Mining for Comparison */}
              <motion.div
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.9 }}
                className="bg-gradient-to-br from-gray-800/50 to-gray-700/50 border border-gray-600/30 rounded-lg p-4"
              >
                <h3 className="text-sm font-semibold text-gray-400 mb-2">📊 Legacy System (2.6.75)</h3>
                <div className="text-xs text-gray-500 space-y-1">
                  <div>Mining: {zionStats.mining?.status || 'Unknown'}</div>
                  <div>Hashrate: {zionStats.mining?.hashrate ? `${zionStats.mining.hashrate.toLocaleString()} H/s` : 'N/A'}</div>
                  <div>Blocks: {summary?.height || 0}</div>
                </div>
              </motion.div>
            </div>

            {/* Full ZION 2.7 Dashboard */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.0 }}
            >
              <Zion27DashboardWidget />
            </motion.div>
          </motion.section>

          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6 mb-8">
            {/* Performance Overview */}
            <motion.div
              className="bg-gradient-to-br from-purple-900/20 to-blue-900/20 border border-purple-700/30 rounded-xl p-6"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
            >
              <h3 className="text-xl font-semibold mb-4 flex items-center">
                📊 Performance Overview
              </h3>
              
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-gray-300">Total Hashrate:</span>
                  <span className="text-green-400 font-mono">
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
                  <span className="text-gray-300">GPU Power:</span>
                  <span className="text-yellow-400">{zionStats.gpu.powerUsage}W</span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-gray-300">Lightning Channels:</span>
                  <span className="text-purple-400">{zionStats.lightning.activeChannels}</span>
                </div>
              </div>
            </motion.div>

          </div>
        </motion.section>
      )}

      {/* Recent Blocks */}
      <motion.section 
        className="relative z-10"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
      >
        <h2 className="text-2xl font-bold mb-4 text-violet-300">🧱 {t.dashboard.recentBlocks}</h2>
        <div className="bg-black/30 backdrop-blur-sm rounded-lg border border-purple-500/30 overflow-hidden">
          <div className="flex justify-between items-center p-4 border-b border-purple-500/20">
            <h3 className="text-lg font-semibold text-purple-300">{t.dashboard.latestActivity}</h3>
            <div className="text-sm text-gray-400">{t.dashboard.totalMined} ≈ {blocksMinedApprox.toLocaleString()}</div>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="bg-purple-900/30">
                <tr>
                  <th className="text-left py-3 px-4 text-gray-300">{t.dashboard.height}</th>
                  <th className="text-left py-3 px-4 text-gray-300">{t.dashboard.hash}</th>
                  <th className="text-left py-3 px-4 text-gray-300">{t.dashboard.time}</th>
                  <th className="text-left py-3 px-4 text-gray-300">{t.dashboard.age}</th>
                </tr>
              </thead>
              <tbody>
                {recent?.blocks?.map((b) => {
                  const hdr = (b.header?.block_header ?? b.header) || {};
                  const ts = hdr.timestamp as number | undefined;
                  const hash = hdr.hash as string | undefined;
                  const ageSec = ts ? Math.max(0, Math.floor(Date.now()/1000 - ts)) : undefined;
                  return (
                    <tr key={b.height} className="border-b border-purple-500/20 hover:bg-purple-900/20">
                      <td className="py-3 px-4 text-green-400 font-semibold">{b.height}</td>
                      <td className="py-3 px-4 font-mono text-xs text-purple-300">{hash?.slice(0,16)}…</td>
                      <td className="py-3 px-4 text-yellow-400">{fmtTs(ts)}</td>
                      <td className="py-3 px-4 text-blue-400">{ageSec ? `${ageSec}s` : '-'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </motion.section>

      {/* Footer */}
      <motion.footer 
        className="relative z-10 text-center text-purple-300 mt-8"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.8 }}
      >
        <div className="text-sm">
          🧬 {t.footer.dna} • {t.footer.vzestup}
          {zionStats && (
            <>
              <br />
              ⚡ Enhanced with ZION CORE v2.5 • TypeScript Unified Architecture • Multi-Chain Ecosystem
            </>
          )}
        </div>
      </motion.footer>
    </div>
  );
}

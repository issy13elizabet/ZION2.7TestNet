"use client";

import { useEffect, useState, useMemo } from "react";
import { motion } from "framer-motion";
import { useLanguage, LanguageSwitcher } from '../components/LanguageContext';
import ZionCoreWidget from '../components/ZionCoreWidget';
import MiningWidget from '../components/MiningWidget';
import GPUWidget from '../components/GPUWidget';
import SystemWidget from '../components/SystemWidget';
import LightningWidget from '../components/LightningWidget';
import NavigationMenu from '../components/NavigationMenu';

interface ZionCoreStats {
  system: {
    cpu: { manufacturer: string; brand: string; cores: number; speed: number };
    memory: { total: number; used: number; free: number };
    network: Record<string, unknown>;
  };
  blockchain: { 
    height: number; 
    difficulty: number; 
    txCount?: number;
    txPoolSize?: number;
  };
  mining: { 
    hashrate: number; 
    miners: number; 
    difficulty: number;
    algorithm: string;
    status: string;
    shares?: { accepted: number; rejected: number };
  };
  gpu: {
    gpus: Array<{
      id: number;
      name: string;
      status: string;
      hashrate: number;
      temperature: number;
      power: number;
    }>;
    totalHashrate: number;
    powerUsage: number;
  };
  lightning: {
    channels: Array<{
      id: string;
      capacity: number;
      localBalance: number;
      remoteBalance: number;
      active: boolean;
      peerAlias?: string;
    }>;
    totalCapacity: number;
    totalLocalBalance: number;
    totalRemoteBalance: number;
    activeChannels: number;
    pendingChannels: number;
    nodeAlias: string;
    nodeId: string;
  };
  _meta?: {
    source: string;
    timestamp: string;
    error?: string;
  };
}

export default function DashboardV2() {
  const { t } = useLanguage();
  const [stats, setStats] = useState<ZionCoreStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Fetch ZION Core stats
  const fetchStats = async () => {
    try {
      setError(null);
      const response = await fetch('/api/zion-core?endpoint=stats', {
        headers: { 'Accept': 'application/json' }
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      setStats(data);
      setLastUpdate(new Date());
      
      // Log connection status for debugging
      console.log('ZION Core Stats:', {
        source: data._meta?.source,
        blockchain_height: data.blockchain?.height,
        mining_status: data.mining?.status,
        gpu_count: data.gpu?.gpus?.length
      });
      
    } catch (err) {
      console.error('Dashboard fetch error:', err);
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  // Auto-refresh every 5 seconds
  useEffect(() => {
    fetchStats();
    
    if (autoRefresh) {
      const interval = setInterval(fetchStats, 5000);
      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  // Computed network status
  const networkStatus = useMemo(() => {
    if (!stats) return 'loading';
    
    const isHealthy = stats.blockchain?.height > 0 && 
                     stats.mining?.status === 'mining' &&
                     stats._meta?.source !== 'fallback';
    
    return isHealthy ? 'active' : 'syncing';
  }, [stats]);

  // Format hashrate for display
  const formatHashrate = (hashrate: number): string => {
    if (hashrate >= 1e9) return `${(hashrate / 1e9).toFixed(2)} GH/s`;
    if (hashrate >= 1e6) return `${(hashrate / 1e6).toFixed(2)} MH/s`;
    if (hashrate >= 1e3) return `${(hashrate / 1e3).toFixed(2)} KH/s`;
    return `${hashrate.toFixed(2)} H/s`;
  };

  // Format ZION currency
  const formatZion = (amount: number): string => {
    if (amount >= 1e8) return `${(amount / 1e8).toFixed(4)} ZION`;
    if (amount >= 1e6) return `${(amount / 1e6).toFixed(2)}M sats`;
    if (amount >= 1e3) return `${(amount / 1e3).toFixed(1)}K sats`;
    return `${amount} sats`;
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-black text-white flex items-center justify-center">
        <motion.div
          className="text-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
        >
          <div className="w-16 h-16 border-4 border-purple-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-xl">🌌 Connecting to ZION CORE v2.5...</p>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-black text-white p-6">
      {/* Navigation */}
      <NavigationMenu />
      
      <div className="pt-24">
      {/* Cosmic Background */}
      <div className="fixed inset-0 opacity-5 pointer-events-none">
        {[...Array(50)].map((_, i) => (
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

      <div className="relative z-10 max-w-7xl mx-auto">
        {/* Header */}
        <motion.header 
          className="flex justify-between items-center mb-8"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-violet-400 to-purple-300 bg-clip-text text-transparent">
              🌐 ZION CORE v2.5 Dashboard
            </h1>
            <p className="text-gray-400 mt-2">
              Unified TypeScript Ecosystem • {t.networkStatus || 'Network Status'}: 
              <span className={`ml-2 px-3 py-1 rounded-full text-sm ${
                networkStatus === 'active' 
                  ? 'bg-green-900/50 text-green-300' 
                  : 'bg-yellow-900/50 text-yellow-300'
              }`}>
                {networkStatus === 'active' ? '🟢 ACTIVE' : '🟡 SYNCING'}
              </span>
            </p>
          </div>
          
          <div className="flex items-center gap-4">
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`px-4 py-2 rounded-lg transition-colors ${
                autoRefresh 
                  ? 'bg-green-900/50 text-green-300 border border-green-700' 
                  : 'bg-gray-800 text-gray-300 border border-gray-600'
              }`}
            >
              {autoRefresh ? '🔄 Auto' : '⏸️ Paused'}
            </button>
            <button
              onClick={fetchStats}
              disabled={loading}
              className="px-4 py-2 bg-purple-900/50 border border-purple-700 rounded-lg hover:bg-purple-800/50 transition-colors disabled:opacity-50"
            >
              {loading ? '⏳' : '🔃'} Refresh
            </button>
            <LanguageSwitcher />
          </div>
        </motion.header>

        {/* Error Banner */}
        {error && (
          <motion.div
            className="mb-6 p-4 bg-red-900/20 border border-red-700 rounded-lg"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <p className="text-red-300">⚠️ Connection Error: {error}</p>
            <p className="text-red-400 text-sm mt-1">
              Using fallback data. Check ZION Core connection.
            </p>
          </motion.div>
        )}

        {/* Meta Information */}
        {stats?._meta && (
          <motion.div
            className="mb-6 p-3 bg-gray-900/30 border border-gray-700 rounded-lg text-sm"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
          >
            <div className="flex justify-between items-center text-gray-400">
              <span>
                📡 Source: {stats._meta.source} • 
                🕒 Last Update: {lastUpdate.toLocaleTimeString()}
              </span>
              {stats._meta.error && (
                <span className="text-yellow-400">⚠️ {stats._meta.error}</span>
              )}
            </div>
          </motion.div>
        )}

        {/* Main Widgets Grid */}
        {stats && (
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            
            {/* System Stats */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
            >
              <SystemWidget stats={stats.system} />
            </motion.div>

            {/* Blockchain Stats */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
            >
              <ZionCoreWidget 
                blockchain={stats.blockchain}
                networkStatus={networkStatus}
              />
            </motion.div>

            {/* Mining Stats */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
            >
              <MiningWidget 
                mining={stats.mining}
                formatHashrate={formatHashrate}
              />
            </motion.div>

            {/* GPU Stats */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
            >
              <GPUWidget 
                gpu={stats.gpu}
                formatHashrate={formatHashrate}
              />
            </motion.div>

            {/* Lightning Network */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
            >
              <LightningWidget 
              lightning={stats.lightning} 
              formatZion={formatZion}
            />
            </motion.div>

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
                    {formatHashrate(stats.mining.hashrate + (stats.gpu.totalHashrate * 1e6))}
                  </span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-gray-300">Active Miners:</span>
                  <span className="text-blue-400">{stats.mining.miners}</span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-gray-300">GPU Power:</span>
                  <span className="text-yellow-400">{stats.gpu.powerUsage}W</span>
                </div>
                
                <div className="flex justify-between">
                  <span className="text-gray-300">Lightning Channels:</span>
                  <span className="text-purple-400">{stats.lightning.activeChannels}</span>
                </div>
              </div>
            </motion.div>

          </div>
        )}

        {/* Footer */}
        <motion.footer
          className="mt-12 text-center text-gray-400 border-t border-gray-800 pt-6"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
        >
          <p>
            🌌 ZION Blockchain v2.5 TestNet • 
            Powered by TypeScript Unified Architecture • 
            Multi-Chain Ecosystem Active
          </p>
        </motion.footer>
      </div>
    </div>
    </div>
  );
}
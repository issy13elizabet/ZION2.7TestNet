'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface MiningStats {
  hashrate: number;
  algorithm: string;
  status: string;
  efficiency: number;
  shares_accepted: number;
  shares_rejected: number;
  blocks_found: number;
}

export default function Zion27MiningWidget() {
  const [stats, setStats] = useState<MiningStats | null>(null);
  const [isExpanded, setIsExpanded] = useState(false);
  const [isActive, setIsActive] = useState(true);

  const fetchMiningStats = async () => {
    try {
      const response = await fetch('/api/zion-2-7-integration');
      const data = await response.json();
      
      if (data.success && data.data.mining) {
        setStats(data.data.mining);
      }
    } catch (err) {
      console.error('Mining stats fetch error:', err);
    }
  };

  useEffect(() => {
    fetchMiningStats();
    const interval = setInterval(fetchMiningStats, 3000);
    return () => clearInterval(interval);
  }, []);

  if (!stats) {
    return (
      <div className="bg-gradient-to-br from-gray-900 to-gray-800 rounded-lg p-4 border border-gray-600">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-600 rounded mb-2"></div>
          <div className="h-8 bg-gray-600 rounded"></div>
        </div>
      </div>
    );
  }

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'active':
        return 'text-green-400';
      case 'paused':
        return 'text-yellow-400';
      case 'error':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };

  const getHashrateColor = (hashrate: number) => {
    if (hashrate >= 7000) return 'text-green-400';
    if (hashrate >= 6000) return 'text-yellow-400';
    return 'text-orange-400';
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="bg-gradient-to-br from-slate-900 to-slate-800 rounded-lg border border-yellow-500/30 overflow-hidden"
    >
      {/* Header */}
      <div 
        className="p-4 cursor-pointer select-none"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={`w-2 h-2 rounded-full ${isActive ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
            <div>
              <h3 className="text-sm font-semibold text-yellow-400">⛏️ ZION 2.7 MINER</h3>
              <div className="flex items-center gap-2 mt-1">
                <span className={`text-lg font-mono font-bold ${getHashrateColor(stats.hashrate)}`}>
                  {stats.hashrate.toLocaleString()}
                </span>
                <span className="text-xs text-gray-400">H/s</span>
                <span className={`text-xs px-1.5 py-0.5 rounded ${getStatusColor(stats.status)} bg-opacity-20`}>
                  {stats.status.toUpperCase()}
                </span>
              </div>
            </div>
          </div>
          
          <div className="text-right">
            <div className="text-xs text-gray-400">Efficiency</div>
            <div className="text-sm font-mono text-green-400">{stats.efficiency}%</div>
          </div>
        </div>
      </div>

      {/* Expanded Stats */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="border-t border-yellow-500/20"
          >
            <div className="p-4 space-y-3">
              
              {/* Algorithm & Stats */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="text-xs text-gray-400 mb-1">Algorithm</div>
                  <div className="text-sm font-mono text-white bg-gray-800 px-2 py-1 rounded">
                    {stats.algorithm}
                  </div>
                </div>
                
                <div>
                  <div className="text-xs text-gray-400 mb-1">Blocks Found</div>
                  <div className="text-sm font-mono text-green-400">
                    {stats.blocks_found}
                  </div>
                </div>
              </div>

              {/* Shares Stats */}
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-xs text-gray-400">Accepted Shares:</span>
                  <span className="text-sm font-mono text-green-400">{stats.shares_accepted}</span>
                </div>
                
                <div className="flex justify-between items-center">
                  <span className="text-xs text-gray-400">Rejected Shares:</span>
                  <span className="text-sm font-mono text-red-400">{stats.shares_rejected}</span>
                </div>
                
                {/* Success Rate Bar */}
                <div className="mt-2">
                  <div className="text-xs text-gray-400 mb-1">Success Rate</div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <div 
                      className="bg-gradient-to-r from-green-500 to-green-400 h-2 rounded-full transition-all duration-500"
                      style={{ 
                        width: `${((stats.shares_accepted / (stats.shares_accepted + stats.shares_rejected)) * 100) || 100}%` 
                      }}
                    ></div>
                  </div>
                  <div className="text-xs text-green-400 mt-1">
                    {(((stats.shares_accepted / (stats.shares_accepted + stats.shares_rejected)) * 100) || 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* Performance Indicator */}
              <div className="flex items-center justify-between pt-2 border-t border-gray-700">
                <div className="flex items-center gap-2">
                  <div className="w-1 h-1 bg-green-500 rounded-full animate-ping"></div>
                  <span className="text-xs text-green-400">REAL MINING ACTIVE</span>
                </div>
                <div className="text-xs text-gray-400">
                  {new Date().toLocaleTimeString()}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
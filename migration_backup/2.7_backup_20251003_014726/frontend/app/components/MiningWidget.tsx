"use client";

import { motion } from "framer-motion";

interface MiningStats {
  hashrate: number;
  miners: number;
  difficulty: number;
  algorithm: string;
  status: string;
  shares?: { accepted: number; rejected: number };
}

interface Props {
  mining: MiningStats;
  formatHashrate: (hashrate: number) => string;
}

export default function MiningWidget({ mining, formatHashrate }: Props) {
  const getStatusColor = (status: string): string => {
    switch (status.toLowerCase()) {
      case 'mining': return 'text-green-400';
      case 'syncing': return 'text-yellow-400';
      case 'stopped': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getStatusIcon = (status: string): string => {
    switch (status.toLowerCase()) {
      case 'mining': return '⚡';
      case 'syncing': return '🔄';
      case 'stopped': return '⏹️';
      default: return '⚪';
    }
  };

  const shareSuccess = mining.shares 
    ? (mining.shares.accepted / (mining.shares.accepted + mining.shares.rejected)) * 100
    : 100;

  return (
    <div className="bg-gradient-to-br from-green-900/30 to-emerald-900/30 border border-green-700/50 rounded-xl p-6">
      <h3 className="text-xl font-semibold mb-4 flex items-center">
        ⛏️ Mining Status
      </h3>
      
      <div className="space-y-4">
        {/* Mining Status */}
        <div className="bg-green-900/20 rounded-lg p-3">
          <div className="flex justify-between items-center">
            <span className="text-gray-300">Status</span>
            <div className="flex items-center gap-2">
              <span className="text-lg">{getStatusIcon(mining.status)}</span>
              <span className={`text-sm font-semibold ${getStatusColor(mining.status)}`}>
                {mining.status.toUpperCase()}
              </span>
            </div>
          </div>
        </div>

        {/* Hashrate */}
        <div className="bg-green-900/20 rounded-lg p-3">
          <div className="flex justify-between items-center mb-1">
            <span className="text-gray-300">Hashrate</span>
            <motion.span 
              className="text-green-400 font-mono text-lg"
              key={mining.hashrate}
              initial={{ scale: 1.1 }}
              animate={{ scale: 1 }}
              transition={{ duration: 0.3 }}
            >
              {formatHashrate(mining.hashrate)}
            </motion.span>
          </div>
          <div className="text-xs text-gray-500">
            Current mining speed
          </div>
        </div>

        {/* Algorithm & Difficulty */}
        <div className="bg-green-900/20 rounded-lg p-3">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <div className="text-gray-300 text-sm mb-1">Algorithm</div>
              <div className="text-purple-400 font-mono">
                {mining.algorithm}
              </div>
            </div>
            <div>
              <div className="text-gray-300 text-sm mb-1">Difficulty</div>
              <div className="text-yellow-400 font-mono">
                {mining.difficulty.toLocaleString()}
              </div>
            </div>
          </div>
        </div>

        {/* Miners Count */}
        <div className="bg-green-900/20 rounded-lg p-3">
          <div className="flex justify-between items-center">
            <span className="text-gray-300">Active Miners</span>
            <span className="text-blue-400 font-mono text-lg">
              {mining.miners}
            </span>
          </div>
        </div>

        {/* Share Statistics */}
        {mining.shares && (
          <div className="bg-green-900/20 rounded-lg p-3">
            <div className="flex justify-between items-center mb-2">
              <span className="text-gray-300">Share Success</span>
              <span className="text-green-400 font-mono">
                {shareSuccess.toFixed(1)}%
              </span>
            </div>
            
            {/* Success Rate Bar */}
            <div className="w-full bg-gray-700 rounded-full h-2 mb-2">
              <motion.div
                className="bg-gradient-to-r from-green-500 to-emerald-400 h-2 rounded-full"
                initial={{ width: 0 }}
                animate={{ width: `${shareSuccess}%` }}
                transition={{ duration: 1 }}
              />
            </div>
            
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div>
                <span className="text-gray-400">Accepted:</span>
                <span className="text-green-400 ml-2">{mining.shares.accepted}</span>
              </div>
              <div>
                <span className="text-gray-400">Rejected:</span>
                <span className="text-red-400 ml-2">{mining.shares.rejected}</span>
              </div>
            </div>
          </div>
        )}

        {/* Mining Animation */}
        {mining.status.toLowerCase() === 'mining' && (
          <div className="bg-green-900/20 rounded-lg p-3">
            <div className="flex items-center justify-center">
              <motion.div
                className="flex space-x-1"
                animate={{ rotate: 360 }}
                transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
              >
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                <div className="w-2 h-2 bg-emerald-400 rounded-full"></div>
                <div className="w-2 h-2 bg-green-300 rounded-full"></div>
              </motion.div>
              <span className="text-green-400 text-sm ml-3">
                Mining in progress...
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
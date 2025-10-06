"use client";

import { motion } from "framer-motion";

interface BlockchainStats {
  height?: number;
  block_height?: number;  // API returns this field
  difficulty?: number;
  txCount?: number;
  txPoolSize?: number;
}

interface Props {
  blockchain: BlockchainStats;
  networkStatus: string;
}

export default function ZionCoreWidget({ blockchain, networkStatus }: Props) {
  const formatNumber = (num: number): string => {
    if (num >= 1e6) return `${(num / 1e6).toFixed(1)}M`;
    if (num >= 1e3) return `${(num / 1e3).toFixed(1)}K`;
    return num.toString();
  };

  const getDifficultyColor = (difficulty: number): string => {
    if (difficulty > 1e6) return 'text-red-400';
    if (difficulty > 1e4) return 'text-yellow-400';
    return 'text-green-400';
  };

  return (
    <div className="bg-gradient-to-br from-purple-900/30 to-blue-900/30 border border-purple-700/50 rounded-xl p-6">
      <h3 className="text-xl font-semibold mb-4 flex items-center">
        ⛓️ ZION Blockchain
      </h3>
      
      <div className="space-y-4">
        {/* Network Status */}
        <div className="bg-purple-900/20 rounded-lg p-3">
          <div className="flex justify-between items-center">
            <span className="text-gray-300">Network</span>
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${
                networkStatus === 'active' 
                  ? 'bg-green-400 animate-pulse' 
                  : 'bg-yellow-400 animate-bounce'
              }`}></div>
              <span className={`text-sm ${
                networkStatus === 'active' ? 'text-green-400' : 'text-yellow-400'
              }`}>
                {networkStatus.toUpperCase()}
              </span>
            </div>
          </div>
        </div>

        {/* Block Height */}
        <div className="bg-purple-900/20 rounded-lg p-3">
          <div className="flex justify-between items-center mb-1">
            <span className="text-gray-300">Block Height</span>
            <motion.span 
              className="text-purple-400 font-mono text-lg"
              key={blockchain.height}
              initial={{ scale: 1.1, color: '#a78bfa' }}
              animate={{ scale: 1, color: '#c4b5fd' }}
              transition={{ duration: 0.3 }}
            >
              #{(blockchain?.height || blockchain?.block_height || 0).toLocaleString()}
            </motion.span>
          </div>
          <div className="text-xs text-gray-500">
            Latest block confirmed
          </div>
        </div>

        {/* Difficulty */}
        <div className="bg-purple-900/20 rounded-lg p-3">
          <div className="flex justify-between items-center mb-1">
            <span className="text-gray-300">Difficulty</span>
            <span className={`font-mono ${getDifficultyColor(blockchain?.difficulty || 0)}`}>
              {formatNumber(blockchain?.difficulty || 0)}
            </span>
          </div>
          <div className="text-xs text-gray-500">
            Network mining difficulty
          </div>
        </div>

        {/* Transaction Stats */}
        {(blockchain.txCount !== undefined || blockchain.txPoolSize !== undefined) && (
          <div className="bg-purple-900/20 rounded-lg p-3">
            <div className="grid grid-cols-2 gap-3">
              {blockchain.txCount !== undefined && (
                <div>
                  <div className="text-gray-300 text-sm">Transactions</div>
                  <div className="text-blue-400 font-mono">
                    {formatNumber(blockchain.txCount)}
                  </div>
                </div>
              )}
              {blockchain.txPoolSize !== undefined && (
                <div>
                  <div className="text-gray-300 text-sm">Mempool</div>
                  <div className="text-yellow-400 font-mono">
                    {blockchain.txPoolSize}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Sync Status Indicator */}
        <div className="bg-purple-900/20 rounded-lg p-3">
          <div className="flex justify-between items-center">
            <span className="text-gray-300">Sync Status</span>
            <div className="flex items-center gap-2">
              {networkStatus === 'active' ? (
                <>
                  <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                  <span className="text-green-400 text-sm">Synced</span>
                </>
              ) : (
                <>
                  <div className="w-4 h-4 border-2 border-yellow-400 border-t-transparent rounded-full animate-spin"></div>
                  <span className="text-yellow-400 text-sm">Syncing</span>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
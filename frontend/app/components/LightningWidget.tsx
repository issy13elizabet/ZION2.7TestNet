"use client";

import { motion } from "framer-motion";

interface LightningChannel {
  id: string;
  capacity: number;
  localBalance: number;
  remoteBalance: number;
  active: boolean;
  peerAlias?: string;
}

interface LightningStats {
  channels: LightningChannel[];
  totalCapacity: number;
  totalLocalBalance: number;
  totalRemoteBalance: number;
  activeChannels: number;
  pendingChannels: number;
  nodeAlias: string;
  nodeId: string;
}

interface Props {
  lightning?: LightningStats;
  formatZion: (amount: number) => string;
}

export default function LightningWidget({ lightning, formatZion }: Props) {
  // Safe access with fallback data
  const lightningData = lightning || {
    channels: [],
    totalCapacity: 0,
    totalLocalBalance: 0,
    totalRemoteBalance: 0,
    activeChannels: 0,
    pendingChannels: 0,
    nodeAlias: 'ZION-Lightning-Node',
    nodeId: 'zion2.7testnet'
  };
  const getChannelHealthColor = (localBalance: number, capacity: number): string => {
    const ratio = localBalance / capacity;
    if (ratio > 0.7) return 'text-green-400';
    if (ratio > 0.3) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getBalancePercentage = (balance: number, capacity: number): number => {
    return capacity > 0 ? (balance / capacity) * 100 : 0;
  };

  const formatCapacity = (sats: number): string => {
    if (sats >= 1e8) return `${(sats / 1e8).toFixed(2)} ZION`;
    if (sats >= 1e6) return `${(sats / 1e6).toFixed(1)}M sats`;
    if (sats >= 1e3) return `${(sats / 1e3).toFixed(1)}K sats`;
    return `${sats} sats`;
  };

  return (
    <div className="bg-gradient-to-br from-blue-900/30 to-purple-900/30 border border-blue-700/50 rounded-xl p-6">
      <h3 className="text-xl font-semibold mb-4 flex items-center">
        ⚡ Lightning Network
      </h3>
      
      <div className="space-y-4">
        {/* Node Information */}
        <div className="bg-blue-900/20 rounded-lg p-3">
          <div className="text-gray-300 text-sm mb-1">Node Alias</div>
          <div className="text-blue-400 font-semibold mb-2">
            {lightningData.nodeAlias || 'ZION-NODE'}
          </div>
          <div className="text-gray-300 text-sm mb-1">Node ID</div>
          <div className="text-gray-400 font-mono text-xs break-all">
            {lightningData.nodeId || '02...'}
          </div>
        </div>

        {/* Network Stats */}
        <div className="bg-blue-900/20 rounded-lg p-3">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-gray-300 text-sm">Total Capacity</div>
              <motion.div 
                className="text-blue-400 font-mono text-lg"
                key={lightningData.totalCapacity}
                initial={{ scale: 1.1 }}
                animate={{ scale: 1 }}
                transition={{ duration: 0.3 }}
              >
                {formatCapacity(lightningData.totalCapacity)}
              </motion.div>
            </div>
            <div>
              <div className="text-gray-300 text-sm">Active Channels</div>
              <div className="text-green-400 font-mono text-lg">
                {lightningData.activeChannels}
                {lightningData.pendingChannels > 0 && (
                  <span className="text-yellow-400 text-sm ml-2">
                    (+{lightningData.pendingChannels} pending)
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Balance Overview */}
        <div className="bg-blue-900/20 rounded-lg p-3">
          <div className="flex justify-between items-center mb-2">
            <span className="text-gray-300 text-sm">Channel Balances</span>
            <span className="text-blue-400 text-sm">
              {lightningData.totalLocalBalance > 0 ? 
                `${((lightningData.totalLocalBalance / lightningData.totalCapacity) * 100).toFixed(1)}% local` : 
                'No balance'
              }
            </span>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-3 relative overflow-hidden">
            <motion.div
              className="bg-gradient-to-r from-green-500 to-green-400 h-3 rounded-full"
              initial={{ width: 0 }}
              animate={{ 
                width: `${lightningData.totalCapacity > 0 ? 
                  (lightningData.totalLocalBalance / lightningData.totalCapacity) * 100 : 0}%` 
              }}
              transition={{ duration: 1 }}
            />
            <motion.div
              className="bg-gradient-to-r from-blue-500 to-blue-400 h-3 rounded-full absolute top-0"
              style={{ 
                left: `${lightningData.totalCapacity > 0 ? 
                  (lightningData.totalLocalBalance / lightningData.totalCapacity) * 100 : 0}%` 
              }}
              initial={{ width: 0 }}
              animate={{ 
                width: `${lightningData.totalCapacity > 0 ? 
                  (lightningData.totalRemoteBalance / lightningData.totalCapacity) * 100 : 0}%` 
              }}
              transition={{ duration: 1, delay: 0.2 }}
            />
          </div>
          <div className="flex justify-between text-xs mt-1">
            <span className="text-green-400">
              Local: {formatCapacity(lightningData.totalLocalBalance)}
            </span>
            <span className="text-blue-400">
              Remote: {formatCapacity(lightningData.totalRemoteBalance)}
            </span>
          </div>
        </div>

        {/* Channels List */}
        <div className="space-y-2">
          <div className="text-gray-300 text-sm font-medium">Channels</div>
          {lightningData.channels.length > 0 ? (
            <div className="max-h-48 overflow-y-auto space-y-2">
              {lightningData.channels.slice(0, 5).map((channel, index) => (
                <motion.div
                  key={channel.id}
                  className="bg-blue-900/20 rounded-lg p-3"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                >
                  <div className="flex justify-between items-center mb-2">
                    <div className="text-white text-sm font-medium">
                      {channel.peerAlias || `Channel ${channel.id.substring(0, 8)}...`}
                    </div>
                    <div className={`text-xs px-2 py-1 rounded ${
                      channel.active ? 'bg-green-900/50 text-green-400' : 'bg-red-900/50 text-red-400'
                    }`}>
                      {channel.active ? 'ACTIVE' : 'INACTIVE'}
                    </div>
                  </div>
                  
                  <div className="text-xs text-gray-400 mb-2">
                    Capacity: {formatCapacity(channel.capacity)}
                  </div>
                  
                  <div className="w-full bg-gray-700 rounded-full h-2 relative">
                    <motion.div
                      className="bg-gradient-to-r from-green-500 to-green-400 h-2 rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: `${getBalancePercentage(channel.localBalance, channel.capacity)}%` }}
                      transition={{ duration: 0.8, delay: index * 0.1 }}
                    />
                  </div>
                  
                  <div className="flex justify-between text-xs mt-1">
                    <span className="text-green-400">
                      Local: {formatCapacity(channel.localBalance)}
                    </span>
                    <span className="text-blue-400">
                      Remote: {formatCapacity(channel.remoteBalance)}
                    </span>
                  </div>
                </motion.div>
              ))}
              
              {lightningData.channels.length > 5 && (
                <div className="text-center text-gray-400 text-sm py-2">
                  +{lightningData.channels.length - 5} more channels
                </div>
              )}
            </div>
          ) : (
            <div className="bg-blue-900/20 rounded-lg p-4 text-center">
              <div className="text-gray-400 mb-2">⚡ No Lightning channels</div>
              <div className="text-gray-500 text-sm">
                Open channels to start Lightning transactions
              </div>
            </div>
          )}
        </div>

        {/* Network Status */}
        <div className="bg-blue-900/20 rounded-lg p-3">
          <div className="flex justify-between items-center">
            <span className="text-gray-300">Lightning Status</span>
            <div className="flex items-center gap-2">
              {lightningData.activeChannels > 0 ? (
                <>
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                  <span className="text-green-400 text-sm">Online</span>
                </>
              ) : (
                <>
                  <div className="w-2 h-2 bg-gray-400 rounded-full"></div>
                  <span className="text-gray-400 text-sm">Offline</span>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
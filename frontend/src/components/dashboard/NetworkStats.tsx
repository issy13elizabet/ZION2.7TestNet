'use client'

import { useZion } from '@/contexts/ZionContext'
import { useEffect, useState } from 'react'

interface NetworkStatsProps {
  expanded?: boolean
}

export function NetworkStats({ expanded = false }: NetworkStatsProps) {
  const { networkInfo, isConnected } = useZion()
  const [hashRateHistory, setHashRateHistory] = useState<number[]>([])

  useEffect(() => {
    if (networkInfo?.hashrate) {
      setHashRateHistory(prev => [...prev.slice(-19), networkInfo.hashrate])
    }
  }, [networkInfo])

  const formatHashrate = (hashrate: number): string => {
    if (hashrate < 1000) return `${hashrate.toFixed(0)} H/s`
    if (hashrate < 1000000) return `${(hashrate / 1000).toFixed(1)} KH/s`
    if (hashrate < 1000000000) return `${(hashrate / 1000000).toFixed(1)} MH/s`
    return `${(hashrate / 1000000000).toFixed(1)} GH/s`
  }

  const formatNumber = (num: number): string => {
    return new Intl.NumberFormat().format(num)
  }

  if (!isConnected) {
    return (
      <div className="card-dark">
        <div className="flex items-center justify-center h-40">
          <div className="text-center">
            <div className="w-8 h-8 border-4 border-purple-500/30 border-t-purple-500 rounded-full animate-spin mx-auto mb-4"></div>
            <p className="text-gray-400">Connecting to ZION Network...</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="card-dark">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold gradient-text">Network Statistics</h2>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
          <span className="text-green-400 text-sm font-medium">Live</span>
        </div>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <div className="bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-lg p-4 border border-blue-500/30">
          <div className="text-blue-300 text-sm font-medium mb-1">Block Height</div>
          <div className="text-2xl font-bold text-white">
            {networkInfo ? formatNumber(networkInfo.height) : '---'}
          </div>
        </div>

        <div className="bg-gradient-to-br from-green-500/20 to-emerald-500/20 rounded-lg p-4 border border-green-500/30">
          <div className="text-green-300 text-sm font-medium mb-1">Network Hashrate</div>
          <div className="text-2xl font-bold text-white">
            {networkInfo ? formatHashrate(networkInfo.hashrate) : '---'}
          </div>
        </div>

        <div className="bg-gradient-to-br from-yellow-500/20 to-orange-500/20 rounded-lg p-4 border border-yellow-500/30">
          <div className="text-yellow-300 text-sm font-medium mb-1">Difficulty</div>
          <div className="text-2xl font-bold text-white">
            {networkInfo ? formatNumber(networkInfo.difficulty) : '---'}
          </div>
        </div>

        <div className="bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-lg p-4 border border-purple-500/30">
          <div className="text-purple-300 text-sm font-medium mb-1">Connected Peers</div>
          <div className="text-2xl font-bold text-white">
            {networkInfo ? networkInfo.peers : '---'}
          </div>
        </div>
      </div>

      {expanded && (
        <div className="space-y-4">
          <div className="bg-black/40 rounded-lg p-4 border border-white/10">
            <h3 className="text-lg font-semibold mb-3 text-white">Network Details</h3>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-gray-400">Version:</span>
                <span className="text-white ml-2 font-mono">
                  {networkInfo?.version || '2.6.75'}
                </span>
              </div>
              <div>
                <span className="text-gray-400">Status:</span>
                <span className="text-green-400 ml-2 font-semibold">
                  {networkInfo?.status || 'OK'}
                </span>
              </div>
            </div>
          </div>

          {hashRateHistory.length > 0 && (
            <div className="bg-black/40 rounded-lg p-4 border border-white/10">
              <h3 className="text-lg font-semibold mb-3 text-white">Hashrate Chart</h3>
              <div className="h-20 flex items-end space-x-1">
                {hashRateHistory.map((rate, index) => (
                  <div
                    key={index}
                    className="bg-gradient-to-t from-purple-500 to-pink-500 rounded-t-sm flex-1 min-w-0 transition-all duration-300"
                    style={{
                      height: `${Math.max(5, (rate / Math.max(...hashRateHistory)) * 100)}%`
                    }}
                    title={`${formatHashrate(rate)}`}
                  ></div>
                ))}
              </div>
              <div className="text-xs text-gray-400 mt-2">
                Real-time hashrate visualization
              </div>
            </div>
          )}
        </div>
      )}

      <div className="flex items-center justify-between mt-4 pt-4 border-t border-white/10">
        <div className="text-sm text-gray-400">
          Last updated: {new Date().toLocaleTimeString()}
        </div>
        <div className="text-sm text-purple-400 font-medium">
          🔗 ZION Network v{networkInfo?.version || '2.6.75'}
        </div>
      </div>
    </div>
  )
}
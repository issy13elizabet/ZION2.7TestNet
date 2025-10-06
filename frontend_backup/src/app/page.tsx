'use client'

import { useEffect, useState } from 'react'
import { useZion } from '@/contexts/ZionContext'
import { NetworkStats } from '@/components/dashboard/NetworkStats'
import { MiningDashboard } from '@/components/dashboard/MiningDashboard'
import { WalletOverview } from '@/components/wallet/WalletOverview'
import { RecentBlocks } from '@/components/explorer/RecentBlocks'
import { RainbowBridge } from '@/components/bridge/RainbowBridge'

export default function HomePage() {
  const { networkInfo, isConnected } = useZion()
  const [activeTab, setActiveTab] = useState('dashboard')

  const tabs = [
    { id: 'dashboard', name: 'Dashboard', icon: '📊' },
    { id: 'wallet', name: 'Wallet', icon: '💳' },
    { id: 'mining', name: 'Mining', icon: '⛏️' },
    { id: 'explorer', name: 'Explorer', icon: '🔍' },
    { id: 'bridge', name: 'Rainbow Bridge', icon: '🌈' },
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900">
      {/* Hero Section */}
      <div className="relative overflow-hidden bg-black/20 backdrop-blur-sm border-b border-purple-500/30">
        <div className="container mx-auto px-4 py-12 text-center">
          <h1 className="text-6xl font-bold mb-4 bg-gradient-to-r from-purple-400 via-pink-500 to-red-500 bg-clip-text text-transparent">
            ZION Network 2.6.75
          </h1>
          <p className="text-xl text-gray-300 mb-6">
            Advanced Blockchain with RandomX Mining & Cross-Chain Bridges
          </p>
          <div className="flex justify-center items-center space-x-6">
            <div className={`flex items-center space-x-2 ${isConnected ? 'text-green-400' : 'text-red-400'}`}>
              <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-400' : 'bg-red-400'} animate-pulse`}></div>
              <span className="font-medium">{isConnected ? 'Connected' : 'Disconnected'}</span>
            </div>
            {networkInfo && (
              <>
                <div className="text-gray-300">
                  Block Height: <span className="text-white font-bold">{networkInfo.height.toLocaleString()}</span>
                </div>
                <div className="text-gray-300">
                  Hashrate: <span className="text-white font-bold">{(networkInfo.hashrate / 1000).toFixed(2)} KH/s</span>
                </div>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="sticky top-0 z-40 bg-black/50 backdrop-blur-md border-b border-purple-500/30">
        <div className="container mx-auto px-4">
          <div className="flex space-x-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center space-x-2 px-6 py-4 font-medium transition-all duration-200 border-b-2 ${
                  activeTab === tab.id
                    ? 'border-purple-500 text-purple-400 bg-purple-500/10'
                    : 'border-transparent text-gray-400 hover:text-white hover:bg-white/5'
                }`}
              >
                <span className="text-lg">{tab.icon}</span>
                <span>{tab.name}</span>
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Content Area */}
      <div className="container mx-auto px-4 py-8">
        {activeTab === 'dashboard' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            <div className="xl:col-span-2">
              <NetworkStats />
            </div>
            <div>
              <WalletOverview />
            </div>
            <div className="lg:col-span-2">
              <MiningDashboard />
            </div>
            <div>
              <RecentBlocks />
            </div>
          </div>
        )}

        {activeTab === 'wallet' && (
          <div className="max-w-4xl mx-auto">
            <WalletOverview expanded />
          </div>
        )}

        {activeTab === 'mining' && (
          <div className="max-w-6xl mx-auto">
            <MiningDashboard expanded />
          </div>
        )}

        {activeTab === 'explorer' && (
          <div className="max-w-6xl mx-auto">
            <RecentBlocks expanded />
          </div>
        )}

        {activeTab === 'bridge' && (
          <div className="max-w-4xl mx-auto">
            <RainbowBridge />
          </div>
        )}
      </div>
    </div>
  )
}
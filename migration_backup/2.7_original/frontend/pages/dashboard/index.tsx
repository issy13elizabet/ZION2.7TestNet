/**
 * ZION 2.7 Mining Dashboard
 * Real-time monitoring and control center
 */
import { useState } from 'react';
import MiningStats from '@/components/dashboard/MiningStats';
import NetworkStatus from '@/components/dashboard/NetworkStatus';

export default function Dashboard() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">
            🌟 ZION 2.7 Mining Dashboard
          </h1>
          <p className="text-xl text-purple-200">
            Real-time Sacred Mining Network Monitor
          </p>
        </div>

        {/* Mining Stats */}
        <div className="mb-8">
          <MiningStats />
        </div>

        {/* Network Status */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <NetworkStatus />

          {/* Placeholder for future components */}
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg">
            <h2 className="text-2xl font-bold mb-4 text-gray-800 dark:text-white">
              🔮 Sacred Flower Analysis
            </h2>
            <p className="text-gray-600 dark:text-gray-400">
              Coming soon: Real-time consciousness monitoring
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

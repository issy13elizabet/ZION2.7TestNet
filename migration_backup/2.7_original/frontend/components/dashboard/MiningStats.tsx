/**
 * Real-time Mining Statistics Dashboard
 */
'use client';

import { useState, useEffect } from 'react';
import { ZionAPI, MiningStats } from '@/lib/api/client';

export default function MiningStats() {
  const [stats, setStats] = useState<MiningStats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Initial load
    ZionAPI.getMiningStats().then(setStats).finally(() => setLoading(false));

    // Real-time updates
    const ws = ZionAPI.createMiningWebSocket((data) => {
      setStats(data);
    });

    return () => ws.close();
  }, []);

  if (loading) return <div className="animate-pulse">Loading mining stats...</div>;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      <div className="bg-gradient-to-br from-purple-500 to-pink-500 p-6 rounded-lg text-white">
        <h3 className="text-lg font-semibold">Network Hashrate</h3>
        <p className="text-3xl font-bold">{stats?.hashrate.toLocaleString()} H/s</p>
      </div>

      <div className="bg-gradient-to-br from-blue-500 to-cyan-500 p-6 rounded-lg text-white">
        <h3 className="text-lg font-semibold">Active Miners</h3>
        <p className="text-3xl font-bold">{stats?.active_miners}</p>
      </div>

      <div className="bg-gradient-to-br from-green-500 to-emerald-500 p-6 rounded-lg text-white">
        <h3 className="text-lg font-semibold">Accepted Shares</h3>
        <p className="text-3xl font-bold">{stats?.accepted_shares.toLocaleString()}</p>
      </div>

      <div className="bg-gradient-to-br from-orange-500 to-red-500 p-6 rounded-lg text-white">
        <h3 className="text-lg font-semibold">Block Height</h3>
        <p className="text-3xl font-bold">{stats?.block_height}</p>
      </div>
    </div>
  );
}

"use client";

import { motion } from "framer-motion";

interface SystemStats {
  cpu: { manufacturer: string; brand: string; cores: number; speed: number };
  memory: { total: number; used: number; free: number };
  network: Record<string, unknown>;
}

interface Props {
  stats: SystemStats;
}

export default function SystemWidget({ stats }: Props) {
  const memoryUsagePercent = (stats.memory.used / stats.memory.total) * 100;
  
  const formatBytes = (bytes: number): string => {
    const gb = bytes / (1024 * 1024 * 1024);
    return `${gb.toFixed(1)} GB`;
  };

  return (
    <div className="bg-gradient-to-br from-gray-900/40 to-gray-800/40 border border-gray-700/50 rounded-xl p-6">
      <h3 className="text-xl font-semibold mb-4 flex items-center">
        🖥️ System Resources
      </h3>
      
      <div className="space-y-4">
        {/* CPU Info */}
        <div className="bg-gray-800/30 rounded-lg p-3">
          <div className="flex justify-between items-center mb-2">
            <span className="text-gray-300">CPU</span>
            <span className="text-green-400 text-sm">{stats.cpu.cores} cores</span>
          </div>
          <div className="text-sm text-gray-400 mb-1">
            {stats.cpu.manufacturer} {stats.cpu.brand}
          </div>
          <div className="text-xs text-gray-500">
            {stats.cpu.speed} MHz
          </div>
        </div>

        {/* Memory Usage */}
        <div className="bg-gray-800/30 rounded-lg p-3">
          <div className="flex justify-between items-center mb-2">
            <span className="text-gray-300">Memory</span>
            <span className="text-blue-400 text-sm">
              {formatBytes(stats.memory.used)} / {formatBytes(stats.memory.total)}
            </span>
          </div>
          
          {/* Memory Bar */}
          <div className="w-full bg-gray-700 rounded-full h-2 mb-1">
            <motion.div
              className="bg-gradient-to-r from-blue-500 to-blue-400 h-2 rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${memoryUsagePercent}%` }}
              transition={{ duration: 1 }}
            />
          </div>
          
          <div className="text-xs text-gray-500">
            {memoryUsagePercent.toFixed(1)}% used • {formatBytes(stats.memory.free)} free
          </div>
        </div>

        {/* System Status */}
        <div className="bg-gray-800/30 rounded-lg p-3">
          <div className="flex justify-between items-center">
            <span className="text-gray-300">Status</span>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-green-400 text-sm">Online</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
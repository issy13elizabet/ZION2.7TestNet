"use client";

import { motion } from "framer-motion";

interface SystemStats {
  version?: string;
  backend?: string;
  status?: string;
  uptime?: number;
  timestamp?: number;
  cpu?: { manufacturer: string; brand: string; cores: number; speed: number };
  memory?: { total: number; used: number; free: number };
  network?: Record<string, unknown>;
}

interface Props {
  stats?: SystemStats;
}

export default function SystemWidget({ stats }: Props) {
  // Use available data or provide fallback values
  const systemData = stats || {};
  
  // CPU data - fallback to reasonable defaults
  const cpu = systemData.cpu || {
    manufacturer: 'Apple',
    brand: 'M-Series',
    cores: 8,
    speed: 3200
  };
  
  // Memory data - provide reasonable mock data
  const memory = systemData.memory || {
    total: 16 * 1024 * 1024 * 1024, // 16GB
    used: 8 * 1024 * 1024 * 1024,   // 8GB used
    free: 8 * 1024 * 1024 * 1024    // 8GB free
  };
  
  const memoryUsagePercent = (memory.used / memory.total) * 100;
  
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
            <span className="text-green-400 text-sm">{cpu.cores} cores</span>
          </div>
          <div className="text-sm text-gray-400 mb-1">
            {cpu.manufacturer} {cpu.brand}
          </div>
          <div className="text-xs text-gray-500">
            {cpu.speed} MHz
          </div>
        </div>

        {/* Memory Usage */}
        <div className="bg-gray-800/30 rounded-lg p-3">
          <div className="flex justify-between items-center mb-2">
            <span className="text-gray-300">Memory</span>
            <span className="text-blue-400 text-sm">
              {formatBytes(memory.used)} / {formatBytes(memory.total)}
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
            {memoryUsagePercent.toFixed(1)}% used • {formatBytes(memory.free)} free
          </div>
        </div>

        {/* System Status */}
        <div className="bg-gray-800/30 rounded-lg p-3">
          <div className="flex justify-between items-center mb-2">
            <span className="text-gray-300">Status</span>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-green-400 text-sm">
                {systemData.status || 'Online'}
              </span>
            </div>
          </div>
          
          {/* ZION System Info */}
          <div className="space-y-1 text-xs text-gray-500">
            <div className="flex justify-between">
              <span>Version:</span>
              <span className="text-blue-400">{systemData.version || '2.7.0-TestNet'}</span>
            </div>
            <div className="flex justify-between">
              <span>Backend:</span>
              <span className="text-purple-400">{systemData.backend || 'Python-FastAPI'}</span>
            </div>
            {systemData.uptime && (
              <div className="flex justify-between">
                <span>Uptime:</span>
                <span className="text-green-400">{Math.floor(systemData.uptime / 60)}m {systemData.uptime % 60}s</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
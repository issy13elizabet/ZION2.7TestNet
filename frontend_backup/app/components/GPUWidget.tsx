"use client";

import { motion } from "framer-motion";

interface GPUDevice {
  id: number;
  name: string;
  status: string;
  hashrate: number;
  temperature: number;
  power: number;
}

interface GPUStats {
  gpus: GPUDevice[];
  totalHashrate: number;
  powerUsage: number;
}

interface Props {
  gpu: GPUStats;
  formatHashrate: (hashrate: number) => string;
}

export default function GPUWidget({ gpu, formatHashrate }: Props) {
  const getStatusColor = (status: string): string => {
    switch (status.toLowerCase()) {
      case 'mining': return 'text-green-400';
      case 'idle': return 'text-blue-400';
      case 'error': return 'text-red-400';
      case 'benchmark': return 'text-yellow-400';
      default: return 'text-gray-400';
    }
  };

  const getTempColor = (temp: number): string => {
    if (temp > 80) return 'text-red-400';
    if (temp > 70) return 'text-yellow-400';
    return 'text-green-400';
  };

  const getPowerEfficiency = (hashrate: number, power: number): number => {
    if (power === 0) return 0;
    return (hashrate * 1e6) / power; // H/s per Watt
  };

  return (
    <div className="bg-gradient-to-br from-yellow-900/30 to-orange-900/30 border border-yellow-700/50 rounded-xl p-6">
      <h3 className="text-xl font-semibold mb-4 flex items-center">
        🎮 GPU Mining
      </h3>
      
      <div className="space-y-4">
        {/* Total GPU Stats */}
        <div className="bg-yellow-900/20 rounded-lg p-3">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <div className="text-gray-300 text-sm">Total Hashrate</div>
              <motion.div 
                className="text-yellow-400 font-mono text-lg"
                key={gpu.totalHashrate}
                initial={{ scale: 1.1 }}
                animate={{ scale: 1 }}
                transition={{ duration: 0.3 }}
              >
                {formatHashrate(gpu.totalHashrate * 1e6)}
              </motion.div>
            </div>
            <div>
              <div className="text-gray-300 text-sm">Power Usage</div>
              <div className="text-red-400 font-mono text-lg">
                {gpu.powerUsage}W
              </div>
            </div>
          </div>
        </div>

        {/* GPU Devices */}
        <div className="space-y-3">
          {gpu.gpus.length > 0 ? (
            gpu.gpus.map((device) => (
              <motion.div
                key={device.id}
                className="bg-yellow-900/20 rounded-lg p-3"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: device.id * 0.1 }}
              >
                {/* GPU Header */}
                <div className="flex justify-between items-center mb-2">
                  <div className="text-white font-semibold truncate">
                    GPU #{device.id} • {device.name}
                  </div>
                  <div className={`text-sm ${getStatusColor(device.status)}`}>
                    {device.status.toUpperCase()}
                  </div>
                </div>

                {/* GPU Metrics */}
                <div className="grid grid-cols-3 gap-3 text-sm">
                  <div>
                    <div className="text-gray-400">Hashrate</div>
                    <div className="text-green-400 font-mono">
                      {formatHashrate(device.hashrate * 1e6)}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-400">Temp</div>
                    <div className={`font-mono ${getTempColor(device.temperature)}`}>
                      {device.temperature}°C
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-400">Power</div>
                    <div className="text-red-400 font-mono">
                      {device.power}W
                    </div>
                  </div>
                </div>

                {/* Efficiency Bar */}
                <div className="mt-2">
                  <div className="flex justify-between text-xs text-gray-400 mb-1">
                    <span>Efficiency</span>
                    <span>{getPowerEfficiency(device.hashrate, device.power).toFixed(1)} H/W</span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-1">
                    <motion.div
                      className="bg-gradient-to-r from-yellow-500 to-orange-400 h-1 rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.min(getPowerEfficiency(device.hashrate, device.power) / 1000 * 100, 100)}%` }}
                      transition={{ duration: 1, delay: device.id * 0.2 }}
                    />
                  </div>
                </div>

                {/* Status Indicator */}
                {device.status.toLowerCase() === 'mining' && (
                  <div className="flex items-center mt-2 text-xs text-green-400">
                    <motion.div
                      className="w-2 h-2 bg-green-400 rounded-full mr-2"
                      animate={{ opacity: [1, 0.3, 1] }}
                      transition={{ duration: 1.5, repeat: Infinity }}
                    />
                    Mining active
                  </div>
                )}
              </motion.div>
            ))
          ) : (
            <div className="bg-yellow-900/20 rounded-lg p-4 text-center">
              <div className="text-gray-400 mb-2">🔍 No GPUs detected</div>
              <div className="text-gray-500 text-sm">
                Check GPU drivers or mining configuration
              </div>
            </div>
          )}
        </div>

        {/* GPU Count Summary */}
        <div className="bg-yellow-900/20 rounded-lg p-3">
          <div className="flex justify-between items-center">
            <span className="text-gray-300">Active GPUs</span>
            <div className="flex items-center gap-2">
              <span className="text-yellow-400 font-mono">
                {gpu.gpus.filter(g => g.status.toLowerCase() === 'mining').length} / {gpu.gpus.length}
              </span>
              {gpu.gpus.length > 0 && (
                <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse"></div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
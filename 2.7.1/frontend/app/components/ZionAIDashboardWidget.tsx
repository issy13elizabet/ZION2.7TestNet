'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';

interface AISystem {
  name: string;
  status: string;
  capabilities: string[];
  metrics: Record<string, any>;
  last_activity: number;
}

interface AIBackendData {
  ai_systems: Record<string, AISystem>;
  system_status: Record<string, boolean>;
  performance_metrics: {
    total_requests: number;
    successful_operations: number;
    failed_operations: number;
    average_response_time: number;
  };
  hardware_status: {
    cpu_usage: string;
    memory_usage: string;
    gpu_count: number;
    gpus: Array<{
      id: number;
      name: string;
      load: string;
      memory_used: string;
      memory_total: string;
      temperature: string;
    }>;
  };
  uptime_formatted: string;
}

const ZionAIDashboardWidget: React.FC = () => {
  const [aiData, setAiData] = useState<AIBackendData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchAIData = async () => {
    try {
      setError(null);
      const response = await fetch('http://localhost:8001/api/ai/overview', {
        method: 'GET',
        cache: 'no-store',
        signal: AbortSignal.timeout(5000)
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      
      const data = await response.json();
      if (data.success) {
        setAiData(data.data);
      } else {
        setError('AI Backend not responding');
      }
    } catch (err) {
      console.error('AI Dashboard fetch error:', err);
      setError('Connection failed');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAIData();
    const interval = setInterval(fetchAIData, 10000); // Refresh every 10 seconds
    return () => clearInterval(interval);
  }, []);

  const getActiveSystemsCount = () => {
    if (!aiData) return 0;
    return Object.values(aiData.system_status).filter(Boolean).length;
  };

  const getSuccessRate = () => {
    if (!aiData || !aiData.performance_metrics) return 100;
    const { successful_operations, failed_operations } = aiData.performance_metrics;
    const total = successful_operations + failed_operations;
    return total > 0 ? (successful_operations / total * 100) : 100;
  };

  const getCriticalSystems = () => {
    if (!aiData) return [];
    return ['afterburner', 'security_monitor', 'blockchain_analytics'].filter(key => 
      aiData.ai_systems[key] && aiData.system_status[key]
    );
  };

  if (loading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {[1, 2, 3].map((i) => (
          <div key={i} className="bg-gray-800/50 border border-gray-700 rounded-xl p-6 animate-pulse">
            <div className="h-4 bg-gray-700 rounded w-3/4 mb-4"></div>
            <div className="h-8 bg-gray-700 rounded w-1/2 mb-2"></div>
            <div className="h-3 bg-gray-700 rounded w-full"></div>
          </div>
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-900/20 border border-red-700/30 rounded-xl p-6">
        <div className="text-center">
          <h3 className="text-xl font-semibold mb-2 text-red-400">
            🤖 AI Backend Status
          </h3>
          <div className="text-red-300 mb-4">❌ {error}</div>
          <button
            onClick={fetchAIData}
            className="px-4 py-2 bg-red-600/50 hover:bg-red-500/50 rounded-lg text-red-200 transition-colors text-sm"
          >
            🔄 Retry Connection
          </button>
        </div>
      </div>
    );
  }

  if (!aiData) return null;

  const activeSystemsCount = getActiveSystemsCount();
  const successRate = getSuccessRate();
  const criticalSystems = getCriticalSystems();

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {/* AI Systems Overview */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-gradient-to-br from-purple-900/20 to-blue-900/20 border border-purple-700/30 rounded-xl p-6"
      >
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-purple-400">🤖 AI Systems</h3>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
            <span className="text-xs text-green-400">ONLINE</span>
          </div>
        </div>

        <div className="text-center mb-4">
          <div className="text-3xl font-bold text-white mb-1">{activeSystemsCount}/10</div>
          <div className="text-sm text-gray-400">Active Systems</div>
        </div>

        <div className="space-y-2 mb-4">
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">Success Rate</span>
            <span className="text-green-400">{successRate.toFixed(1)}%</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">Avg Response</span>
            <span className="text-blue-400">
              {aiData.performance_metrics.average_response_time ? 
                `${(aiData.performance_metrics.average_response_time * 1000).toFixed(0)}ms` : 
                '0ms'
              }
            </span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">Uptime</span>
            <span className="text-yellow-400">{aiData.uptime_formatted}</span>
          </div>
        </div>

        <Link 
          href="/ai"
          className="block w-full text-center py-2 px-3 bg-purple-600/50 hover:bg-purple-500/50 rounded-lg transition-colors text-purple-200 hover:text-white text-sm"
        >
          🚀 Full AI Dashboard
        </Link>
      </motion.div>

      {/* Critical Systems Status */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="bg-gradient-to-br from-orange-900/20 to-red-900/20 border border-orange-700/30 rounded-xl p-6"
      >
        <h3 className="text-lg font-semibold text-orange-400 mb-4">🛡️ Critical Systems</h3>

        <div className="space-y-3">
          {['afterburner', 'security_monitor', 'blockchain_analytics'].map((systemKey) => {
            const system = aiData.ai_systems[systemKey];
            const isActive = aiData.system_status[systemKey];
            
            if (!system) return null;
            
            return (
              <div key={systemKey} className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${isActive ? 'bg-green-400' : 'bg-red-400'}`}></div>
                  <span className="text-sm text-white">
                    {systemKey === 'afterburner' ? '🚀 Afterburner' :
                     systemKey === 'security_monitor' ? '🛡️ Security' :
                     '📊 Analytics'}
                  </span>
                </div>
                <span className={`text-xs ${isActive ? 'text-green-400' : 'text-red-400'}`}>
                  {isActive ? 'ACTIVE' : 'OFFLINE'}
                </span>
              </div>
            );
          })}
        </div>

        <div className="mt-4 pt-4 border-t border-gray-600">
          <div className="text-sm text-gray-400 mb-2">Critical Systems Health</div>
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div 
              className={`h-2 rounded-full transition-all ${
                criticalSystems.length === 3 ? 'bg-green-500' :
                criticalSystems.length >= 2 ? 'bg-yellow-500' : 'bg-red-500'
              }`}
              style={{ width: `${(criticalSystems.length / 3) * 100}%` }}
            />
          </div>
          <div className="text-xs text-gray-400 mt-1">
            {criticalSystems.length}/3 systems operational
          </div>
        </div>
      </motion.div>

      {/* Hardware & Performance */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="bg-gradient-to-br from-blue-900/20 to-cyan-900/20 border border-blue-700/30 rounded-xl p-6"
      >
        <h3 className="text-lg font-semibold text-blue-400 mb-4">💻 Hardware Status</h3>

        {aiData.hardware_status && (
          <>
            <div className="space-y-3 mb-4">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-400">CPU</span>
                  <span className="text-white">{aiData.hardware_status.cpu_usage}</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div 
                    className="bg-blue-500 h-2 rounded-full transition-all"
                    style={{ width: aiData.hardware_status.cpu_usage }}
                  />
                </div>
              </div>
              
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-400">Memory</span>
                  <span className="text-white">{aiData.hardware_status.memory_usage}</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div 
                    className="bg-cyan-500 h-2 rounded-full transition-all"
                    style={{ width: aiData.hardware_status.memory_usage }}
                  />
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3 text-center">
              <div className="bg-gray-900/50 p-3 rounded-lg">
                <div className="text-lg font-bold text-white">{aiData.hardware_status.gpu_count}</div>
                <div className="text-xs text-gray-400">GPUs</div>
              </div>
              <div className="bg-gray-900/50 p-3 rounded-lg">
                <div className="text-lg font-bold text-cyan-400">
                  {aiData.performance_metrics.total_requests}
                </div>
                <div className="text-xs text-gray-400">Requests</div>
              </div>
            </div>
          </>
        )}

        <Link 
          href="/miner"
          className="block w-full text-center py-2 px-3 bg-blue-600/50 hover:bg-blue-500/50 rounded-lg transition-colors text-blue-200 hover:text-white text-sm mt-4"
        >
          ⛏️ AI Mining Center
        </Link>
      </motion.div>
    </div>
  );
};

export default ZionAIDashboardWidget;
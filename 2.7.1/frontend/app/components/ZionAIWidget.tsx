'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';

interface AISystemMetrics {
  operations_count: number;
  optimization_level?: number;
  health_score?: number;
  processing_speed?: number;
  success_rate?: number;
  [key: string]: any;
}

interface AISystem {
  name: string;
  status: string;
  capabilities: string[];
  metrics: AISystemMetrics;
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
  uptime_formatted: string;
}

const ZionAIWidget: React.FC = () => {
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
      console.error('AI Backend fetch error:', err);
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

  const getTotalOperations = () => {
    if (!aiData) return 0;
    return Object.values(aiData.ai_systems).reduce((total, system) => {
      return total + (system.metrics.operations_count || 0);
    }, 0);
  };

  const getSuccessRate = () => {
    if (!aiData || !aiData.performance_metrics) return 100;
    const { successful_operations, failed_operations } = aiData.performance_metrics;
    const total = successful_operations + failed_operations;
    return total > 0 ? (successful_operations / total * 100) : 100;
  };

  const getTopPerformingSystem = () => {
    if (!aiData) return null;
    const systems = Object.entries(aiData.ai_systems);
    if (systems.length === 0) return null;
    
    return systems.reduce((best, [key, system]) => {
      const currentOps = system.metrics.operations_count || 0;
      const bestOps = best.system.metrics.operations_count || 0;
      return currentOps > bestOps ? { key, system } : best;
    }, { key: systems[0][0], system: systems[0][1] });
  };

  if (loading) {
    return (
      <div className="bg-gradient-to-br from-purple-900/20 to-blue-900/20 border border-purple-700/30 rounded-xl p-6">
        <div className="flex items-center justify-center h-48">
          <div className="text-center">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
              className="w-8 h-8 border-2 border-purple-500 border-t-transparent rounded-full mx-auto mb-4"
            />
            <p className="text-purple-400">🤖 Connecting to AI Backend...</p>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-gradient-to-br from-red-900/20 to-orange-900/20 border border-red-700/30 rounded-xl p-6">
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
  const totalOperations = getTotalOperations();
  const successRate = getSuccessRate();
  const topSystem = getTopPerformingSystem();

  return (
    <div className="bg-gradient-to-br from-purple-900/20 to-blue-900/20 border border-purple-700/30 rounded-xl p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xl font-semibold flex items-center">
          🤖 AI Backend Status
        </h3>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
          <span className="text-xs text-green-400">ACTIVE</span>
        </div>
      </div>

      {/* AI Stats Grid */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="bg-black/20 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-green-400">{activeSystemsCount}</div>
          <div className="text-xs text-gray-400">Active Systems</div>
        </div>
        
        <div className="bg-black/20 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-blue-400">{totalOperations}</div>
          <div className="text-xs text-gray-400">Total Operations</div>
        </div>
        
        <div className="bg-black/20 rounded-lg p-3 text-center">
          <div className="text-2xl font-bold text-purple-400">{successRate.toFixed(1)}%</div>
          <div className="text-xs text-gray-400">Success Rate</div>
        </div>
        
        <div className="bg-black/20 rounded-lg p-3 text-center">
          <div className="text-lg font-bold text-yellow-400">
            {aiData.performance_metrics.average_response_time ? 
              `${(aiData.performance_metrics.average_response_time * 1000).toFixed(0)}ms` : 
              '0ms'
            }
          </div>
          <div className="text-xs text-gray-400">Avg Response</div>
        </div>
      </div>

      {/* Top Performing System */}
      {topSystem && (
        <div className="bg-black/20 rounded-lg p-3 mb-4">
          <div className="text-sm text-gray-400 mb-1">🏆 Top Performer:</div>
          <div className="text-sm font-medium text-white">{topSystem.system.name}</div>
          <div className="text-xs text-purple-400">
            {topSystem.system.metrics.operations_count} operations
          </div>
        </div>
      )}

      {/* System Status Indicators */}
      <div className="mb-4">
        <div className="text-sm text-gray-400 mb-2">🔧 System Status:</div>
        <div className="grid grid-cols-5 gap-1">
          {Object.entries(aiData.ai_systems).slice(0, 10).map(([key, system]) => (
            <div
              key={key}
              className={`w-3 h-3 rounded-full ${
                aiData.system_status[key] ? 'bg-green-400' : 'bg-gray-600'
              }`}
              title={system.name}
            />
          ))}
        </div>
      </div>

      {/* Uptime */}
      <div className="text-center mb-4">
        <div className="text-sm text-gray-400">⏱️ Uptime: <span className="text-cyan-400">{aiData.uptime_formatted}</span></div>
      </div>

      {/* Navigation to AI Dashboard */}
      <Link 
        href="/ai"
        className="block w-full text-center py-2 px-3 bg-purple-600/50 hover:bg-purple-500/50 rounded-lg transition-colors text-purple-200 hover:text-white text-sm"
      >
        🚀 Open AI Dashboard
      </Link>
    </div>
  );
};

export default ZionAIWidget;
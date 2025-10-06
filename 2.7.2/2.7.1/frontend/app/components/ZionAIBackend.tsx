'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface AISystem {
  name: string;
  status: string;
  capabilities: string[];
  last_activity: number;
  metrics: Record<string, any>;
  activation_time?: string;
}

interface SystemStatus {
  [key: string]: boolean;
}

interface PerformanceMetrics {
  total_requests: number;
  successful_operations: number;
  failed_operations: number;
  average_response_time: number;
  uptime_start: number;
}

interface HardwareStatus {
  cpu_usage: string;
  memory_usage: string;
  memory_available: string;
  gpu_count: number;
  gpus: Array<{
    id: number;
    name: string;
    load: string;
    memory_used: string;
    memory_total: string;
    temperature: string;
  }>;
}

interface AIBackendData {
  ai_systems: Record<string, AISystem>;
  system_status: SystemStatus;
  performance_metrics: PerformanceMetrics;
  hardware_status: HardwareStatus;
  uptime_seconds: number;
  uptime_formatted: string;
  timestamp: string;
}

const ZionAIBackend: React.FC = () => {
  const [aiData, setAiData] = useState<AIBackendData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedSystem, setSelectedSystem] = useState<string | null>(null);
  const [taskResults, setTaskResults] = useState<Record<string, any>>({});
  const [activeTab, setActiveTab] = useState<'overview' | 'systems' | 'metrics' | 'hardware'>('overview');

  // Fetch AI backend data
  const fetchAIData = async () => {
    try {
      setError(null);
      const response = await fetch('http://localhost:8001/api/ai/overview');
      const data = await response.json();
      
      if (data.success) {
        setAiData(data.data);
      } else {
        setError('Failed to fetch AI data');
      }
    } catch (err) {
      console.error('AI Backend fetch error:', err);
      setError('Failed to connect to AI backend');
    } finally {
      setLoading(false);
    }
  };

  // Toggle AI system
  const toggleAISystem = async (systemName: string, currentStatus: boolean) => {
    try {
      const endpoint = currentStatus ? 'deactivate' : 'activate';
      const response = await fetch(`http://localhost:8001/api/ai/${endpoint}/${systemName}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config: { power_level: 'optimal' } })
      });
      
      const result = await response.json();
      if (result.success) {
        fetchAIData(); // Refresh data
      }
    } catch (err) {
      console.error('Toggle AI system error:', err);
    }
  };

  // Execute AI task
  const executeAITask = async (systemName: string, taskType: string) => {
    try {
      const response = await fetch(`http://localhost:8001/api/ai/execute/${systemName}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task_type: taskType,
          parameters: { priority: 'high', optimize: true }
        })
      });
      
      const result = await response.json();
      if (result.success) {
        setTaskResults(prev => ({
          ...prev,
          [systemName]: result.data
        }));
        fetchAIData(); // Refresh data
      }
    } catch (err) {
      console.error('Execute AI task error:', err);
    }
  };

  useEffect(() => {
    fetchAIData();
    const interval = setInterval(fetchAIData, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-green-400';
      case 'ready': return 'text-yellow-400';
      case 'error': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active': return '🟢';
      case 'ready': return '🟡';
      case 'error': return '🔴';
      default: return '⚪';
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 flex items-center justify-center">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
          className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full"
        />
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 flex items-center justify-center">
        <div className="bg-red-900/50 border border-red-500 rounded-lg p-6 text-center">
          <h2 className="text-2xl font-bold text-red-400 mb-4">❌ AI Backend Error</h2>
          <p className="text-red-300 mb-4">{error}</p>
          <button
            onClick={fetchAIData}
            className="bg-red-600 hover:bg-red-700 text-white px-6 py-2 rounded-lg transition-colors"
          >
            🔄 Retry Connection
          </button>
        </div>
      </div>
    );
  }

  if (!aiData) return null;

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 text-white">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <h1 className="text-5xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent mb-4">
            🤖 ZION AI Backend
          </h1>
          <p className="text-xl text-blue-300">
            Advanced AI System Management Dashboard
          </p>
          <div className="text-sm text-gray-400 mt-2">
            Uptime: {aiData.uptime_formatted} | Systems: {Object.keys(aiData.ai_systems).length}
          </div>
        </motion.div>

        {/* Tab Navigation */}
        <div className="flex justify-center mb-8">
          <div className="flex bg-gray-800/50 rounded-lg p-1">
            {(['overview', 'systems', 'metrics', 'hardware'] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-6 py-2 rounded-lg font-medium transition-all ${
                  activeTab === tab
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-400 hover:text-white hover:bg-gray-700'
                }`}
              >
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
              </button>
            ))}
          </div>
        </div>

        <AnimatePresence mode="wait">
          {/* Overview Tab */}
          {activeTab === 'overview' && (
            <motion.div
              key="overview"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
            >
              {/* System Status Cards */}
              {Object.entries(aiData.ai_systems).map(([systemKey, system]) => (
                <motion.div
                  key={systemKey}
                  whileHover={{ scale: 1.02 }}
                  className="bg-gray-800/50 border border-gray-700 rounded-xl p-6 backdrop-blur-sm"
                >
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-xl font-bold text-blue-400">
                      {getStatusIcon(system.status)} {system.name}
                    </h3>
                    <span className={`text-sm font-medium ${getStatusColor(system.status)}`}>
                      {system.status}
                    </span>
                  </div>
                  
                  <div className="space-y-2 text-sm">
                    <div className="text-gray-300">
                      Capabilities: {system.capabilities.length}
                    </div>
                    <div className="text-gray-300">
                      Operations: {system.metrics.operations_count || 0}
                    </div>
                  </div>
                  
                  <div className="mt-4 flex gap-2">
                    <button
                      onClick={() => toggleAISystem(systemKey, aiData.system_status[systemKey])}
                      className={`flex-1 py-2 px-4 rounded-lg font-medium transition-colors ${
                        aiData.system_status[systemKey]
                          ? 'bg-red-600 hover:bg-red-700 text-white'
                          : 'bg-green-600 hover:bg-green-700 text-white'
                      }`}
                    >
                      {aiData.system_status[systemKey] ? '🔻 Deactivate' : '🚀 Activate'}
                    </button>
                    
                    {aiData.system_status[systemKey] && (
                      <button
                        onClick={() => executeAITask(systemKey, 'optimize')}
                        className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors"
                      >
                        ⚡ Task
                      </button>
                    )}
                  </div>
                </motion.div>
              ))}
            </motion.div>
          )}

          {/* Systems Tab */}
          {activeTab === 'systems' && (
            <motion.div
              key="systems"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              {Object.entries(aiData.ai_systems).map(([systemKey, system]) => (
                <motion.div
                  key={systemKey}
                  whileHover={{ scale: 1.01 }}
                  className="bg-gray-800/50 border border-gray-700 rounded-xl p-6 backdrop-blur-sm"
                >
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-2xl font-bold text-blue-400">
                      {getStatusIcon(system.status)} {system.name}
                    </h3>
                    <div className="flex items-center gap-4">
                      <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusColor(system.status)}`}>
                        {system.status}
                      </span>
                      <button
                        onClick={() => setSelectedSystem(selectedSystem === systemKey ? null : systemKey)}
                        className="text-blue-400 hover:text-blue-300"
                      >
                        {selectedSystem === systemKey ? '🔼' : '🔽'}
                      </button>
                    </div>
                  </div>

                  {/* Capabilities */}
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-2 mb-4">
                    {system.capabilities.map((capability, index) => (
                      <span
                        key={index}
                        className="px-3 py-1 bg-blue-900/50 border border-blue-700 rounded-lg text-sm text-blue-300 text-center"
                      >
                        {capability.replace(/_/g, ' ')}
                      </span>
                    ))}
                  </div>

                  {/* Expanded Details */}
                  <AnimatePresence>
                    {selectedSystem === systemKey && (
                      <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        className="mt-4 p-4 bg-gray-900/50 rounded-lg"
                      >
                        <h4 className="text-lg font-bold text-green-400 mb-3">📊 System Metrics</h4>
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                          {Object.entries(system.metrics).map(([key, value]) => (
                            <div key={key} className="bg-gray-800/50 p-3 rounded-lg">
                              <div className="text-gray-400 capitalize">
                                {key.replace(/_/g, ' ')}
                              </div>
                              <div className="text-white font-medium">
                                {typeof value === 'number' ? 
                                  (value < 1 ? `${(value * 100).toFixed(1)}%` : value.toLocaleString()) : 
                                  value
                                }
                              </div>
                            </div>
                          ))}
                        </div>

                        {taskResults[systemKey] && (
                          <div className="mt-4">
                            <h5 className="text-lg font-bold text-purple-400 mb-2">⚡ Last Task Result</h5>
                            <div className="bg-purple-900/30 border border-purple-700 rounded-lg p-3">
                              <pre className="text-sm text-purple-300 whitespace-pre-wrap">
                                {JSON.stringify(taskResults[systemKey].result, null, 2)}
                              </pre>
                              <div className="text-xs text-purple-400 mt-2">
                                Execution time: {taskResults[systemKey].execution_time_ms}ms
                              </div>
                            </div>
                          </div>
                        )}
                      </motion.div>
                    )}
                  </AnimatePresence>
                </motion.div>
              ))}
            </motion.div>
          )}

          {/* Metrics Tab */}
          {activeTab === 'metrics' && (
            <motion.div
              key="metrics"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
            >
              {/* Performance Metrics */}
              <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-6 backdrop-blur-sm">
                <h3 className="text-xl font-bold text-green-400 mb-4">📈 Performance</h3>
                <div className="space-y-3">
                  <div>
                    <div className="text-gray-400 text-sm">Total Requests</div>
                    <div className="text-2xl font-bold text-white">
                      {aiData.performance_metrics.total_requests.toLocaleString()}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-400 text-sm">Success Rate</div>
                    <div className="text-2xl font-bold text-green-400">
                      {(
                        (aiData.performance_metrics.successful_operations / 
                        (aiData.performance_metrics.successful_operations + aiData.performance_metrics.failed_operations) * 100) || 100
                      ).toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-400 text-sm">Avg Response Time</div>
                    <div className="text-2xl font-bold text-blue-400">
                      {(aiData.performance_metrics.average_response_time * 1000).toFixed(1)}ms
                    </div>
                  </div>
                </div>
              </div>

              {/* System Statistics */}
              <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-6 backdrop-blur-sm">
                <h3 className="text-xl font-bold text-purple-400 mb-4">🎯 Statistics</h3>
                <div className="space-y-3">
                  <div>
                    <div className="text-gray-400 text-sm">Active Systems</div>
                    <div className="text-2xl font-bold text-white">
                      {Object.values(aiData.system_status).filter(Boolean).length} / {Object.keys(aiData.system_status).length}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-400 text-sm">Successful Operations</div>
                    <div className="text-2xl font-bold text-green-400">
                      {aiData.performance_metrics.successful_operations.toLocaleString()}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-400 text-sm">Failed Operations</div>
                    <div className="text-2xl font-bold text-red-400">
                      {aiData.performance_metrics.failed_operations.toLocaleString()}
                    </div>
                  </div>
                </div>
              </div>

              {/* Uptime Info */}
              <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-6 backdrop-blur-sm">
                <h3 className="text-xl font-bold text-yellow-400 mb-4">⏱️ Uptime</h3>
                <div className="space-y-3">
                  <div>
                    <div className="text-gray-400 text-sm">System Uptime</div>
                    <div className="text-xl font-bold text-white">
                      {aiData.uptime_formatted}
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-400 text-sm">Last Updated</div>
                    <div className="text-sm text-gray-300">
                      {new Date(aiData.timestamp).toLocaleString()}
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {/* Hardware Tab */}
          {activeTab === 'hardware' && (
            <motion.div
              key="hardware"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="grid grid-cols-1 md:grid-cols-2 gap-6"
            >
              {/* System Resources */}
              <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-6 backdrop-blur-sm">
                <h3 className="text-xl font-bold text-orange-400 mb-4">💻 System Resources</h3>
                <div className="space-y-4">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-400">CPU Usage</span>
                      <span className="text-white">{aiData.hardware_status.cpu_usage}</span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2">
                      <div 
                        className="bg-orange-500 h-2 rounded-full transition-all"
                        style={{ width: aiData.hardware_status.cpu_usage }}
                      />
                    </div>
                  </div>
                  
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-400">Memory Usage</span>
                      <span className="text-white">{aiData.hardware_status.memory_usage}</span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2">
                      <div 
                        className="bg-blue-500 h-2 rounded-full transition-all"
                        style={{ width: aiData.hardware_status.memory_usage }}
                      />
                    </div>
                  </div>
                  
                  <div className="text-sm text-gray-400">
                    Available Memory: {aiData.hardware_status.memory_available}
                  </div>
                </div>
              </div>

              {/* GPU Information */}
              <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-6 backdrop-blur-sm">
                <h3 className="text-xl font-bold text-green-400 mb-4">
                  🎮 GPU Status ({aiData.hardware_status.gpu_count})
                </h3>
                {aiData.hardware_status.gpus.length > 0 ? (
                  <div className="space-y-4">
                    {aiData.hardware_status.gpus.map((gpu) => (
                      <div key={gpu.id} className="bg-gray-900/50 rounded-lg p-4">
                        <div className="text-sm font-medium text-white mb-2">
                          GPU {gpu.id}: {gpu.name}
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-xs text-gray-400">
                          <div>Load: {gpu.load}</div>
                          <div>Temp: {gpu.temperature}</div>
                          <div>Memory: {gpu.memory_used}/{gpu.memory_total}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-gray-400 text-center py-8">
                    No GPU devices detected
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Refresh Button */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="fixed bottom-6 right-6"
        >
          <button
            onClick={fetchAIData}
            className="bg-blue-600 hover:bg-blue-700 text-white p-4 rounded-full shadow-lg transition-colors"
          >
            🔄
          </button>
        </motion.div>
      </div>
    </div>
  );
};

export default ZionAIBackend;
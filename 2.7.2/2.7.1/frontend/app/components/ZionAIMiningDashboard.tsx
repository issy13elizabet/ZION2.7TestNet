'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

interface AIMinerSystem {
  name: string;
  status: string;
  capabilities: string[];
  metrics: {
    operations_count: number;
    optimization_level?: number;
    performance_gain?: string;
    energy_efficiency?: number;
    [key: string]: any;
  };
  last_activity: number;
}

interface AIBackendData {
  ai_systems: Record<string, AIMinerSystem>;
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

const ZionAIMiningDashboard: React.FC = () => {
  const [aiData, setAiData] = useState<AIBackendData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedMiner, setSelectedMiner] = useState<string>('afterburner');
  const [taskResults, setTaskResults] = useState<Record<string, any>>({});

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
        setError('AI Backend nedostupný');
      }
    } catch (err) {
      console.error('AI Mining fetch error:', err);
      setError('Chyba připojení k AI');
    } finally {
      setLoading(false);
    }
  };

  const activateAIMiner = async (minerName: string) => {
    try {
      const response = await fetch(`http://localhost:8001/api/ai/activate/${minerName}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ power_level: 'maximum', mining_mode: true })
      });
      
      const result = await response.json();
      if (result.success) {
        fetchAIData(); // Refresh data
      }
    } catch (err) {
      console.error('AI Miner activation error:', err);
    }
  };

  const executeAITask = async (minerName: string, taskType: string) => {
    try {
      const response = await fetch(`http://localhost:8001/api/ai/execute/${minerName}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task_type: taskType,
          parameters: { 
            mining_target: 'maximum_efficiency',
            gpu_optimization: true,
            power_management: true
          }
        })
      });
      
      const result = await response.json();
      if (result.success) {
        setTaskResults(prev => ({
          ...prev,
          [minerName]: result.data
        }));
        fetchAIData(); // Refresh data
      }
    } catch (err) {
      console.error('AI Task execution error:', err);
    }
  };

  useEffect(() => {
    fetchAIData();
    const interval = setInterval(fetchAIData, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const getMiningAISystems = () => {
    if (!aiData) return [];
    return Object.entries(aiData.ai_systems).filter(([key, system]) => 
      system.capabilities.includes('mining_optimization') || 
      key === 'afterburner' || 
      key === 'lightning_ai' ||
      key === 'quantum_ai'
    );
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
          <h2 className="text-2xl font-bold text-red-400 mb-4">❌ AI Mining Error</h2>
          <p className="text-red-300 mb-4">{error}</p>
          <button
            onClick={fetchAIData}
            className="bg-red-600 hover:bg-red-700 text-white px-6 py-2 rounded-lg transition-colors"
          >
            🔄 Reconnect to AI
          </button>
        </div>
      </div>
    );
  }

  const miningAISystems = getMiningAISystems();

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
            ⛏️ ZION AI Mining Center
          </h1>
          <p className="text-xl text-blue-300">
            Advanced AI-Powered Mining with Afterburner Technology
          </p>
          <div className="text-sm text-gray-400 mt-2">
            Uptime: {aiData?.uptime_formatted} | Active AI Miners: {miningAISystems.filter(([key]) => aiData?.system_status[key]).length}
          </div>
        </motion.div>

        {/* AI Miner Selection */}
        <div className="mb-8">
          <div className="flex justify-center">
            <div className="flex bg-gray-800/50 rounded-lg p-1">
              {miningAISystems.map(([key, system]) => (
                <button
                  key={key}
                  onClick={() => setSelectedMiner(key)}
                  className={`px-6 py-2 rounded-lg font-medium transition-all ${
                    selectedMiner === key
                      ? 'bg-blue-600 text-white'
                      : 'text-gray-400 hover:text-white hover:bg-gray-700'
                  }`}
                >
                  {key === 'afterburner' ? '🚀 Afterburner' : 
                   key === 'lightning_ai' ? '⚡ Lightning AI' :
                   key === 'quantum_ai' ? '🔮 Quantum AI' : system.name}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Main Mining Dashboard */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* AI Miner Control Panel */}
          <div className="lg:col-span-2">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-gray-800/50 border border-gray-700 rounded-xl p-6 backdrop-blur-sm"
            >
              {miningAISystems.find(([key]) => key === selectedMiner) && (
                <>
                  {(() => {
                    const [key, system] = miningAISystems.find(([k]) => k === selectedMiner)!;
                    const isActive = aiData?.system_status[key];
                    
                    return (
                      <>
                        <div className="flex items-center justify-between mb-6">
                          <h2 className="text-2xl font-bold text-blue-400">
                            {key === 'afterburner' ? '🚀 AI Afterburner' : 
                             key === 'lightning_ai' ? '⚡ Lightning AI Miner' :
                             key === 'quantum_ai' ? '🔮 Quantum AI Miner' : system.name}
                          </h2>
                          <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                            isActive ? 'bg-green-900/50 text-green-400' : 'bg-gray-900/50 text-gray-400'
                          }`}>
                            {isActive ? '🟢 ACTIVE' : '⚪ STANDBY'}
                          </div>
                        </div>

                        {/* Capabilities */}
                        <div className="mb-6">
                          <h3 className="text-lg font-bold text-purple-400 mb-3">🛠️ Capabilities</h3>
                          <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                            {system.capabilities.map((capability, index) => (
                              <span
                                key={index}
                                className="px-3 py-2 bg-purple-900/30 border border-purple-700 rounded-lg text-sm text-purple-300 text-center"
                              >
                                {capability.replace(/_/g, ' ')}
                              </span>
                            ))}
                          </div>
                        </div>

                        {/* Performance Metrics */}
                        <div className="mb-6">
                          <h3 className="text-lg font-bold text-green-400 mb-3">📊 Performance Metrics</h3>
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div className="bg-gray-900/50 p-4 rounded-lg text-center">
                              <div className="text-2xl font-bold text-white">
                                {system.metrics.operations_count || 0}
                              </div>
                              <div className="text-sm text-gray-400">Operations</div>
                            </div>
                            
                            {system.metrics.optimization_level && (
                              <div className="bg-gray-900/50 p-4 rounded-lg text-center">
                                <div className="text-2xl font-bold text-blue-400">
                                  {(system.metrics.optimization_level * 100).toFixed(1)}%
                                </div>
                                <div className="text-sm text-gray-400">Optimization</div>
                              </div>
                            )}
                            
                            {system.metrics.energy_efficiency && (
                              <div className="bg-gray-900/50 p-4 rounded-lg text-center">
                                <div className="text-2xl font-bold text-green-400">
                                  {(system.metrics.energy_efficiency * 100).toFixed(1)}%
                                </div>
                                <div className="text-sm text-gray-400">Efficiency</div>
                              </div>
                            )}
                            
                            <div className="bg-gray-900/50 p-4 rounded-lg text-center">
                              <div className="text-2xl font-bold text-purple-400">
                                {aiData?.performance_metrics?.average_response_time ? 
                                  `${(aiData.performance_metrics.average_response_time * 1000).toFixed(0)}ms` : 
                                  '0ms'
                                }
                              </div>
                              <div className="text-sm text-gray-400">Response Time</div>
                            </div>
                          </div>
                        </div>

                        {/* Control Buttons */}
                        <div className="flex gap-4 mb-6">
                          <button
                            onClick={() => activateAIMiner(key)}
                            className={`flex-1 py-3 px-6 rounded-lg font-medium transition-colors ${
                              isActive
                                ? 'bg-orange-600 hover:bg-orange-700 text-white'
                                : 'bg-green-600 hover:bg-green-700 text-white'
                            }`}
                          >
                            {isActive ? '🔻 Deactivate Miner' : '🚀 Activate Miner'}
                          </button>
                          
                          {isActive && (
                            <button
                              onClick={() => executeAITask(key, 'optimize_mining')}
                              className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors"
                            >
                              ⚡ Optimize Mining
                            </button>
                          )}
                        </div>

                        {/* Task Results */}
                        {taskResults[key] && (
                          <div className="bg-blue-900/30 border border-blue-700 rounded-lg p-4">
                            <h4 className="text-lg font-bold text-blue-400 mb-3">⚡ Last Optimization Result</h4>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                              {Object.entries(taskResults[key].result || {}).map(([resultKey, value]) => (
                                <div key={resultKey} className="bg-blue-900/50 p-3 rounded">
                                  <div className="text-blue-300 capitalize">
                                    {resultKey.replace(/_/g, ' ')}
                                  </div>
                                  <div className="text-white font-medium">
                                    {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                                  </div>
                                </div>
                              ))}
                            </div>
                            <div className="text-xs text-blue-400 mt-2">
                              Execution time: {taskResults[key].execution_time_ms}ms
                            </div>
                          </div>
                        )}
                      </>
                    );
                  })()}
                </>
              )}
            </motion.div>
          </div>

          {/* Hardware Status & Stats */}
          <div className="space-y-6">
            {/* Hardware Monitor */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="bg-gray-800/50 border border-gray-700 rounded-xl p-6 backdrop-blur-sm"
            >
              <h3 className="text-xl font-bold text-orange-400 mb-4">💻 Hardware Status</h3>
              
              {aiData?.hardware_status && (
                <>
                  <div className="space-y-3 mb-4">
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
                  </div>

                  <div className="text-sm text-gray-400 mb-4">
                    🎮 GPUs Detected: {aiData.hardware_status.gpu_count}
                  </div>

                  {aiData.hardware_status.gpus.length > 0 && (
                    <div className="space-y-2">
                      {aiData.hardware_status.gpus.map((gpu) => (
                        <div key={gpu.id} className="bg-gray-900/50 rounded-lg p-3">
                          <div className="text-sm font-medium text-white mb-1">
                            GPU {gpu.id}: {gpu.name}
                          </div>
                          <div className="grid grid-cols-2 gap-2 text-xs text-gray-400">
                            <div>Load: {gpu.load}</div>
                            <div>Temp: {gpu.temperature}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </>
              )}
            </motion.div>

            {/* Overall AI Mining Stats */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="bg-gray-800/50 border border-gray-700 rounded-xl p-6 backdrop-blur-sm"
            >
              <h3 className="text-xl font-bold text-cyan-400 mb-4">📈 AI Mining Statistics</h3>
              
              {aiData && (
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Total AI Requests</span>
                    <span className="text-cyan-400">{aiData.performance_metrics.total_requests}</span>
                  </div>
                  
                  <div className="flex justify-between">
                    <span className="text-gray-400">Success Rate</span>
                    <span className="text-green-400">
                      {(
                        (aiData.performance_metrics.successful_operations / 
                        (aiData.performance_metrics.successful_operations + aiData.performance_metrics.failed_operations) * 100) || 100
                      ).toFixed(1)}%
                    </span>
                  </div>
                  
                  <div className="flex justify-between">
                    <span className="text-gray-400">Active Mining AI</span>
                    <span className="text-purple-400">
                      {miningAISystems.filter(([key]) => aiData.system_status[key]).length}/{miningAISystems.length}
                    </span>
                  </div>
                  
                  <div className="flex justify-between">
                    <span className="text-gray-400">System Uptime</span>
                    <span className="text-yellow-400">{aiData.uptime_formatted}</span>
                  </div>
                </div>
              )}
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ZionAIMiningDashboard;
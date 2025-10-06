"use client";
import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

interface SystemStatus {
  height: any;
  balance: any;
  blocks: any;
  aiBackend: any;
  error?: string;
}

export default function WalletStatusPage() {
  const [status, setStatus] = useState<SystemStatus>({
    height: null,
    balance: null,
    blocks: { tip: 0, count: 0, blocks: [] },
    aiBackend: null
  });
  const [loading, setLoading] = useState(true);

  const loadSystemStatus = async () => {
    try {
      setLoading(true);
      
      // Load traditional wallet/blockchain data
      const [h, b, bl] = await Promise.all([
        fetch('/api/chain/height').then(r => r.json()).catch(() => ({ height: 0, error: 'Chain API unavailable' })),
        fetch('/api/wallet/balance').then(r => r.json()).catch(() => ({ balance: 0, error: 'Wallet API unavailable' })),
        fetch('/api/pool/blocks-recent?n=8').then(r => r.json()).catch(() => ({ tip: 0, count: 0, blocks: [] })),
      ]);

      // Load AI Backend status
      let aiStatus = null;
      try {
        const aiResponse = await fetch('http://localhost:8001/api/ai/overview', {
          method: 'GET',
          cache: 'no-store',
          signal: AbortSignal.timeout(3000)
        });
        if (aiResponse.ok) {
          const aiData = await aiResponse.json();
          aiStatus = aiData.success ? aiData.data : { error: 'AI Backend not responding' };
        }
      } catch (aiError) {
        aiStatus = { error: 'AI Backend connection failed' };
      }

      setStatus({
        height: h,
        balance: b,
        blocks: bl,
        aiBackend: aiStatus,
        error: undefined
      });
      
    } catch (e: any) {
      setStatus(prev => ({
        ...prev,
        error: String(e?.message || e)
      }));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { 
    loadSystemStatus(); 
    const id = setInterval(loadSystemStatus, 5000); 
    return () => clearInterval(id); 
  }, []);

  const getStatusColor = (hasError: boolean, hasData: boolean) => {
    if (hasError) return 'border-red-500 bg-red-900/20';
    if (hasData) return 'border-green-500 bg-green-900/20';
    return 'border-yellow-500 bg-yellow-900/20';
  };

  const getActiveAISystems = () => {
    if (!status.aiBackend?.system_status) return 0;
    return Object.values(status.aiBackend.system_status).filter(Boolean).length;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900 text-white">
      <div className="container mx-auto px-4 py-8 max-w-6xl">
        
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent mb-4">
            📊 ZION System Status
          </h1>
          <p className="text-xl text-blue-300">
            Complete system health monitoring with AI integration
          </p>
          <div className="text-sm text-gray-400 mt-2">
            {loading ? '🔄 Refreshing...' : '✅ Last updated: ' + new Date().toLocaleTimeString()}
          </div>
        </motion.div>

        {/* Global Error */}
        {status.error && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-red-900/50 border border-red-500 rounded-lg p-4 mb-6 text-center"
          >
            <div className="text-red-400 font-bold">❌ System Error</div>
            <div className="text-red-300">{status.error}</div>
          </motion.div>
        )}

        {/* Status Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          
          {/* Blockchain Status */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className={`border rounded-xl p-6 backdrop-blur-sm ${getStatusColor(
              status.height?.error, 
              status.height?.height > 0
            )}`}
          >
            <h3 className="text-xl font-bold mb-4 flex items-center">
              ⛓️ Blockchain Status
            </h3>
            
            {status.height?.error ? (
              <div className="text-red-300">❌ {status.height.error}</div>
            ) : (
              <div className="space-y-3">
                <div className="text-center">
                  <div className="text-3xl font-bold text-white mb-1">
                    {status.height?.height || 0}
                  </div>
                  <div className="text-sm text-gray-400">Current Height</div>
                </div>
                
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div className="bg-black/30 p-2 rounded">
                    <div className="text-gray-400">Status</div>
                    <div className="text-green-400">🟢 Active</div>
                  </div>
                  <div className="bg-black/30 p-2 rounded">
                    <div className="text-gray-400">Network</div>
                    <div className="text-blue-400">ZION 2.7.1</div>
                  </div>
                </div>
              </div>
            )}
          </motion.div>

          {/* Wallet Status */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className={`border rounded-xl p-6 backdrop-blur-sm ${getStatusColor(
              status.balance?.error, 
              status.balance?.balance !== undefined
            )}`}
          >
            <h3 className="text-xl font-bold mb-4 flex items-center">
              💎 Wallet Status
            </h3>
            
            {status.balance?.error ? (
              <div className="text-red-300">❌ {status.balance.error}</div>
            ) : (
              <div className="space-y-3">
                <div className="text-center">
                  <div className="text-3xl font-bold text-white mb-1">
                    {typeof status.balance?.balance === 'number' 
                      ? (status.balance.balance / 1e8).toFixed(4)
                      : '0.0000'
                    }
                  </div>
                  <div className="text-sm text-gray-400">ZION Balance</div>
                </div>
                
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div className="bg-black/30 p-2 rounded">
                    <div className="text-gray-400">Status</div>
                    <div className="text-green-400">🟢 Online</div>
                  </div>
                  <div className="bg-black/30 p-2 rounded">
                    <div className="text-gray-400">Type</div>
                    <div className="text-purple-400">HD Wallet</div>
                  </div>
                </div>
              </div>
            )}
          </motion.div>

          {/* AI Backend Status */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className={`border rounded-xl p-6 backdrop-blur-sm ${getStatusColor(
              status.aiBackend?.error, 
              status.aiBackend?.ai_systems
            )}`}
          >
            <h3 className="text-xl font-bold mb-4 flex items-center">
              🤖 AI Backend Status
            </h3>
            
            {status.aiBackend?.error ? (
              <div className="text-red-300">❌ {status.aiBackend.error}</div>
            ) : status.aiBackend ? (
              <div className="space-y-3">
                <div className="text-center">
                  <div className="text-3xl font-bold text-white mb-1">
                    {getActiveAISystems()}/10
                  </div>
                  <div className="text-sm text-gray-400">Active AI Systems</div>
                </div>
                
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div className="bg-black/30 p-2 rounded">
                    <div className="text-gray-400">Uptime</div>
                    <div className="text-cyan-400">{status.aiBackend.uptime_formatted}</div>
                  </div>
                  <div className="bg-black/30 p-2 rounded">
                    <div className="text-gray-400">Success Rate</div>
                    <div className="text-green-400">
                      {status.aiBackend.performance_metrics ? 
                        ((status.aiBackend.performance_metrics.successful_operations / 
                        (status.aiBackend.performance_metrics.successful_operations + status.aiBackend.performance_metrics.failed_operations) * 100) || 100).toFixed(1)
                        : '100'
                      }%
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-yellow-300">⚡ Connecting to AI Backend...</div>
            )}
          </motion.div>
        </div>

        {/* Recent Blocks */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="bg-gray-800/50 border border-gray-700 rounded-xl p-6 backdrop-blur-sm mb-8"
        >
          <h3 className="text-xl font-bold mb-4 text-purple-400">
            📦 Recent Blocks
          </h3>
          
          {status.blocks?.blocks?.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {status.blocks.blocks.slice(0, 8).map((block: any, index: number) => (
                <div key={block.height || index} className="bg-black/30 rounded-lg p-3">
                  <div className="text-lg font-bold text-white">#{block.height || 0}</div>
                  <div className="text-sm text-gray-400">
                    {block.header?.timestamp ? 
                      new Date(block.header.timestamp * 1000).toLocaleTimeString() : 
                      'Unknown time'
                    }
                  </div>
                  {block.error && <div className="text-red-400 text-xs">Error: {block.error}</div>}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center text-gray-400 py-8">
              📭 No recent blocks available
            </div>
          )}
        </motion.div>

        {/* AI Systems Detail */}
        {status.aiBackend?.ai_systems && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="bg-gray-800/50 border border-gray-700 rounded-xl p-6 backdrop-blur-sm mb-8"
          >
            <h3 className="text-xl font-bold mb-4 text-cyan-400">
              🔧 AI Systems Detail
            </h3>
            
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              {Object.entries(status.aiBackend.ai_systems).map(([key, system]: [string, any]) => (
                <div key={key} className="bg-black/30 rounded-lg p-3 text-center">
                  <div className={`w-3 h-3 rounded-full mx-auto mb-2 ${
                    status.aiBackend.system_status[key] ? 'bg-green-400' : 'bg-gray-600'
                  }`}></div>
                  <div className="text-sm font-medium text-white truncate" title={system.name}>
                    {system.name.replace('ZION ', '').substring(0, 12)}
                  </div>
                  <div className="text-xs text-gray-400">
                    {system.metrics?.operations_count || 0} ops
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        )}

        {/* Notes */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="bg-blue-900/20 border border-blue-700/30 rounded-xl p-6 backdrop-blur-sm"
        >
          <h3 className="text-lg font-bold text-blue-400 mb-3">📝 System Notes</h3>
          <div className="space-y-2 text-sm text-blue-200">
            <div>• Výplaty se spustí až po coinbase maturitě na chainu (~60 bloků)</div>
            <div>• AI Backend poskytuje optimalizaci výkonu a bezpečnostní monitoring</div>
            <div>• Status se automaticky obnovuje každých 5 sekund</div>
            <div>• Pro detailní AI metriky navštivte AI Dashboard</div>
          </div>
        </motion.div>

      </div>
    </div>
  );
}

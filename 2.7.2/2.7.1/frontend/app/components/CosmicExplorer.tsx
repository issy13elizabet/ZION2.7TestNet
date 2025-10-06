"use client";

import { motion } from "framer-motion";
import { useState, useEffect } from "react";
import { useCosmicSounds } from "../hooks/useCosmicSounds";
import { 
  CosmicParticles, 
  FloatingElement, 
  PulsingGlow, 
  TypewriterText, 
  CosmicButton 
} from "./CosmicAnimations";

interface CosmicBlock {
  height: number;
  hash: string;
  timestamp: number;
  transactions: number;
  miner: string;
  difficulty: number;
  reward: number;
}

interface CosmicTransaction {
  hash: string;
  from: string;
  to: string;
  amount: number;
  fee: number;
  timestamp: number;
  cosmic_energy: number;
}

const cosmicMantras = [
  "🕉️ Jai Shree Ram! Cosmic blocks flow through eternal dharma! 🕉️",
  "⚡ Divine synchronicity guides every transaction hash! ⚡",
  "🌟 In the cosmic ledger, all karmic debts are recorded! 🌟",
  "🔮 Blockchain of consciousness expands through infinite dimensions! 🔮",
  "🌌 From stardust to satoshis, the universe computes! 🌌"
];

export default function CosmicExplorer() {
  const [blocks, setBlocks] = useState<CosmicBlock[]>([]);
  const [transactions, setTransactions] = useState<CosmicTransaction[]>([]);
  const [currentMantra, setCurrentMantra] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const [cosmicEnergy, setCosmicEnergy] = useState(108);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<any>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [soundEnabled, setSoundEnabled] = useState(false);

  // Cosmic Sound Effects
  const sounds = useCosmicSounds();

  // Search functionality
  const handleSearch = async (query: string) => {
    if (!query.trim()) {
      setSearchResults(null);
      return;
    }

    if (soundEnabled) {
      sounds.playSearch();
    }

    setIsSearching(true);
    try {
      const response = await fetch(`/api/search?q=${encodeURIComponent(query)}&limit=10`);
      const data = await response.json();
      
      if (data.success) {
        setSearchResults(data.data);
        if (soundEnabled && data.data.totalResults > 0) {
          sounds.playSuccess();
        }
      }
    } catch (error) {
      console.error('Search failed:', error);
    } finally {
      setIsSearching(false);
    }
  };

  // Simulate cosmic blockchain data
  useEffect(() => {
    const generateCosmicBlocks = () => {
      const newBlocks: CosmicBlock[] = Array.from({ length: 10 }, (_, i) => ({
        height: 144000 + i,
        hash: `0x${Math.random().toString(16).substr(2, 64)}`,
        timestamp: Date.now() - (i * 120000), // 2 min intervals
        transactions: Math.floor(Math.random() * 50) + 1,
        miner: `cosmic_validator_${Math.floor(Math.random() * 108) + 1}`,
        difficulty: Math.floor(Math.random() * 1000000) + 500000,
        reward: 50.0 + Math.random() * 10
      }));
      setBlocks(newBlocks);
      
      // Play block found sound when new blocks are generated
      if (soundEnabled && !isLoading) {
        sounds.playBlockMined();
      }

      const newTransactions: CosmicTransaction[] = Array.from({ length: 20 }, (_, i) => ({
        hash: `tx_${Math.random().toString(16).substr(2, 40)}`,
        from: `Z3${Math.random().toString(36).substr(2, 20)}`,
        to: `Z3${Math.random().toString(36).substr(2, 20)}`,
        amount: Math.random() * 1000,
        fee: Math.random() * 0.01,
        timestamp: Date.now() - (i * 30000),
        cosmic_energy: Math.floor(Math.random() * 108) + 1
      }));
      setTransactions(newTransactions);
      setIsLoading(false);
    };

    generateCosmicBlocks();
    const interval = setInterval(generateCosmicBlocks, 30000); // Update every 30s

    return () => clearInterval(interval);
  }, []);

  // Rotate cosmic mantras
  useEffect(() => {
    const mantraInterval = setInterval(() => {
      setCurrentMantra((prev) => (prev + 1) % cosmicMantras.length);
      setCosmicEnergy(prev => (prev % 108) + 1); // Sacred 108 cycle
    }, 5000);

    return () => clearInterval(mantraInterval);
  }, []);

  const formatHash = (hash: string) => `${hash.slice(0, 6)}...${hash.slice(-6)}`;
  const formatTime = (timestamp: number) => new Date(timestamp).toLocaleTimeString();

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-indigo-900">
      {/* Multi-layer Cosmic Particles */}
      <CosmicParticles color="cyan" size={2} speed={20} count={30} />
      <CosmicParticles color="purple" size={3} speed={15} count={20} />
      <CosmicParticles color="gold" size={1} speed={25} count={40} />
      <CosmicParticles color="pink" size={2} speed={18} count={25} />

      <div className="relative z-10 container mx-auto px-6 py-8">
        {/* Cosmic Header */}
        <motion.header
          className="text-center mb-12"
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, type: "spring" }}
        >
          <FloatingElement amplitude={15} duration={6}>
            <PulsingGlow color="gold" intensity={2}>
              <motion.h1 
                className="text-7xl font-bold bg-gradient-to-r from-yellow-400 via-pink-400 to-purple-400 bg-clip-text text-transparent mb-6"
                animate={{ 
                  backgroundPosition: ["0%", "100%", "0%"],
                  scale: [1, 1.02, 1]
                }}
                transition={{ duration: 4, repeat: Infinity }}
              >
                🌌 ZION COSMIC EXPLORER 🌌
              </motion.h1>
            </PulsingGlow>
          </FloatingElement>
          
          <FloatingElement amplitude={8} duration={4}>
            <motion.div
              className="text-xl text-cyan-300 mb-4 font-mono h-16 flex items-center justify-center"
              animate={{ opacity: [0.7, 1, 0.7] }}
              transition={{ duration: 3, repeat: Infinity }}
            >
              <TypewriterText 
                text={cosmicMantras[currentMantra]}
                speed={30}
                className="text-center"
              />
            </motion.div>
          </FloatingElement>

          {/* Cosmic Search Bar */}
          <motion.div
            className="max-w-2xl mx-auto mb-8"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <div className="relative">
              <input
                type="text"
                placeholder="🔍 Search hash, address, block height, or transaction..."
                value={searchQuery}
                onChange={(e) => {
                  setSearchQuery(e.target.value);
                  if (e.target.value.length > 3) {
                    handleSearch(e.target.value);
                  } else {
                    setSearchResults(null);
                  }
                }}
                className="w-full px-6 py-4 bg-black/40 backdrop-blur-md border border-purple-500/50 rounded-xl text-white placeholder-gray-400 focus:border-cyan-400 focus:outline-none focus:ring-2 focus:ring-cyan-400/50 transition-all"
              />
              {isSearching && (
                <motion.div
                  className="absolute right-4 top-1/2 transform -translate-y-1/2"
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                >
                  <span className="text-cyan-400">🌀</span>
                </motion.div>
              )}
            </div>
          </motion.div>

          <div className="flex justify-center items-center space-x-8 text-lg mb-8">
            <div className="flex items-center space-x-2">
              <span className="text-green-400">⚡ Cosmic Energy:</span>
              <motion.span 
                className="text-yellow-300 font-bold"
                animate={{ scale: [1, 1.1, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                {cosmicEnergy}/108
              </motion.span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-blue-400">🔮 Network:</span>
              <span className="text-purple-300">ZION TestNet v2.6</span>
            </div>
            <CosmicButton
              variant={soundEnabled ? "cosmic" : "secondary"}
              onClick={() => {
                setSoundEnabled(!soundEnabled);
                if (!soundEnabled) {
                  sounds.playAmbient();
                  sounds.playSuccess();
                } else {
                  sounds.stopAmbient();
                }
              }}
              className="text-sm flex items-center space-x-2"
            >
              <span>{soundEnabled ? '🔊' : '🔇'}</span>
              <span>Cosmic Audio</span>
            </CosmicButton>
          </div>
        </motion.header>

        {/* Loading Animation */}
        {isLoading && (
          <motion.div 
            className="text-center py-20"
            animate={{ rotate: 360 }}
            transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
          >
            <div className="text-6xl">🌀</div>
            <p className="text-purple-300 mt-4">Channeling cosmic data streams...</p>
          </motion.div>
        )}

        {/* Search Results */}
        {searchResults && (
          <motion.div
            className="mb-8 bg-black/40 backdrop-blur-md rounded-xl p-6 border border-yellow-500/30"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
          >
            <h2 className="text-2xl font-bold text-yellow-300 mb-4">
              🔍 Search Results for "{searchQuery}"
            </h2>
            
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Blocks Results */}
              {searchResults.blocks?.length > 0 && (
                <div>
                  <h3 className="text-lg font-semibold text-cyan-300 mb-3">🧊 Blocks ({searchResults.blocks.length})</h3>
                  {searchResults.blocks.map((block: any, index: number) => (
                    <motion.div
                      key={block.height}
                      className="bg-cyan-900/20 rounded-lg p-3 mb-2 border border-cyan-500/20"
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.05 }}
                    >
                      <div className="text-sm">
                        <div className="text-cyan-300 font-bold">#{block.height}</div>
                        <div className="text-purple-300 font-mono text-xs">{formatHash(block.hash)}</div>
                        <div className="text-green-300 text-xs">{block.transactions} txs</div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              )}

              {/* Transactions Results */}
              {searchResults.transactions?.length > 0 && (
                <div>
                  <h3 className="text-lg font-semibold text-pink-300 mb-3">⚡ Transactions ({searchResults.transactions.length})</h3>
                  {searchResults.transactions.map((tx: any, index: number) => (
                    <motion.div
                      key={tx.hash}
                      className="bg-pink-900/20 rounded-lg p-3 mb-2 border border-pink-500/20"
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.05 }}
                    >
                      <div className="text-sm">
                        <div className="text-pink-300 font-mono text-xs">{formatHash(tx.hash)}</div>
                        <div className="text-orange-300 text-xs">{tx.amount.toFixed(4)} ZION</div>
                        <div className="text-yellow-300 text-xs">🔮 {tx.cosmic_energy}</div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              )}

              {/* Addresses Results */}
              {searchResults.addresses?.length > 0 && (
                <div>
                  <h3 className="text-lg font-semibold text-green-300 mb-3">👛 Addresses ({searchResults.addresses.length})</h3>
                  {searchResults.addresses.map((addr: any, index: number) => (
                    <motion.div
                      key={addr.address}
                      className="bg-green-900/20 rounded-lg p-3 mb-2 border border-green-500/20"
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.05 }}
                    >
                      <div className="text-sm">
                        <div className="text-green-300 font-mono text-xs">{formatHash(addr.address)}</div>
                        <div className="text-yellow-300 text-xs">{addr.balance.toFixed(2)} ZION</div>
                        <div className="text-purple-300 text-xs">{addr.transactionCount} txs</div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              )}
            </div>
            
            {searchResults.totalResults === 0 && (
              <div className="text-center py-8">
                <div className="text-4xl mb-2">🔮</div>
                <p className="text-gray-400">No cosmic results found in the dharmic ledger...</p>
              </div>
            )}
          </motion.div>
        )}

        {!isLoading && !searchResults && (
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
            {/* Latest Blocks */}
            <motion.section
              className="bg-black/30 backdrop-blur-md rounded-xl p-6 border border-purple-500/30"
              initial={{ opacity: 0, x: -50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              <h2 className="text-2xl font-bold text-cyan-300 mb-6 flex items-center">
                🧊 Latest Cosmic Blocks
                <motion.div
                  className="ml-3 w-3 h-3 bg-green-400 rounded-full"
                  animate={{ opacity: [0.3, 1, 0.3] }}
                  transition={{ duration: 2, repeat: Infinity }}
                />
              </h2>

              <div className="space-y-4 max-h-96 overflow-y-auto">
                {blocks.map((block, index) => (
                  <motion.div
                    key={block.height}
                    className="bg-purple-900/20 rounded-lg p-4 border border-cyan-500/20 hover:border-cyan-400/50 transition-all"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                    whileHover={{ scale: 1.02, y: -2 }}
                  >
                    <div className="flex justify-between items-start mb-2">
                      <div className="flex items-center space-x-2">
                        <span className="text-yellow-300 font-bold">#{block.height}</span>
                        <motion.span 
                          className="text-xs text-green-400"
                          animate={{ opacity: [0.5, 1, 0.5] }}
                          transition={{ duration: 3, repeat: Infinity }}
                        >
                          ✨ COSMIC
                        </motion.span>
                      </div>
                      <span className="text-xs text-gray-400">{formatTime(block.timestamp)}</span>
                    </div>
                    
                    <div className="text-sm space-y-1">
                      <div className="text-purple-300">🔗 Hash: <span className="font-mono text-cyan-300">{formatHash(block.hash)}</span></div>
                      <div className="text-blue-300">⛏️ Miner: <span className="text-green-300">{block.miner}</span></div>
                      <div className="flex justify-between">
                        <span className="text-pink-300">📊 Txs: {block.transactions}</span>
                        <span className="text-orange-300">💎 Reward: {block.reward.toFixed(2)} ZION</span>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.section>

            {/* Recent Transactions */}
            <motion.section
              className="bg-black/30 backdrop-blur-md rounded-xl p-6 border border-purple-500/30"
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
            >
              <h2 className="text-2xl font-bold text-cyan-300 mb-6 flex items-center">
                ⚡ Cosmic Transactions
                <motion.div
                  className="ml-3 w-3 h-3 bg-yellow-400 rounded-full"
                  animate={{ opacity: [0.3, 1, 0.3] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                />
              </h2>

              <div className="space-y-3 max-h-96 overflow-y-auto">
                {transactions.map((tx, index) => (
                  <motion.div
                    key={tx.hash}
                    className="bg-indigo-900/20 rounded-lg p-3 border border-pink-500/20 hover:border-pink-400/50 transition-all"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.05 }}
                    whileHover={{ scale: 1.01, x: 5 }}
                  >
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-xs text-purple-300 font-mono">{formatHash(tx.hash)}</span>
                      <div className="flex items-center space-x-2">
                        <span className="text-xs text-yellow-400">🔮 {tx.cosmic_energy}</span>
                        <span className="text-xs text-gray-400">{formatTime(tx.timestamp)}</span>
                      </div>
                    </div>
                    
                    <div className="text-xs space-y-1">
                      <div className="flex items-center space-x-2">
                        <span className="text-green-300">From:</span>
                        <span className="font-mono text-cyan-300">{formatHash(tx.from)}</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className="text-red-300">To:</span>
                        <span className="font-mono text-cyan-300">{formatHash(tx.to)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-orange-300">💰 {tx.amount.toFixed(4)} ZION</span>
                        <span className="text-pink-300">Fee: {tx.fee.toFixed(6)}</span>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.section>
          </div>
        )}

        {/* Cosmic Footer */}
        <motion.footer
          className="text-center mt-16 py-8"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
        >
          <motion.div
            className="text-lg text-purple-400 mb-2"
            animate={{ y: [0, -5, 0] }}
            transition={{ duration: 4, repeat: Infinity }}
          >
            🚀 "Through cosmic blockchain, we transcend the material realm!" 🚀
          </motion.div>
          <p className="text-sm text-gray-500">
            ZION Network • Powered by Cosmic Consciousness • Jai Ram Ram Ram! 🕉️
          </p>
        </motion.footer>
      </div>
    </div>
  );
}
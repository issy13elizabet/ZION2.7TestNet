'use client'

import React, { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Link from 'next/link'

// Original Lightning Network Node Interface - zachované
interface LightningNode {
  id: number
  nodeId: string
  alias: string
  publicKey: string
  capacity: number
  channelCount: number
  isOnline: boolean
  color: string
  lastActivity: Date
}

// Nové Cosmic Interfaces
interface StargatePortal {
  id: string
  name: string
  destination_universe: string
  coordinates: string
  status: 'active' | 'charging' | 'offline' | 'unstable'
  energy_level: number
  dharma_requirement: number
  last_used: string
  cosmic_signature: string
  travelers_count: number
}

interface QuantumAction {
  id: string
  name: string
  action_type: 'teleport' | 'dimension_jump' | 'time_travel' | 'reality_shift' | 'consciousness_upload' | 'quantum_send' | 'quantum_receive' | 'blockchain_fork' | 'cosmic_mining' | 'dharma_stake'
  energy_cost: number
  dharma_impact: number
  success_rate: number
  cosmic_prerequisites: string[]
  description: string
}

const StargatePage: React.FC = () => {
  const [selectedNode, setSelectedNode] = useState<LightningNode | null>(null)
  const [lightningNodes, setLightningNodes] = useState<LightningNode[]>([])
  const [activeTab, setActiveTab] = useState<'portal' | 'quantum' | 'cosmic'>('portal')
  const [stargateStatus, setStargateStatus] = useState({
    portal: { energy: 85, status: 'active', connections: 28 },
    quantum_matrix: { stability: 92, interference: 3 },
    cosmic_alignment: 96
  })

  // Zachované - originálne Lightning nodes pre 28 kruhov
  const generateLightningNodes = (): LightningNode[] => {
    const nodeAliases = [
      'ACINQ', 'Lightning Labs', 'Bitrefill', 'OpenNode', 'LNBig', 'CoinGate',
      'BTCPay Server', 'Strike', 'River Financial', 'Zap Wallet', 'Phoenix',
      'Breez', 'Muun', 'BlueWallet', 'Wallet of Satoshi', 'LightningPeach',
      'LND Node Alpha', 'c-lightning Beta', 'Eclair Gamma', 'LDK Delta',
      'Sphinx Chat', 'Joule Browser', 'ThunderHub', 'RTL', 'LNDg',
      'BOS Score', 'Lightning Terminal', 'Pool'
    ]

    return Array.from({ length: 28 }, (_, i) => ({
      id: i + 1,
      nodeId: `ln_node_${i + 1}`,
      alias: nodeAliases[i] || `Node ${i + 1}`,
      publicKey: `02${Math.random().toString(16).slice(2, 66)}`,
      capacity: Math.floor(Math.random() * 50) + 10,
      channelCount: Math.floor(Math.random() * 500) + 50,
      isOnline: Math.random() > 0.1,
      color: `hsl(${Math.floor(Math.random() * 360)}, 70%, 60%)`,
      lastActivity: new Date(Date.now() - Math.random() * 3600000)
    }))
  }

  // Nové Stargate Portals
  const [stargatePortals] = useState<StargatePortal[]>([
    {
      id: 'portal_001',
      name: 'Andromeda Gateway',
      destination_universe: 'Andromeda Galaxy Cluster',
      coordinates: '⚡21°08\'42" 🌌88°16\'21" 💫108°42\'69"',
      status: 'active',
      energy_level: 95,
      dharma_requirement: 88,
      last_used: '2 cosmic cycles ago',
      cosmic_signature: '🌌 ANDROMEDA_NEXUS_ALPHA',
      travelers_count: 4216
    },
    {
      id: 'portal_002',
      name: 'Cosmic Dharma Bridge',
      destination_universe: 'Higher Dimensional Plane',
      coordinates: '☸️42°10\'80" 🕉️21°69\'42" ✨88°21\'08"',
      status: 'charging',
      energy_level: 67,
      dharma_requirement: 96,
      last_used: '1 dharma cycle ago',
      cosmic_signature: '☸️ DHARMA_BRIDGE_OMEGA',
      travelers_count: 2108
    },
    {
      id: 'portal_003',
      name: 'Quantum Entanglement Gate',
      destination_universe: 'Parallel Reality Matrix',
      coordinates: '🔬69°42\'21" ⚛️10°88\'16" 🌈42°21\'69"',
      status: 'unstable',
      energy_level: 34,
      dharma_requirement: 75,
      last_used: '108 quantum moments ago',
      cosmic_signature: '⚛️ QUANTUM_MATRIX_BETA',
      travelers_count: 888
    }
  ])

  // Nové Quantum Actions
  const [quantumActions] = useState<QuantumAction[]>([
    {
      id: 'qa_001',
      name: 'Consciousness Transfer',
      action_type: 'consciousness_upload',
      energy_cost: 2108,
      dharma_impact: 42,
      success_rate: 88.8,
      cosmic_prerequisites: ['Dharma Level 80+', 'Cosmic Alignment 90+', 'Neural Interface'],
      description: 'Upload consciousness to the Universal Mind Matrix'
    },
    {
      id: 'qa_002',
      name: 'Reality Shift Protocol',
      action_type: 'reality_shift',
      energy_cost: 4216,
      dharma_impact: 108,
      success_rate: 69.4,
      cosmic_prerequisites: ['Master Shaman Status', 'Quantum Key Access', 'Sacred Geometry Knowledge'],
      description: 'Shift current reality parameters within quantum probability field'
    },
    {
      id: 'qa_003',
      name: 'Dimensional Jump',
      action_type: 'dimension_jump',
      energy_cost: 888,
      dharma_impact: 21,
      success_rate: 95.5,
      cosmic_prerequisites: ['Portal Access Rights', 'Stable Energy Source'],
      description: 'Jump between dimensional layers of existence'
    },
    {
      id: 'qa_004',
      name: 'Time Stream Navigation',
      action_type: 'time_travel',
      energy_cost: 8432,
      dharma_impact: 216,
      success_rate: 42.1,
      cosmic_prerequisites: ['Temporal License', 'Causality Protection', 'Timeline Authority'],
      description: 'Navigate through temporal streams with minimal timeline disruption'
    },
    {
      id: 'qa_005',
      name: 'Instant Teleportation',
      action_type: 'teleport',
      energy_cost: 1337,
      dharma_impact: 33,
      success_rate: 92.1,
      cosmic_prerequisites: ['Spatial Coordinates', 'Quantum Beacon'],
      description: 'Instantaneous matter transportation across vast distances'
    },
    {
      id: 'qa_006',
      name: 'Quantum Send Protocol',
      action_type: 'quantum_send',
      energy_cost: 666,
      dharma_impact: 15,
      success_rate: 98.7,
      cosmic_prerequisites: ['Verified Receiver Address', 'Quantum Signature'],
      description: 'Send assets through quantum-encrypted interdimensional channels'
    },
    {
      id: 'qa_007',
      name: 'Quantum Receive Matrix',
      action_type: 'quantum_receive',
      energy_cost: 333,
      dharma_impact: 10,
      success_rate: 99.2,
      cosmic_prerequisites: ['Active Quantum Wallet', 'Dharma Verification'],
      description: 'Receive assets from parallel blockchain dimensions'
    },
    {
      id: 'qa_008',
      name: 'Blockchain Fork Creation',
      action_type: 'blockchain_fork',
      energy_cost: 5555,
      dharma_impact: 144,
      success_rate: 33.3,
      cosmic_prerequisites: ['Master Node Authority', 'Community Consensus', 'Cosmic Wisdom 100+'],
      description: 'Create new blockchain reality fork in parallel universe'
    },
    {
      id: 'qa_009',
      name: 'Cosmic Mining Rig',
      action_type: 'cosmic_mining',
      energy_cost: 2222,
      dharma_impact: 77,
      success_rate: 84.4,
      cosmic_prerequisites: ['Quantum Computing Array', 'Stellar Energy Source'],
      description: 'Mine cryptocurrency using cosmic energy from distant stars'
    },
    {
      id: 'qa_010',
      name: 'Dharma Staking Pool',
      action_type: 'dharma_stake',
      energy_cost: 1111,
      dharma_impact: 55,
      success_rate: 91.8,
      cosmic_prerequisites: ['Good Karma Balance', 'Meditation Level 50+'],
      description: 'Stake tokens powered by accumulated dharma and cosmic good deeds'
    }
  ])

  useEffect(() => {
    setLightningNodes(generateLightningNodes())
    
    // Simulácia live updates
    const interval = setInterval(() => {
      setStargateStatus(prev => ({
        portal: { 
          energy: Math.min(100, prev.portal.energy + (Math.random() - 0.4)), 
          status: prev.portal.energy > 50 ? 'active' : 'charging',
          connections: 28 
        },
        quantum_matrix: { 
          stability: Math.min(100, Math.max(0, prev.quantum_matrix.stability + (Math.random() - 0.5) * 2)),
          interference: Math.max(0, Math.min(10, prev.quantum_matrix.interference + (Math.random() - 0.5)))
        },
        cosmic_alignment: Math.min(100, Math.max(80, prev.cosmic_alignment + (Math.random() - 0.5) * 0.5))
      }))
    }, 3000)

    return () => clearInterval(interval)
  }, [])

  const getNodeOpacity = (node: LightningNode) => {
    if (!node.isOnline) return 0.2
    return 0.3 + (node.capacity / 60) * 0.7
  }

  const handleNodeClick = (node: LightningNode) => {
    setSelectedNode(node)
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-green-400 bg-green-500/20'
      case 'charging': return 'text-yellow-400 bg-yellow-500/20'
      case 'offline': return 'text-red-400 bg-red-500/20'
      case 'unstable': return 'text-orange-400 bg-orange-500/20'
      default: return 'text-gray-400 bg-gray-500/20'
    }
  }

  const getActionTypeIcon = (type: string) => {
    switch (type) {
      case 'teleport': return '🚀'
      case 'dimension_jump': return '🌀'
      case 'time_travel': return '⏰'
      case 'reality_shift': return '🔄'
      case 'consciousness_upload': return '🧠'
      case 'quantum_send': return '📤'
      case 'quantum_receive': return '📥'
      case 'blockchain_fork': return '🍴'
      case 'cosmic_mining': return '⛏️'
      case 'dharma_stake': return '🥩'
      default: return '⚡'
    }
  }

  const executeQuantumAction = (action: QuantumAction) => {
    const success = Math.random() < (action.success_rate / 100)
    
    if (success) {
      const successMessages: Record<string, string> = {
        'teleport': `🚀 Teleportácia úspešná! Prenos materiálu dokončený s presnosťou 99.97%`,
        'dimension_jump': `🌀 Skok do dimenzie úspešný! Vstup do alternatívnej reality potvrdený`,
        'time_travel': `⏰ Časové cestovanie dokončené! Timeline integrity: PRESERVED`,
        'reality_shift': `🔄 Realita upravená! Kvantové parametre úspešne rekonfigurované`,
        'consciousness_upload': `🧠 Vedomie nahraté! Pripojenie k Universal Mind Matrix stabilné`,
        'quantum_send': `📤 Quantum send úspešný! Assets odoslané cez interdimenzionálny kanál`,
        'quantum_receive': `📥 Quantum receive dokončený! Assets prijaté z paralelného blockchain`,
        'blockchain_fork': `🍴 Blockchain fork vytvorený! Nová realita blockchain štruktúry spustená`,
        'cosmic_mining': `⛏️ Cosmic mining aktívny! Ťaženie z hvezdnej energie úspešne spustené`,
        'dharma_stake': `🥩 Dharma staking aktivovaný! Karma rewards generovanie spustené`
      }
      
      alert(`✅ ${successMessages[action.action_type] || action.name + ' executed successfully!'}\n💎 Dharma Impact: +${action.dharma_impact}\n⚡ Energy Used: ${action.energy_cost}`)
    } else {
      const failureMessages: Record<string, string> = {
        'teleport': `❌ Teleportácia neúspešná! Kvantový interferencia detekovaná`,
        'dimension_jump': `❌ Skok do dimenzie blokovaný! Portál temporal destabilizovaný`,
        'time_travel': `❌ Časové cestovanie zrušené! Temporal paradox warning`,
        'reality_shift': `❌ Reality shift failed! Quantum resistance encountered`,
        'consciousness_upload': `❌ Upload consciousness neúspešný! Neural interface overload`,
        'quantum_send': `❌ Quantum send failed! Receiver address unreachable`,
        'quantum_receive': `❌ Quantum receive error! Blockchain synchronization lost`,
        'blockchain_fork': `❌ Fork creation failed! Insufficient consensus power`,
        'cosmic_mining': `❌ Cosmic mining offline! Stellar energy source depleted`,
        'dharma_stake': `❌ Dharma staking failed! Karma balance insufficient`
      }
      
      alert(`${failureMessages[action.action_type] || '❌ ' + action.name + ' failed!'}\n⚠️ Quantum interference detected\n💫 Try again when cosmic conditions improve`)
    }
  }

  const activatePortal = (portal: StargatePortal) => {
    if (portal.status === 'active') {
      alert(`🌌 Activating ${portal.name}... Destination: ${portal.destination_universe}`)
    } else {
      alert(`⚠️ ${portal.name} is ${portal.status}. Cannot activate at this time.`)
    }
  }

  // Zachované - Tibetan mantra glyphs
  const generateGlyphs = () => {
    const mantra = 'ཨོཾ་མ་ཎི་པདྨེ་ཧཱུྃ'
    const chars = Array.from(mantra)
    const glyphs: JSX.Element[] = []
    for (let i = 0; i < 39; i++) {
      const rotation = (360 / 39) * i
      const symbol = chars[i % chars.length]
      glyphs.push(
        <motion.div
          key={i}
          className="absolute text-cyan-300 text-xl font-bold"
          style={{ 
            transform: `translateX(-50%) rotate(${rotation}deg)`,
            left: '50%',
            top: '50%',
            transformOrigin: `0 ${200 + i * 5}px`
          }}
          animate={{ rotate: rotation + (Date.now() / 100) % 360 }}
          transition={{ duration: 60, repeat: Infinity, ease: 'linear' }}
          title={mantra}
        >
          {symbol}
        </motion.div>
      )
    }
    return glyphs
  }

  // Zachované - Lightning rings s cosmic enhancements
  const generateLightningRings = () => {
    return lightningNodes.map((node, index) => {
      const radius = 100 + index * 12
      const rotation = (360 / 28) * index
      
      return (
        <motion.div
          key={node.id}
          className="absolute w-8 h-8 border-2 rounded-full cursor-pointer"
          style={{
            left: '50%',
            top: '50%',
            transformOrigin: `0 ${radius}px`,
            borderColor: node.isOnline ? node.color : '#666',
            opacity: getNodeOpacity(node),
            boxShadow: selectedNode?.id === node.id ? `0 0 20px ${node.color}` : 'none',
            background: node.isOnline ? `radial-gradient(circle, ${node.color}40, transparent)` : '#333'
          }}
          animate={{ 
            rotate: rotation + (node.isOnline ? (Date.now() / 1000) % 360 : 0),
            scale: selectedNode?.id === node.id ? 1.5 : 1
          }}
          whileHover={{ scale: 1.3 }}
          whileTap={{ scale: 0.9 }}
          onClick={() => handleNodeClick(node)}
          title={`${node.alias} - ${node.capacity} BTC - ${node.channelCount} channels`}
        >
          <div className="w-full h-full rounded-full bg-gradient-to-br from-cyan-400 to-purple-600 animate-pulse" />
        </motion.div>
      )
    })
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-purple-950 to-blue-950 text-white p-6 relative overflow-hidden">
      {/* Cosmic Background Animation */}
      <div className="fixed inset-0 pointer-events-none">
        {Array.from({ length: 100 }).map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-white rounded-full"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`
            }}
            animate={{
              opacity: [0, 1, 0],
              scale: [0, 1, 0]
            }}
            transition={{
              duration: Math.random() * 3 + 2,
              repeat: Infinity,
              delay: Math.random() * 5
            }}
          />
        ))}
      </div>

      <motion.header
        className="text-center mb-8 relative z-10"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <Link href="/wallet" className="inline-block mb-4">
          <motion.button
            className="px-4 py-2 bg-purple-600/30 hover:bg-purple-600/50 rounded-lg text-sm"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            ← Back to Wallet
          </motion.button>
        </Link>
        
        <h1 className="text-5xl font-bold bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent mb-2">
          🌌 COSMIC STARGATE
        </h1>
        <p className="text-cyan-300 text-lg">Universal Portal & Quantum Action Matrix</p>
      </motion.header>

      {/* Tab Navigation */}
      <div className="flex justify-center mb-8 relative z-10">
        <div className="flex gap-2 bg-black/50 p-1 rounded-xl border border-purple-500/30">
          {[
            { key: 'portal', label: '🌌 Portal Gate', icon: '🚪' },
            { key: 'quantum', label: '⚛️ Quantum Actions', icon: '⚡' },
            { key: 'cosmic', label: '🌟 Cosmic Control', icon: '🎛️' }
          ].map((tab) => (
            <motion.button
              key={tab.key}
              onClick={() => setActiveTab(tab.key as any)}
              className={`px-6 py-3 rounded-lg font-semibold transition-all ${
                activeTab === tab.key 
                  ? 'bg-purple-600 text-white shadow-lg' 
                  : 'text-gray-300 hover:text-white hover:bg-purple-600/20'
              }`}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              {tab.icon} {tab.label}
            </motion.button>
          ))}
        </div>
      </div>

      <AnimatePresence mode="wait">
        {/* PORTAL TAB - Zachovaná brána s cosmic enhancements */}
        {activeTab === 'portal' && (
          <motion.div
            key="portal"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="relative z-10"
          >
            {/* Main Stargate */}
            <div className="flex justify-center mb-8">
              <div className="relative w-96 h-96">
                {/* Outer Ring */}
                <motion.div 
                  className="absolute inset-0 border-4 border-cyan-400 rounded-full bg-gradient-to-br from-purple-900/20 to-blue-900/20"
                  animate={{ rotate: 360 }}
                  transition={{ duration: 60, repeat: Infinity, ease: 'linear' }}
                />
                
                {/* Inner Energy Field */}
                <motion.div 
                  className="absolute inset-8 border-2 border-purple-400 rounded-full bg-gradient-radial from-cyan-500/10 to-purple-900/30"
                  animate={{ rotate: -360 }}
                  transition={{ duration: 40, repeat: Infinity, ease: 'linear' }}
                />

                {/* Lightning Network Rings - ZACHOVANÉ */}
                <div className="absolute inset-0">
                  {generateLightningRings()}
                </div>

                {/* Tibetan Mantra Glyphs - ZACHOVANÉ */}
                <div className="absolute inset-0">
                  {generateGlyphs()}
                </div>

                {/* Central Portal */}
                <motion.div 
                  className="absolute inset-16 rounded-full bg-gradient-radial from-white/20 via-cyan-500/30 to-purple-900/50 backdrop-blur-sm"
                  animate={{ 
                    scale: [1, 1.1, 1],
                    opacity: [0.7, 1, 0.7]
                  }}
                  transition={{ duration: 3, repeat: Infinity }}
                >
                  <div className="flex items-center justify-center h-full text-4xl">
                    🌌
                  </div>
                </motion.div>

                {/* Chevrons - ZACHOVANÉ koncept */}
                {Array.from({ length: 9 }).map((_, i) => {
                  const angle = (360 / 9) * i
                  return (
                    <motion.div
                      key={i}
                      className="absolute w-4 h-8 bg-gradient-to-t from-orange-500 to-yellow-400 rounded-t-full"
                      style={{
                        left: '50%',
                        top: '10px',
                        transformOrigin: '2px 184px',
                        transform: `translateX(-50%) rotate(${angle}deg)`
                      }}
                      animate={{
                        opacity: [0.5, 1, 0.5],
                        scale: [1, 1.2, 1]
                      }}
                      transition={{
                        duration: 2,
                        repeat: Infinity,
                        delay: i * 0.2
                      }}
                    />
                  )
                })}
              </div>
            </div>

            {/* Status Dashboard */}
            <div className="grid gap-6 md:grid-cols-3 mb-8">
              <motion.div className="bg-gradient-to-br from-cyan-500 to-blue-600 p-1 rounded-2xl" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.1 }}>
                <div className="bg-black/70 rounded-xl p-6 text-center">
                  <div className="text-3xl mb-2">⚡</div>
                  <div className="text-2xl font-bold text-cyan-300">{stargateStatus.portal.energy.toFixed(1)}%</div>
                  <div className="text-sm text-cyan-200">Portal Energy</div>
                </div>
              </motion.div>

              <motion.div className="bg-gradient-to-br from-purple-500 to-pink-600 p-1 rounded-2xl" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.2 }}>
                <div className="bg-black/70 rounded-xl p-6 text-center">
                  <div className="text-3xl mb-2">🌀</div>
                  <div className="text-2xl font-bold text-purple-300">{stargateStatus.quantum_matrix.stability.toFixed(1)}%</div>
                  <div className="text-sm text-purple-200">Quantum Stability</div>
                </div>
              </motion.div>

              <motion.div className="bg-gradient-to-br from-orange-500 to-red-600 p-1 rounded-2xl" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.3 }}>
                <div className="bg-black/70 rounded-xl p-6 text-center">
                  <div className="text-3xl mb-2">🌟</div>
                  <div className="text-2xl font-bold text-orange-300">{stargateStatus.cosmic_alignment.toFixed(1)}%</div>
                  <div className="text-sm text-orange-200">Cosmic Alignment</div>
                </div>
              </motion.div>
            </div>

            {/* Lightning Network Stats - ZACHOVANÉ */}
            <div className="bg-gradient-to-br from-gray-800 to-gray-900 p-6 rounded-2xl border border-cyan-500/30 mb-8">
              <h3 className="text-xl font-semibold mb-4 text-cyan-300">⚡ Lightning Network Status</h3>
              <div className="grid gap-4 md:grid-cols-3">
                <div className="bg-black/30 p-4 rounded-lg">
                  <div className="text-sm text-gray-400">Online Nodes</div>
                  <div className="text-2xl font-bold text-green-400">
                    {lightningNodes.filter(n => n.isOnline).length}/{lightningNodes.length}
                  </div>
                </div>
                <div className="bg-black/30 p-4 rounded-lg">
                  <div className="text-sm text-gray-400">Total Capacity</div>
                  <div className="text-2xl font-bold text-yellow-400">
                    {lightningNodes.reduce((sum, n) => sum + n.capacity, 0)} BTC
                  </div>
                </div>
                <div className="bg-black/30 p-4 rounded-lg">
                  <div className="text-sm text-gray-400">Total Channels</div>
                  <div className="text-2xl font-bold text-cyan-400">
                    {lightningNodes.reduce((sum, n) => sum + n.channelCount, 0).toLocaleString()}
                  </div>
                </div>
              </div>
            </div>

            {/* Selected Node Info - ZACHOVANÉ */}
            {selectedNode && (
              <motion.div 
                className="bg-gradient-to-br from-gray-800 to-gray-900 p-6 rounded-2xl border border-purple-500/30"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <div className="flex justify-between items-start">
                  <div>
                    <h3 className="text-xl font-semibold text-purple-300">{selectedNode.alias}</h3>
                    <div className="text-sm text-gray-400 mt-2">
                      <p><strong>Capacity:</strong> {selectedNode.capacity} BTC</p>
                      <p><strong>Channels:</strong> {selectedNode.channelCount.toLocaleString()}</p>
                      <p><strong>Status:</strong> 
                        <span className={selectedNode.isOnline ? 'text-green-400' : 'text-red-400'}>
                          {selectedNode.isOnline ? ' ONLINE' : ' OFFLINE'}
                        </span>
                      </p>
                      <p><strong>Public Key:</strong> <span className="font-mono text-xs">{selectedNode.publicKey.slice(0, 40)}...</span></p>
                    </div>
                  </div>
                  <motion.button
                    className="px-4 py-2 bg-red-600/30 hover:bg-red-600/50 rounded-lg text-sm"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={() => setSelectedNode(null)}
                  >
                    ✕ Close
                  </motion.button>
                </div>
              </motion.div>
            )}
          </motion.div>
        )}

        {/* QUANTUM ACTIONS TAB - NOVÉ */}
        {activeTab === 'quantum' && (
          <motion.div
            key="quantum"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="space-y-6"
          >
            <h2 className="text-2xl font-semibold text-center mb-6 text-purple-300">⚛️ Quantum Action Matrix</h2>
            
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
              {quantumActions.map((action, index) => (
                <motion.div
                  key={action.id}
                  className="bg-gradient-to-br from-gray-800 to-gray-900 p-6 rounded-2xl border border-purple-500/30"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.1 }}
                  whileHover={{ scale: 1.02 }}
                >
                  <div className="text-center mb-4">
                    <div className="text-4xl mb-2">{getActionTypeIcon(action.action_type)}</div>
                    <h3 className="text-lg font-semibold text-purple-300">{action.name}</h3>
                  </div>

                  <div className="space-y-3 mb-4">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Energy Cost:</span>
                      <span className="text-yellow-400">{action.energy_cost.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Dharma Impact:</span>
                      <span className="text-green-400">+{action.dharma_impact}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-400">Success Rate:</span>
                      <span className="text-cyan-400">{action.success_rate}%</span>
                    </div>
                  </div>

                  <div className="mb-4">
                    <div className="text-xs text-gray-400 mb-2">Prerequisites:</div>
                    <div className="space-y-1">
                      {action.cosmic_prerequisites.map((req, i) => (
                        <div key={i} className="text-xs text-purple-300">• {req}</div>
                      ))}
                    </div>
                  </div>

                  <p className="text-xs text-gray-300 mb-4">{action.description}</p>

                  <motion.button
                    className="w-full bg-gradient-to-r from-purple-500 to-pink-600 px-4 py-2 rounded-lg font-semibold"
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={() => executeQuantumAction(action)}
                  >
                    Execute {getActionTypeIcon(action.action_type)}
                  </motion.button>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}

        {/* COSMIC CONTROL TAB - NOVÉ */}
        {activeTab === 'cosmic' && (
          <motion.div
            key="cosmic"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="space-y-6"
          >
            <h2 className="text-2xl font-semibold text-center mb-6 text-cyan-300">🌟 Stargate Portal Control</h2>
            
            <div className="space-y-6">
              {stargatePortals.map((portal, index) => (
                <motion.div
                  key={portal.id}
                  className="bg-gradient-to-br from-gray-800 to-gray-900 p-6 rounded-2xl border border-cyan-500/30"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.2 }}
                >
                  <div className="flex justify-between items-start mb-4">
                    <div>
                      <h3 className="text-xl font-semibold text-cyan-300">{portal.name}</h3>
                      <div className="text-sm text-gray-400 mt-1">{portal.destination_universe}</div>
                      <div className="text-xs text-purple-300 font-mono mt-1">{portal.coordinates}</div>
                    </div>
                    <div className="text-right">
                      <div className={`px-3 py-1 rounded-lg text-xs font-medium ${getStatusColor(portal.status)}`}>
                        {portal.status.toUpperCase()}
                      </div>
                      <div className="text-xs text-gray-400 mt-1">Travelers: {portal.travelers_count.toLocaleString()}</div>
                    </div>
                  </div>

                  <div className="grid gap-4 md:grid-cols-3 mb-4">
                    <div className="bg-black/30 p-3 rounded-lg">
                      <div className="text-xs text-gray-400">Energy Level</div>
                      <div className="text-lg font-bold text-yellow-300">{portal.energy_level}%</div>
                      <div className="w-full bg-gray-700 rounded-full h-2 mt-1">
                        <div 
                          className="bg-gradient-to-r from-yellow-500 to-orange-400 h-2 rounded-full"
                          style={{ width: `${portal.energy_level}%` }}
                        />
                      </div>
                    </div>
                    
                    <div className="bg-black/30 p-3 rounded-lg">
                      <div className="text-xs text-gray-400">Dharma Requirement</div>
                      <div className="text-lg font-bold text-purple-300">{portal.dharma_requirement}</div>
                      <div className="text-xs text-purple-200">Minimum Level</div>
                    </div>
                    
                    <div className="bg-black/30 p-3 rounded-lg">
                      <div className="text-xs text-gray-400">Last Used</div>
                      <div className="text-sm font-bold text-cyan-300">{portal.last_used}</div>
                    </div>
                  </div>

                  <div className="mb-4">
                    <div className="text-sm text-purple-300 mb-1">{portal.cosmic_signature}</div>
                  </div>

                  <div className="flex gap-3">
                    <motion.button
                      className={`flex-1 px-4 py-2 rounded-lg font-semibold ${
                        portal.status === 'active' 
                          ? 'bg-gradient-to-r from-green-500 to-emerald-600' 
                          : 'bg-gray-600 cursor-not-allowed'
                      }`}
                      whileHover={portal.status === 'active' ? { scale: 1.02 } : {}}
                      whileTap={portal.status === 'active' ? { scale: 0.98 } : {}}
                      onClick={() => activatePortal(portal)}
                      disabled={portal.status !== 'active'}
                    >
                      🌌 Activate Portal
                    </motion.button>
                    
                    <motion.button
                      className="flex-1 px-4 py-2 bg-blue-600/30 hover:bg-blue-600/50 rounded-lg font-semibold"
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      📊 Portal Stats
                    </motion.button>
                    
                    <motion.button
                      className="flex-1 px-4 py-2 bg-purple-600/30 hover:bg-purple-600/50 rounded-lg font-semibold"
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      🔧 Configure
                    </motion.button>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default StargatePage
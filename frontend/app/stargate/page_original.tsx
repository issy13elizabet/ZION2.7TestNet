"use client";
import { motion } from "framer-motion";
import { useState, useEffect } from "react";
import Link from "next/link";

interface StargatePortal {
  id: string;
  name: string;
  destination_universe: string;
  coordinates: string;
  status: 'active' | 'charging' | 'offline' | 'unstable';
  energy_level: number;
  dharma_requirement: number;
  last_used: string;
  cosmic_signature: string;
  travelers_count: number;
}

interface QuantumTunnel {
  id: string;
  source: string;
  destination: string;
  stability: number;
  bandwidth: number;
  latency: number;
  cosmic_interference: number;
  active_connections: number;
}

interface DimensionalBeacon {
  id: string;
  universe_id: string;
  frequency: number;
  signal_strength: number;
  beacon_type: 'navigation' | 'communication' | 'emergency' | 'trade';
  cosmic_alignment: number;
}

const StargatePage: React.FC = () => {
  const { t } = useLanguage()
  const { status, loading, error, refresh } = useStargateStatus()
  const [selectedNode, setSelectedNode] = useState<LightningNode | null>(null)
  const [lightningNodes, setLightningNodes] = useState<LightningNode[]>([])
  const [activeTab, setActiveTab] = useState<'portal' | 'swap' | 'radio'>('portal')

  // Generate mock Lightning nodes for 28 rings
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
      capacity: Math.floor(Math.random() * 50) + 10, // 10-60 BTC
      channelCount: Math.floor(Math.random() * 500) + 50, // 50-550 channels
      isOnline: Math.random() > 0.1, // 90% online
      color: `hsl(${Math.floor(Math.random() * 360)}, 70%, 60%)`,
      lastActivity: new Date(Date.now() - Math.random() * 3600000) // Last hour
    }))
  }

  useEffect(() => {
    setLightningNodes(generateLightningNodes())
  }, [])

  const getStatusColor = (serviceStatus: string) => {
    switch (serviceStatus) {
      case 'active': return '#00ff00'
      case 'maintenance': return '#ffff00'
      case 'inactive': return '#ff0000'
      default: return '#666666'
    }
  }

  const getNodeOpacity = (node: LightningNode) => {
    if (!node.isOnline) return 0.2
    return 0.3 + (node.capacity / 60) * 0.7 // Based on capacity
  }

  const handleNodeClick = (node: LightningNode) => {
    setSelectedNode(node)
  }

  // Generate glyphs as Tibetan mantra: ཨོཾ་མ་ཎི་པདྨེ་ཧཱུྃ repeated to fill 39 slots
  const generateGlyphs = () => {
    const mantra = 'ཨོཾ་མ་ཎི་པདྨེ་ཧཱུྃ'
    // Rozložit mantru na jednotlivé znaky včetně kombinujících znaků – použijeme Array.from pro Unicode
    const chars = Array.from(mantra)
    const glyphs: JSX.Element[] = []
    for (let i = 0; i < 39; i++) {
      const rotation = (360 / 39) * i
      const symbol = chars[i % chars.length]
      glyphs.push(
        <div
          key={i}
          className={styles.glyph}
          style={{ transform: `translateX(-50%) rotate(${rotation}deg)` }}
          title={mantra}
        >
          {symbol}
        </div>
      )
    }
    return glyphs
  }

  // Generate rings with Lightning node data
  const generateLightningRings = () => {
    return lightningNodes.map((node, index) => {
      const isInner = index >= 22 // Last 6 are inner Sri Yantra rings
  const rotateClass = (styles as Record<string, string>)[`rotate${index + 1}`]
  const stateClass = node.isOnline ? styles.ringOnline : styles.ringOffline
  const className = [rotateClass, styles.lightningRing, stateClass].filter(Boolean).join(' ')

      return (
        <img
          key={index}
          src={isInner ? '/stargate/1.png' : '/stargate/2.png'}
          alt={`Lightning Node: ${node.alias}`}
          className={className}
          title={`${node.alias} - ${node.capacity} BTC - ${node.channelCount} channels`}
          style={{
            opacity: getNodeOpacity(node),
            cursor: 'pointer',
            boxShadow: selectedNode?.id === node.id ? `0 0 20px ${node.color}` : 'none'
          }}
          onClick={() => handleNodeClick(node)}
        />
      )
    })
  }

  return (
    <div className={styles.stargatePage}>
      {/* Tab Navigation */}
      <div className={styles.tabNav}>
        <button
          onClick={() => setActiveTab('portal')}
          className={`${styles.tabBtn} ${activeTab === 'portal' ? styles.tabBtnActive : ''}`}
        >
          🌌 Stargate Portal
        </button>
        <button
          onClick={() => setActiveTab('swap')}
          className={`${styles.tabBtn} ${activeTab === 'swap' ? styles.tabBtnActive : ''}`}
        >
          ⚡🔄 Atomic Swap
        </button>
        <button
          onClick={() => setActiveTab('radio')}
          className={`${styles.tabBtn} ${activeTab === 'radio' ? styles.tabBtnActive : ''}`}
        >
          🎵🔮 Cosmic Radio
        </button>
      </div>

      {/* Health Widget fixed under tabs, centered */}
      <div className={styles.healthWrap}>
        <HealthWidget />
      </div>

      {/* Portal Tab */}
      {activeTab === 'portal' && (
        <div className={styles.portalContainer}>
          {/* Main Portal Gate */}
          <div className={styles.gate}>
            <div className={styles.container}>
              {/* Lightning Network Rings */}
              {generateLightningRings()}

              {/* Lightning Merchant Glyphs */}
              <div className={styles.glyphs}>
                {generateGlyphs()}
              </div>

              {/* Chevrons (outer only) */}
              <div className={styles.chevrons}>
                {Array.from({ length: 9 }).map((_, i) => (
                  <div key={i} className={styles.chevron} />
                ))}
              </div>
            </div>
          </div>

          {/* Lightning Node Info Panel */}
          {selectedNode && (
            <div className={styles.lightningPanel}>
              <h3>{selectedNode.alias}</h3>
              <div className={styles.nodeDetails}>
                <p><strong>Capacity:</strong> {selectedNode.capacity} BTC</p>
                <p><strong>Channels:</strong> {selectedNode.channelCount}</p>
                <p><strong>Status:</strong>
                  <span style={{ color: selectedNode.isOnline ? '#00ff00' : '#ff0000' }}>
                    {selectedNode.isOnline ? ' ONLINE' : ' OFFLINE'}
                  </span>
                </p>
                <p><strong>Public Key:</strong> {selectedNode.publicKey.slice(0, 20)}...</p>
              </div>
              <button
                className={styles.closePanel}
                onClick={() => setSelectedNode(null)}
              >
                ✕ Close
              </button>
            </div>
          )}

          {/* Status Dashboard placed below the gate */}
          <div className={styles.statusDashboard}>
            <h3 className={styles.statusTitle}>⚡ {t.stargate.status} - Lightning Network</h3>
            {status && (
              <div className={styles.portalEnergy}>
                <div className={styles.energyBar}>
                  <div className={styles.fill} style={{ width: `${status.portal.energy || 0}%` }} />
                </div>
                <div className={styles.energyInfo}>
                  <span>Portal: {status.portal.status.toUpperCase()}</span>
                  <span>Energy: {status.portal.energy}%</span>
                  <span>Connections: {status.portal.connections}</span>
                </div>
              </div>
            )}

            {loading && (
              <div className="status-loading">
                {t.stargate.connecting}...
              </div>
            )}

            {error && (
              <div className="status-error">
                Error: {error}
              </div>
            )}

            <div className={styles.lightningStats}>
              <div className={styles.statItem}>
                <span>Online Nodes:</span>
                <span style={{ color: '#00ff00' }}>
                  {lightningNodes.filter(n => n.isOnline).length}/{lightningNodes.length}
                </span>
              </div>
              <div className={styles.statItem}>
                <span>Total Capacity:</span>
                <span style={{ color: '#ffff00' }}>
                  {lightningNodes.reduce((sum, n) => sum + n.capacity, 0)} BTC
                </span>
              </div>
              <div className={styles.statItem}>
                <span>Total Channels:</span>
                <span style={{ color: '#00ffff' }}>
                  {lightningNodes.reduce((sum, n) => sum + n.channelCount, 0)}
                </span>
              </div>
            </div>

            {status?.services && (
              <div className={styles.servicesGrid}>
                {status.services.map((service: StargateService, index: number) => (
                  <div key={index} className={styles.serviceItem}>
                    <span>{service.name}</span>
                    <span
                      className={styles.serviceStatus}
                      style={{ color: getStatusColor(service.status) }}
                    >
                      {service.status.toUpperCase()}
                    </span>
                  </div>
                ))}
              </div>
            )}

            <div className={styles.refreshWrap}>
              <button onClick={refresh} className={styles.refreshBtn}>
                ⚡ {t.stargate.refresh} Lightning
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Atomic Swap Tab */}
      {activeTab === 'swap' && (
        <div className={styles.swapContainer}>
          <AtomicSwapWidget />
        </div>
      )}

      {/* Cosmic Radio Tab */}
      {activeTab === 'radio' && (
        <div className={styles.radioContainer}>
          <CosmicRadio />
        </div>
      )}
    </div>
  )
}

export default StargatePage
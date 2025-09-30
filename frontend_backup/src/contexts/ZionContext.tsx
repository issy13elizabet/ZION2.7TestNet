'use client'

import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react'
import { ZionAPI } from '@/lib/zion-api'
import { ZionWebSocket } from '@/lib/zion-websocket'

interface NetworkInfo {
  height: number
  difficulty: number
  hashrate: number
  version: string
  status: string
  peers: number
}

interface ZionContextType {
  networkInfo: NetworkInfo | null
  isConnected: boolean
  api: ZionAPI
  ws: ZionWebSocket | null
  refreshNetworkInfo: () => Promise<void>
}

const ZionContext = createContext<ZionContextType | undefined>(undefined)

interface ZionProviderProps {
  children: ReactNode
}

export function ZionProvider({ children }: ZionProviderProps) {
  const [networkInfo, setNetworkInfo] = useState<NetworkInfo | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [api] = useState(() => new ZionAPI())
  const [ws, setWs] = useState<ZionWebSocket | null>(null)

  const refreshNetworkInfo = async () => {
    try {
      const info = await api.getInfo()
      setNetworkInfo({
        height: info.height || 0,
        difficulty: info.difficulty || 0,
        hashrate: info.hashrate || 0,
        version: info.version || '2.6.75',
        status: info.status || 'unknown',
        peers: info.outgoing_connections_count || 0
      })
      setIsConnected(true)
    } catch (error) {
      console.error('Failed to fetch network info:', error)
      setIsConnected(false)
    }
  }

  useEffect(() => {
    // Initialize WebSocket connection
    const websocket = new ZionWebSocket()
    websocket.connect()
    setWs(websocket)

    // Initial network info fetch
    refreshNetworkInfo()

    // Set up periodic refresh
    const interval = setInterval(refreshNetworkInfo, 10000) // Every 10 seconds

    return () => {
      clearInterval(interval)
      websocket.disconnect()
    }
  }, [])

  const value: ZionContextType = {
    networkInfo,
    isConnected,
    api,
    ws,
    refreshNetworkInfo
  }

  return <ZionContext.Provider value={value}>{children}</ZionContext.Provider>
}

export function useZion() {
  const context = useContext(ZionContext)
  if (context === undefined) {
    throw new Error('useZion must be used within a ZionProvider')
  }
  return context
}
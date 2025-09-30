'use client'

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react'
import { generateMnemonic, mnemonicToSeedSync } from 'bip39'
import CryptoJS from 'crypto-js'

interface WalletState {
  address: string | null
  balance: number
  isLocked: boolean
  mnemonic: string | null
  privateKey: string | null
}

interface Transaction {
  id: string
  type: 'sent' | 'received'
  amount: number
  fee: number
  to: string
  from: string
  timestamp: number
  confirmations: number
  status: 'pending' | 'confirmed' | 'failed'
}

interface WalletContextType {
  wallet: WalletState
  transactions: Transaction[]
  isConnected: boolean
  createWallet: (password: string) => Promise<string>
  importWallet: (mnemonic: string, password: string) => Promise<void>
  unlockWallet: (password: string) => Promise<boolean>
  lockWallet: () => void
  sendTransaction: (to: string, amount: number, fee?: number) => Promise<string>
  refreshBalance: () => Promise<void>
  refreshTransactions: () => Promise<void>
}

const WalletContext = createContext<WalletContextType | undefined>(undefined)

interface WalletProviderProps {
  children: ReactNode
}

export function WalletProvider({ children }: WalletProviderProps) {
  const [wallet, setWallet] = useState<WalletState>({
    address: null,
    balance: 0,
    isLocked: true,
    mnemonic: null,
    privateKey: null
  })
  const [transactions, setTransactions] = useState<Transaction[]>([])
  const [isConnected, setIsConnected] = useState(false)

  // Generate ZION address from seed
  const generateAddress = (seed: Buffer): string => {
    const hash = CryptoJS.SHA256(seed.toString('hex')).toString()
    return 'ZION' + hash.substring(0, 60) // ZION address format
  }

  // Generate private key from seed
  const generatePrivateKey = (seed: Buffer): string => {
    return CryptoJS.SHA256(seed.toString('hex') + 'private').toString()
  }

  const createWallet = async (password: string): Promise<string> => {
    try {
      const mnemonic = generateMnemonic(256) // 24 words
      const seed = mnemonicToSeedSync(mnemonic)
      const address = generateAddress(seed)
      const privateKey = generatePrivateKey(seed)
      
      // Encrypt and store wallet
      const encryptedMnemonic = CryptoJS.AES.encrypt(mnemonic, password).toString()
      const encryptedPrivateKey = CryptoJS.AES.encrypt(privateKey, password).toString()
      
      localStorage.setItem('zion_wallet_mnemonic', encryptedMnemonic)
      localStorage.setItem('zion_wallet_private_key', encryptedPrivateKey)
      localStorage.setItem('zion_wallet_address', address)
      
      setWallet({
        address,
        balance: 0,
        isLocked: false,
        mnemonic,
        privateKey
      })
      
      setIsConnected(true)
      return mnemonic
    } catch (error) {
      console.error('Failed to create wallet:', error)
      throw new Error('Wallet creation failed')
    }
  }

  const importWallet = async (mnemonic: string, password: string): Promise<void> => {
    try {
      const seed = mnemonicToSeedSync(mnemonic)
      const address = generateAddress(seed)
      const privateKey = generatePrivateKey(seed)
      
      // Encrypt and store wallet
      const encryptedMnemonic = CryptoJS.AES.encrypt(mnemonic, password).toString()
      const encryptedPrivateKey = CryptoJS.AES.encrypt(privateKey, password).toString()
      
      localStorage.setItem('zion_wallet_mnemonic', encryptedMnemonic)
      localStorage.setItem('zion_wallet_private_key', encryptedPrivateKey)
      localStorage.setItem('zion_wallet_address', address)
      
      setWallet({
        address,
        balance: 0,
        isLocked: false,
        mnemonic,
        privateKey
      })
      
      setIsConnected(true)
      await refreshBalance()
    } catch (error) {
      console.error('Failed to import wallet:', error)
      throw new Error('Wallet import failed')
    }
  }

  const unlockWallet = async (password: string): Promise<boolean> => {
    try {
      const encryptedMnemonic = localStorage.getItem('zion_wallet_mnemonic')
      const encryptedPrivateKey = localStorage.getItem('zion_wallet_private_key')
      const address = localStorage.getItem('zion_wallet_address')
      
      if (!encryptedMnemonic || !encryptedPrivateKey || !address) {
        return false
      }
      
      const mnemonic = CryptoJS.AES.decrypt(encryptedMnemonic, password).toString(CryptoJS.enc.Utf8)
      const privateKey = CryptoJS.AES.decrypt(encryptedPrivateKey, password).toString(CryptoJS.enc.Utf8)
      
      if (!mnemonic || !privateKey) {
        return false
      }
      
      setWallet({
        address,
        balance: wallet.balance,
        isLocked: false,
        mnemonic,
        privateKey
      })
      
      setIsConnected(true)
      await refreshBalance()
      return true
    } catch (error) {
      console.error('Failed to unlock wallet:', error)
      return false
    }
  }

  const lockWallet = (): void => {
    setWallet(prev => ({
      ...prev,
      isLocked: true,
      mnemonic: null,
      privateKey: null
    }))
  }

  const sendTransaction = async (to: string, amount: number, fee: number = 1000): Promise<string> => {
    if (!wallet.address || !wallet.privateKey || wallet.isLocked) {
      throw new Error('Wallet not available')
    }
    
    // TODO: Implement actual transaction sending via API
    console.log('Sending transaction:', { from: wallet.address, to, amount, fee })
    
    // Mock transaction ID
    const txId = CryptoJS.SHA256(`${wallet.address}-${to}-${amount}-${Date.now()}`).toString()
    
    // Add to pending transactions
    const newTx: Transaction = {
      id: txId,
      type: 'sent',
      amount,
      fee,
      to,
      from: wallet.address,
      timestamp: Date.now(),
      confirmations: 0,
      status: 'pending'
    }
    
    setTransactions(prev => [newTx, ...prev])
    
    return txId
  }

  const refreshBalance = async (): Promise<void> => {
    if (!wallet.address) return
    
    try {
      // TODO: Implement actual balance fetching via API
      console.log('Refreshing balance for:', wallet.address)
      
      // Mock balance for demo
      setWallet(prev => ({ ...prev, balance: 1000 + Math.random() * 9000 }))
    } catch (error) {
      console.error('Failed to refresh balance:', error)
    }
  }

  const refreshTransactions = async (): Promise<void> => {
    if (!wallet.address) return
    
    try {
      // TODO: Implement actual transaction fetching via API
      console.log('Refreshing transactions for:', wallet.address)
      
      // Mock transactions for demo
      const mockTxs: Transaction[] = [
        {
          id: 'tx1',
          type: 'received',
          amount: 500,
          fee: 0,
          to: wallet.address,
          from: 'ZION1234...5678',
          timestamp: Date.now() - 3600000,
          confirmations: 10,
          status: 'confirmed'
        },
        {
          id: 'tx2',
          type: 'sent',
          amount: 250,
          fee: 1000,
          to: 'ZION9876...4321',
          from: wallet.address,
          timestamp: Date.now() - 7200000,
          confirmations: 5,
          status: 'confirmed'
        }
      ]
      
      setTransactions(mockTxs)
    } catch (error) {
      console.error('Failed to refresh transactions:', error)
    }
  }

  // Check for existing wallet on mount
  useEffect(() => {
    const address = localStorage.getItem('zion_wallet_address')
    if (address) {
      setWallet(prev => ({ ...prev, address }))
    }
  }, [])

  const value: WalletContextType = {
    wallet,
    transactions,
    isConnected,
    createWallet,
    importWallet,
    unlockWallet,
    lockWallet,
    sendTransaction,
    refreshBalance,
    refreshTransactions
  }

  return <WalletContext.Provider value={value}>{children}</WalletContext.Provider>
}

export function useWallet() {
  const context = useContext(WalletContext)
  if (context === undefined) {
    throw new Error('useWallet must be used within a WalletProvider')
  }
  return context
}
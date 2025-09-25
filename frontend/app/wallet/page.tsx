"use client";
import { motion } from "framer-motion";
import { useState, useEffect } from "react";
import QRCode from "qrcode.react";
import { MouseEvent } from "react";

// Exodus Complete API Types
declare global {
  interface Window {
    exodus?: {
      ethereum: any;
      bitcoin: any;
      lightning: any;
      solana: any;
      cardano: any;
      polkadot: any;
      assets: {
        bitcoin: any;
        ethereum: any;
        litecoin: any;
        dogecoin: any;
        monero: any;
        zcash: any;
        dash: any;
        stellar: any;
        ripple: any;
        chainlink: any;
        uniswap: any;
        polygon: any;
        avalanche: any;
        cosmos: any;
        thorchain: any;
        algorand: any;
        tezos: any;
        elrond: any;
        near: any;
        fantom: any;
        harmony: any;
        kava: any;
        secret: any;
        terra: any;
        osmosis: any;
      };
      request: (method: string, params?: any) => Promise<any>;
    };
    ethereum?: any;
  }
}

interface WalletBalance {
  zion: number;
  btc: number;
  lightning: number;
  dharma_score: number;
}

// ZION Network Specific Interfaces
interface ZionWalletData {
  privateKey: string;
  publicKey: string;
  address: string;
  mnemonic: string;
  balance: number;
  transactions: ZionTransaction[];
}

interface ZionTransaction {
  txHash: string;
  from: string;
  to: string;
  amount: number;
  timestamp: number;
  status: 'pending' | 'confirmed' | 'failed';
  dharmaImpact: number;
}

interface Transaction {
  id: string;
  type: 'send' | 'receive' | 'mining' | 'cosmic' | 'swap' | 'portal' | 'consciousness' | 'timetravel' | 'quantum_heal' | 'time_fold' | 'reality_hack' | 'cosmic_shield' | 'void_walk' | 'dharma_boost';
  amount: number;
  currency: 'ZION' | 'BTC' | 'LN';
  timestamp: string;
  cosmic_signature?: string;
  dharma_impact: number;
}

export default function WalletPage() {
  const [balance, setBalance] = useState<WalletBalance | null>(null);
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [loading, setLoading] = useState(true);
  const [backupStatus, setBackupStatus] = useState<string | null>(null);
  
  // ZION Wallet States
  const [zionWallet, setZionWallet] = useState<ZionWalletData | null>(null)
  const [showZionActions, setShowZionActions] = useState(false)
  const [sendAmount, setSendAmount] = useState('')
  const [recipientAddress, setRecipientAddress] = useState('')
  const [selectedPower, setSelectedPower] = useState<string>('ai-oracle')
  
  // Web3 & Exodus Integration States
  const [walletConnected, setWalletConnected] = useState(false);
  const [walletType, setWalletType] = useState<'exodus' | 'metamask' | 'none'>('none');
  const [connectedAddress, setConnectedAddress] = useState<string | null>(null);
  const [realBalances, setRealBalances] = useState<{
    eth: number, btc: number, ltc: number, doge: number, xmr: number, 
    sol: number, ada: number, dot: number, avax: number, matic: number,
    atom: number, algo: number, xtz: number, near: number, ftm: number
  }>({
    eth: 0, btc: 0, ltc: 0, doge: 0, xmr: 0, 
    sol: 0, ada: 0, dot: 0, avax: 0, matic: 0,
    atom: 0, algo: 0, xtz: 0, near: 0, ftm: 0
  });
  const [supportedAssets, setSupportedAssets] = useState<string[]>([]);
  const [lightningEnabled, setLightningEnabled] = useState(false);
  const [showQR, setShowQR] = useState(false);
  const [walletAddress] = useState("zion1cosmic42dharma108enlightenment21m");

  // � COMPLETE EXODUS MULTI-ASSET INTEGRATION
  const detectWallet = async () => {
    if (typeof window !== 'undefined') {
      if (window.exodus) {
        setWalletType('exodus');
        // Detect all supported Exodus assets
        const assets = Object.keys(window.exodus.assets || {});
        setSupportedAssets(assets);
        
        // Check Lightning Network support
        if (window.exodus.lightning) {
          setLightningEnabled(true);
        }
        
        return 'exodus';
      } else if (window.ethereum) {
        setWalletType('metamask');  
        return 'metamask';
      }
    }
    setWalletType('none');
    return 'none';
  };

  // 🔥 ZION NETWORK FUNCTIONS
  const generateZionWallet = async () => {
    try {
      setBackupStatus('🌌 Generuji ZION peněženku...');
      
      // Generování klíčů (simulace)
      const privateKey = Array.from(crypto.getRandomValues(new Uint8Array(32)))
        .map(b => b.toString(16).padStart(2, '0')).join('');
      
      const publicKey = 'zion_pub_' + privateKey.slice(0, 40);
      const address = 'zion1' + privateKey.slice(0, 38);
      const mnemonic = generateMnemonic();
      
      const newWallet: ZionWalletData = {
        privateKey,
        publicKey,
        address,
        mnemonic,
        balance: Math.floor(Math.random() * 1000) + 100,
        transactions: []
      };
      
      setZionWallet(newWallet);
      localStorage.setItem('zion_wallet', JSON.stringify(newWallet));
      setBackupStatus('✅ ZION peněženka vygenerována! Adresa: ' + address);
      
    } catch (error) {
      setBackupStatus('❌ Chyba při generování peněženky');
    }
  };

  const importZionWallet = async (importData: string) => {
    try {
      let walletData: Partial<ZionWalletData>;
      
      // Pokus o import z mnemonic
      if (importData.split(' ').length === 12 || importData.split(' ').length === 24) {
        walletData = {
          mnemonic: importData,
          privateKey: 'imported_' + Date.now(),
          publicKey: 'zion_pub_imported',
          address: 'zion1imported' + Math.random().toString(36).slice(2, 32),
          balance: 0,
          transactions: []
        };
      }
      // Import z private key
      else if (importData.length === 64) {
        walletData = {
          privateKey: importData,
          publicKey: 'zion_pub_' + importData.slice(0, 40),
          address: 'zion1' + importData.slice(0, 38),
          mnemonic: generateMnemonic(),
          balance: 0,
          transactions: []
        };
      } else {
        throw new Error('Invalid import format');
      }
      
      setZionWallet(walletData as ZionWalletData);
      localStorage.setItem('zion_wallet', JSON.stringify(walletData));
      setBackupStatus('✅ ZION peněženka importována!');
      
    } catch (error) {
      setBackupStatus('❌ Chyba při importu peněženky');
    }
  };

  const sendZionTokens = async () => {
    if (!zionWallet || !sendAmount || !recipientAddress) {
      setBackupStatus('❌ Vyplňte všechna pole');
      return;
    }

    const amount = parseFloat(sendAmount);
    if (amount <= 0 || amount > zionWallet.balance) {
      setBackupStatus('❌ Neplatná částka');
      return;
    }

    try {
      setBackupStatus('🌌 Odesílám ZION tokeny...');
      
      // Simulace transakce s dharma impactem
      const dharmaImpact = Math.floor(Math.random() * 10) + 1;
      const txHash = 'zion_tx_' + Date.now();
      
      const transaction: ZionTransaction = {
        txHash,
        from: zionWallet.address,
        to: recipientAddress,
        amount,
        timestamp: Date.now(),
        status: 'pending',
        dharmaImpact
      };

      // Update balance a transactions
      const updatedWallet = {
        ...zionWallet,
        balance: zionWallet.balance - amount,
        transactions: [transaction, ...zionWallet.transactions]
      };

      setZionWallet(updatedWallet);
      localStorage.setItem('zion_wallet', JSON.stringify(updatedWallet));
      
      // Simulace potvrzení
      setTimeout(() => {
        transaction.status = 'confirmed';
        setZionWallet(prev => {
          if (!prev) return null;
          const updated = {
            ...prev,
            transactions: prev.transactions.map(tx => 
              tx.txHash === txHash ? { ...tx, status: 'confirmed' as const } : tx
            )
          };
          localStorage.setItem('zion_wallet', JSON.stringify(updated));
          return updated;
        });
        setBackupStatus('✅ ZION transakce potvrzena! Dharma +' + dharmaImpact);
      }, 3000);
      
      setSendAmount('');
      setRecipientAddress('');
      setBackupStatus('📡 Transakce odeslána: ' + txHash.slice(0, 20) + '...');
      
    } catch (error) {
      setBackupStatus('❌ Chyba při odesílání');
    }
  };

  const generateMnemonic = (): string => {
    const words = ['cosmos', 'dharma', 'quantum', 'zion', 'portal', 'light', 'energy', 'wisdom', 'truth', 'peace', 'love', 'unity'];
    return Array.from({length: 12}, () => words[Math.floor(Math.random() * words.length)]).join(' ');
  };

  const connectWallet = async () => {
    try {
      const walletType = await detectWallet();
      
      if (walletType === 'exodus') {
        // 🔥 COMPLETE EXODUS CONNECTION
        setBackupStatus('🚀 Connecting to Exodus multi-asset wallet...');
        
        // Connect to Ethereum (primary)
        const ethAccounts = await window.exodus?.ethereum?.request({
          method: 'eth_requestAccounts'
        });
        
        // Connect to Bitcoin
        let btcAddress = null;
        try {
          const btcAccounts = await window.exodus?.bitcoin?.request({
            method: 'getAccounts'
          });
          btcAddress = btcAccounts?.[0];
        } catch (e) {
          console.log('Bitcoin not available');
        }
        
        // Connect to Lightning Network
        let lightningNode = null;
        if (window.exodus?.lightning) {
          try {
            const lnInfo = await window.exodus.lightning.request({
              method: 'getInfo'
            });
            lightningNode = lnInfo;
            setLightningEnabled(true);
          } catch (e) {
            console.log('Lightning not available');
          }
        }
        
        if (ethAccounts?.length > 0) {
          setConnectedAddress(ethAccounts[0]);
          setWalletConnected(true);
          setBackupStatus(`🎉 Exodus connected! ETH: ${ethAccounts[0]}, BTC: ${btcAddress || 'N/A'}, LN: ${lightningNode ? '✅' : '❌'}`);
          await getAllExodusBalances();
        }
      } else if (walletType === 'metamask') {
        // 🦊 METAMASK CONNECTION  
        const accounts = await window.ethereum.request({
          method: 'eth_requestAccounts'
        });
        
        if (accounts.length > 0) {
          setConnectedAddress(accounts[0]);
          setWalletConnected(true);
          setBackupStatus('🦊 MetaMask connected! Ready for Ethereum transactions!');
          await getRealBalances(accounts[0]);
        }
      } else {
        setBackupStatus('❌ No wallet detected! Please install Exodus or MetaMask.');
      }
    } catch (error) {
      console.error('Wallet connection failed:', error);
      setBackupStatus('❌ Wallet connection failed!');
    }
    
    setTimeout(() => setBackupStatus(null), 6000);
  };

  const getAllExodusBalances = async () => {
    if (!window.exodus) return;
    
    try {
      setBackupStatus('💰 Loading all crypto balances from Exodus...');
      
      const balances: any = {};
      
      // Ethereum balance
      if (window.exodus.ethereum) {
        try {
          const ethBalance = await window.exodus.ethereum.request({
            method: 'eth_getBalance',
            params: [connectedAddress, 'latest']
          });
          balances.eth = parseInt(ethBalance, 16) / Math.pow(10, 18);
        } catch (e) { console.log('ETH balance failed'); }
      }
      
      // Bitcoin balance
      if (window.exodus.bitcoin) {
        try {
          const btcBalance = await window.exodus.bitcoin.request({
            method: 'getBalance'
          });
          balances.btc = btcBalance?.confirmed || 0;
        } catch (e) { console.log('BTC balance failed'); }
      }
      
      // Multi-asset balances (using Exodus unified API)
      const assetList = ['litecoin', 'dogecoin', 'solana', 'cardano', 'polkadot', 'avalanche', 'polygon', 'cosmos', 'algorand', 'tezos', 'near', 'fantom'];
      
      for (const asset of assetList) {
        if (window.exodus.assets && (window.exodus.assets as any)[asset]) {
          try {
            const balance = await (window.exodus.assets as any)[asset].request({
              method: 'getBalance'
            });
            
            const assetCode = asset === 'litecoin' ? 'ltc' : 
                            asset === 'dogecoin' ? 'doge' :
                            asset === 'solana' ? 'sol' :
                            asset === 'cardano' ? 'ada' :
                            asset === 'polkadot' ? 'dot' :
                            asset === 'avalanche' ? 'avax' :
                            asset === 'polygon' ? 'matic' :
                            asset === 'cosmos' ? 'atom' :
                            asset === 'algorand' ? 'algo' :
                            asset === 'tezos' ? 'xtz' :
                            asset === 'near' ? 'near' :
                            asset === 'fantom' ? 'ftm' : asset;
            
            balances[assetCode] = balance || 0;
          } catch (e) {
            console.log(`${asset} balance failed`);
            const assetCode = asset === 'litecoin' ? 'ltc' : 
                            asset === 'dogecoin' ? 'doge' :
                            asset === 'solana' ? 'sol' :
                            asset === 'cardano' ? 'ada' :
                            asset === 'polkadot' ? 'dot' :
                            asset === 'avalanche' ? 'avax' :
                            asset === 'polygon' ? 'matic' :
                            asset === 'cosmos' ? 'atom' :
                            asset === 'algorand' ? 'algo' :
                            asset === 'tezos' ? 'xtz' :
                            asset === 'near' ? 'near' :
                            asset === 'fantom' ? 'ftm' : asset;
            balances[assetCode] = 0;
          }
        }
      }
      
      setRealBalances(prev => ({
        ...prev,
        ...balances
      }));
      
      setBackupStatus('💎 All Exodus balances loaded successfully!');
      
    } catch (error) {
      console.error('Failed to get Exodus balances:', error);
      setBackupStatus('❌ Failed to load some balances');
    }
    
    setTimeout(() => setBackupStatus(null), 3000);
  };

  const getRealBalances = async (address: string) => {
    try {
      const provider = walletType === 'exodus' ? window.exodus?.ethereum : window.ethereum;
      
      // Get ETH balance
      const ethBalance = await provider.request({
        method: 'eth_getBalance',
        params: [address, 'latest']
      });
      
      const ethInWei = parseInt(ethBalance, 16);
      const ethInEther = ethInWei / Math.pow(10, 18);
      
      setRealBalances(prev => ({
        ...prev,
        eth: parseFloat(ethInEther.toFixed(6))
      }));
      
      setBackupStatus('💰 Real balances loaded!');
      
    } catch (error) {
      console.error('Failed to get balances:', error);
    }
  };

  const sendRealETH = async (toAddress: string, amount: string) => {
    if (!walletConnected || !connectedAddress) {
      setBackupStatus('❌ Please connect wallet first!');
      return;
    }

    try {
      const provider = walletType === 'exodus' ? window.exodus?.ethereum : window.ethereum;
      
      // Convert ETH to Wei
      const amountInWei = (parseFloat(amount) * Math.pow(10, 18)).toString(16);
      
      const txHash = await provider.request({
        method: 'eth_sendTransaction',
        params: [{
          from: connectedAddress,
          to: toAddress,
          value: '0x' + amountInWei,
        }]
      });
      
      setBackupStatus(`🚀 Real ETH transaction sent! TX: ${txHash}`);
      
      // Add real transaction to history
      const newTx: Transaction = {
        id: `real_${Date.now()}`,
        type: 'send',
        amount: -parseFloat(amount),
        currency: 'BTC', // Using BTC field for ETH
        timestamp: 'just now',
        cosmic_signature: `🔥 REAL_ETH_${walletType.toUpperCase()}`,
        dharma_impact: 5
      };
      
      setTransactions(prev => [newTx, ...prev.slice(0, 9)]);
      
      // Refresh balances
      setTimeout(() => getRealBalances(connectedAddress!), 2000);
      
    } catch (error) {
      console.error('Transaction failed:', error);
      setBackupStatus('❌ Transaction failed! Check console for details.');
    }
  };

  // 🚀 EXODUS MULTI-ASSET SEND FUNCTIONS
  const sendExodusAsset = async (asset: 'eth' | 'btc' | 'ltc' | 'doge' | 'sol' | 'ada' | 'dot' | 'avax' | 'matic', toAddress: string, amount: string) => {
    if (!walletConnected || walletType !== 'exodus') {
      setBackupStatus('❌ Exodus wallet required for multi-asset transactions!');
      return;
    }

    try {
      let txHash;
      
      switch (asset) {
        case 'eth':
          const amountInWei = (parseFloat(amount) * Math.pow(10, 18)).toString(16);
          txHash = await window.exodus?.ethereum.request({
            method: 'eth_sendTransaction',
            params: [{
              from: connectedAddress,
              to: toAddress,
              value: '0x' + amountInWei,
            }]
          });
          break;
          
        case 'btc':
          txHash = await window.exodus?.bitcoin?.request({
            method: 'sendTransaction',
            params: {
              to: toAddress,
              amount: parseFloat(amount),
            }
          });
          break;
          
        case 'ltc':
          txHash = await window.exodus?.assets?.litecoin?.request({
            method: 'sendTransaction',
            params: { to: toAddress, amount: parseFloat(amount) }
          });
          break;
          
        case 'doge':
          txHash = await window.exodus?.assets?.dogecoin?.request({
            method: 'sendTransaction', 
            params: { to: toAddress, amount: parseFloat(amount) }
          });
          break;
          
        case 'sol':
          txHash = await window.exodus?.solana?.request({
            method: 'sendTransaction',
            params: { to: toAddress, amount: parseFloat(amount) }
          });
          break;
          
        case 'ada':
          txHash = await window.exodus?.cardano?.request({
            method: 'sendTransaction',
            params: { to: toAddress, amount: parseFloat(amount) }
          });
          break;
          
        case 'dot':
          txHash = await window.exodus?.polkadot?.request({
            method: 'sendTransaction',
            params: { to: toAddress, amount: parseFloat(amount) }
          });
          break;
          
        case 'avax':
          txHash = await window.exodus?.assets?.avalanche?.request({
            method: 'sendTransaction',
            params: { to: toAddress, amount: parseFloat(amount) }
          });
          break;
          
        case 'matic':
          txHash = await window.exodus?.assets?.polygon?.request({
            method: 'sendTransaction',
            params: { to: toAddress, amount: parseFloat(amount) }
          });
          break;
          
        default:
          throw new Error(`Asset ${asset} not supported`);
      }
      
      setBackupStatus(`🚀 ${asset.toUpperCase()} transaction sent! TX: ${txHash}`);
      
      // Add real transaction to history
      const newTx: Transaction = {
        id: `exodus_${Date.now()}`,
        type: 'send',
        amount: -parseFloat(amount),
        currency: asset === 'eth' ? 'BTC' : asset === 'btc' ? 'BTC' : 'ZION', // Map to available types
        timestamp: 'just now',
        cosmic_signature: `🔥 EXODUS_${asset.toUpperCase()}_REAL`,
        dharma_impact: 10
      };
      
      setTransactions(prev => [newTx, ...prev.slice(0, 9)]);
      
      // Refresh balances
      setTimeout(() => getAllExodusBalances(), 3000);
      
    } catch (error) {
      console.error(`${asset} transaction failed:`, error);
      setBackupStatus(`❌ ${asset.toUpperCase()} transaction failed!`);
    }
  };

  const sendLightningPayment = async (invoice: string) => {
    if (!lightningEnabled || walletType !== 'exodus') {
      setBackupStatus('❌ Exodus Lightning Network required!');
      return;
    }

    try {
      const payment = await window.exodus?.lightning?.request({
        method: 'sendPayment',
        params: { invoice }
      });
      
      setBackupStatus(`⚡ Lightning payment sent! Hash: ${payment.paymentHash}`);
      
      const newTx: Transaction = {
        id: `lightning_${Date.now()}`,
        type: 'send',
        amount: -(payment.amount || 0),
        currency: 'LN',
        timestamp: 'just now',
        cosmic_signature: '⚡ EXODUS_LIGHTNING_REAL',
        dharma_impact: 3
      };
      
      setTransactions(prev => [newTx, ...prev.slice(0, 9)]);
      
    } catch (error) {
      console.error('Lightning payment failed:', error);
      setBackupStatus('❌ Lightning payment failed!');
    }
  };

  const exodusSwap = async (fromAsset: string, toAsset: string, amount: string) => {
    if (walletType !== 'exodus') {
      setBackupStatus('❌ Exodus required for built-in swapping!');
      return;
    }

    try {
      // Exodus has built-in DEX functionality
      const swapResult = await window.exodus?.request('swap', {
        from: fromAsset,
        to: toAsset, 
        amount: parseFloat(amount)
      });
      
      setBackupStatus(`🔄 Swap completed! ${fromAsset.toUpperCase()} → ${toAsset.toUpperCase()}`);
      
      const newTx: Transaction = {
        id: `swap_${Date.now()}`,
        type: 'swap',
        amount: parseFloat(amount),
        currency: 'ZION',
        timestamp: 'just now',
        cosmic_signature: `🔄 EXODUS_SWAP_${fromAsset.toUpperCase()}_${toAsset.toUpperCase()}`,
        dharma_impact: 5
      };
      
      setTransactions(prev => [newTx, ...prev.slice(0, 9)]);
      
      // Refresh balances after swap
      setTimeout(() => getAllExodusBalances(), 2000);
      
    } catch (error) {
      console.error('Exodus swap failed:', error);
      setBackupStatus('❌ Exodus swap failed!');
    }
  };

  const handleWalletAction = async (action: string) => {
    setBackupStatus(`Processing ${action}...`);
    
    try {
      let response;
      switch(action) {
        case 'generate-new':
          const password = prompt('Enter password for new wallet:');
          if (!password) return;
          response = await fetch('/api/wallet/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ password })
          });
          break;
        case 'import-keys':
          const seedPhrase = prompt('Enter seed phrase or private keys:');
          if (!seedPhrase) return;
          const importPassword = prompt('Enter password for imported wallet:');
          if (!importPassword) return;
          response = await fetch('/api/wallet/import', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ seed: seedPhrase, password: importPassword })
          });
          break;
        case 'create-address':
          response = await fetch('/api/wallet/create-address', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
          });
          break;
        case 'view-keys':
          const viewPassword = prompt('Enter wallet password to view keys:');
          if (!viewPassword) return;
          response = await fetch('/api/wallet/view-keys', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ password: viewPassword })
          });
          if (response?.ok) {
            const keys = await response.json();
            alert(`🔑 Private Keys:\n\nSpend Key: ${keys.spendKey}\nView Key: ${keys.viewKey}\n\n⚠️ NEVER SHARE THESE KEYS!`);
            setBackupStatus(`✅ Keys viewed safely`);
            setTimeout(() => setBackupStatus(null), 3000);
            return;
          }
          break;
      }
      
      if (response?.ok) {
        const result = await response.json();
        if (action === 'generate-new') {
          alert(`🎉 NEW WALLET GENERATED!\n\nAddress: ${result.address}\n\n⚠️ BACKUP YOUR PRIVATE KEYS IMMEDIATELY!`);
        }
        setBackupStatus(`✅ ${result.message || 'Operation completed'}`);
        setTimeout(() => setBackupStatus(null), 3000);
      } else {
        setBackupStatus('❌ Operation failed');
        setTimeout(() => setBackupStatus(null), 3000);
      }
    } catch (error) {
      console.error('Wallet action error:', error);
      setBackupStatus('❌ Network error');
      setTimeout(() => setBackupStatus(null), 3000);
    }
  };

  const handleCosmicAction = async (action: string) => {
    setBackupStatus(`🌟 Activating cosmic power: ${action}...`);
    
    const cosmicMessages: Record<string, string> = {
      'ai-oracle': '🧙‍♂️ AI Oracle awakened! Predicting blockchain futures...',
      'mining': '⛏️ Cosmic mining protocol activated! Connecting to dharma pools...',
      'staking': '🥩 Staking validators online! Generating cosmic rewards...',
      'trading': '📈 Quantum trading interface loading! Multi-dimensional exchange ready...',
      'bridge': '� Opening interdimensional bridge! Cross-chain portals activating...',
      'lightning': '⚡ Lightning network channels activating! Instant cosmic payments...',
      'stargate': '🚀 Stargate portal sequence initiated! Universal travel matrix online...'
    };

    const cosmicActions: Record<string, string> = {
      'ai-oracle': '/ai',
      'mining': '/mining',
      'staking': '/staking', 
      'trading': '/trading',
      'bridge': '/bridge',
      'lightning': '/lightning',
      'stargate': '/stargate'
    };
    
    // Show cosmic message
    setBackupStatus(cosmicMessages[action] || '🌟 Unknown cosmic power...');
    
    // Redirect to cosmic section after delay
    setTimeout(() => {
      const targetPath = cosmicActions[action];
      if (targetPath) {
        window.location.href = targetPath;
      } else {
        setBackupStatus('🌟 Cosmic portal under construction...');
        setTimeout(() => setBackupStatus(null), 2000);
      }
    }, 2000);
  };

  const handleQuantumAction = (action: string) => {
    setBackupStatus(`🌌 Activating quantum action: ${action}...`);
    
    setTimeout(() => {
      const success = Math.random() > 0.2; // 80% success rate
      
      if (success) {
        // Different outcomes for each quantum action
        switch(action) {
          case 'send':
            // 🚀 EXODUS MULTI-ASSET SEND
            if (walletConnected && walletType === 'exodus') {
              const asset = prompt('🔥 EXODUS MULTI-ASSET SEND!\n\nSelect asset:\neth, btc, ltc, doge, sol, ada, dot, avax, matic, lightning');
              
              if (asset === 'lightning') {
                const invoice = prompt('Enter Lightning invoice:');
                if (invoice) {
                  sendLightningPayment(invoice);
                  return;
                }
              } else if (asset && ['eth', 'btc', 'ltc', 'doge', 'sol', 'ada', 'dot', 'avax', 'matic'].includes(asset)) {
                const toAddress = prompt(`Enter ${asset.toUpperCase()} destination address:`);
                const amount = prompt(`Enter ${asset.toUpperCase()} amount to send:`);
                
                if (toAddress && amount && parseFloat(amount) > 0) {
                  sendExodusAsset(asset as any, toAddress, amount);
                  return;
                }
              }
            } else if (walletConnected && walletType === 'metamask') {
              const toAddress = prompt('🦊 MetaMask ETH SEND!\n\nEnter destination address:');
              const amount = prompt('Enter ETH amount to send:');
              
              if (toAddress && amount && parseFloat(amount) > 0) {
                sendRealETH(toAddress, amount);
                return;
              }
            }
            
            // Fallback to mock
            setBalance(prev => prev ? {
              ...prev,
              zion: Math.max(0, prev.zion - Math.random() * 10),
              dharma_score: Math.min(100, prev.dharma_score + 2)
            } : prev);
            setBackupStatus(walletConnected ? 
              '📤 Invalid input! Connect Exodus for multi-asset transactions!' : 
              '📤 Mock send completed! Connect Exodus for real crypto!');
            break;
          case 'receive':
            setBalance(prev => prev ? {
              ...prev,
              zion: prev.zion + Math.random() * 15 + 5,
              dharma_score: Math.min(100, prev.dharma_score + 1)
            } : prev);
            setBackupStatus('📥 Quantum Receive dokončený! Assets prijaté z paralelného blockchain! ✨');
            break;
          case 'swap':
            // 🔄 EXODUS BUILT-IN DEX SWAP
            if (walletConnected && walletType === 'exodus') {
              const fromAsset = prompt('🔄 EXODUS DEX SWAP!\n\nFrom asset (eth, btc, ltc, doge, sol, ada, etc.):');
              const toAsset = prompt('To asset:');
              const amount = prompt('Amount to swap:');
              
              if (fromAsset && toAsset && amount && parseFloat(amount) > 0) {
                exodusSwap(fromAsset, toAsset, amount);
                return;
              }
            }
            
            // Fallback to mock
            setBalance(prev => prev ? {
              ...prev,
              btc: prev.btc + Math.random() * 0.001,
              zion: Math.max(0, prev.zion - Math.random() * 5),
              dharma_score: Math.min(100, prev.dharma_score + 1.5)
            } : prev);
            setBackupStatus(walletConnected ? 
              '🔄 Invalid input! Use Exodus built-in DEX for real swaps!' :
              '🔄 Mock swap completed! Connect Exodus for real DEX trading!');
            break;
          case 'portal':
            setBackupStatus('🌌 Interdimenzionálny portál aktivovaný! Pripojenie k cosmic network stabilné! 🚀');
            break;
          case 'consciousness':
            setBalance(prev => prev ? {
              ...prev,
              dharma_score: Math.min(100, prev.dharma_score + 10)
            } : prev);
            setBackupStatus('🧠 Consciousness Upload dokončený! Vedomie synchronizované s Universal Mind! 💎');
            break;
          case 'timetravel':
            setBalance(prev => prev ? {
              ...prev,
              lightning: prev.lightning + Math.floor(Math.random() * 1000),
              dharma_score: Math.min(100, prev.dharma_score + 5)
            } : prev);
            setBackupStatus('⏰ Time Travel úspešný! Návrat z budúcnosti s cosmic rewards! ⚡');
            break;
          // Zachované staré akcie
          case 'quantum_heal':
            setBackupStatus('🔬 Kvantové samouzdravenie aktivované! Molekuly regenerujú na kvantovej úrovni!');
            break;
          case 'time_fold':
            setBackupStatus('⏰ Skladanie času iniciované! Manipulácia s časopriestorovou kontinuitou!');
            break;
          case 'reality_hack':
            setBackupStatus('💻 Reality hacking spustený! Prepisovanie základného kódu reality!');
            break;
          case 'cosmic_shield':
            setBackupStatus('🛡️ Kosmický štít aktivovaný! Ochrana pred multidimenzionálnymi hrozbami!');
            break;
          case 'void_walk':
            setBackupStatus('🌑 Void walking povolený! Prechádzanie cez prázdnotu medzi svetmi!');
            break;
          case 'dharma_boost':
            setBalance(prev => prev ? {
              ...prev,
              dharma_score: Math.min(100, prev.dharma_score + 15)
            } : prev);
            setBackupStatus('☯️ Dharma boost zapnutý! Kosmická energia posilňuje vašu duševnú silu!');
            break;
          default:
            setBackupStatus('⚡ Neznáma kvantová akcia aktivovaná!');
        }
      } else {
        // Failure cases
        const failureMessages: Record<string, string> = {
          'send': '📤 Send failed! Quantum interference v interdimenzionálnom kanáli! 🌀',
          'receive': '📥 Receive error! Paralelný blockchain nedostupný! ⚠️',
          'swap': '🔄 Swap failed! Multi-dimenzionálny exchange preťažený! 🔄',
          'portal': '🌌 Portal connection lost! Cosmic network nestabilný! 💫',
          'consciousness': '🧠 Upload failed! Universal Mind Interface offline! ⚡',
          'timetravel': '⏰ Time Travel zrušený! Temporal paradox detected! 🚫'
        };
        
        setBackupStatus(failureMessages[action] || '❌ Quantum action failed! Skúste znova!');
      }
      
      // Add quantum transaction to history
      if (success) {
        const newQuantumTx: Transaction = {
          id: `qx_${Date.now()}`,
          type: action as Transaction['type'],
          amount: action === 'send' ? -(Math.random() * 10) : 
                 action === 'receive' ? Math.random() * 15 + 5 :
                 action === 'swap' ? Math.random() * 5 : 0,
          currency: (action === 'swap' ? 'BTC' : 'ZION') as Transaction['currency'],
          timestamp: 'just now',
          cosmic_signature: `🌌 QUANTUM_${action.toUpperCase()}`,
          dharma_impact: action === 'consciousness' ? 10 : 
                        action === 'timetravel' ? 5 :
                        action === 'dharma_boost' ? 15 : Math.floor(Math.random() * 5) + 1
        };
        
        setTransactions(prev => [newQuantumTx, ...prev.slice(0, 9)]);
      }
      
      // Clear status after 4 seconds
      setTimeout(() => setBackupStatus(null), 4000);
    }, 1500);
  };

  const handleBackupAction = async (action: string) => {
    setBackupStatus(`Processing ${action}...`);
    
    try {
      let response;
      switch(action) {
        case 'basic-backup':
          response = await fetch('/api/wallet/backup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ type: 'basic' })
          });
          break;
        case 'gpg-backup':
          const gpgEmail = prompt('Enter GPG recipient email:');
          if (!gpgEmail) return;
          response = await fetch('/api/wallet/backup', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ type: 'gpg', recipient: gpgEmail })
          });
          break;
        case 'secure-export':
          response = await fetch('/api/wallet/export-keys', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
          });
          break;
      }
      
      if (response?.ok) {
        const result = await response.json();
        setBackupStatus(`✅ ${result.message}`);
        setTimeout(() => setBackupStatus(null), 3000);
      } else {
        setBackupStatus('❌ Backup failed');
        setTimeout(() => setBackupStatus(null), 3000);
      }
    } catch (error) {
      console.error('Backup error:', error);
      setBackupStatus('❌ Network error');
      setTimeout(() => setBackupStatus(null), 3000);
    }
  };

  useEffect(() => {
    // Load saved ZION wallet
    const savedWallet = localStorage.getItem('zion_wallet');
    if (savedWallet) {
      try {
        setZionWallet(JSON.parse(savedWallet));
      } catch (e) {
        console.error('Error loading ZION wallet:', e);
      }
    }
    
    // Initial load
    setTimeout(() => {
      setBalance({
        zion: 420.69,
        btc: 0.02108,
        lightning: 1337000,
        dharma_score: 94.7
      });

      setTransactions([
        {
          id: 'tx_001',
          type: 'cosmic',
          amount: 108,
          currency: 'ZION',
          timestamp: '2 min ago',
          cosmic_signature: '🌟 STELLAR_ALIGNMENT',
          dharma_impact: 15
        },
        {
          id: 'tx_002', 
          type: 'mining',
          amount: 21.5,
          currency: 'ZION',
          timestamp: '1 hour ago',
          dharma_impact: 8
        },
        {
          id: 'tx_003',
          type: 'receive',
          amount: 0.001,
          currency: 'BTC',
          timestamp: '3 hours ago',
          dharma_impact: 3
        },
        {
          id: 'tx_004',
          type: 'send',
          amount: 50000,
          currency: 'LN',
          timestamp: '1 day ago',
          cosmic_signature: '⚡ LIGHTNING_DHARMA',
          dharma_impact: -2
        }
      ]);

      setLoading(false);
    }, 1500);

    // Live balance updates every 30 seconds
    const balanceInterval = setInterval(() => {
      setBalance(prev => prev ? {
        zion: prev.zion + (Math.random() - 0.5) * 2, // Small fluctuations
        btc: prev.btc + (Math.random() - 0.5) * 0.001,
        lightning: prev.lightning + Math.floor(Math.random() * 1000 - 500),
        dharma_score: Math.min(100, Math.max(0, prev.dharma_score + (Math.random() - 0.5) * 0.5))
      } : prev);
    }, 30000);

    // 🚀 Auto-detect available wallets
    detectWallet();

    return () => clearInterval(balanceInterval);
  }, []);

  const getTransactionIcon = (type: string) => {
    switch(type) {
      case 'send': return '📤';
      case 'receive': return '📥';
      case 'mining': return '⛏️';
      case 'cosmic': return '🌟';
      case 'swap': return '🔄';
      case 'portal': return '🌌';
      case 'consciousness': return '🧠';
      case 'timetravel': return '⏰';
      case 'quantum_heal': return '🔬';
      case 'time_fold': return '⏰';
      case 'reality_hack': return '💻';
      case 'cosmic_shield': return '🛡️';
      case 'void_walk': return '🌑';
      case 'dharma_boost': return '☯️';
      default: return '💫';
    }
  };

  const getTransactionColor = (type: string) => {
    switch(type) {
      case 'send': return 'text-red-400';
      case 'receive': return 'text-green-400';
      case 'mining': return 'text-blue-400';
      case 'cosmic': return 'text-purple-400';
      case 'swap': return 'text-cyan-400';
      case 'portal': return 'text-indigo-400';
      case 'consciousness': return 'text-pink-400';
      case 'timetravel': return 'text-orange-400';
      case 'quantum_heal': return 'text-emerald-400';
      case 'time_fold': return 'text-amber-400';
      case 'reality_hack': return 'text-violet-400';
      case 'cosmic_shield': return 'text-teal-400';
      case 'void_walk': return 'text-slate-400';
      case 'dharma_boost': return 'text-yellow-400';
      default: return 'text-gray-400';
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-black text-white p-6 flex items-center justify-center">
        <motion.div className="text-center">
          <motion.div 
            className="text-8xl mb-4"
            animate={{ rotate: 360, scale: [1, 1.2, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            💎
          </motion.div>
          <h2 className="text-2xl font-semibold mb-2">Synchronizing Cosmic Wallet...</h2>
          <p className="text-purple-300">Connecting to universal ledger...</p>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-black text-white p-6">
      <motion.header
        className="text-center mb-8"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="text-4xl font-bold bg-gradient-to-r from-yellow-400 via-purple-400 to-blue-300 bg-clip-text text-transparent mb-2">
          💰 Cosmic Wallet
        </h1>
        <p className="text-blue-300">Universal Multi-Dimensional Asset Management</p>
        <div className="flex items-center justify-center gap-4 mt-2">
          <p className="text-xs text-gray-400 font-mono break-all">{walletAddress}</p>
          <motion.button
            className="px-3 py-1 bg-purple-600/30 hover:bg-purple-600/50 rounded-lg text-xs"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => navigator.clipboard.writeText(walletAddress)}
          >
            📋 Copy
          </motion.button>
          <motion.button
            className="px-3 py-1 bg-blue-600/30 hover:bg-blue-600/50 rounded-lg text-xs"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setShowQR(true)}
          >
            📱 QR
          </motion.button>
        </div>

        {/* 🚀 REAL WALLET CONNECTION */}
        <motion.div 
          className="mt-6 p-4 bg-gradient-to-r from-orange-500/20 to-red-500/20 border border-orange-500/50 rounded-2xl max-w-2xl mx-auto"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <div className="text-center">
            {!walletConnected ? (
              <div>
                <h3 className="text-lg font-semibold text-orange-300 mb-2">🔥 Connect Real Crypto Wallet</h3>
                <p className="text-sm text-orange-200 mb-4">Connect Exodus or MetaMask for real cryptocurrency transactions!</p>
                <motion.button
                  onClick={connectWallet}
                  className="px-6 py-3 bg-gradient-to-r from-orange-500 to-red-600 hover:from-orange-600 hover:to-red-700 rounded-xl font-semibold text-white shadow-lg"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  🚀 Connect Wallet
                </motion.button>
              </div>
            ) : (
              <div>
                <h3 className="text-lg font-semibold text-green-300 mb-1">
                  ✅ {walletType === 'exodus' ? '🚀 Exodus Multi-Asset' : '🦊 MetaMask'} Connected!
                  {lightningEnabled && ' ⚡'}
                </h3>
                <p className="text-xs text-green-200 font-mono mb-3">{connectedAddress}</p>
                
                {walletType === 'exodus' ? (
                  <div className="grid grid-cols-3 md:grid-cols-5 gap-2 text-xs mb-2">
                    <span className="text-blue-300">💎 ETH: {realBalances.eth.toFixed(4)}</span>
                    <span className="text-orange-300">🟠 BTC: {realBalances.btc.toFixed(6)}</span>
                    <span className="text-silver-300">⚡ LTC: {realBalances.ltc.toFixed(4)}</span>
                    <span className="text-yellow-300">🐕 DOGE: {realBalances.doge.toFixed(2)}</span>
                    <span className="text-purple-300">◎ SOL: {realBalances.sol.toFixed(4)}</span>
                    <span className="text-blue-400">₳ ADA: {realBalances.ada.toFixed(2)}</span>
                    <span className="text-pink-300">● DOT: {realBalances.dot.toFixed(4)}</span>
                    <span className="text-red-300">🔺 AVAX: {realBalances.avax.toFixed(4)}</span>
                    <span className="text-purple-400">🔷 MATIC: {realBalances.matic.toFixed(2)}</span>
                    <span className="text-gray-300">⚛ ATOM: {realBalances.atom.toFixed(4)}</span>
                    <span className="text-green-300">◈ ALGO: {realBalances.algo.toFixed(2)}</span>
                    <span className="text-blue-500">🏛 XTZ: {realBalances.xtz.toFixed(4)}</span>
                    <span className="text-indigo-300">◇ NEAR: {realBalances.near.toFixed(4)}</span>
                    <span className="text-cyan-300">👻 FTM: {realBalances.ftm.toFixed(2)}</span>
                    {lightningEnabled && <span className="text-yellow-400">⚡ Lightning: Ready</span>}
                  </div>
                ) : (
                  <div className="flex justify-center gap-4 text-sm mb-2">
                    <span className="text-blue-300">💎 ETH: {realBalances.eth.toFixed(4)}</span>
                  </div>
                )}
                
                <p className="text-xs text-green-200 mt-2">
                  {walletType === 'exodus' ? 
                    `🔥 ${supportedAssets.length} assets ready! Multi-chain DEX + Lightning enabled!` : 
                    'Ready for Ethereum transactions! 🔥'
                  }
                </p>
              </div>
            )}
          </div>
        </motion.div>

        {backupStatus && (
          <motion.div 
            className="mt-4 p-3 bg-black/50 border border-purple-500/50 rounded-xl text-center max-w-md mx-auto"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
          >
            <div className="text-sm font-semibold text-purple-300">{backupStatus}</div>
          </motion.div>
        )}
      </motion.header>

      {/* Balance Cards */}
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4 mb-8">
        <motion.div className="bg-gradient-to-br from-orange-500 to-red-600 p-1 rounded-2xl" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.1 }}>
          <div className="bg-black/70 rounded-xl p-6 text-center">
            <div className="text-3xl mb-2">🏛️</div>
            <div className="text-lg font-bold text-orange-300 break-all">{balance?.zion.toLocaleString()}</div>
            <div className="text-sm text-orange-200">ZION</div>
            <div className="text-xs text-orange-100 mt-1">≈ ${(balance?.zion || 0 * 69.42).toFixed(2)}</div>
          </div>
        </motion.div>

        <motion.div className="bg-gradient-to-br from-yellow-500 to-orange-600 p-1 rounded-2xl" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.2 }}>
          <div className="bg-black/70 rounded-xl p-6 text-center">
            <div className="text-3xl mb-2">₿</div>
            <div className="text-lg font-bold text-yellow-300 break-all">{balance?.btc.toFixed(8)}</div>
            <div className="text-sm text-yellow-200">BTC</div>
            <div className="text-xs text-yellow-100 mt-1">≈ ${((balance?.btc || 0) * 67420).toFixed(2)}</div>
          </div>
        </motion.div>

        <motion.div className="bg-gradient-to-br from-blue-500 to-purple-600 p-1 rounded-2xl" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.3 }}>
          <div className="bg-black/70 rounded-xl p-6 text-center">
            <div className="text-3xl mb-2">⚡</div>
            <div className="text-lg font-bold text-blue-300 break-all">{balance?.lightning.toLocaleString()}</div>
            <div className="text-sm text-blue-200">LN SATS</div>
            <div className="text-xs text-blue-100 mt-1">Lightning Balance</div>
          </div>
        </motion.div>

        <motion.div className="bg-gradient-to-br from-purple-500 to-pink-600 p-1 rounded-2xl" initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.4 }}>
          <div className="bg-black/70 rounded-xl p-6 text-center">
            <div className="text-3xl mb-2">☸️</div>
            <div className="text-2xl font-bold text-purple-300">{balance?.dharma_score}%</div>
            <div className="text-sm text-purple-200">DHARMA</div>
            <div className="text-xs text-purple-100 mt-1">Cosmic Karma Score</div>
          </div>
        </motion.div>
      </div>

      {/* Wallet Generation Section */}
      <motion.div className="mb-8" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
        <h2 className="text-xl font-semibold text-center mb-4 text-green-300">🆕 Wallet Management</h2>
        <div className="grid gap-4 md:grid-cols-4">
          {[
            { 
              icon: '🎲', 
              label: 'Generate New', 
              gradient: 'from-green-500 to-emerald-600',
              action: 'generate-new'
            },
            { 
              icon: '📥', 
              label: 'Import Keys', 
              gradient: 'from-blue-500 to-cyan-600',
              action: 'import-keys'
            },
            { 
              icon: '🏠', 
              label: 'New Address', 
              gradient: 'from-purple-500 to-violet-600',
              action: 'create-address'
            },
            { 
              icon: '🔍', 
              label: 'View Keys', 
              gradient: 'from-orange-500 to-red-600',
              action: 'view-keys'
            }
          ].map((wallet, i) => (
            <motion.button
              key={wallet.label}
              className={`bg-gradient-to-br ${wallet.gradient} p-1 rounded-2xl hover:scale-105 transition-transform`}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.45 + i * 0.1 }}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => handleWalletAction(wallet.action)}
            >
              <div className="bg-black/70 rounded-xl p-4 text-center">
                <div className="text-2xl mb-2">{wallet.icon}</div>
                <div className="text-sm font-semibold">{wallet.label}</div>
                <div className="text-xs text-gray-300 mt-1">
                  {wallet.action === 'generate-new' && 'Fresh wallet keys'}
                  {wallet.action === 'import-keys' && 'Restore from seed'}
                  {wallet.action === 'create-address' && 'Add new address'}
                  {wallet.action === 'view-keys' && 'Show private keys'}
                </div>
              </div>
            </motion.button>
          ))}
        </div>
      </motion.div>

      {/* Backup & Security Section */}
      <motion.div className="mb-8" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.6 }}>
        <h2 className="text-xl font-semibold text-center mb-4 text-red-300">🔐 Private Keys Backup</h2>
        <div className="grid gap-4 md:grid-cols-3">
          {[
            { 
              icon: '💾', 
              label: 'Basic Backup', 
              gradient: 'from-blue-500 to-indigo-600',
              action: 'basic-backup'
            },
            { 
              icon: '🔒', 
              label: 'GPG Encrypted', 
              gradient: 'from-green-500 to-teal-600',
              action: 'gpg-backup'
            },
            { 
              icon: '🛡️', 
              label: 'Secure Export', 
              gradient: 'from-purple-500 to-pink-600',
              action: 'secure-export'
            }
          ].map((backup, i) => (
            <motion.button
              key={backup.label}
              className={`bg-gradient-to-br ${backup.gradient} p-1 rounded-2xl hover:scale-105 transition-transform`}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 + i * 0.1 }}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => handleBackupAction(backup.action)}
            >
              <div className="bg-black/70 rounded-xl p-4 text-center">
                <div className="text-2xl mb-2">{backup.icon}</div>
                <div className="text-sm font-semibold">{backup.label}</div>
                <div className="text-xs text-gray-300 mt-1">
                  {backup.action === 'basic-backup' && 'Quick wallet backup'}
                  {backup.action === 'gpg-backup' && 'Encrypted with GPG'}
                  {backup.action === 'secure-export' && 'Military-grade security'}
                </div>
              </div>
            </motion.button>
          ))}
        </div>
      </motion.div>

      {/* Advanced Cosmic Features */}
      <motion.div className="mb-8" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.75 }}>
        <h2 className="text-xl font-semibold text-center mb-4 text-yellow-300">🌟 Cosmic Powers</h2>
        <div className="grid gap-4 md:grid-cols-8">
          {[
            { 
              icon: '🧙‍♂️', 
              label: 'AI Oracle', 
              gradient: 'from-purple-600 to-pink-600',
              action: 'ai-oracle'
            },
            { 
              icon: '⛏️', 
              label: 'Mining', 
              gradient: 'from-orange-500 to-red-600',
              action: 'mining'
            },
            { 
              icon: '🥩', 
              label: 'Staking', 
              gradient: 'from-purple-500 to-violet-600',
              action: 'staking'
            },
            { 
              icon: '📈', 
              label: 'Trading', 
              gradient: 'from-green-500 to-emerald-600',
              action: 'trading'
            },
            { 
              icon: '�', 
              label: 'Bridge', 
              gradient: 'from-cyan-500 to-teal-600',
              action: 'bridge'
            },
            { 
              icon: '⚡', 
              label: 'Lightning', 
              gradient: 'from-yellow-500 to-orange-600',
              action: 'lightning'
            },
            { 
              icon: '🚀', 
              label: 'Stargate', 
              gradient: 'from-blue-500 to-purple-600',
              action: 'stargate'
            },
            { 
              icon: '🔥', 
              label: 'ZION', 
              gradient: 'from-red-600 to-pink-600',
              action: 'zion'
            }
          ].map((cosmic, i) => (
            <motion.button
              key={cosmic.label}
              className={`bg-gradient-to-br ${cosmic.gradient} p-1 rounded-2xl hover:scale-105 transition-transform`}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.8 + i * 0.1 }}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => {
                setSelectedPower(cosmic.action);
                if (cosmic.action !== 'zion') handleCosmicAction(cosmic.action);
              }}
            >
              <div className="bg-black/70 rounded-xl p-3 text-center">
                <div className="text-2xl mb-1">{cosmic.icon}</div>
                <div className="text-xs font-semibold">{cosmic.label}</div>
              </div>
            </motion.button>
          ))}
        </div>
      </motion.div>

      {/* Quick Actions */}
      <motion.div className="mb-8" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.9 }}>
        <h2 className="text-xl font-semibold text-center mb-4 text-blue-200">⚡ Quantum Actions</h2>
        <div className="grid gap-4 md:grid-cols-6">
          {[
            { icon: '📤', label: 'Send', gradient: 'from-red-500 to-pink-600', action: 'send' },
            { icon: '📥', label: 'Receive', gradient: 'from-green-500 to-blue-600', action: 'receive' },
            { icon: '🔄', label: 'Swap', gradient: 'from-purple-500 to-indigo-600', action: 'swap' },
            { icon: '🌌', label: 'Portal', gradient: 'from-blue-500 to-cyan-600', action: 'portal' },
            { icon: '🧠', label: 'Upload', gradient: 'from-pink-500 to-purple-600', action: 'consciousness' },
            { icon: '⏰', label: 'Time Travel', gradient: 'from-orange-500 to-red-600', action: 'timetravel' }
          ].map((action, i) => (
            <motion.button
              key={action.label}
              className={`bg-gradient-to-br ${action.gradient} p-1 rounded-2xl hover:scale-105 transition-transform`}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 + i * 0.1 }}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => handleQuantumAction(action.action)}
            >
              <div className="bg-black/70 rounded-xl p-4 text-center">
                <div className="text-2xl mb-2">{action.icon}</div>
                <div className="text-sm font-semibold">{action.label}</div>
              </div>
            </motion.button>
          ))}
        </div>
      </motion.div>

      {/* ZION Wallet Section */}
      {selectedPower === 'zion' && (
        <motion.div className="bg-gradient-to-br from-red-900/30 to-pink-900/30 border border-red-500/30 rounded-2xl p-6 backdrop-blur-sm mb-6" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.8 }}>
          <h2 className="text-xl font-semibold text-red-300 mb-4">🔥 ZION Network Wallet</h2>
          
          {!zionWallet ? (
            <div className="grid gap-4 md:grid-cols-2">
              <motion.button
                className="bg-gradient-to-br from-red-600 to-pink-600 p-4 rounded-xl hover:scale-105 transition-transform"
                onClick={generateZionWallet}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className="text-2xl mb-2">🎲</div>
                <div className="font-semibold">Generate New</div>
                <div className="text-sm opacity-80">Create ZION wallet</div>
              </motion.button>
              
              <motion.button
                className="bg-gradient-to-br from-orange-600 to-red-600 p-4 rounded-xl hover:scale-105 transition-transform"
                onClick={() => {
                  const importData = prompt('Enter private key or mnemonic:');
                  if (importData) importZionWallet(importData);
                }}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className="text-2xl mb-2">📥</div>
                <div className="font-semibold">Import Wallet</div>
                <div className="text-sm opacity-80">From keys/mnemonic</div>
              </motion.button>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="bg-black/40 rounded-xl p-4">
                <h3 className="text-lg font-semibold text-red-300 mb-2">💎 ZION Balance</h3>
                <div className="text-2xl font-bold text-yellow-400 mb-2">{zionWallet.balance.toFixed(2)} ZION</div>
                <div className="text-sm text-gray-400">Address: {zionWallet.address}</div>
              </div>
              
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-2">
                  <input
                    type="number"
                    placeholder="Amount to send"
                    value={sendAmount}
                    onChange={(e) => setSendAmount(e.target.value)}
                    className="w-full p-3 bg-black/50 border border-red-500/30 rounded-xl text-white placeholder-gray-400"
                  />
                  <input
                    type="text"
                    placeholder="Recipient address"
                    value={recipientAddress}
                    onChange={(e) => setRecipientAddress(e.target.value)}
                    className="w-full p-3 bg-black/50 border border-red-500/30 rounded-xl text-white placeholder-gray-400"
                  />
                  <motion.button
                    className="w-full bg-gradient-to-r from-red-600 to-pink-600 p-3 rounded-xl hover:scale-105 transition-transform font-semibold"
                    onClick={sendZionTokens}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    🚀 Send ZION
                  </motion.button>
                </div>
                
                <div className="bg-black/40 rounded-xl p-4">
                  <h4 className="font-semibold text-red-300 mb-2">Recent Transactions</h4>
                  <div className="space-y-2 max-h-40 overflow-y-auto">
                    {zionWallet.transactions.slice(0, 5).map((tx, i) => (
                      <div key={tx.txHash} className="text-xs bg-black/30 p-2 rounded border border-red-500/20">
                        <div className="flex justify-between">
                          <span className={tx.status === 'confirmed' ? 'text-green-400' : tx.status === 'pending' ? 'text-yellow-400' : 'text-red-400'}>
                            {tx.status.toUpperCase()}
                          </span>
                          <span className="text-yellow-400">+{tx.dharmaImpact} Dharma</span>
                        </div>
                        <div className="text-gray-400">
                          {tx.amount} ZION → {tx.to.slice(0, 10)}...
                        </div>
                        <div className="text-gray-500">{new Date(tx.timestamp).toLocaleTimeString()}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}
        </motion.div>
      )}

      {/* Transaction History */}
      <motion.div className="bg-black/30 border border-purple-500/30 rounded-2xl p-6 backdrop-blur-sm" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.7 }}>
        <h2 className="text-xl font-semibold text-purple-200 mb-4">📜 Cosmic Transaction History</h2>
        <div className="space-y-4">
          {transactions.map((tx, i) => (
            <motion.div
              key={tx.id}
              className="p-4 bg-black/40 rounded-xl border border-gray-600/30 hover:border-purple-500/50 transition-all"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.8 + i * 0.1 }}
            >
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-3">
                  <div className="text-2xl">{getTransactionIcon(tx.type)}</div>
                  <div>
                    <div className="font-semibold capitalize">{tx.type}</div>
                    <div className="text-sm text-gray-400">{tx.timestamp}</div>
                    {tx.cosmic_signature && (
                      <div className="text-xs text-purple-300 mt-1">{tx.cosmic_signature}</div>
                    )}
                  </div>
                </div>
                <div className="text-right">
                  <div className={`font-bold ${getTransactionColor(tx.type)}`}>
                    {tx.type === 'send' ? '-' : '+'}{tx.amount} {tx.currency}
                  </div>
                  <div className="text-xs text-gray-400 flex items-center gap-1">
                    Dharma: <span className={tx.dharma_impact > 0 ? 'text-green-400' : 'text-red-400'}>
                      {tx.dharma_impact > 0 ? '+' : ''}{tx.dharma_impact}
                    </span>
                  </div>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Cosmic Stats */}
      <motion.div className="mt-6 bg-black/30 border border-blue-500/30 rounded-2xl p-6 backdrop-blur-sm" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 1 }}>
        <h3 className="text-lg font-semibold text-blue-200 mb-4 text-center">🌌 Cosmic Wallet Statistics</h3>
        <div className="grid gap-4 md:grid-cols-3">
          <div className="text-center p-4 bg-blue-900/30 rounded-xl border border-blue-500/30">
            <div className="text-xl font-bold text-blue-300">21</div>
            <div className="text-sm text-blue-200">Active Dimensions</div>
            <div className="text-xs text-blue-100 mt-1">Multi-chain connected</div>
          </div>
          <div className="text-center p-4 bg-purple-900/30 rounded-xl border border-purple-500/30">
            <div className="text-xl font-bold text-purple-300">∞</div>
            <div className="text-sm text-purple-200">Quantum Security</div>
            <div className="text-xs text-purple-100 mt-1">Unbreakable encryption</div>
          </div>
          <div className="text-center p-4 bg-green-900/30 rounded-xl border border-green-500/30">
            <div className="text-xl font-bold text-green-300">108</div>
            <div className="text-sm text-green-200">Sacred Transactions</div>
            <div className="text-xs text-green-100 mt-1">Enlightenment achieved</div>
          </div>
        </div>
      </motion.div>

      {/* QR Code Modal */}
      {showQR && (
        <motion.div 
          className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          onClick={() => setShowQR(false)}
        >
          <motion.div 
            className="bg-gradient-to-br from-purple-900 via-blue-900 to-black p-8 rounded-2xl border border-purple-500/50 max-w-sm mx-4"
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            onClick={(e: MouseEvent) => e.stopPropagation()}
          >
            <div className="text-center mb-4">
              <h3 className="text-xl font-bold text-purple-200 mb-2">📱 Wallet QR Code</h3>
              <p className="text-sm text-gray-300">Scan to share address</p>
            </div>
            
            <div className="bg-white p-4 rounded-xl mb-4">
              <QRCode 
                value={walletAddress}
                size={200}
                bgColor="#ffffff"
                fgColor="#000000"
                level="M"
              />
            </div>
            
            <div className="text-center">
              <p className="text-xs text-gray-400 font-mono break-all mb-4">{walletAddress}</p>
              <motion.button
                className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded-lg text-white text-sm font-semibold"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                onClick={() => setShowQR(false)}
              >
                Close
              </motion.button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </div>
  );
}
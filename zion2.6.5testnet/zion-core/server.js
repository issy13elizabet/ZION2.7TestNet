#!/usr/bin/env node
import express from 'express';
import { createServer } from 'http';
import { WebSocketServer } from 'ws';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import rateLimit from 'express-rate-limit';
import dotenv from 'dotenv';
import { createHash } from 'crypto';
import net from 'net';
import cron from 'node-cron';
import si from 'systeminformation';

// Load environment variables
dotenv.config();

const PORT = process.env.ZION_PORT || 8888;
const NODE_ENV = process.env.NODE_ENV || 'development';

console.log(`
╔══════════════════════════════════════════════╗
║              🌈 ZION CORE v2.5               ║
║      Unified Multi-Chain Dharma Node         ║
║                                              ║
║  🔗 Mining Pool + RPC + Wallet + Lightning   ║
║  ⚡ GPU Support + RandomX + Cross-Chain       ║
╚══════════════════════════════════════════════╝
`);

// Core modules
import MiningPool from './modules/mining-pool.js';
import RPCAdapter from './modules/rpc-adapter.js';
import WalletService from './modules/wallet-service.js';
import LightningNetwork from './modules/lightning-network.js';
import GPUMining from './modules/gpu-mining.js';
import P2PNetwork from './modules/p2p-network.js';
import BlockchainCore from './modules/blockchain-core.js';

class ZionCore {
  constructor() {
    this.app = express();
    this.server = createServer(this.app);
    this.wss = new WebSocketServer({ server: this.server });
    this.isRunning = false;
    
    // Initialize core modules
    this.blockchain = new BlockchainCore();
    this.miningPool = new MiningPool();
    this.rpcAdapter = new RPCAdapter();
    this.walletService = new WalletService();
    this.lightningNetwork = new LightningNetwork();
    this.gpuMining = new GPUMining();
    this.p2pNetwork = new P2PNetwork();
    
    this.setupMiddleware();
    this.setupRoutes();
    this.setupWebSocket();
    this.setupCronJobs();
  }

  setupMiddleware() {
    // Security
    this.app.use(helmet({
      contentSecurityPolicy: false // Allow for development
    }));
    
    // CORS for cross-origin requests
    this.app.use(cors({
      origin: process.env.CORS_ORIGINS?.split(',') || ['*'],
      credentials: true
    }));
    
    // Rate limiting
    const limiter = rateLimit({
      windowMs: 15 * 60 * 1000, // 15 minutes
      max: 1000, // limit each IP to 1000 requests per windowMs
      standardHeaders: true,
      legacyHeaders: false,
    });
    this.app.use(limiter);
    
    // Logging
    this.app.use(morgan('combined'));
    
    // JSON parsing
    this.app.use(express.json({ limit: '10mb' }));
    this.app.use(express.urlencoded({ extended: true, limit: '10mb' }));
  }

  setupRoutes() {
    // Core status endpoint
    this.app.get('/', (req, res) => {
      res.json({
        name: 'ZION Core',
        version: '2.5.0',
        status: 'running',
        uptime: process.uptime(),
        modules: {
          blockchain: this.blockchain.getStatus(),
          mining: this.miningPool.getStatus(),
          wallet: this.walletService.getStatus(),
          lightning: this.lightningNetwork.getStatus(),
          gpu: this.gpuMining.getStatus(),
          p2p: this.p2pNetwork.getStatus()
        }
      });
    });

    // Health check
    this.app.get('/health', (req, res) => {
      res.json({ 
        status: 'healthy',
        timestamp: new Date().toISOString(),
        modules: Object.keys(this).filter(key => key.endsWith('Service') || key.endsWith('Pool') || key.endsWith('Network'))
      });
    });

    // Mining API
    this.app.use('/api/mining', this.miningPool.getRouter());
    
    // RPC API (Monero-compatible)
    this.app.use('/json_rpc', this.rpcAdapter.getRouter());
    
    // Wallet API
    this.app.use('/api/wallet', this.walletService.getRouter());
    
    // Lightning Network API
    this.app.use('/api/lightning', this.lightningNetwork.getRouter());
    
    // GPU Mining API
    this.app.use('/api/gpu', this.gpuMining.getRouter());
    
    // Blockchain API
    this.app.use('/api/blockchain', this.blockchain.getRouter());
    
    // P2P Network API
    this.app.use('/api/p2p', this.p2pNetwork.getRouter());

    // Stats endpoint
    this.app.get('/api/stats', async (req, res) => {
      try {
        const stats = await this.getSystemStats();
        res.json(stats);
      } catch (error) {
        res.status(500).json({ error: error.message });
      }
    });
  }

  setupWebSocket() {
    this.wss.on('connection', (ws) => {
      console.log('🔗 WebSocket client connected');
      
      // Send welcome message
      ws.send(JSON.stringify({
        type: 'welcome',
        message: 'Connected to ZION Core',
        timestamp: Date.now()
      }));
      
      // Handle messages
      ws.on('message', (data) => {
        try {
          const message = JSON.parse(data);
          this.handleWebSocketMessage(ws, message);
        } catch (error) {
          console.error('WebSocket message error:', error);
        }
      });
      
      ws.on('close', () => {
        console.log('🔌 WebSocket client disconnected');
      });
    });
  }

  handleWebSocketMessage(ws, message) {
    const { type, data } = message;
    
    switch (type) {
      case 'subscribe':
        // Handle subscriptions to real-time data
        ws.subscription = data.channel;
        break;
      case 'mining_status':
        ws.send(JSON.stringify({
          type: 'mining_status',
          data: this.miningPool.getStatus()
        }));
        break;
      case 'blockchain_status':
        ws.send(JSON.stringify({
          type: 'blockchain_status',
          data: this.blockchain.getStatus()
        }));
        break;
    }
  }

  setupCronJobs() {
    // Update stats every minute
    cron.schedule('* * * * *', () => {
      this.broadcastStats();
    });
    
    // Cleanup every hour
    cron.schedule('0 * * * *', () => {
      this.performCleanup();
    });
  }

  async broadcastStats() {
    const stats = await this.getSystemStats();
    
    this.wss.clients.forEach((client) => {
      if (client.readyState === 1) { // WebSocket.OPEN
        client.send(JSON.stringify({
          type: 'stats_update',
          data: stats,
          timestamp: Date.now()
        }));
      }
    });
  }

  async getSystemStats() {
    const cpuInfo = await si.cpu();
    const memInfo = await si.mem();
    const networkInfo = await si.networkStats();
    
    return {
      system: {
        cpu: {
          manufacturer: cpuInfo.manufacturer,
          brand: cpuInfo.brand,
          cores: cpuInfo.cores,
          speed: cpuInfo.speed
        },
        memory: {
          total: memInfo.total,
          used: memInfo.used,
          free: memInfo.free
        },
        network: networkInfo[0] || {}
      },
      blockchain: this.blockchain.getStats(),
      mining: this.miningPool.getStats(),
      lightning: this.lightningNetwork.getStats(),
      gpu: this.gpuMining.getStats()
    };
  }

  performCleanup() {
    console.log('🧹 Performing system cleanup...');
    // Implement cleanup logic
  }

  async start() {
    try {
      // Initialize all modules
      await Promise.all([
        this.blockchain.initialize(),
        this.miningPool.initialize(),
        this.rpcAdapter.initialize(),
        this.walletService.initialize(),
        this.lightningNetwork.initialize(),
        this.gpuMining.initialize(),
        this.p2pNetwork.initialize()
      ]);

      // Start HTTP server
      this.server.listen(PORT, () => {
        console.log(`🚀 ZION Core running on port ${PORT}`);
        console.log(`🌐 Environment: ${NODE_ENV}`);
        console.log(`📡 WebSocket server ready`);
        console.log(`⚡ All modules initialized`);
        console.log(`💫 Jai Ram Ram Ram Sita Ram Ram Ram Hanuman! 💫`);
        this.isRunning = true;
      });

      // Handle graceful shutdown
      process.on('SIGINT', () => this.shutdown());
      process.on('SIGTERM', () => this.shutdown());

    } catch (error) {
      console.error('❌ Failed to start ZION Core:', error);
      process.exit(1);
    }
  }

  async shutdown() {
    console.log('\n🛑 Shutting down ZION Core...');
    
    this.isRunning = false;
    
    // Close WebSocket server
    this.wss.close();
    
    // Close HTTP server
    this.server.close();
    
    // Shutdown all modules
    await Promise.all([
      this.blockchain.shutdown(),
      this.miningPool.shutdown(),
      this.rpcAdapter.shutdown(),
      this.walletService.shutdown(),
      this.lightningNetwork.shutdown(),
      this.gpuMining.shutdown(),
      this.p2pNetwork.shutdown()
    ]);
    
    console.log('✅ ZION Core shutdown complete');
    process.exit(0);
  }
}

// Start ZION Core if this file is run directly
if (import.meta.url === `file://${process.argv[1]}`) {
  const zionCore = new ZionCore();
  zionCore.start();
}

export default ZionCore;
#!/usr/bin/env node

import express, { Application, Request, Response, NextFunction } from 'express';
import cors from 'cors';
import rateLimit from 'express-rate-limit';
import { WebSocketServer, WebSocket } from 'ws';
import { createServer, Server as HTTPServer } from 'http';
import * as cron from 'node-cron';
import cluster from 'cluster';
import os from 'os';
import process from 'process';
import {
  ZionConfig,
  SystemStats,
  WebSocketMessage,
  IZionModule,
  ZionError,
  ZION_CONSTANTS
} from './types.js';
import { BlockchainCore } from './modules/blockchain-core.js';
import { MiningPool } from './modules/mining-pool.js';
import { GPUMining } from './modules/gpu-mining.js';
import { LightningNetwork } from './modules/lightning-network.js';
import { WalletService } from './modules/wallet-service.js';
import { P2PNetwork } from './modules/p2p-network.js';
import { RPCAdapter } from './modules/rpc-adapter.js';
import GalacticDebugger from './modules/galactic-debugger.js';

/**
 * ZION CORE v2.5.0 - Unified Multi-Chain Dharma Blockchain Platform
 * 
 * Integrating all essential services into a single TypeScript application:
 * - GPU Mining with Lightning Network acceleration
 * - Stratum Mining Pool with share validation
 * - Lightning Network payment channels
 * - RPC proxy for Monero compatibility
 * - Wallet service with transaction management
 * - P2P network for peer communication
 * - Real-time WebSocket statistics
 * 
 * Built with TypeScript for type safety and maintainability.
 */
class ZionCore {
  private readonly config: ZionConfig;
  private readonly app: Application;
  private readonly server: HTTPServer;
  private readonly wss: WebSocketServer;
  
  // Core modules
  private readonly blockchain: BlockchainCore;
  private readonly mining: MiningPool;
  private readonly gpu: GPUMining;
  private readonly lightning: LightningNetwork;
  private readonly wallet: WalletService;
  private readonly p2p: P2PNetwork;
  private readonly rpc: RPCAdapter;
  private readonly galacticDebugger: GalacticDebugger;
  
  private readonly modules: Map<string, IZionModule> = new Map();
  private readonly connectedClients: Set<WebSocket> = new Set();
  
  private isShuttingDown = false;

  constructor() {
    this.config = {
      port: parseInt(process.env.PORT || '8888'),
      nodeEnv: (process.env.NODE_ENV as 'development' | 'production' | 'test') || 'development',
      corsOrigins: process.env.CORS_ORIGINS?.split(',') || ['http://localhost:3000'],
      rateLimit: {
        windowMs: 15 * 60 * 1000, // 15 minutes
        max: 100 // limit each IP to 100 requests per windowMs
      }
    };
    
    this.app = express();
    this.server = createServer(this.app);
    this.wss = new WebSocketServer({ server: this.server });
    
    // Initialize modules
    this.blockchain = new BlockchainCore();
    this.mining = new MiningPool();
    this.gpu = new GPUMining();
    this.lightning = new LightningNetwork();
    this.wallet = new WalletService();
    this.p2p = new P2PNetwork();
    this.rpc = new RPCAdapter();
    this.galacticDebugger = new GalacticDebugger();
    
    // Register modules
    this.modules.set('blockchain', this.blockchain);
    this.modules.set('mining', this.mining);
    this.modules.set('gpu', this.gpu);
    this.modules.set('lightning', this.lightning);
    this.modules.set('wallet', this.wallet);
    this.modules.set('p2p', this.p2p);
    this.modules.set('rpc', this.rpc);
    this.modules.set('galactic', this.galacticDebugger);
  }

  private setupMiddleware(): void {
    // Basic middleware
    this.app.use(express.json());
    this.app.use(express.urlencoded({ extended: true }));
    
    // CORS configuration
    this.app.use(cors({
      origin: this.config.corsOrigins,
      credentials: true
    }));
    
    // Rate limiting
    const limiter = rateLimit(this.config.rateLimit);
    this.app.use('/api/', limiter);
    
    // Request logging
    this.app.use((req: Request, res: Response, next: NextFunction) => {
      console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
      next();
    });
  }

  private setupRoutes(): void {
    // Health check
    this.app.get('/health', (req: Request, res: Response) => {
      const health = {
        status: 'healthy',
        timestamp: Date.now(),
        version: ZION_CONSTANTS.VERSION,
        uptime: process.uptime(),
        modules: Object.fromEntries(
          Array.from(this.modules.entries()).map(([name, module]) => [
            name,
            module.getStatus()
          ])
        )
      };
      res.json(health);
    });

    // System statistics
    this.app.get('/api/stats', async (req: Request, res: Response) => {
      try {
        const stats: SystemStats = await this.getSystemStats();
        res.json(stats);
      } catch (error) {
        const err = error as Error;
        res.status(500).json({ error: err.message });
      }
    });

    // Module routes
    this.modules.forEach((module, name) => {
      if (module.getRouter) {
        this.app.use(`/api/${name}`, module.getRouter() as any);
      }
    });

    // RPC endpoints (replaces rpc-shim) 
    this.app.post('/json_rpc', async (req: Request, res: Response) => {
      try {
        const { method, params, id } = req.body;
        const result = await this.rpc.handleRequest(method, params);
        
        res.json({
          jsonrpc: '2.0',
          result,
          id
        });
      } catch (error) {
        const err = error as Error;
        res.json({
          jsonrpc: '2.0',
          error: {
            code: -1,
            message: err.message
          },
          id: req.body.id
        });
      }
    });

    // Legacy daemon endpoints (for compatibility)
    this.app.get('/get_info', async (req: Request, res: Response) => {
      const result = await this.rpc.handleRequest('get_info', {});
      res.json(result);
    });

    this.app.post('/getblocktemplate', async (req: Request, res: Response) => {
      const result = await this.rpc.handleRequest('getblocktemplate', req.body);
      res.json(result);
    });

    // Catch-all error handler
    this.app.use((error: Error, req: Request, res: Response, next: NextFunction) => {
      console.error('Unhandled error:', error);
      
      if (error instanceof ZionError) {
        res.status(400).json({
          error: error.message,
          code: error.code,
          module: error.module
        });
      } else {
        res.status(500).json({
          error: 'Internal server error',
          code: 'INTERNAL_ERROR'
        });
      }
    });
  }

  private setupWebSocket(): void {
    this.wss.on('connection', (ws: WebSocket, req) => {
      console.log(`WebSocket client connected from ${req.socket.remoteAddress}`);
      this.connectedClients.add(ws);

      ws.on('message', (data: Buffer) => {
        try {
          const message: WebSocketMessage = JSON.parse(data.toString());
          this.handleWebSocketMessage(ws, message);
        } catch (error) {
          const err = error as Error;
          console.error('WebSocket message error:', err.message);
        }
      });

      ws.on('close', () => {
        this.connectedClients.delete(ws);
        console.log('WebSocket client disconnected');
      });

      ws.on('error', (error: Error) => {
        console.error('WebSocket error:', error.message);
        this.connectedClients.delete(ws);
      });

      // Add to galactic debugger if requested
      if (req.url === '/galactic-debug') {
        this.galacticDebugger.addDebugSession(ws);
      }

      // Send welcome message
      this.sendToClient(ws, {
        type: 'welcome',
        data: {
          version: ZION_CONSTANTS.VERSION,
          timestamp: Date.now(),
          galactic_center: req.url === '/galactic-debug' ? 'ZION CORE - CENTER OF GALAXY' : undefined
        }
      });
    });
  }

  private async handleWebSocketMessage(ws: WebSocket, message: WebSocketMessage): Promise<void> {
    switch (message.type) {
      case 'subscribe':
        // Client wants to subscribe to updates
        this.sendToClient(ws, {
          type: 'subscribed',
          data: { success: true }
        });
        break;
        
      case 'stats':
        // Send current stats
        const stats = await this.getSystemStats();
        this.sendToClient(ws, {
          type: 'stats',
          data: stats
        });
        break;
        
      default:
        this.sendToClient(ws, {
          type: 'error',
          data: { message: `Unknown message type: ${message.type}` }
        });
    }
  }

  private sendToClient(ws: WebSocket, message: WebSocketMessage): void {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        ...message,
        timestamp: Date.now()
      }));
    }
  }

  private broadcast(message: WebSocketMessage): void {
    this.connectedClients.forEach(client => {
      this.sendToClient(client, message);
    });
  }

  private async getSystemStats(): Promise<SystemStats> {
    const [cpus] = await Promise.all([os.cpus()]);
    const memory = {
      total: os.totalmem(),
      used: os.totalmem() - os.freemem(),
      free: os.freemem()
    };

    return {
      system: {
        cpu: {
          manufacturer: 'Unknown',
          brand: cpus[0]?.model || 'Unknown',
          cores: os.cpus().length,
          speed: cpus[0]?.speed || 0
        },
        memory,
        network: os.networkInterfaces()
      },
      blockchain: this.blockchain.getStats(),
      mining: this.mining.getStats(),
      lightning: this.lightning.getStats(),
      gpu: this.gpu.getStats()
    };
  }

  private setupScheduledTasks(): void {
    // Broadcast stats every 30 seconds
    cron.schedule('*/30 * * * * *', async () => {
      if (this.connectedClients.size > 0) {
        const stats = await this.getSystemStats();
        this.broadcast({
          type: 'stats_update',
          data: stats
        });
      }
    });

    // Cleanup tasks every 5 minutes
    cron.schedule('*/5 * * * *', () => {
      // Cleanup old mining jobs, expired invoices, etc.
      console.log('Running maintenance tasks...');
    });
  }

  private setupGracefulShutdown(): void {
    const shutdown = async (signal: string) => {
      if (this.isShuttingDown) return;
      this.isShuttingDown = true;
      
      console.log(`\nReceived ${signal}. Gracefully shutting down ZION CORE...`);
      
      // Notify clients
      this.broadcast({
        type: 'shutdown',
        data: { message: 'Server shutting down' }
      });
      
      // Close WebSocket connections
      this.connectedClients.forEach(client => {
        client.close(1000, 'Server shutdown');
      });
      
      // Stop modules
      for (const [name, module] of this.modules) {
        try {
          console.log(`Shutting down ${name} module...`);
          await module.shutdown();
        } catch (error) {
          const err = error as Error;
          console.error(`Error shutting down ${name}:`, err.message);
        }
      }
      
      // Close server
      this.server.close(() => {
        console.log('ZION CORE shut down successfully');
        process.exit(0);
      });
      
      // Force exit after timeout
      setTimeout(() => {
        console.error('Force shutting down...');
        process.exit(1);
      }, 10000);
    };

    process.on('SIGTERM', () => shutdown('SIGTERM'));
    process.on('SIGINT', () => shutdown('SIGINT'));
    process.on('uncaughtException', (error: Error) => {
      console.error('Uncaught exception:', error);
      shutdown('uncaughtException');
    });
    process.on('unhandledRejection', (reason: unknown) => {
      console.error('Unhandled rejection:', reason);
      shutdown('unhandledRejection');
    });
  }

  public async initialize(): Promise<void> {
    console.log(`🚀 Initializing ZION CORE v${ZION_CONSTANTS.VERSION}...`);
    
    try {
      // Setup Express middleware and routes
      this.setupMiddleware();
      this.setupRoutes();
      this.setupWebSocket();
      this.setupScheduledTasks();
      this.setupGracefulShutdown();
      
      // Initialize all modules
      for (const [name, module] of this.modules) {
        console.log(`Initializing ${name} module...`);
        await module.initialize();
        console.log(`✅ ${name} module ready`);
      }
      
      console.log('🌟 All modules initialized successfully');
      
    } catch (error) {
      const err = error as Error;
      console.error('❌ Failed to initialize ZION CORE:', err.message);
      throw error;
    }
  }

  public async start(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.server.listen(this.config.port, () => {
        console.log(`
╔══════════════════════════════════════════╗
║           ZION CORE v${ZION_CONSTANTS.VERSION}                ║
║      Multi-Chain Dharma Platform         ║
╠══════════════════════════════════════════╣
║ HTTP Server: http://localhost:${this.config.port}     ║
║ WebSocket: ws://localhost:${this.config.port}          ║
║ Environment: ${this.config.nodeEnv.padEnd(12)}          ║
║ Mining Pool: :${ZION_CONSTANTS.DEFAULT_PORTS.POOL}                      ║
║ Lightning: :${ZION_CONSTANTS.DEFAULT_PORTS.LIGHTNING}                       ║
╚══════════════════════════════════════════╝

🎯 Ready to serve the multi-chain dharma ecosystem!
        `);
        resolve();
      });
      
      this.server.on('error', (error: Error) => {
        console.error('❌ Server error:', error.message);
        reject(error);
      });
    });
  }
}

// Unified server mode - single process for all services
async function main(): Promise<void> {
  console.log('🚀 Starting ZION CORE in unified mode...');
  
  const zionCore = new ZionCore();
  
  try {
    await zionCore.initialize();
    await zionCore.start();
  } catch (error) {
    const err = error as Error;
    console.error('❌ Failed to start ZION CORE:', err.message);
    process.exit(1);
  }
}

// Start the application
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((error: Error) => {
    console.error('❌ Fatal error:', error);
    process.exit(1);
  });
}

export default ZionCore;
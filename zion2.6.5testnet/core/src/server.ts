#!/usr/bin/env node
/// <reference types="node" />
// Fallback declaration if @types/node not yet installed in environment (will be shadowed once types present)
// eslint-disable-next-line @typescript-eslint/no-explicit-any
declare const process: any;

import express, { Application, Request, Response } from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import rateLimit from 'express-rate-limit';
import dotenv from 'dotenv';
import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { StratumServer } from './pool/StratumServer.js';

// Import zion-core modules
import { BlockchainCore } from './modules/blockchain-core.js';
import { GPUMining } from './modules/gpu-mining.js'; 
import { LightningNetwork } from './modules/lightning-network.js';
import { WalletService } from './modules/wallet-service.js';
import { P2PNetwork } from './modules/p2p-network.js';
import { RPCAdapter } from './modules/rpc-adapter.js';
import { DaemonBridge } from './modules/daemon-bridge.js';
import { MiningPool } from './modules/mining-pool.js';

// Load environment
dotenv.config();

const __dirname = dirname(fileURLToPath(import.meta.url));

/**
 * Zion Core 2.6.5 - Integrovaný blockchain node
 * 
 * Kombinuje:
 * - Original StratumServer (skeleton z migrace)
 * - Blockchain core logiku z zion-core
 * - GPU mining support
 * - Lightning Network integration  
 * - Wallet service
 * - P2P networking
 * - RPC compatibility layer
 */
class ZionCore {
  private readonly app: Application;
  private readonly stratum: StratumServer;
  
  // Integrated modules z original zion-core
  private readonly blockchain: BlockchainCore;
  private readonly gpu: GPUMining;
  private readonly lightning: LightningNetwork;
  private readonly wallet: WalletService;
  private readonly p2p: P2PNetwork;
  private readonly rpc: RPCAdapter;
  private readonly daemonBridge: DaemonBridge;
  private readonly miningPool: MiningPool;
  
  private readonly version: string;
  private readonly port: number;
  private readonly stratumPort: number;

  constructor() {
    // Read version
    this.version = this.readVersion();
    
    // Environment config
    this.port = parseInt(process.env.PORT || '8601', 10);
    this.stratumPort = parseInt(process.env.STRATUM_PORT || '3333', 10);
    const initialDifficulty = parseInt(process.env.INITIAL_DIFFICULTY || '1000', 10);
    
    // Initialize Express app
    this.app = express();
    this.setupMiddleware();
    
    // Initialize integrated modules FIRST
    // Initialize external daemon bridge (legacy real chain) if enabled (before rpc & pool)
    this.daemonBridge = new DaemonBridge();
    if (this.daemonBridge.isEnabled()) {
      this.daemonBridge.isAvailable()
        .then(avail => console.log(`[bridge] external daemon ${avail ? 'available ✅' : 'unreachable ⚠️'}`))
        .catch(e => console.warn('[bridge] availability check error:', (e as Error).message));
    } else {
      console.log('[bridge] external daemon bridge disabled (set EXTERNAL_DAEMON_ENABLED=true to enable)');
    }
    // Core modules
    this.blockchain = new BlockchainCore();
    this.gpu = new GPUMining();
    this.lightning = new LightningNetwork();
    this.wallet = new WalletService();
    this.p2p = new P2PNetwork();
    this.rpc = new RPCAdapter(this.daemonBridge.isEnabled() ? this.daemonBridge : undefined);
    // Mining pool (injected bridge if enabled)
    this.miningPool = new MiningPool(this.daemonBridge.isEnabled() ? this.daemonBridge : undefined);
    
    // Setup routes AFTER modules are initialized
    this.setupRoutes();
    
    // Initialize Stratum server (preserved from migration skeleton)
    this.stratum = new StratumServer({ 
      port: this.stratumPort, 
      initialDifficulty,
      log: (...args) => console.log('[stratum]', ...args)
    });
    
    // Setup Stratum events
    this.setupStratumEvents();
  }
  
  private readVersion(): string {
    try {
      return readFileSync(join(__dirname, '..', '..', 'VERSION')).toString().trim();
    } catch {
      return 'unknown';
    }
  }
  
  private setupMiddleware(): void {
    // Security
    this.app.use(helmet());
    this.app.use(cors({
      origin: process.env.CORS_ORIGINS?.split(',') || ['http://localhost:3000']
    }));
    
    // Rate limiting
    this.app.use(rateLimit({
      windowMs: 15 * 60 * 1000, // 15 minutes
      max: 100 // limit each IP to 100 requests per windowMs
    }));
    
    // Logging
    if (process.env.NODE_ENV !== 'test') {
      this.app.use(morgan('combined'));
    }
    
    // Body parsing
    this.app.use(express.json());
    this.app.use(express.urlencoded({ extended: true }));
  }
  
  private setupRoutes(): void {
    // Health check (preserved from skeleton)
  this.app.get('/healthz', (_req: Request, res: Response) => {
      res.json({ 
        status: 'ok', 
        version: this.version, 
        service: 'zion-core-integrated',
        modules: {
          stratum: 'active',
          blockchain: this.blockchain.getStatus().status, 
          gpu: this.gpu.getStatus().status,
          lightning: this.lightning.getStatus().status,
          wallet: this.wallet.getStatus().status,
          p2p: this.p2p.getStatus().status,
          rpc: this.rpc.getStatus().status
          ,miningPool: this.miningPool.getStatus().status
        }
      });
    });
    
    // Module status endpoints
  this.app.get('/api/status', (_req: Request, res: Response) => {
      res.json({
        version: this.version,
        uptime: process.uptime(),
        memory: process.memoryUsage(),
        stratum: {
          port: this.stratumPort,
          connections: 0 // TODO: get real count from StratumServer
        },
        blockchain: this.blockchain.getStats(),
        p2p: {
          peers: this.p2p.getPeerCount()
        },
        bridge: this.daemonBridge.isEnabled() ? 'enabled' : 'disabled'
      });
    });
    
    // Mount module routers (only if getRouter method exists)
    if (this.blockchain.getRouter) {
      this.app.use('/api/blockchain', this.blockchain.getRouter());
    }
    if (this.p2p.getRouter) {
      this.app.use('/api/p2p', this.p2p.getRouter());
    }
    if (this.gpu.getRouter) {
      this.app.use('/api/gpu', this.gpu.getRouter());
    }
    if (this.lightning.getRouter) {
      this.app.use('/api/lightning', this.lightning.getRouter());
    }
    if (this.wallet.getRouter) {
      this.app.use('/api/wallet', this.wallet.getRouter());
    }
    if (this.rpc.getRouter) {
      this.app.use('/api/rpc', this.rpc.getRouter());
    }

    // Mining pool router
    if (this.miningPool.getRouter) {
      this.app.use('/api/pool', this.miningPool.getRouter());
    }

    // Bridge status endpoint
  this.app.get('/api/bridge/status', async (_req: Request, res: Response) => {
      if (!this.daemonBridge || !this.daemonBridge.isEnabled()) {
        return res.json({ enabled: false });
      }
      try {
        const info = await this.daemonBridge.getInfo();
        res.json({
          enabled: true,
          height: info.height,
          difficulty: info.difficulty,
          status: info.status
        });
      } catch (e) {
        res.status(503).json({ enabled: true, error: (e as Error).message });
      }
    });
  }
  
  private setupStratumEvents(): void {
    // Preserve original share handling (placeholder)
  // Cast to any to access EventEmitter .on (StratumServer extends EventEmitter)
  (this.stratum as any).on('share', (shareData: any) => {
      console.log('[stratum] share received:', shareData);
      // TODO: Forward to blockchain module for real validation
    });
  }
  
  public async start(): Promise<void> {
    try {
      // Start HTTP server
      this.app.listen(this.port, () => {
        console.log(`[zion-core] HTTP server running on port ${this.port} (version ${this.version})`);
      });
      
      // Initialize blockchain core (včetně node sync)
      await this.blockchain.initialize();
      
      // Initialize P2P network (včetně seed node connections)  
      await this.p2p.initialize();
      
      // Initialize other integrated modules
      await this.gpu.initialize();
      await this.lightning.initialize();
      await this.wallet.initialize();
      await this.rpc.initialize();
  await this.miningPool.initialize();
      
      // Start Stratum server (po inicializaci blockchain pro real block templates)
      await this.stratum.start();
      console.log(`[stratum] Mining pool active on port ${this.stratumPort}`);
      
      console.log('[zion-core] All integrated modules initialized and ready!');
      console.log(`[zion-core] Seed nodes: connecting to P2P network`);
      console.log(`[zion-core] Blockchain sync: started`);
      
    } catch (error) {
      console.error('[zion-core] Startup failed:', error);
      // Process object je dostupné v Docker containeru
      if (typeof process !== 'undefined') {
        process.exit(1);
      }
    }
  }
  
  public async stop(): Promise<void> {
    console.log('[zion-core] Shutting down all modules...');
    
    try {
      // Stop Stratum server first
      await this.stratum.stop();
      
      // Shutdown integrated modules
      await this.rpc.shutdown();
  await this.miningPool.shutdown();
      await this.wallet.shutdown();
      await this.lightning.shutdown();
      await this.gpu.shutdown();
      
      // Shutdown P2P network (disconnect from peers)
      await this.p2p.shutdown();
      
      // Shutdown blockchain core last (save state)
      await this.blockchain.shutdown();
      
      console.log('[zion-core] All modules shut down successfully');
    } catch (error) {
      console.error('[zion-core] Error during shutdown:', error);
    }
  }
}

// Start the integrated Zion Core
const zionCore = new ZionCore();
zionCore.start().catch(error => {
  console.error('[zion-core] Failed to start:', error);
  process.exit(1);
});

// Graceful shutdown
process.on('SIGINT', async () => {
  await zionCore.stop();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  await zionCore.stop();
  process.exit(0);
});

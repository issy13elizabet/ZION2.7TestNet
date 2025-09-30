import { Router, Request, Response } from 'express';
import {
  IBlockchainCore,
  BlockchainStats,
  ModuleStatus,
  ZionError,
  ZION_CONSTANTS
} from '../types.js';
import { DaemonBridge } from './daemon-bridge.js';

/**
 * ZION Blockchain Core Module
 * 
 * Manages the blockchain state, synchronization with network peers,
 * and provides blockchain statistics and information.
 */
export class BlockchainCore implements IBlockchainCore {
  private status: ModuleStatus['status'] = 'stopped';
  private startTime: number = 0;
  private currentHeight: number = 0;
  private currentDifficulty: number = 1;
  private txCount: number = 0;
  private txPoolSize: number = 0;
  private router: Router;
  private bridge: DaemonBridge | null = null;
  private syncTimer: any = null; // NodeJS.Timeout (simplified)

  constructor(bridge?: DaemonBridge) {
    this.router = Router();
    if (bridge && bridge.isEnabled()) {
      this.bridge = bridge;
      console.log('[bridge] BlockchainCore: external daemon bridge enabled');
    }
    this.setupRoutes();
  }

  private setupRoutes(): void {
  this.router.get('/info', (req: Request, res: Response) => {
      res.json({
        height: this.currentHeight,
        difficulty: this.currentDifficulty,
        txCount: this.txCount,
        txPoolSize: this.txPoolSize,
        version: ZION_CONSTANTS.VERSION,
        maxSupply: ZION_CONSTANTS.MAX_SUPPLY,
        blockTime: ZION_CONSTANTS.BLOCK_TIME
      });
    });

  this.router.get('/stats', (req: Request, res: Response) => {
      res.json(this.getStats());
    });

  this.router.get('/status', (req: Request, res: Response) => {
      res.json(this.getStatus());
    });
  }

  public async initialize(): Promise<void> {
    console.log('🔗 Initializing Blockchain Core...');
    
    try {
      this.status = 'starting';
      this.startTime = Date.now();
      
      if (this.bridge) {
        // Immediate fetch from real daemon
        await this.refreshFromBridge(true);
        this.startBridgeSyncLoop();
      } else {
        // Legacy mock bootstrap
        await this.connectToNetwork();
        await this.loadBlockchainState();
        await this.startSync();
      }
      
      this.status = 'ready';
      console.log('✅ Blockchain Core initialized successfully');
      
    } catch (error) {
      this.status = 'error';
      const err = error as Error;
      console.error('❌ Failed to initialize Blockchain Core:', err.message);
      throw new ZionError(`Blockchain initialization failed: ${err.message}`, 'BLOCKCHAIN_INIT_FAILED', 'blockchain');
    }
  }

  public async shutdown(): Promise<void> {
    console.log('🔗 Shutting down Blockchain Core...');
    
    try {
      this.status = 'stopped';
      
  if (this.syncTimer) clearInterval(this.syncTimer);
      
      console.log('✅ Blockchain Core shut down successfully');
      
    } catch (error) {
      const err = error as Error;
      console.error('❌ Error shutting down Blockchain Core:', err.message);
      throw error;
    }
  }

  public getStatus(): ModuleStatus {
    return {
      status: this.status,
      uptime: this.startTime ? Date.now() - this.startTime : 0
    };
  }

  public getStats(): BlockchainStats {
    return {
      height: this.currentHeight,
      difficulty: this.currentDifficulty,
      txCount: this.txCount,
      txPoolSize: this.txPoolSize
    };
  }

  // Real blockchain data methods (Phase 3 enhancement)
  
  /**
   * Get real block data from legacy daemon
   */
  public async getBlock(height: number): Promise<any> {
    if (this.bridge?.isEnabled()) {
      try {
        return await this.bridge.getBlock(height);
      } catch (error) {
        console.warn(`[blockchain] Failed to get block ${height} from bridge:`, (error as Error).message);
      }
    }
    
    // Fallback to synthetic block generation
    return this.generateSyntheticBlock(height);
  }

  /**
   * Get block by hash
   */
  public async getBlockByHash(hash: string): Promise<any> {
    if (this.bridge?.isEnabled()) {
      try {
        return await this.bridge.getBlockByHash(hash);
      } catch (error) {
        console.warn(`[blockchain] Failed to get block ${hash} from bridge:`, (error as Error).message);
      }
    }
    
    throw new ZionError('Block not found', 'BLOCK_NOT_FOUND', 'blockchain');
  }

  /**
   * Get real transaction pool from legacy daemon
   */
  public async getTransactionPool(): Promise<any[]> {
    if (this.bridge?.isEnabled()) {
      try {
        const poolData = await this.bridge.getTxPool();
        this.txPoolSize = poolData?.transactions?.length || 0;
        return poolData?.transactions || [];
      } catch (error) {
        console.warn('[blockchain] Failed to get tx pool from bridge:', (error as Error).message);
      }
    }
    
    // Fallback to empty pool
    return [];
  }

  /**
   * Submit raw transaction to network
   */
  public async submitTransaction(txHex: string): Promise<{ success: boolean; txId?: string; error?: string }> {
    if (this.bridge?.isEnabled()) {
      try {
        const result = await this.bridge.sendRawTransaction(txHex);
        return {
          success: true,
          txId: result?.tx_hash || result?.tx_id
        };
      } catch (error) {
        return {
          success: false,
          error: (error as Error).message
        };
      }
    }
    
    // Mock submission for testing
    return {
      success: true,
      txId: 'mock_tx_' + Date.now().toString(16)
    };
  }

  /**
   * Sync blockchain state to network tip
   */
  public async syncToTip(): Promise<void> {
    if (this.bridge?.isEnabled()) {
      try {
        const info = await this.bridge.getInfo();
        
        // Update state from real daemon
        this.currentHeight = info.height || this.currentHeight;
        this.currentDifficulty = info.difficulty || this.currentDifficulty;
        this.txCount = info.tx_count || this.txCount;
        
        console.log(`[blockchain] Synced to height ${this.currentHeight}, difficulty ${this.currentDifficulty}`);
        
      } catch (error) {
        console.warn('[blockchain] Sync failed, using cached values:', (error as Error).message);
      }
    } else {
      // Mock sync progression
      this.currentHeight += Math.floor(Math.random() * 2);
      this.currentDifficulty += Math.floor(Math.random() * 100 - 50);
    }
  }

  /**
   * Get last block header
   */
  public async getLastBlockHeader(): Promise<any> {
    if (this.bridge?.isEnabled()) {
      try {
        return await this.bridge.getLastBlockHeader();
      } catch (error) {
        console.warn('[blockchain] Failed to get last block header:', (error as Error).message);
      }
    }
    
    return this.generateSyntheticBlockHeader(this.currentHeight);
  }

  public getHeight(): number {
    return this.currentHeight;
  }

  public getDifficulty(): number {
    return this.currentDifficulty;
  }

  public getRouter(): Router {
    return this.router;
  }

  private async connectToNetwork(): Promise<void> {
    if (this.bridge) return; // handled by bridge
    console.log('🌐 Connecting to ZION network (mock)...');
    this.currentHeight = 1;
    this.currentDifficulty = 100;
    console.log('✅ Connected (mock)');
  }

  private async loadBlockchainState(): Promise<void> {
    if (this.bridge) return; // real state via bridge
    console.log('📦 Loading blockchain state (mock)...');
    this.currentHeight = 1;
    this.currentDifficulty = 100;
    this.txCount = 0;
    this.txPoolSize = 0;
    console.log(`✅ Loaded mock state - Height: ${this.currentHeight}, Difficulty: ${this.currentDifficulty}`);
  }

  private async startSync(): Promise<void> {
    if (this.bridge) return; // bridge drives sync
    console.log('🔄 Starting blockchain sync (mock)...');
    console.log('✅ Blockchain sync completed (mock bootstrap)');
  }

  private startBridgeSyncLoop() {
    if (!this.bridge) return;
    if (this.syncTimer) clearInterval(this.syncTimer);
    this.syncTimer = setInterval(() => this.refreshFromBridge(false).catch(() => {}), 5000);
  }

  private async refreshFromBridge(force: boolean) {
    if (!this.bridge) return;
    try {
      const info = await this.bridge.getInfo(force);
      if (info) {
        this.currentHeight = info.height ?? this.currentHeight;
        this.currentDifficulty = info.difficulty ?? info.wide_difficulty ?? this.currentDifficulty;
        this.txCount = info.tx_count ?? this.txCount;
        this.txPoolSize = info.tx_pool_size ?? this.txPoolSize;
      }
    } catch (e:any) {
      console.warn('[bridge] blockchain refresh failed:', e.message);
    }
  }

  // New daemon functionality methods
  public async getBlockTemplate(): Promise<any> {
    // Provide block template for mining
    return {
      blocktemplate_blob: this.generateBlockTemplate(),
      difficulty: this.currentDifficulty,
      height: this.currentHeight,
      status: 'OK'
    };
  }

  private generateBlockTemplate(): string {
    // Generate mock block template blob
    const timestamp = Math.floor(Date.now() / 1000);
    const nonce = Math.floor(Math.random() * 0xFFFFFFFF);
    return `0100${this.currentHeight.toString(16).padStart(8, '0')}${timestamp.toString(16).padStart(8, '0')}${nonce.toString(16).padStart(8, '0')}${'00'.repeat(32)}`;
  }

  public async submitBlock(blockBlob: string): Promise<any> {
    if (this.bridge) {
      try {
        const res = await this.bridge.submitBlock(blockBlob);
        // After submit, force refresh of new height
        await this.refreshFromBridge(true);
        return res;
      } catch (e:any) {
        throw new ZionError(`Bridge submit failed: ${e.message}`, 'BLOCK_SUBMIT_FAILED', 'blockchain');
      }
    }
    console.log(`🎉 (mock) Block submitted height=${this.currentHeight + 1}`);
    this.currentHeight += 1;
    return { status: 'OK' };
  }

  public getInfo(): any {
    return {
      height: this.currentHeight,
      difficulty: this.currentDifficulty,
      tx_count: this.txCount,
      tx_pool_size: this.txPoolSize,
      status: 'OK'
    };
  }

  // Public methods for other modules to use
  public async submitTransaction(tx: any): Promise<string> {
    if (this.status !== 'ready') {
      throw new ZionError('Blockchain not ready', 'BLOCKCHAIN_NOT_READY', 'blockchain');
    }
    
    // Add transaction to mempool
    this.txPoolSize++;
    
    // Return transaction ID
    return `tx_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  public async getBlock(height: number): Promise<any> {
    if (this.status !== 'ready') {
      throw new ZionError('Blockchain not ready', 'BLOCKCHAIN_NOT_READY', 'blockchain');
    }
    
    if (this.bridge) {
      try {
        return await this.bridge.getBlock(height);
      } catch (e:any) {
        console.warn('[bridge] getBlock fallback to mock:', e.message);
      }
    }
    return { height, hash: `mock_block_${height}`, timestamp: Date.now(), difficulty: this.currentDifficulty, reward: ZION_CONSTANTS.INITIAL_REWARD };
  }

  public isReady(): boolean {
    return this.status === 'ready';
  }
}
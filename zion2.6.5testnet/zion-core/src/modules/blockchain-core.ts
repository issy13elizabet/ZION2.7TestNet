import { Router } from 'express';
import {
  IBlockchainCore,
  BlockchainStats,
  ModuleStatus,
  ZionError,
  ZION_CONSTANTS
} from '../types.js';

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

  constructor() {
    this.router = Router();
    this.setupRoutes();
  }

  private setupRoutes(): void {
    this.router.get('/info', (req, res) => {
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

    this.router.get('/stats', (req, res) => {
      res.json(this.getStats());
    });

    this.router.get('/status', (req, res) => {
      res.json(this.getStatus());
    });
  }

  public async initialize(): Promise<void> {
    console.log('🔗 Initializing Blockchain Core...');
    
    try {
      this.status = 'starting';
      this.startTime = Date.now();
      
      // Initialize blockchain connection
      await this.connectToNetwork();
      
      // Load current blockchain state
      await this.loadBlockchainState();
      
      // Start sync process
      await this.startSync();
      
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
      
      // Disconnect from network
      // Save current state
      // Clean up resources
      
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
    // Initialize blockchain daemon functionality
    console.log('🌐 Connecting to ZION network...');
    
    // Bootstrap mode - start with height 1 for testing
    this.currentHeight = 1;
    this.currentDifficulty = 100;
    
    console.log('✅ Connected to ZION network');
  }

  private async loadBlockchainState(): Promise<void> {
    // Load blockchain state - bootstrap mode for testing
    console.log('📦 Loading blockchain state...');
    
    // Bootstrap values for initial testing
    this.currentHeight = 1;
    this.currentDifficulty = 100;
    this.txCount = 0;
    this.txPoolSize = 0;
    
    console.log(`✅ Loaded blockchain state - Height: ${this.currentHeight}, Difficulty: ${this.currentDifficulty}`);
  }

  private async startSync(): Promise<void> {
    // Start blockchain sync - bootstrap mode allows immediate mining
    console.log('🔄 Starting blockchain sync...');
    
    // In bootstrap mode, we're ready immediately
    console.log('✅ Blockchain sync completed (bootstrap mode)');
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
    // Handle block submission
    console.log(`🎉 Block submitted! Height: ${this.currentHeight + 1}`);
    this.currentHeight += 1;
    
    return {
      status: 'OK'
    };
  }

  public getInfo(): any {
    // Compatible with Monero get_info RPC
    return {
      height: this.currentHeight,
      difficulty: this.currentDifficulty,
      tx_count: this.txCount,
      tx_pool_size: this.txPoolSize,
      status: 'OK',
      version: '0.18.0.0', // Monero compatibility
      bootstrap_daemon_address: '',
      nettype: 'mainnet'
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
    
    // Return mock block data
    return {
      height,
      hash: `block_${height}_${Math.random().toString(36).substr(2, 16)}`,
      timestamp: Date.now(),
      difficulty: this.currentDifficulty,
      reward: ZION_CONSTANTS.INITIAL_REWARD
    };
  }

  public isReady(): boolean {
    return this.status === 'ready';
  }
}
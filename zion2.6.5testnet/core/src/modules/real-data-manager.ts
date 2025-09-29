import { DaemonBridge } from './daemon-bridge.js';
import { RandomXValidator } from './randomx-validator.js';
import { ZionError } from '../types.js';

/**
 * Real Data Integration Manager
 * 
 * Manages integration between TypeScript layer and legacy C++ daemon.
 * Provides enhanced data synchronization, validation, and monitoring.
 */
export class RealDataManager {
  private bridge: DaemonBridge;
  private validator: RandomXValidator;
  private syncInterval: NodeJS.Timeout | null = null;
  private lastSyncTime: number = 0;
  private syncMetrics: {
    totalSyncs: number;
    successfulSyncs: number;
    failedSyncs: number;
    avgLatency: number;
  } = {
    totalSyncs: 0,
    successfulSyncs: 0,
    failedSyncs: 0,
    avgLatency: 0
  };

  constructor(bridge: DaemonBridge) {
    this.bridge = bridge;
    this.validator = new RandomXValidator();
  }

  /**
   * Initialize real data integration
   */
  public async initialize(): Promise<void> {
    console.log('🔗 Initializing Real Data Integration...');
    
    try {
      // Check bridge availability
      await this.bridge.requireAvailable();
      
      // Initialize RandomX validator with current seed
      const info = await this.bridge.getInfo();
      if (info?.top_block_hash) {
        await this.validator.initialize(info.top_block_hash);
      }
      
      // Start periodic sync
      this.startPeriodicSync();
      
      console.log('✅ Real Data Integration initialized successfully');
      
    } catch (error) {
      const err = error as Error;
      console.error('❌ Failed to initialize Real Data Integration:', err.message);
      throw new ZionError(`Real data initialization failed: ${err.message}`, 'REAL_DATA_INIT_FAILED', 'realdata');
    }
  }

  /**
   * Enhanced block data retrieval with caching and validation
   */
  public async getEnhancedBlockData(height: number): Promise<{
    block: any;
    validated: boolean;
    fromCache: boolean;
    metadata: {
      retrievalTime: number;
      size: number;
      txCount: number;
    };
  }> {
    const startTime = Date.now();
    
    try {
      // Get block from bridge
      const block = await this.bridge.getBlock(height);
      const retrievalTime = Date.now() - startTime;
      
      // Basic validation
      let validated = false;
      if (block?.hash && block?.height === height) {
        validated = true;
      }
      
      return {
        block,
        validated,
        fromCache: false, // TODO: Implement block caching
        metadata: {
          retrievalTime,
          size: JSON.stringify(block).length,
          txCount: block?.tx_hashes?.length || 0
        }
      };
      
    } catch (error) {
      throw new ZionError(
        `Failed to get enhanced block data for height ${height}: ${(error as Error).message}`,
        'ENHANCED_BLOCK_FETCH_FAILED',
        'realdata'
      );
    }
  }

  /**
   * Real-time transaction pool monitoring
   */
  public async getTransactionPoolWithMetrics(): Promise<{
    transactions: any[];
    metrics: {
      poolSize: number;
      totalSize: number; // bytes
      avgFee: number;
      oldestTxAge: number; // seconds
    };
  }> {
    try {
      const poolData = await this.bridge.getTxPool();
      const transactions = poolData?.transactions || [];
      
      // Calculate metrics
      const now = Date.now() / 1000; // Unix timestamp
      let totalSize = 0;
      let totalFee = 0;
      let oldestTx = now;
      
      transactions.forEach((tx: any) => {
        if (tx.blob_size) totalSize += tx.blob_size;
        if (tx.fee) totalFee += tx.fee;
        if (tx.receive_time && tx.receive_time < oldestTx) {
          oldestTx = tx.receive_time;
        }
      });
      
      return {
        transactions,
        metrics: {
          poolSize: transactions.length,
          totalSize,
          avgFee: transactions.length > 0 ? totalFee / transactions.length : 0,
          oldestTxAge: now - oldestTx
        }
      };
      
    } catch (error) {
      throw new ZionError(
        `Failed to get transaction pool with metrics: ${(error as Error).message}`,
        'TX_POOL_METRICS_FAILED',
        'realdata'
      );
    }
  }

  /**
   * Enhanced share validation using RandomX
   */
  public async validateMiningShare(
    blockBlob: string,
    nonce: string,
    resultHash: string,
    target: string
  ): Promise<{
    valid: boolean;
    hash?: string;
    difficulty?: number;
    validationTime: number;
    error?: string;
  }> {
    const startTime = Date.now();
    
    try {
      const result = await this.validator.validateShare(blockBlob, nonce, resultHash, target);
      
      return {
        ...result,
        validationTime: Date.now() - startTime
      };
      
    } catch (error) {
      return {
        valid: false,
        validationTime: Date.now() - startTime,
        error: (error as Error).message
      };
    }
  }

  /**
   * Network health monitoring
   */
  public async getNetworkHealth(): Promise<{
    daemonConnected: boolean;
    peerCount: number;
    syncStatus: 'synced' | 'syncing' | 'behind';
    lastBlockAge: number; // seconds
    chainHeight: number;
    difficulty: number;
  }> {
    try {
      const [info, connections] = await Promise.all([
        this.bridge.getInfo(),
        this.bridge.getConnections().catch(() => ({ connections: [] }))
      ]);
      
      const now = Date.now() / 1000;
      const lastBlockAge = info?.top_block_hash ? now - (info.timestamp || now) : 0;
      
      let syncStatus: 'synced' | 'syncing' | 'behind' = 'synced';
      if (lastBlockAge > 300) syncStatus = 'behind'; // 5+ minutes old
      else if (lastBlockAge > 120) syncStatus = 'syncing'; // 2+ minutes old
      
      return {
        daemonConnected: true,
        peerCount: connections?.connections?.length || 0,
        syncStatus,
        lastBlockAge,
        chainHeight: info?.height || 0,
        difficulty: info?.difficulty || info?.wide_difficulty || 0
      };
      
    } catch (error) {
      return {
        daemonConnected: false,
        peerCount: 0,
        syncStatus: 'behind',
        lastBlockAge: 0,
        chainHeight: 0,
        difficulty: 0
      };
    }
  }

  /**
   * Sync blockchain state from daemon
   */
  public async syncFromDaemon(): Promise<{
    height: number;
    difficulty: number;
    txCount: number;
    poolSize: number;
    syncTime: number;
    success: boolean;
  }> {
    const startTime = Date.now();
    this.syncMetrics.totalSyncs++;
    
    try {
      const [info, poolData] = await Promise.all([
        this.bridge.getInfo(),
        this.bridge.getTxPool().catch(() => ({ transactions: [] }))
      ]);
      
      const syncTime = Date.now() - startTime;
      this.syncMetrics.successfulSyncs++;
      this.syncMetrics.avgLatency = (this.syncMetrics.avgLatency + syncTime) / 2;
      this.lastSyncTime = Date.now();
      
      return {
        height: info?.height || 0,
        difficulty: info?.difficulty || info?.wide_difficulty || 0,
        txCount: info?.tx_count || 0,
        poolSize: poolData?.transactions?.length || 0,
        syncTime,
        success: true
      };
      
    } catch (error) {
      this.syncMetrics.failedSyncs++;
      
      return {
        height: 0,
        difficulty: 0,
        txCount: 0,
        poolSize: 0,
        syncTime: Date.now() - startTime,
        success: false
      };
    }
  }

  /**
   * Start periodic synchronization with daemon
   */
  private startPeriodicSync(): void {
    if (this.syncInterval) {
      clearInterval(this.syncInterval);
    }
    
    this.syncInterval = setInterval(async () => {
      try {
        await this.syncFromDaemon();
      } catch (error) {
        console.warn('[realdata] Periodic sync failed:', (error as Error).message);
      }
    }, 5000); // Sync every 5 seconds
    
    console.log('[realdata] Periodic sync started (5s interval)');
  }

  /**
   * Get synchronization metrics
   */
  public getSyncMetrics(): {
    totalSyncs: number;
    successRate: number;
    avgLatency: number;
    lastSyncTime: number;
    timeSinceLastSync: number;
  } {
    return {
      ...this.syncMetrics,
      successRate: this.syncMetrics.totalSyncs > 0 
        ? this.syncMetrics.successfulSyncs / this.syncMetrics.totalSyncs 
        : 0,
      lastSyncTime: this.lastSyncTime,
      timeSinceLastSync: Date.now() - this.lastSyncTime
    };
  }

  /**
   * Shutdown real data integration
   */
  public async shutdown(): Promise<void> {
    console.log('[realdata] Shutting down...');
    
    if (this.syncInterval) {
      clearInterval(this.syncInterval);
      this.syncInterval = null;
    }
    
    await this.validator.cleanup();
    
    console.log('[realdata] Shutdown completed');
  }

  /**
   * Force seed reinitialization (for new blocks)
   */
  public async reinitializeValidatorSeed(): Promise<void> {
    try {
      const info = await this.bridge.getInfo();
      if (info?.top_block_hash) {
        await this.validator.reinitialize(info.top_block_hash);
        console.log('[realdata] RandomX validator reinitialized with new seed');
      }
    } catch (error) {
      console.warn('[realdata] Failed to reinitialize validator seed:', (error as Error).message);
    }
  }
}

export default RealDataManager;
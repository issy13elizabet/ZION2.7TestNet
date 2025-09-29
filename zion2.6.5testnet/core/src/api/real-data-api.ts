import { Router, Request, Response } from 'express';
import { RealDataManager } from '../modules/real-data-manager.js';
import { EnhancedMiningPool } from '../modules/enhanced-mining-pool.js';
import { DaemonBridge } from '../modules/daemon-bridge.js';

/**
 * Real Data API Routes
 * 
 * Provides API endpoints for accessing real blockchain data,
 * enhanced mining pool functionality, and network monitoring.
 */
export class RealDataAPI {
  private router: Router;
  private realDataManager: RealDataManager;
  private enhancedPool: EnhancedMiningPool;
  private bridge: DaemonBridge;

  constructor(
    realDataManager: RealDataManager,
    enhancedPool: EnhancedMiningPool,
    bridge: DaemonBridge
  ) {
    this.router = Router();
    this.realDataManager = realDataManager;
    this.enhancedPool = enhancedPool;
    this.bridge = bridge;
    
    this.setupRoutes();
  }

  private setupRoutes(): void {
    // Bridge health and status
    this.router.get('/bridge/health', this.getBridgeHealth.bind(this));
    this.router.get('/bridge/status', this.getBridgeStatus.bind(this));
    this.router.get('/bridge/cache-stats', this.getCacheStats.bind(this));
    
    // Real blockchain data
    this.router.get('/blockchain/info', this.getBlockchainInfo.bind(this));
    this.router.get('/blockchain/block/:height', this.getBlockByHeight.bind(this));
    this.router.get('/blockchain/block-hash/:hash', this.getBlockByHash.bind(this));
    this.router.get('/blockchain/network-health', this.getNetworkHealth.bind(this));
    
    // Transaction pool
    this.router.get('/txpool/info', this.getTransactionPoolInfo.bind(this));
    this.router.post('/txpool/submit', this.submitTransaction.bind(this));
    
    // Enhanced mining pool
    this.router.get('/mining/enhanced-stats', this.getEnhancedMiningStats.bind(this));
    this.router.get('/mining/template', this.getEnhancedTemplate.bind(this));
    this.router.post('/mining/validate-share', this.validateShare.bind(this));
    this.router.post('/mining/submit-block', this.submitBlock.bind(this));
    
    // Real-time monitoring
    this.router.get('/monitoring/sync-metrics', this.getSyncMetrics.bind(this));
    this.router.get('/monitoring/validation-stats', this.getValidationStats.bind(this));
    
    // Administrative
    this.router.post('/admin/clear-cache', this.clearCache.bind(this));
    this.router.post('/admin/force-sync', this.forceSync.bind(this));
  }

  // Bridge endpoints
  
  private async getBridgeHealth(req: Request, res: Response): Promise<void> {
    try {
      const health = await this.bridge.healthCheck();
      res.json({
        status: 'success',
        data: health
      });
    } catch (error) {
      res.status(500).json({
        status: 'error',
        message: (error as Error).message
      });
    }
  }

  private async getBridgeStatus(req: Request, res: Response): Promise<void> {
    try {
      const enabled = this.bridge.isEnabled();
      const available = enabled ? await this.bridge.isAvailable() : false;
      
      res.json({
        status: 'success',
        data: {
          enabled,
          available,
          uptime: enabled && available ? Date.now() : 0
        }
      });
    } catch (error) {
      res.status(500).json({
        status: 'error',
        message: (error as Error).message
      });
    }
  }

  private getCacheStats(req: Request, res: Response): void {
    try {
      const stats = this.bridge.getCacheStats();
      res.json({
        status: 'success',
        data: stats
      });
    } catch (error) {
      res.status(500).json({
        status: 'error',
        message: (error as Error).message
      });
    }
  }

  // Blockchain data endpoints

  private async getBlockchainInfo(req: Request, res: Response): Promise<void> {
    try {
      const info = await this.bridge.getInfo();
      res.json({
        status: 'success',
        data: info
      });
    } catch (error) {
      res.status(500).json({
        status: 'error',
        message: (error as Error).message
      });
    }
  }

  private async getBlockByHeight(req: Request, res: Response): Promise<void> {
    try {
      const height = parseInt(req.params.height);
      if (isNaN(height)) {
        res.status(400).json({
          status: 'error',
          message: 'Invalid height parameter'
        });
        return;
      }

      const blockData = await this.realDataManager.getEnhancedBlockData(height);
      res.json({
        status: 'success',
        data: blockData
      });
    } catch (error) {
      res.status(500).json({
        status: 'error',
        message: (error as Error).message
      });
    }
  }

  private async getBlockByHash(req: Request, res: Response): Promise<void> {
    try {
      const hash = req.params.hash;
      if (!hash || hash.length !== 64) {
        res.status(400).json({
          status: 'error',
          message: 'Invalid hash parameter'
        });
        return;
      }

      const block = await this.bridge.getBlockByHash(hash);
      res.json({
        status: 'success',
        data: block
      });
    } catch (error) {
      res.status(500).json({
        status: 'error',
        message: (error as Error).message
      });
    }
  }

  private async getNetworkHealth(req: Request, res: Response): Promise<void> {
    try {
      const health = await this.realDataManager.getNetworkHealth();
      res.json({
        status: 'success',
        data: health
      });
    } catch (error) {
      res.status(500).json({
        status: 'error',
        message: (error as Error).message
      });
    }
  }

  // Transaction pool endpoints

  private async getTransactionPoolInfo(req: Request, res: Response): Promise<void> {
    try {
      const poolData = await this.realDataManager.getTransactionPoolWithMetrics();
      res.json({
        status: 'success',
        data: poolData
      });
    } catch (error) {
      res.status(500).json({
        status: 'error',
        message: (error as Error).message
      });
    }
  }

  private async submitTransaction(req: Request, res: Response): Promise<void> {
    try {
      const { txHex } = req.body;
      if (!txHex) {
        res.status(400).json({
          status: 'error',
          message: 'Missing txHex parameter'
        });
        return;
      }

      const result = await this.bridge.sendRawTransaction(txHex);
      res.json({
        status: 'success',
        data: result
      });
    } catch (error) {
      res.status(500).json({
        status: 'error',
        message: (error as Error).message
      });
    }
  }

  // Enhanced mining endpoints

  private async getEnhancedMiningStats(req: Request, res: Response): Promise<void> {
    try {
      const stats = await this.enhancedPool.getEnhancedStats();
      res.json({
        status: 'success',
        data: stats
      });
    } catch (error) {
      res.status(500).json({
        status: 'error',
        message: (error as Error).message
      });
    }
  }

  private async getEnhancedTemplate(req: Request, res: Response): Promise<void> {
    try {
      const { walletAddress } = req.query;
      const template = await this.enhancedPool.getEnhancedBlockTemplate(walletAddress as string);
      res.json({
        status: 'success',
        data: template
      });
    } catch (error) {
      res.status(500).json({
        status: 'error',
        message: (error as Error).message
      });
    }
  }

  private async validateShare(req: Request, res: Response): Promise<void> {
    try {
      const { jobId, nonce, result, minerId } = req.body;
      if (!jobId || !nonce || !result || !minerId) {
        res.status(400).json({
          status: 'error',
          message: 'Missing required parameters: jobId, nonce, result, minerId'
        });
        return;
      }

      const validation = await this.enhancedPool.validateShareEnhanced({
        jobId,
        nonce,
        result,
        minerId
      });

      res.json({
        status: 'success',
        data: validation
      });
    } catch (error) {
      res.status(500).json({
        status: 'error',
        message: (error as Error).message
      });
    }
  }

  private async submitBlock(req: Request, res: Response): Promise<void> {
    try {
      const { blockData } = req.body;
      if (!blockData) {
        res.status(400).json({
          status: 'error',
          message: 'Missing blockData parameter'
        });
        return;
      }

      const submission = await this.enhancedPool.submitBlockEnhanced(blockData);
      res.json({
        status: 'success',
        data: submission
      });
    } catch (error) {
      res.status(500).json({
        status: 'error',
        message: (error as Error).message
      });
    }
  }

  // Monitoring endpoints

  private getSyncMetrics(req: Request, res: Response): void {
    try {
      const metrics = this.realDataManager.getSyncMetrics();
      res.json({
        status: 'success',
        data: metrics
      });
    } catch (error) {
      res.status(500).json({
        status: 'error',
        message: (error as Error).message
      });
    }
  }

  private async getValidationStats(req: Request, res: Response): Promise<void> {
    try {
      const stats = await this.enhancedPool.getEnhancedStats();
      res.json({
        status: 'success',
        data: {
          validation: stats.enhancedStats,
          network: stats.networkHealth
        }
      });
    } catch (error) {
      res.status(500).json({
        status: 'error',
        message: (error as Error).message
      });
    }
  }

  // Administrative endpoints

  private clearCache(req: Request, res: Response): void {
    try {
      this.bridge.clearCache();
      res.json({
        status: 'success',
        message: 'Cache cleared successfully'
      });
    } catch (error) {
      res.status(500).json({
        status: 'error',
        message: (error as Error).message
      });
    }
  }

  private async forceSync(req: Request, res: Response): Promise<void> {
    try {
      const syncResult = await this.realDataManager.syncFromDaemon();
      res.json({
        status: 'success',
        data: syncResult
      });
    } catch (error) {
      res.status(500).json({
        status: 'error',
        message: (error as Error).message
      });
    }
  }

  public getRouter(): Router {
    return this.router;
  }
}

export default RealDataAPI;
import { Router, Request, Response } from 'express';
// eslint-disable-next-line @typescript-eslint/no-explicit-any
declare const process: any;
import { DaemonBridge } from './daemon-bridge.js';
import {
  IRPCAdapter,
  ModuleStatus,
  ZionError
} from '../types.js';

/**
 * ZION RPC Adapter Module
 * 
 * Monero-compatible JSON-RPC interface for seamless integration
 * with existing mining tools and wallets.
 */
export class RPCAdapter implements IRPCAdapter {
  private status: ModuleStatus['status'] = 'stopped';
  private startTime: number = 0;
  private router: Router;
  private bridge: DaemonBridge | null = null;
  private lastInfo: any = null;
  private lastInfoTs = 0;
  private readonly INFO_TTL = 3000; // ms

  constructor(bridge?: DaemonBridge) {
    this.router = Router();
    if (bridge && bridge.isEnabled()) {
      this.bridge = bridge;
      console.log('[bridge] RPCAdapter: external daemon bridge enabled');
    }
    this.setupRoutes();
  }

  private setupRoutes(): void {
    // JSON-RPC endpoint
  this.router.post('/json_rpc', async (_req: Request, res: Response) => {
      try {
        const { method, params, id } = _req.body;
        const result = await this.handleRequest(method, params);
        
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
          id: _req.body.id
        });
      }
    });

    // Legacy daemon endpoints
  this.router.get('/get_info', async (_req: Request, res: Response) => {
      const result = await this.handleRequest('get_info', {});
      res.json(result);
    });

  this.router.get('/get_height', async (_req: Request, res: Response) => {
      const result = await this.handleRequest('get_height', {});
      res.json(result);
    });
  }

  public async initialize(): Promise<void> {
    console.log('🔌 Initializing RPC Adapter...');
    
    try {
      this.status = 'starting';
      this.startTime = Date.now();
      
      // Setup RPC interface
      await this.setupRPCInterface();
      
      this.status = 'ready';
      console.log('✅ RPC Adapter initialized successfully');
      
    } catch (error) {
      this.status = 'error';
      const err = error as Error;
      console.error('❌ Failed to initialize RPC Adapter:', err.message);
      throw new ZionError(`RPC initialization failed: ${err.message}`, 'RPC_INIT_FAILED', 'rpc');
    }
  }

  public async shutdown(): Promise<void> {
    console.log('🔌 Shutting down RPC Adapter...');
    
    try {
      this.status = 'stopped';
      
      console.log('✅ RPC Adapter shut down successfully');
      
    } catch (error) {
      const err = error as Error;
      console.error('❌ Error shutting down RPC Adapter:', err.message);
      throw error;
    }
  }

  public getStatus(): ModuleStatus {
    return {
      status: this.status,
      uptime: this.startTime ? Date.now() - this.startTime : 0
    };
  }

  public async handleRequest(method: string, params: unknown): Promise<unknown> {
    console.log(`🔌 RPC call: ${method}`);
    
    switch (method) {
      case 'get_info':
        return this.getInfo();
      
      case 'get_height':
        return this.getHeight();
      
      case 'get_block':
        return this.getBlock(params);
      
      case 'get_block_template':
      case 'getblocktemplate':
        return this.getBlockTemplate(params);
      
      case 'submit_block':
        return this.submitBlock(params);
      
      case 'get_connections':
        return this.getConnections();
      
      default:
        throw new ZionError(`Unknown RPC method: ${method}`, 'UNKNOWN_METHOD', 'rpc');
    }
  }

  public getRouter(): Router {
    return this.router;
  }

  private async setupRPCInterface(): Promise<void> {
    // Setup Monero-compatible RPC interface
    await new Promise(resolve => setTimeout(resolve, 500));
  }

  private async getInfo(): Promise<any> {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const strictRequired = (process as any)?.env?.STRICT_BRIDGE_REQUIRED === 'true';
    if (this.bridge) {
      try {
        const now = Date.now();
        if (now - this.lastInfoTs > this.INFO_TTL) {
          this.lastInfo = await this.bridge.getInfo();
          this.lastInfoTs = now;
        }
        return this.lastInfo;
      } catch (e:any) {
        if (strictRequired) {
          throw new ZionError('STRICT mode: get_info failed via bridge: ' + e.message, 'STRICT_RPC_GET_INFO_FAIL', 'rpc');
        }
        console.warn('[bridge] get_info fallback:', e.message);
      }
    }
    if (strictRequired) {
      throw new ZionError('STRICT mode: no mock fallback for get_info', 'STRICT_NO_FALLBACK', 'rpc');
    }
    // Non-strict mock fallback
    return {
      status: 'OK',
      height: 142857,
      target_height: 142857,
      difficulty: 1000000,
      target: 120,
      tx_count: 123456,
      tx_pool_size: 42,
      alt_blocks_count: 0,
      outgoing_connections_count: 8,
      incoming_connections_count: 12,
      rpc_connections_count: 1,
      white_peerlist_size: 50,
      grey_peerlist_size: 100,
      mainnet: false,
      testnet: true,
      stagenet: false,
      nettype: 'testnet',
      top_block_hash: 'mock_top_block_hash_' + Math.random().toString(36).substr(2, 32),
      cumulative_difficulty: 50000000,
      block_size_limit: 600000,
      block_weight_limit: 600000,
      block_size_median: 300000,
      block_weight_median: 300000,
      start_time: Math.floor(Date.now() / 1000) - 86400,
      free_space: 1000000000,
      offline: false,
      untrusted: false,
      bootstrap_daemon_address: '',
      height_without_bootstrap: 142857,
      was_bootstrap_ever_used: false,
      database_size: 500000000,
      update_available: false,
      version: '2.5.0-testnet'
    };
  }

  private async getHeight(): Promise<any> {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const strictRequired = (process as any)?.env?.STRICT_BRIDGE_REQUIRED === 'true';
    if (this.bridge) {
      try {
        const h = await this.bridge.getHeight();
        return { status: 'OK', height: h, hash: 'N/A' };
      } catch (e:any) {
        if (strictRequired) {
          throw new ZionError('STRICT mode: get_height failed via bridge: ' + e.message, 'STRICT_RPC_GET_HEIGHT_FAIL', 'rpc');
        }
        console.warn('[bridge] get_height fallback:', e.message);
      }
    }
    if (strictRequired) {
      throw new ZionError('STRICT mode: no mock fallback for get_height', 'STRICT_NO_FALLBACK', 'rpc');
    }
    return { status: 'OK', height: 142857, hash: 'mock_block_hash_' + Math.random().toString(36).substr(2, 32) };
  }

  private async getBlock(params: unknown): Promise<any> {
    const height = (params as any)?.height || 142857;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const strictRequired = (process as any)?.env?.STRICT_BRIDGE_REQUIRED === 'true';
    if (this.bridge) {
      try {
        return await this.bridge.getBlock(height);
      } catch (e:any) {
        if (strictRequired) {
          throw new ZionError('STRICT mode: get_block failed via bridge: ' + e.message, 'STRICT_RPC_GET_BLOCK_FAIL', 'rpc');
        }
        console.warn('[bridge] get_block fallback:', e.message);
      }
    }
    if (strictRequired) {
      throw new ZionError('STRICT mode: no mock fallback for get_block', 'STRICT_NO_FALLBACK', 'rpc');
    }
    // Non-strict mock fallback
    return {
      status: 'OK',
      block_header: {
        block_size: 300000,
        block_weight: 300000,
        cumulative_difficulty: 50000000,
        depth: 0,
        difficulty: 1000000,
        hash: 'mock_block_hash_' + Math.random().toString(36).substr(2, 32),
        height,
        major_version: 16,
        minor_version: 16,
        nonce: Math.floor(Math.random() * 4294967295),
        num_txes: 1,
        orphan_status: false,
        prev_hash: 'mock_prev_hash_' + Math.random().toString(36).substr(2, 32),
        reward: 333000000,
        timestamp: Math.floor(Date.now() / 1000)
      }
    };
  }

  private async getBlockTemplate(params: unknown): Promise<any> {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const strictRequired = (process as any)?.env?.STRICT_BRIDGE_REQUIRED === 'true';
    if (this.bridge) {
      try {
        return await this.bridge.getBlockTemplate();
      } catch (e:any) {
        if (strictRequired) {
          throw new ZionError('STRICT mode: get_block_template failed via bridge: ' + e.message, 'STRICT_RPC_GET_TEMPLATE_FAIL', 'rpc');
        }
        console.warn('[bridge] get_block_template fallback:', e.message);
      }
    }
    if (strictRequired) {
      throw new ZionError('STRICT mode: no mock fallback for get_block_template', 'STRICT_NO_FALLBACK', 'rpc');
    }
    const walletAddress = (params as any)?.wallet_address || 'ZiONTestWalletAddress';
    return {
      status: 'OK',
      blocktemplate_blob: 'mock_blocktemplate_' + Math.random().toString(36).substr(2, 64),
      difficulty: 1000000,
      expected_reward: 333000000,
      height: 142858,
      prev_hash: 'mock_prev_hash_' + Math.random().toString(36).substr(2, 32),
      reserved_offset: 130,
      seed_hash: 'mock_seed_hash_' + Math.random().toString(36).substr(2, 32),
      next_seed_hash: 'mock_next_seed_hash_' + Math.random().toString(36).substr(2, 32),
      wallet_address: walletAddress
    };
  }

  private async submitBlock(params: unknown): Promise<any> {
    const blockBlob = (params as any)?.[0] || '';
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const strictRequired = (process as any)?.env?.STRICT_BRIDGE_REQUIRED === 'true';
    if (this.bridge) {
      try {
        return await this.bridge.submitBlock(blockBlob);
      } catch (e:any) {
        if (strictRequired) {
          throw new ZionError('STRICT mode: submit_block failed via bridge: ' + e.message, 'STRICT_RPC_SUBMIT_FAIL', 'rpc');
        }
        console.warn('[bridge] submit_block fallback:', e.message);
      }
    }
    if (strictRequired) {
      throw new ZionError('STRICT mode: no mock fallback for submit_block', 'STRICT_NO_FALLBACK', 'rpc');
    }
    if (blockBlob.length < 64) {
      return { status: 'FAIL', error: 'Invalid block blob' };
    }
    return { status: 'OK' };
  }

  private async getConnections(): Promise<any> {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const strictRequired = (process as any)?.env?.STRICT_BRIDGE_REQUIRED === 'true';
    if (this.bridge) {
      try {
        return await this.bridge.getConnections();
      } catch (e:any) {
        if (strictRequired) {
          throw new ZionError('STRICT mode: get_connections failed via bridge: ' + e.message, 'STRICT_RPC_CONNECTIONS_FAIL', 'rpc');
        }
        console.warn('[bridge] get_connections fallback:', e.message);
      }
    }
    if (strictRequired) {
      throw new ZionError('STRICT mode: no mock fallback for get_connections', 'STRICT_NO_FALLBACK', 'rpc');
    }
    return { status: 'OK', connections: [] };
  }
}
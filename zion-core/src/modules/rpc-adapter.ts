import { Router } from 'express';
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

  constructor() {
    this.router = Router();
    this.setupRoutes();
  }

  private setupRoutes(): void {
    // JSON-RPC endpoint
    this.router.post('/json_rpc', async (_req, res) => {
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
    this.router.get('/get_info', async (_req, res) => {
      const result = await this.handleRequest('get_info', {});
      res.json(result);
    });

    this.router.get('/get_height', async (_req, res) => {
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

  private getInfo(): unknown {
    return {
      status: 'OK',
      height: 142857,
      target_height: 142857,
      difficulty: 1000000,
      target: 120, // 2 minutes
      tx_count: 123456,
      tx_pool_size: 42,
      alt_blocks_count: 0,
      outgoing_connections_count: 8,
      incoming_connections_count: 12,
      rpc_connections_count: 1,
      white_peerlist_size: 50,
      grey_peerlist_size: 100,
      mainnet: false, // testnet
      testnet: true,
      stagenet: false,
      nettype: 'testnet',
      top_block_hash: 'mock_top_block_hash_' + Math.random().toString(36).substr(2, 32),
      cumulative_difficulty: 50000000,
      block_size_limit: 600000,
      block_weight_limit: 600000,
      block_size_median: 300000,
      block_weight_median: 300000,
      start_time: Math.floor(Date.now() / 1000) - 86400, // 24 hours ago
      free_space: 1000000000, // 1GB
      offline: false,
      untrusted: false,
      bootstrap_daemon_address: '',
      height_without_bootstrap: 142857,
      was_bootstrap_ever_used: false,
      database_size: 500000000, // 500MB
      update_available: false,
      version: '2.5.0-testnet'
    };
  }

  private getHeight(): unknown {
    return {
      status: 'OK',
      height: 142857,
      hash: 'mock_block_hash_' + Math.random().toString(36).substr(2, 32)
    };
  }

  private getBlock(params: unknown): unknown {
    const height = (params as any)?.height || 142857;
    
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
        reward: 333000000, // 333 ZION
        timestamp: Math.floor(Date.now() / 1000)
      },
      json: JSON.stringify({
        major_version: 16,
        minor_version: 16,
        timestamp: Math.floor(Date.now() / 1000),
        prev_id: 'mock_prev_hash_' + Math.random().toString(36).substr(2, 32),
        nonce: Math.floor(Math.random() * 4294967295),
        miner_tx: {
          version: 2,
          unlock_time: 0,
          vin: [{
            gen: {
              height
            }
          }],
          vout: [{
            amount: 333000000,
            target: {
              key: 'mock_key_' + Math.random().toString(36).substr(2, 32)
            }
          }],
          extra: [],
          rct_signatures: {
            type: 0
          }
        },
        tx_hashes: []
      }),
      miner_tx_hash: 'mock_miner_tx_hash_' + Math.random().toString(36).substr(2, 32)
    };
  }

  private getBlockTemplate(params: unknown): unknown {
    const walletAddress = (params as any)?.wallet_address || 'ZiONTestWalletAddress';
    
    return {
      status: 'OK',
      blocktemplate_blob: 'mock_blocktemplate_' + Math.random().toString(36).substr(2, 64),
      blockhashing_blob: 'mock_blockhashing_' + Math.random().toString(36).substr(2, 64),
      difficulty: 1000000,
      expected_reward: 333000000,
      height: 142858, // Next block
      prev_hash: 'mock_prev_hash_' + Math.random().toString(36).substr(2, 32),
      reserved_offset: 130,
      wide_difficulty: '0x' + (1000000).toString(16),
      seed_hash: 'mock_seed_hash_' + Math.random().toString(36).substr(2, 32),
      next_seed_hash: 'mock_next_seed_hash_' + Math.random().toString(36).substr(2, 32)
    };
  }

  private submitBlock(params: unknown): unknown {
    const blockBlob = (params as any)?.[0] || '';
    
    console.log(`🔌 Block submitted: ${blockBlob.substr(0, 32)}...`);
    
    // Validate block (mock)
    if (blockBlob.length < 64) {
      return {
        status: 'FAIL',
        error: 'Invalid block blob'
      };
    }
    
    return {
      status: 'OK'
    };
  }

  private getConnections(): unknown {
    return {
      status: 'OK',
      connections: [
        {
          address: '127.0.0.1:18080',
          avg_download: 1024,
          avg_upload: 512,
          connection_id: 'conn_1',
          current_download: 0,
          current_upload: 0,
          height: 142857,
          host: '127.0.0.1',
          incoming: false,
          ip: '127.0.0.1',
          live_time: 3600,
          local_ip: false,
          localhost: true,
          peer_id: 'peer_1',
          port: '18080',
          recv_count: 1000,
          recv_idle_time: 30,
          send_count: 800,
          send_idle_time: 30,
          state: 'active',
          support_flags: 1
        }
      ]
    };
  }
}
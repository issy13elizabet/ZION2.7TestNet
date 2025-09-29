import { Router } from 'express';
import {
  IP2PNetwork,
  ModuleStatus,
  ZionError
} from '../types.js';

/**
 * ZION P2P Network Module
 * 
 * Peer-to-peer network management for blockchain synchronization
 * and transaction propagation.
 */
export class P2PNetwork implements IP2PNetwork {
  private status: ModuleStatus['status'] = 'stopped';
  private startTime: number = 0;
  private router: Router;
  
  // P2P state
  private peers: Map<string, any> = new Map();
  private peerCount: number = 0;

  constructor() {
    this.router = Router();
    this.setupRoutes();
  }

  private setupRoutes(): void {
    this.router.get('/peers', (_req, res) => {
      res.json({
        count: this.peerCount,
        peers: Array.from(this.peers.values())
      });
    });

    this.router.post('/connect', async (_req, res) => {
      try {
        const { address } = _req.body;
        const result = await this.connectPeer(address);
        res.json({ success: result });
      } catch (error) {
        const err = error as Error;
        res.status(400).json({ error: err.message });
      }
    });
  }

  public async initialize(): Promise<void> {
    console.log('🌐 Initializing P2P Network...');
    
    try {
      this.status = 'starting';
      this.startTime = Date.now();
      
      await this.connectToSeeds();
      
      this.status = 'ready';
      console.log('✅ P2P Network initialized successfully');
      
    } catch (error) {
      this.status = 'error';
      const err = error as Error;
      console.error('❌ Failed to initialize P2P Network:', err.message);
      throw new ZionError(`P2P initialization failed: ${err.message}`, 'P2P_INIT_FAILED', 'p2p');
    }
  }

  public async shutdown(): Promise<void> {
    console.log('🌐 Shutting down P2P Network...');
    
    try {
      this.status = 'stopped';
      this.peers.clear();
      this.peerCount = 0;
      
      console.log('✅ P2P Network shut down successfully');
      
    } catch (error) {
      const err = error as Error;
      console.error('❌ Error shutting down P2P Network:', err.message);
      throw error;
    }
  }

  public getStatus(): ModuleStatus {
    return {
      status: this.status,
      uptime: this.startTime ? Date.now() - this.startTime : 0
    };
  }

  public getPeerCount(): number {
    return this.peerCount;
  }

  public async connectPeer(address: string): Promise<boolean> {
    console.log(`🌐 Connecting to peer: ${address}`);
    
    // Mock peer connection
    const peerId = `peer_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`;
    this.peers.set(peerId, {
      id: peerId,
      address,
      connected: true,
      connectedAt: Date.now()
    });
    
    this.peerCount++;
    return true;
  }

  public async disconnectPeer(peerId: string): Promise<boolean> {
    console.log(`🌐 Disconnecting peer: ${peerId}`);
    
    const success = this.peers.delete(peerId);
    if (success) {
      this.peerCount--;
    }
    
    return success;
  }

  public getRouter(): Router {
    return this.router;
  }

  private async connectToSeeds(): Promise<void> {
    // Connect to seed nodes
    const seeds = [
      '127.0.0.1:18080',
      'seed1.zion.network:18080',
      'seed2.zion.network:18080'
    ];
    
    for (const seed of seeds) {
      try {
        await this.connectPeer(seed);
      } catch (error) {
        console.error(`Failed to connect to seed ${seed}:`, error);
      }
    }
  }
}
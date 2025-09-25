import { Router } from 'express';
import { createServer, Server as NetServer } from 'net';
import {
  IMiningPool,
  MiningStats,
  ModuleStatus,
  Miner,
  MiningJob,
  Share,
  Transaction,
  ZionError,
  MiningError,
  ZION_CONSTANTS
} from '../types.js';

/**
 * ZION Mining Pool Module
 * 
 * Stratum mining pool implementation with share validation,
 * payout management, and real-time miner statistics.
 */
export class MiningPool implements IMiningPool {
  private status: ModuleStatus['status'] = 'stopped';
  private startTime: number = 0;
  private router: Router;
  private stratumServer: NetServer | null = null;
  
  // Mining state
  private miners: Map<string, Miner> = new Map();
  private currentJob: MiningJob | null = null;
  private jobs: Map<string, MiningJob> = new Map();
  private shares: Share[] = [];
  private blocksFound: number = 0;
  private lastBlockTime: number | null = null;
  
  // Pool configuration
  private readonly poolPort: number = ZION_CONSTANTS.DEFAULT_PORTS.POOL;
  private readonly poolAddress: string = 'Z321vzLnnRu15AF82Kwx5DVZAX8rf1jZdDThTFZERRwKV23rXUqcV5u5mvT45ZDpW71kG3zEGuM5F2n3uytvAx5G9jFSY5HcFU'; // Real ZION wallet
  private readonly poolFee: number = 0.025; // 2.5% pool fee
  private readonly minPayout: number = 0.1; // Minimum 0.1 ZION payout
  private readonly networkId: string = 'zion-mainnet-v2';
  
  // Multi-algo support
  private readonly algorithmPorts: Map<string, number> = new Map([
    ['randomx', 3333],    // RandomX (Monero-like)
    ['kawpow', 3334],     // KawPow (Ravencoin)
    ['ethash', 3335],     // Ethash (Ethereum Classic) 
    ['cryptonight', 3336], // CryptoNight (ZION native)
    ['octopus', 3337],    // Octopus (Conflux)
    ['ergo', 3338]        // Ergo (Autolykos2)
  ]);
  private algorithmServers: Map<string, NetServer> = new Map();
  private currentAlgorithm: string = 'randomx';

  constructor() {
    this.router = Router();
    this.setupRoutes();
  }

  private setupRoutes(): void {
    // Pool statistics
    this.router.get('/stats', (_req, res) => {
      res.json(this.getStats());
    });

    // Active miners
    this.router.get('/miners', (_req, res) => {
      res.json(this.getMiners());
    });

    // Current job
    this.router.get('/job', (_req, res) => {
      res.json(this.currentJob);
    });

    // Shares history
    this.router.get('/shares', (_req, res) => {
      const limit = parseInt(String(_req.query.limit)) || 100;
      res.json(this.shares.slice(-limit));
    });

    // Process payout
    this.router.post('/payout', async (_req, res) => {
      try {
        const { minerId, amount } = _req.body;
        const transaction = await this.processPayout(minerId, amount);
        res.json(transaction);
      } catch (error) {
        const err = error as Error;
        res.status(400).json({ error: err.message });
      }
    });
  }

  public async initialize(): Promise<void> {
    console.log('⛏️  Initializing Mining Pool...');
    
    try {
      this.status = 'starting';
      this.startTime = Date.now();
      
      // Start Stratum server
      await this.startStratumServer();
      
      // Create initial mining job
      this.generateNewJob();
      
      // Start job update interval
      this.startJobUpdates();
      
      this.status = 'ready';
      console.log(`✅ Mining Pool initialized - Listening on port ${this.poolPort}`);
      
    } catch (error) {
      this.status = 'error';
      const err = error as Error;
      console.error('❌ Failed to initialize Mining Pool:', err.message);
      throw new MiningError(`Pool initialization failed: ${err.message}`, 'POOL_INIT_FAILED');
    }
  }

  public async shutdown(): Promise<void> {
    console.log('⛏️  Shutting down Mining Pool...');
    
    try {
      this.status = 'stopped';
      
      // Close Stratum server
      if (this.stratumServer) {
        this.stratumServer.close();
        this.stratumServer = null;
      }
      
      // Disconnect all miners
      this.miners.clear();
      
      console.log('✅ Mining Pool shut down successfully');
      
    } catch (error) {
      const err = error as Error;
      console.error('❌ Error shutting down Mining Pool:', err.message);
      throw error;
    }
  }

  public getStatus(): ModuleStatus {
    return {
      status: this.status,
      uptime: this.startTime ? Date.now() - this.startTime : 0
    };
  }

  public getStats(): MiningStats {
    const totalHashrate = Array.from(this.miners.values())
      .reduce((sum, miner) => sum + miner.hashrate, 0);
    
    const totalShares = this.shares.length;
    const acceptedShares = this.shares.filter(share => share.difficulty > 0).length;
    const rejectedShares = totalShares - acceptedShares;
    
    return {
      hashrate: totalHashrate,
      minersActive: this.miners.size,
      sharesAccepted: acceptedShares,
      sharesRejected: rejectedShares,
      blocksFound: this.blocksFound,
      lastBlock: this.lastBlockTime
    };
  }

  public getMiners(): Miner[] {
    return Array.from(this.miners.values());
  }

  public getRouter(): Router {
    return this.router;
  }

  private validateZionAddress(address: string): boolean {
    // ZION address validation
    // Standard format: Z + 86 characters (base58)
    if (!address || typeof address !== 'string') {
      return false;
    }
    
    // Check prefix
    if (!address.startsWith('Z')) {
      return false;
    }
    
    // Check length (ZION addresses are typically 87 characters)
    if (address.length !== 87) {
      return false;
    }
    
    // Check base58 character set
    const base58Regex = /^[123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz]+$/;
    if (!base58Regex.test(address.substring(1))) {
      return false;
    }
    
    return true;
  }

  public async processPayout(minerId: string, amount: number): Promise<Transaction> {
    const miner = this.miners.get(minerId);
    if (!miner || !miner.address) {
      throw new MiningError(`Miner ${minerId} not found or no address set`, 'MINER_NOT_FOUND');
    }

    // Validate ZION address
    if (!this.validateZionAddress(miner.address)) {
      throw new MiningError(`Invalid ZION address: ${miner.address}`, 'INVALID_ADDRESS');
    }

    // Check minimum payout
    if (amount < this.minPayout) {
      throw new MiningError(`Amount ${amount} below minimum payout ${this.minPayout} ZION`, 'AMOUNT_TOO_SMALL');
    }

    // Calculate actual payout after pool fee
    const feeAmount = amount * this.poolFee;
    const payoutAmount = amount - feeAmount;

    // Create mock transaction
    const transaction: Transaction = {
      id: `payout_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      amount: payoutAmount,
      address: miner.address,
      confirmations: 0,
      timestamp: Date.now(),
      fee: feeAmount
    };

    console.log(`💰 Processing payout: ${payoutAmount} ZION to ${miner.address} (Fee: ${feeAmount})`);
    
    return transaction;
  }

  private async startStratumServer(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.stratumServer = createServer((socket) => {
        const minerId = `miner_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`;
        
        console.log(`⛏️  New miner connected: ${minerId} from ${socket.remoteAddress}`);
        
        // Initialize miner
        const miner: Miner = {
          id: minerId,
          address: null,
          worker: null,
          hashrate: 0,
          shares: 0,
          lastActivity: Date.now(),
          connectedAt: Date.now(),
          difficulty: 1000,
          subscribed: false,
          socket
        };
        
        this.miners.set(minerId, miner);

        socket.on('data', (data) => {
          this.handleStratumMessage(minerId, data.toString());
        });

        socket.on('close', () => {
          console.log(`⛏️  Miner disconnected: ${minerId}`);
          this.miners.delete(minerId);
        });

        socket.on('error', (error) => {
          console.error(`⛏️  Miner ${minerId} error:`, error.message);
          this.miners.delete(minerId);
        });

        // Send initial job
        if (this.currentJob) {
          this.sendJobToMiner(minerId, this.currentJob);
        }
      });

      this.stratumServer.listen(this.poolPort, () => {
        console.log(`⛏️  Stratum server listening on port ${this.poolPort}`);
        resolve();
      });

      this.stratumServer.on('error', (error) => {
        console.error('⛏️  Stratum server error:', error.message);
        reject(error);
      });
    });
  }

  private handleStratumMessage(minerId: string, message: string): void {
    try {
      const lines = message.trim().split('\n');
      
      for (const line of lines) {
        const request = JSON.parse(line);
        this.processStratumRequest(minerId, request);
      }
    } catch (error) {
      const err = error as Error;
      console.error(`⛏️  Invalid Stratum message from ${minerId}:`, err.message);
    }
  }

  private processStratumRequest(minerId: string, request: any): void {
    const miner = this.miners.get(minerId);
    if (!miner) return;

    switch (request.method) {
      case 'mining.subscribe':
        this.handleSubscribe(minerId, request);
        break;
      
      case 'mining.authorize':
        this.handleAuthorize(minerId, request);
        break;
      
      case 'mining.submit':
        this.handleSubmit(minerId, request);
        break;
      
      default:
        console.log(`⛏️  Unknown Stratum method: ${request.method}`);
    }
  }

  private handleSubscribe(minerId: string, request: any): void {
    const miner = this.miners.get(minerId);
    if (!miner || !miner.socket) return;

    // Update miner subscription status
    this.miners.set(minerId, { ...miner, subscribed: true });

    // Send subscription response
    const response = {
      id: request.id,
      result: [null, `${minerId}`],
      error: null
    };

    (miner.socket as any).write(JSON.stringify(response) + '\n');
  }

  private handleAuthorize(minerId: string, request: any): void {
    const miner = this.miners.get(minerId);
    if (!miner || !miner.socket) return;

    const [address, worker] = request.params;
    
    // Validate ZION address if provided
    let validatedAddress = this.poolAddress; // Default to pool address
    let authResult = true;
    let errorMessage: string | null = null;
    
    if (address && address !== this.poolAddress) {
      if (this.validateZionAddress(address)) {
        validatedAddress = address;
        console.log(`✅ Valid ZION address: ${address}`);
      } else {
        authResult = false;
        errorMessage = `Invalid ZION address format: ${address}`;
        console.log(`❌ ${errorMessage}`);
      }
    }
    
    if (authResult) {
      // Update miner info
      this.miners.set(minerId, { 
        ...miner, 
        address: validatedAddress,
        worker: worker || 'worker1'
      });
    }

    // Send authorization response
    const response = {
      id: request.id,
      result: authResult,
      error: errorMessage
    };

    (miner.socket as any).write(JSON.stringify(response) + '\n');
    
    if (authResult) {
      console.log(`⛏️  Miner ${minerId} authorized with ZION address: ${validatedAddress}`);
    } else {
      console.log(`❌ Miner ${minerId} authorization failed: ${errorMessage}`);
    }
  }

  private handleSubmit(minerId: string, request: any): void {
    const miner = this.miners.get(minerId);
    if (!miner || !miner.socket) return;

    const [worker, jobId, nonce] = request.params;
    
    // Validate share
    const isValid = this.validateShare(jobId, nonce);
    
    // Create share record
    const share: Share = {
      minerId,
      jobId,
      nonce,
      timestamp: Date.now(),
      difficulty: miner.difficulty
    };
    
    this.shares.push(share);
    
    // Update miner stats
    const updatedMiner = { 
      ...miner, 
      shares: miner.shares + 1,
      lastActivity: Date.now(),
      hashrate: this.calculateHashrate(minerId)
    };
    this.miners.set(minerId, updatedMiner);

    // Send submit response
    const response = {
      id: request.id,
      result: isValid,
      error: isValid ? null : 'Invalid share'
    };

    (miner.socket as any).write(JSON.stringify(response) + '\n');

    if (isValid) {
      console.log(`⛏️  Valid share from ${minerId}: ${nonce}`);
      
      // Check if it's a block
      if (this.isBlockShare(share)) {
        this.handleBlockFound(share);
      }
    }
  }

  private validateShare(jobId: string, nonce: string): boolean {
    // Simple validation (in real implementation, this would check the actual hash)
    const job = this.jobs.get(jobId);
    return job !== undefined && nonce.length === 8;
  }

  private isBlockShare(share: Share): boolean {
    // Check if share meets block difficulty
    // This is a mock implementation
    return Math.random() < 0.001; // 0.1% chance for demo
  }

  private handleBlockFound(share: Share): void {
    console.log(`🎉 BLOCK FOUND by miner ${share.minerId}!`);
    
    this.blocksFound++;
    this.lastBlockTime = Date.now();
    
    // Generate new job for new block
    this.generateNewJob();
  }

  private calculateHashrate(minerId: string): number {
    const miner = this.miners.get(minerId);
    if (!miner) return 0;
    
    // Calculate hashrate based on shares submitted in last 10 minutes
    const tenMinutesAgo = Date.now() - (10 * 60 * 1000);
    const recentShares = this.shares.filter(
      share => share.minerId === minerId && share.timestamp > tenMinutesAgo
    );
    
    return recentShares.length * miner.difficulty / 600; // shares per second
  }

  private generateNewJob(): void {
    const jobId = `job_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`;
    
    this.currentJob = {
      id: jobId,
      prevhash: `prevhash_${Math.random().toString(36).substr(2, 16)}`,
      coinb1: 'coinbase1_data',
      coinb2: 'coinbase2_data',
      merkleBranch: [],
      version: '01000000',
      nbits: '1d00ffff',
      ntime: Math.floor(Date.now() / 1000).toString(16),
      cleanJobs: true,
      createdAt: Date.now()
    };
    
    this.jobs.set(jobId, this.currentJob);
    
    // Send new job to all miners
    this.miners.forEach((miner, minerId) => {
      if (miner.subscribed && this.currentJob) {
        this.sendJobToMiner(minerId, this.currentJob);
      }
    });
    
    console.log(`⛏️  New job generated: ${jobId}`);
  }

  private sendJobToMiner(minerId: string, job: MiningJob): void {
    const miner = this.miners.get(minerId);
    if (!miner || !miner.socket || !miner.subscribed) return;

    const notification = {
      id: null,
      method: 'mining.notify',
      params: [
        job.id,
        job.prevhash,
        job.coinb1,
        job.coinb2,
        job.merkleBranch,
        job.version,
        job.nbits,
        job.ntime,
        job.cleanJobs
      ]
    };

    (miner.socket as any).write(JSON.stringify(notification) + '\n');
  }

  private startJobUpdates(): void {
    // Generate new job every 30 seconds
    setInterval(() => {
      this.generateNewJob();
    }, 30000);
  }

  // Multi-Algorithm Mining Support
  public async startMultiAlgoServers(): Promise<void> {
    console.log('🎯 Starting Multi-Algorithm Mining Servers...');
    
    for (const [algorithm, port] of this.algorithmPorts.entries()) {
      try {
        await this.startAlgorithmServer(algorithm, port);
        console.log(`✅ ${algorithm.toUpperCase()} server started on port ${port}`);
      } catch (error) {
        const err = error as Error;
        console.error(`❌ Failed to start ${algorithm} server:`, err.message);
      }
    }
  }

  private async startAlgorithmServer(algorithm: string, port: number): Promise<void> {
    return new Promise((resolve, reject) => {
      const server = createServer((socket) => {
        const minerId = `${algorithm}_miner_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`;
        
        console.log(`🎮 New ${algorithm.toUpperCase()} miner: ${minerId} from ${socket.remoteAddress}`);
        
        // Initialize algorithm-specific miner
        const miner: Miner = {
          id: minerId,
          address: null,
          worker: null,
          hashrate: 0,
          shares: 0,
          lastActivity: Date.now(),
          connectedAt: Date.now(),
          difficulty: this.getAlgorithmDifficulty(algorithm),
          subscribed: false,
          socket,
          algorithm
        };
        
        this.miners.set(minerId, miner);

        socket.on('data', (data) => {
          this.handleAlgorithmStratumMessage(minerId, algorithm, data.toString());
        });

        socket.on('close', () => {
          console.log(`🎮 ${algorithm.toUpperCase()} miner disconnected: ${minerId}`);
          this.miners.delete(minerId);
        });

        socket.on('error', (error) => {
          console.error(`🎮 ${algorithm.toUpperCase()} miner ${minerId} error:`, error.message);
          this.miners.delete(minerId);
        });

        // Send algorithm-specific job
        const job = this.generateAlgorithmJob(algorithm);
        if (job) {
          this.sendAlgorithmJobToMiner(minerId, algorithm, job);
        }
      });

      server.listen(port, () => {
        this.algorithmServers.set(algorithm, server);
        resolve();
      });

      server.on('error', reject);
    });
  }

  private getAlgorithmDifficulty(algorithm: string): number {
    const difficulties = {
      randomx: 1000,
      kawpow: 500000,
      ethash: 2000000000,
      cryptonight: 5000,
      octopus: 800000000,
      ergo: 1500000000
    };
    return difficulties[algorithm as keyof typeof difficulties] || 1000;
  }

  private handleAlgorithmStratumMessage(minerId: string, algorithm: string, message: string): void {
    const miner = this.miners.get(minerId);
    if (!miner) return;

    miner.lastActivity = Date.now();

    try {
      const lines = message.trim().split('\n');
      for (const line of lines) {
        if (!line) continue;
        
        const request = JSON.parse(line);
        console.log(`🎮 ${algorithm.toUpperCase()} Stratum [${minerId}]:`, request);

        switch (request.method) {
          case 'mining.subscribe':
            this.handleAlgorithmSubscribe(minerId, algorithm, request);
            break;
          case 'mining.authorize':
            this.handleAlgorithmAuthorize(minerId, algorithm, request);
            break;
          case 'mining.submit':
            this.handleAlgorithmSubmit(minerId, algorithm, request);
            break;
          default:
            console.log(`🎮 Unknown ${algorithm} method:`, request.method);
        }
      }
    } catch (error) {
      const err = error as Error;
      console.error(`🎮 ${algorithm.toUpperCase()} Stratum parse error [${minerId}]:`, err.message);
    }
  }

  private handleAlgorithmSubscribe(minerId: string, algorithm: string, request: any): void {
    const miner = this.miners.get(minerId);
    if (!miner || !miner.socket) return;

    const response = {
      id: request.id,
      result: [
        [
          ["mining.set_difficulty", "subscription_id_1"],
          ["mining.notify", "subscription_id_2"]
        ],
        "subscription_id",
        4 // extranonce1 size
      ],
      error: null
    };

    (miner.socket as any).write(JSON.stringify(response) + '\n');
    
    // Set difficulty for this algorithm
    const difficultyResponse = {
      id: null,
      method: 'mining.set_difficulty',
      params: [miner.difficulty]
    };
    
    (miner.socket as any).write(JSON.stringify(difficultyResponse) + '\n');
    
    console.log(`🎮 ${algorithm.toUpperCase()} miner ${minerId} subscribed`);
  }

  private handleAlgorithmAuthorize(minerId: string, algorithm: string, request: any): void {
    const miner = this.miners.get(minerId);
    if (!miner || !miner.socket) return;

    const [username, password] = request.params;
    miner.address = username;
    miner.worker = password;
    miner.subscribed = true;

    const response = {
      id: request.id,
      result: true,
      error: null
    };

    (miner.socket as any).write(JSON.stringify(response) + '\n');
    
    console.log(`🎮 ${algorithm.toUpperCase()} miner authorized: ${username} (${password})`);
    
    // Send job immediately after authorization
    const job = this.generateAlgorithmJob(algorithm);
    if (job) {
      this.sendAlgorithmJobToMiner(minerId, algorithm, job);
    }
  }

  private handleAlgorithmSubmit(minerId: string, algorithm: string, request: any): void {
    const miner = this.miners.get(minerId);
    if (!miner || !miner.socket) return;

    const [worker, jobId, extranonce2, ntime, nonce] = request.params;

    // Validate share (simplified)
    const isValid = this.validateAlgorithmShare(algorithm, jobId, nonce);
    
    const response = {
      id: request.id,
      result: isValid,
      error: isValid ? null : "Invalid share"
    };

    (miner.socket as any).write(JSON.stringify(response) + '\n');

    if (isValid) {
      miner.shares++;
      miner.hashrate = this.estimateHashrate(miner, algorithm);
      
      const share: Share = {
        minerId,
        jobId,
        nonce,
        difficulty: miner.difficulty,
        timestamp: Date.now(),
        algorithm
      };
      
      this.shares.push(share);
      console.log(`✅ ${algorithm.toUpperCase()} Valid share from ${minerId}: ${nonce}`);
    } else {
      console.log(`❌ ${algorithm.toUpperCase()} Invalid share from ${minerId}: ${nonce}`);
    }
  }

  private generateAlgorithmJob(algorithm: string): MiningJob | null {
    const jobId = `${algorithm}_job_${Date.now()}_${Math.random().toString(36).substr(2, 8)}`;
    
    return {
      id: jobId,
      prevhash: `${algorithm}_prevhash_${Math.random().toString(36).substr(2, 16)}`,
      coinb1: `${algorithm}_coinbase1_data`,
      coinb2: `${algorithm}_coinbase2_data`,
      merkleBranch: [],
      version: '01000000',
      nbits: '1d00ffff',
      ntime: Math.floor(Date.now() / 1000).toString(16),
      cleanJobs: true,
      createdAt: Date.now(),
      algorithm
    };
  }

  private sendAlgorithmJobToMiner(minerId: string, algorithm: string, job: MiningJob): void {
    const miner = this.miners.get(minerId);
    if (!miner || !miner.socket || !miner.subscribed) return;

    const notification = {
      id: null,
      method: 'mining.notify',
      params: [
        job.id,
        job.prevhash,
        job.coinb1,
        job.coinb2,
        job.merkleBranch,
        job.version,
        job.nbits,
        job.ntime,
        job.cleanJobs
      ]
    };

    (miner.socket as any).write(JSON.stringify(notification) + '\n');
    console.log(`🎮 ${algorithm.toUpperCase()} job sent to ${minerId}: ${job.id}`);
  }

  private validateAlgorithmShare(algorithm: string, jobId: string, nonce: string): boolean {
    // Algorithm-specific share validation (simplified)
    const algorithms = {
      randomx: () => nonce.length >= 8 && !nonce.startsWith('00000'),
      kawpow: () => nonce.length >= 8 && parseInt(nonce, 16) > 0,
      ethash: () => nonce.length >= 8 && parseInt(nonce, 16) > 1000,
      cryptonight: () => nonce.length >= 8 && !nonce.startsWith('ff'),
      octopus: () => nonce.length >= 8 && parseInt(nonce, 16) > 500000,
      ergo: () => nonce.length >= 8 && !nonce.startsWith('000') && parseInt(nonce, 16) > 100000
    };
    
    const validator = algorithms[algorithm as keyof typeof algorithms];
    return validator ? validator() : true;
  }

  private estimateHashrate(miner: Miner, algorithm: string): number {
    // Algorithm-specific hashrate estimation
    const baseRates = {
      randomx: 100,     // H/s
      kawpow: 8500000,  // H/s (8.5 MH/s)
      ethash: 22000000, // H/s (22 MH/s)  
      cryptonight: 800  // H/s
    };
    
    const baseRate = baseRates[algorithm as keyof typeof baseRates] || 100;
    const timeSinceConnect = (Date.now() - miner.connectedAt) / 1000;
    const sharesPerSecond = miner.shares / Math.max(timeSinceConnect, 1);
    
    return Math.floor(sharesPerSecond * miner.difficulty * baseRate / 1000);
  }

  public switchAlgorithm(newAlgorithm: string): void {
    if (this.algorithmPorts.has(newAlgorithm)) {
      this.currentAlgorithm = newAlgorithm;
      console.log(`🔄 Switched to ${newAlgorithm.toUpperCase()} algorithm`);
    }
  }

  public getMultiAlgoStats(): any {
    const algoStats: any = {};
    
    for (const algorithm of this.algorithmPorts.keys()) {
      const algoMiners = Array.from(this.miners.values()).filter(m => m.algorithm === algorithm);
      const totalHashrate = algoMiners.reduce((sum, miner) => sum + miner.hashrate, 0);
      
      algoStats[algorithm] = {
        miners: algoMiners.length,
        hashrate: totalHashrate,
        shares: algoMiners.reduce((sum, miner) => sum + miner.shares, 0)
      };
    }
    
    return {
      currentAlgorithm: this.currentAlgorithm,
      algorithms: algoStats,
      totalMiners: this.miners.size
    };
  }
}
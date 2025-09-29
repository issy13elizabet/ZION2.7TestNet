import express from 'express';
import net from 'net';
import { createHash, randomBytes } from 'crypto';

// Import our ZION Cosmic Harmony algorithm
import { ZionCosmicHarmonyTest } from './zion-cosmic-harmony-mining.js';

class MiningPool {
  constructor() {
    this.status = 'stopped';
    this.miners = new Map();
    this.jobs = new Map();
    this.shares = [];
    
    // Initialize ZION Cosmic Harmony algorithm
    this.zionAlgorithm = new ZionCosmicHarmonyTest();
    console.log('🌟 ZION Cosmic Harmony algorithm initialized for hybrid mining');
    this.blocks = [];
    
    this.config = {
      port: process.env.POOL_PORT || 3333,
      difficulty: process.env.POOL_DIFFICULTY || 1000,
      fee: process.env.POOL_FEE || 1.0,
      payout_threshold: process.env.PAYOUT_THRESHOLD || 1000000
    };
    
    this.stats = {
      hashrate: 0,
      miners_active: 0,
      shares_accepted: 0,
      shares_rejected: 0,
      blocks_found: 0,
      last_block: null
    };
    
    this.router = express.Router();
    this.setupRoutes();
  }

  async initialize() {
    console.log('🏊 Initializing mining pool...');
    
    try {
      await this.startStratumServer();
      console.log(`✅ Mining pool initialized on port ${this.config.port}`);
      this.status = 'ready';
    } catch (error) {
      console.error('❌ Mining pool initialization failed:', error);
      this.status = 'error';
    }
  }

  setupRoutes() {
    // Pool status
    this.router.get('/status', (req, res) => {
      res.json({
        status: this.status,
        config: this.config,
        stats: this.stats,
        miners: this.miners.size
      });
    });

    // Pool statistics
    this.router.get('/stats', (req, res) => {
      res.json({
        ...this.stats,
        miners: Array.from(this.miners.values()).map(miner => ({
          id: miner.id,
          address: miner.address,
          hashrate: miner.hashrate,
          shares: miner.shares,
          connected_at: miner.connected_at
        }))
      });
    });

    // Miner details
    this.router.get('/miner/:id', (req, res) => {
      const miner = this.miners.get(req.params.id);
      if (miner) {
        res.json(miner);
      } else {
        res.status(404).json({ error: 'Miner not found' });
      }
    });

    // Pool configuration
    this.router.post('/config', (req, res) => {
      try {
        const { difficulty, fee, payout_threshold } = req.body;
        
        if (difficulty) this.config.difficulty = difficulty;
        if (fee) this.config.fee = fee;
        if (payout_threshold) this.config.payout_threshold = payout_threshold;
        
        console.log('⚙️ Pool configuration updated');
        res.json({ success: true, config: this.config });
      } catch (error) {
        res.status(500).json({ error: error.message });
      }
    });

    // Manual payout
    this.router.post('/payout', async (req, res) => {
      try {
        const { miner_id, amount } = req.body;
        const result = await this.processPayout(miner_id, amount);
        res.json(result);
      } catch (error) {
        res.status(500).json({ error: error.message });
      }
    });
  }

  async startStratumServer() {
    return new Promise((resolve, reject) => {
      this.stratumServer = net.createServer((socket) => {
        const miner = this.createMiner(socket);
        
        socket.on('data', (data) => {
          this.handleStratumMessage(miner, data);
        });
        
        socket.on('close', () => {
          this.removeMiner(miner.id);
        });
        
        socket.on('error', (error) => {
          console.error('Miner socket error:', error);
          this.removeMiner(miner.id);
        });
      });
      
      this.stratumServer.listen(this.config.port, (error) => {
        if (error) {
          reject(error);
        } else {
          console.log(`🌊 Stratum server listening on port ${this.config.port}`);
          resolve();
        }
      });
    });
  }

  createMiner(socket) {
    const miner_id = randomBytes(8).toString('hex');
    
    const miner = {
      id: miner_id,
      socket,
      address: null,
      worker: null,
      hashrate: 0,
      shares: 0,
      last_activity: Date.now(),
      connected_at: Date.now(),
      difficulty: this.config.difficulty,
      subscribed: false
    };
    
    this.miners.set(miner_id, miner);
    console.log(`⛏️  New miner connected: ${miner_id}`);
    
    return miner;
  }

  removeMiner(miner_id) {
    if (this.miners.has(miner_id)) {
      console.log(`👋 Miner disconnected: ${miner_id}`);
      this.miners.delete(miner_id);
      this.updateStats();
    }
  }

  handleStratumMessage(miner, data) {
    const lines = data.toString().split('\n');
    
    for (const line of lines) {
      if (line.trim()) {
        try {
          const message = JSON.parse(line);
          this.processStratumMessage(miner, message);
        } catch (error) {
          console.error('Invalid JSON from miner:', error);
        }
      }
    }
  }

  processStratumMessage(miner, message) {
    const { id, method, params } = message;
    
    switch (method) {
      case 'mining.subscribe':
        this.handleMiningSubscribe(miner, id, params);
        break;
        
      case 'mining.authorize':
        this.handleMiningAuthorize(miner, id, params);
        break;
        
      case 'mining.submit':
        this.handleMiningSubmit(miner, id, params);
        break;
        
      // XMRig/CryptoNote protocol support
      case 'login':
        this.handleXMRigLogin(miner, id, params);
        break;
        
      case 'getjob':
        this.handleXMRigGetJob(miner, id, params);
        break;
        
      case 'submit':
        this.handleXMRigSubmit(miner, id, params);
        break;
        
      default:
        console.log(`Unknown method: ${method}`);
    }
  }

  handleMiningSubscribe(miner, id, params) {
    miner.subscribed = true;
    
    const response = {
      id,
      result: [
        ["mining.notify", miner.id],
        miner.id,
        4 // ExtraNonce1 size
      ],
      error: null
    };
    
    miner.socket.write(JSON.stringify(response) + '\n');
    
    // Send initial job
    this.sendJob(miner);
  }

  handleMiningAuthorize(miner, id, params) {
    const [username, password] = params;
    
    miner.address = username.split('.')[0];
    miner.worker = username.split('.')[1] || 'default';
    
    const response = {
      id,
      result: true,
      error: null
    };
    
    miner.socket.write(JSON.stringify(response) + '\n');
    
    console.log(`✅ Miner authorized: ${miner.address}.${miner.worker}`);
  }

  handleMiningSubmit(miner, id, params) {
    const [username, job_id, extranonce2, ntime, nonce] = params;
    
    miner.last_activity = Date.now();
    miner.shares++;
    
    // Validate share (simplified)
    const isValid = this.validateShare(miner, job_id, extranonce2, ntime, nonce);
    
    const response = {
      id,
      result: isValid,
      error: isValid ? null : { code: 23, message: 'Low difficulty share' }
    };
    
    miner.socket.write(JSON.stringify(response) + '\n');
    
    if (isValid) {
      this.stats.shares_accepted++;
      this.processValidShare(miner, job_id, nonce);
    } else {
      this.stats.shares_rejected++;
    }
    
    this.updateMinerHashrate(miner);
  }

  validateShare(miner, job_id, extranonce2, ntime, nonce) {
    // Simplified share validation
    const job = this.jobs.get(job_id);
    if (!job) return false;
    
    // In a real implementation, this would verify the hash meets difficulty
    return Math.random() > 0.05; // 95% valid shares
  }

  processValidShare(miner, job_id, nonce) {
    const share = {
      miner_id: miner.id,
      job_id,
      nonce,
      timestamp: Date.now(),
      difficulty: miner.difficulty
    };
    
    this.shares.push(share);
    
    // Check if it's a block
    if (Math.random() < 0.001) { // 0.1% chance of finding a block
      this.processFoundBlock(share);
    }
  }

  processFoundBlock(share) {
    const block = {
      height: this.blocks.length + 1,
      hash: randomBytes(32).toString('hex'),
      miner_id: share.miner_id,
      timestamp: Date.now(),
      difficulty: share.difficulty,
      reward: 333000000 // 333 ZION in atomic units
    };
    
    this.blocks.push(block);
    this.stats.blocks_found++;
    this.stats.last_block = Date.now();
    
    console.log(`🎉 Block found! Height: ${block.height}, Miner: ${share.miner_id}`);
    
    // Distribute rewards
    this.distributeBlockReward(block);
  }

  distributeBlockReward(block) {
    const totalShares = this.shares.length;
    if (totalShares === 0) return;
    
    const reward = block.reward * (1 - this.config.fee / 100);
    
    // Group shares by miner
    const minerShares = new Map();
    this.shares.forEach(share => {
      const count = minerShares.get(share.miner_id) || 0;
      minerShares.set(share.miner_id, count + 1);
    });
    
    // Calculate payouts
    minerShares.forEach((shares, miner_id) => {
      const payout = Math.floor((shares / totalShares) * reward);
      console.log(`💰 Payout: ${miner_id} gets ${payout} atomic units`);
    });
    
    // Clear shares for next round
    this.shares = [];
  }

  sendJob(miner) {
    const job_id = randomBytes(4).toString('hex');
    const prevhash = randomBytes(32).toString('hex');
    const coinb1 = randomBytes(32).toString('hex');
    const coinb2 = randomBytes(32).toString('hex');
    const merkle_branch = [];
    const version = '00000001';
    const nbits = '1e0ffff0';
    const ntime = Math.floor(Date.now() / 1000).toString(16);
    const clean_jobs = true;
    
    const job = {
      id: job_id,
      prevhash,
      coinb1,
      coinb2,
      merkle_branch,
      version,
      nbits,
      ntime,
      clean_jobs,
      created_at: Date.now()
    };
    
    this.jobs.set(job_id, job);
    
    const notification = {
      id: null,
      method: 'mining.notify',
      params: [
        job_id,
        prevhash,
        coinb1,
        coinb2,
        merkle_branch,
        version,
        nbits,
        ntime,
        clean_jobs
      ]
    };
    
    miner.socket.write(JSON.stringify(notification) + '\n');
  }

  updateMinerHashrate(miner) {
    const now = Date.now();
    const timeWindow = 60000; // 1 minute
    
    // Simple hashrate calculation based on shares
    miner.hashrate = (miner.shares * miner.difficulty) / (timeWindow / 1000);
    
    this.updateStats();
  }

  updateStats() {
    this.stats.miners_active = this.miners.size;
    this.stats.hashrate = Array.from(this.miners.values())
      .reduce((sum, miner) => sum + miner.hashrate, 0);
  }

  async processPayout(miner_id, amount) {
    console.log(`💳 Processing payout: ${amount} to ${miner_id}`);
    
    // Simulate payout processing
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    return {
      miner_id,
      amount,
      transaction_id: randomBytes(32).toString('hex'),
      status: 'completed',
      timestamp: Date.now()
    };
  }

  getRouter() {
    return this.router;
  }

  getStatus() {
    return {
      status: this.status,
      miners: this.stats.miners_active,
      hashrate: this.stats.hashrate,
      blocks_found: this.stats.blocks_found
    };
  }

  getStats() {
    return this.stats;
  }

  // XMRig/CryptoNote protocol handlers
  handleXMRigLogin(miner, id, params) {
    console.log(`🎯 XMRig login from ${miner.id}:`, params);
    
    // Extract wallet address and worker name
    const { login, pass, agent } = params;
    miner.wallet = login;
    miner.worker = pass || 'default';
    miner.agent = agent || 'xmrig';
    miner.authorized = true;
    
    const response = {
      id,
      jsonrpc: "2.0",
      result: {
        id: miner.id,
        job: this.generateRandomXJob(),
        status: "OK"
      }
    };
    
    miner.socket.write(JSON.stringify(response) + '\n');
    console.log(`✅ XMRig miner ${miner.id} logged in successfully`);
  }

  handleXMRigGetJob(miner, id, params) {
    const response = {
      id,
      jsonrpc: "2.0",
      result: this.generateRandomXJob()
    };
    
    miner.socket.write(JSON.stringify(response) + '\n');
  }

  handleXMRigSubmit(miner, id, params) {
    console.log(`📤 Share submitted by ${miner.id}:`, params);
    
    // Process share submission
    const { job_id, nonce, result } = params;
    
    // Validate share (simplified)
    const isValid = this.validateRandomXShare(job_id, nonce, result);
    
    if (isValid) {
      this.stats.shares_accepted++;
      miner.shares_accepted = (miner.shares_accepted || 0) + 1;
      console.log(`✅ ACCEPTED SHARE from ${miner.id}! Total: ${this.stats.shares_accepted}`);
    } else {
      this.stats.shares_rejected++;
      miner.shares_rejected = (miner.shares_rejected || 0) + 1;
      console.log(`❌ Rejected share from ${miner.id}`);
    }

    const response = {
      id,
      jsonrpc: "2.0",
      result: {
        status: isValid ? "OK" : "REJECTED"
      }
    };
    
    miner.socket.write(JSON.stringify(response) + '\n');
  }

  generateRandomXJob() {
    const jobId = `zion_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    // Generate cosmic seed using our algorithm
    const cosmicSeed = this.zionAlgorithm.hash(`cosmic_job_${jobId}_${Date.now()}`);
    
    console.log(`🌟 Generated ZION Cosmic job: ${jobId}`);
    
    return {
      blob: "0606e4a4d005b27e470181a2928b5a147c717b4f1c1a5420bb2cf88b7b92c95c6a03b95a2a00000000b194c01576ba2124bef7b76fce73ffb5b53a20e1cc6a23a1e4a0ba7e20ee7030101",
      job_id: jobId,
      target: "00ffffff", // Easier target for testing
      height: Math.floor(Date.now() / 1000), // Use timestamp as height
      seed_hash: cosmicSeed.toString('hex').substr(0, 64),
      algorithm: "zion-cosmic-harmony"
    };
  }

  validateRandomXShare(jobId, nonce, result) {
    console.log(`🔮 Validating share using ZION Cosmic Harmony algorithm...`);
    console.log(`   Job ID: ${jobId}`);
    console.log(`   Nonce: ${nonce}`);
    console.log(`   Result: ${result}`);
    
    // Basic format validation
    if (!result || result.length !== 64 || !/^[a-f0-9]+$/i.test(result)) {
      console.log('❌ Invalid result format');
      return false;
    }
    
    try {
      // Use our ZION Cosmic Harmony algorithm for validation
      const inputData = `${jobId}:${nonce}:${result}`;
      const zionHash = this.zionAlgorithm.hash(inputData);
      
      // Check if the hash meets our difficulty target
      const hashString = zionHash.toString('hex');
      const difficulty = this.calculateDifficulty(hashString);
      
      console.log(`🌟 ZION hash: ${hashString}`);
      console.log(`⚡ Difficulty: ${difficulty}`);
      
      // Accept shares that meet minimum difficulty (4 leading zeros for now)
      const isValid = hashString.startsWith('0000');
      
      if (isValid) {
        console.log('✅ ZION Cosmic Harmony validation PASSED!');
      } else {
        console.log('❌ ZION Cosmic Harmony validation failed - insufficient difficulty');
      }
      
      return isValid;
      
    } catch (error) {
      console.error('❌ ZION validation error:', error);
      return false;
    }
  }
  
  calculateDifficulty(hashString) {
    // Count leading zeros as simple difficulty measure
    let difficulty = 0;
    for (const char of hashString) {
      if (char === '0') difficulty++;
      else break;
    }
    return difficulty;
  }

  async shutdown() {
    console.log('🏊 Shutting down mining pool...');
    
    if (this.stratumServer) {
      this.stratumServer.close();
    }
    
    // Disconnect all miners
    this.miners.forEach(miner => {
      miner.socket.end();
    });
    
    this.miners.clear();
    this.status = 'stopped';
  }
}

export default MiningPool;
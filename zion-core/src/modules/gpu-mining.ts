import { Router } from 'express';
import { spawn, ChildProcess } from 'child_process';
import {
  IGPUMining,
  GPUStats,
  GPUDevice,
  GPUBenchmarkResult,
  MiningConfig,
  ModuleStatus,
  ZionError,
  GPUError
} from '../types.js';

/**
 * ZION GPU Mining Module
 * 
 * Multi-vendor GPU support (NVIDIA, AMD, Intel) with performance monitoring,
 * benchmarking, and Lightning Network acceleration capabilities.
 */
export class GPUMining implements IGPUMining {
  private status: ModuleStatus['status'] = 'stopped';
  private startTime: number = 0;
  private router: Router;
  
  // GPU state
  private gpus: Map<number, GPUDevice> = new Map();
  private miningProcesses: Map<number, ChildProcess> = new Map();
  private benchmarkResults: Map<number, GPUBenchmarkResult> = new Map();
  
  // Performance tracking
  private performanceInterval: NodeJS.Timeout | null = null;

  constructor() {
    this.router = Router();
    this.setupRoutes();
  }

  private setupRoutes(): void {
    // GPU statistics
    this.router.get('/stats', (_req, res) => {
      res.json(this.getStats());
    });

    // List available GPUs
    this.router.get('/devices', (_req, res) => {
      res.json(Array.from(this.gpus.values()));
    });

    // Start mining
    this.router.post('/start', async (_req, res) => {
      try {
        const config: MiningConfig = _req.body;
        const result = await this.startMining(config);
        res.json(result);
      } catch (error) {
        const err = error as Error;
        res.status(400).json({ error: err.message });
      }
    });

    // Stop mining
    this.router.post('/stop', async (_req, res) => {
      try {
        const { gpuIds } = _req.body;
        const result = await this.stopMining(gpuIds);
        res.json(result);
      } catch (error) {
        const err = error as Error;
        res.status(400).json({ error: err.message });
      }
    });

    // Run benchmark
    this.router.post('/benchmark', async (_req, res) => {
      try {
        const { gpuId, duration = 60 } = _req.body;
        const result = await this.runBenchmark(gpuId, duration);
        res.json(result);
      } catch (error) {
        const err = error as Error;
        res.status(400).json({ error: err.message });
      }
    });

    // GPU performance data
    this.router.get('/performance', (_req, res) => {
      res.json(Array.from(this.benchmarkResults.values()));
    });
  }

  public async initialize(): Promise<void> {
    console.log('🖥️  Initializing GPU Mining...');
    
    try {
      this.status = 'starting';
      this.startTime = Date.now();
      
      // Detect available GPUs
      await this.detectGPUs();
      
      // Initialize GPU monitoring
      this.startPerformanceMonitoring();
      
      this.status = 'ready';
      console.log(`✅ GPU Mining initialized - Found ${this.gpus.size} GPUs`);
      
    } catch (error) {
      this.status = 'error';
      const err = error as Error;
      console.error('❌ Failed to initialize GPU Mining:', err.message);
      throw new GPUError(`GPU initialization failed: ${err.message}`, 'GPU_INIT_FAILED');
    }
  }

  public async shutdown(): Promise<void> {
    console.log('🖥️  Shutting down GPU Mining...');
    
    try {
      this.status = 'stopped';
      
      // Stop all mining processes
      await this.stopMining();
      
      // Clear performance monitoring
      if (this.performanceInterval) {
        clearInterval(this.performanceInterval);
        this.performanceInterval = null;
      }
      
      console.log('✅ GPU Mining shut down successfully');
      
    } catch (error) {
      const err = error as Error;
      console.error('❌ Error shutting down GPU Mining:', err.message);
      throw error;
    }
  }

  public getStatus(): ModuleStatus {
    return {
      status: this.status,
      uptime: this.startTime ? Date.now() - this.startTime : 0
    };
  }

  public getStats(): GPUStats {
    const gpus = Array.from(this.gpus.values());
    
    return {
      gpus,
      totalHashrate: gpus.reduce((sum, gpu) => sum + gpu.hashrate, 0),
      powerUsage: gpus.reduce((sum, gpu) => sum + gpu.power, 0),
      averageTemperature: gpus.length > 0 
        ? gpus.reduce((sum, gpu) => sum + gpu.temperature, 0) / gpus.length 
        : 0
    };
  }

  public getRouter(): Router {
    return this.router;
  }

  public async startMining(config: MiningConfig): Promise<{ results: Array<{ gpuId: number; success: boolean; error?: string }> }> {
    console.log(`🖥️  Starting mining on GPUs: ${config.gpuIds.join(', ')}`);
    
    const results: Array<{ gpuId: number; success: boolean; error?: string }> = [];
    
    for (const gpuId of config.gpuIds) {
      try {
        const gpu = this.gpus.get(gpuId);
        if (!gpu) {
          results.push({
            gpuId,
            success: false,
            error: `GPU ${gpuId} not found`
          });
          continue;
        }

        if (!gpu.supported) {
          results.push({
            gpuId,
            success: false,
            error: `GPU ${gpuId} not supported for mining`
          });
          continue;
        }

        // Start mining process for this GPU
        await this.startGPUMining(gpuId, config);
        
        // Update GPU status
        this.gpus.set(gpuId, { ...gpu, status: 'mining' });
        
        results.push({
          gpuId,
          success: true
        });
        
        console.log(`✅ Started mining on GPU ${gpuId}: ${gpu.name}`);
        
      } catch (error) {
        const err = error as Error;
        results.push({
          gpuId,
          success: false,
          error: err.message
        });
        console.error(`❌ Failed to start mining on GPU ${gpuId}:`, err.message);
      }
    }
    
    return { results };
  }

  public async stopMining(gpuIds?: number[]): Promise<{ results: Array<{ gpuId: number; success: boolean }> }> {
    const targetGpuIds = gpuIds || Array.from(this.gpus.keys());
    console.log(`🖥️  Stopping mining on GPUs: ${targetGpuIds.join(', ')}`);
    
    const results: Array<{ gpuId: number; success: boolean }> = [];
    
    for (const gpuId of targetGpuIds) {
      try {
        // Stop mining process
        const process = this.miningProcesses.get(gpuId);
        if (process) {
          process.kill('SIGTERM');
          this.miningProcesses.delete(gpuId);
        }
        
        // Update GPU status
        const gpu = this.gpus.get(gpuId);
        if (gpu) {
          this.gpus.set(gpuId, { 
            ...gpu, 
            status: 'idle',
            hashrate: 0
          });
        }
        
        results.push({
          gpuId,
          success: true
        });
        
        console.log(`✅ Stopped mining on GPU ${gpuId}`);
        
      } catch (error) {
        const err = error as Error;
        results.push({
          gpuId,
          success: false
        });
        console.error(`❌ Failed to stop mining on GPU ${gpuId}:`, err.message);
      }
    }
    
    return { results };
  }

  public async runBenchmark(gpuId: number, duration: number): Promise<GPUBenchmarkResult> {
    console.log(`🖥️  Running benchmark on GPU ${gpuId} for ${duration} seconds...`);
    
    const gpu = this.gpus.get(gpuId);
    if (!gpu) {
      throw new GPUError(`GPU ${gpuId} not found`, 'GPU_NOT_FOUND');
    }

    if (!gpu.supported) {
      throw new GPUError(`GPU ${gpuId} not supported for benchmarking`, 'GPU_NOT_SUPPORTED');
    }

    // Update GPU status
    this.gpus.set(gpuId, { ...gpu, status: 'benchmark' });

    try {
      // Run benchmark (mock implementation)
      const startTime = Date.now();
      
      // Simulate benchmark process
      await new Promise(resolve => setTimeout(resolve, Math.min(duration * 1000, 5000)));
      
      const actualDuration = (Date.now() - startTime) / 1000;
      
      // Generate mock benchmark results
      const hashrate = this.calculateBenchmarkHashrate(gpu);
      const powerUsage = this.calculatePowerUsage(gpu, hashrate);
      const temperature = 65 + Math.random() * 20; // 65-85°C
      const efficiency = hashrate / powerUsage; // MH/s per Watt
      const score = hashrate * efficiency; // Overall performance score
      
      const result: GPUBenchmarkResult = {
        gpuId,
        gpu: gpu.name,
        duration: actualDuration,
        hashrate,
        powerUsage,
        temperature,
        efficiency,
        score
      };
      
      // Store benchmark result
      this.benchmarkResults.set(gpuId, result);
      
      // Update GPU with benchmark data
      this.gpus.set(gpuId, {
        ...gpu,
        status: 'idle',
        hashrate: 0,
        power: powerUsage,
        temperature
      });
      
      console.log(`✅ Benchmark completed for GPU ${gpuId}: ${hashrate.toFixed(2)} MH/s`);
      
      return result;
      
    } catch (error) {
      // Reset GPU status on error
      this.gpus.set(gpuId, { ...gpu, status: 'error' });
      const err = error as Error;
      throw new GPUError(`Benchmark failed for GPU ${gpuId}: ${err.message}`, 'BENCHMARK_FAILED');
    }
  }

  public async accelerateLightningOperation(operation: string, data: unknown): Promise<unknown> {
    console.log(`⚡ Accelerating Lightning Network operation: ${operation}`);
    
    // Select best GPU for Lightning Network acceleration
    const bestGpu = this.selectBestGPUForAcceleration();
    
    if (!bestGpu) {
      console.log('⚡ No suitable GPU available for acceleration, using CPU');
      return this.cpuFallback(operation, data);
    }
    
    try {
      // GPU-accelerated Lightning Network operations
      switch (operation) {
        case 'route_calculation':
          return await this.accelerateRouteCalculation(bestGpu.id, data);
        
        case 'cryptography':
          return await this.accelerateCryptography(bestGpu.id, data);
        
        case 'payment_verification':
          return await this.acceleratePaymentVerification(bestGpu.id, data);
        
        default:
          console.log(`⚡ Unknown operation: ${operation}, using CPU fallback`);
          return this.cpuFallback(operation, data);
      }
      
    } catch (error) {
      const err = error as Error;
      console.error(`⚡ GPU acceleration failed: ${err.message}, falling back to CPU`);
      return this.cpuFallback(operation, data);
    }
  }

  private async detectGPUs(): Promise<void> {
    console.log('🔍 Detecting GPUs...');
    
    // Mock GPU detection (in real implementation, this would use nvidia-ml-py, rocm-smi, etc.)
    const mockGPUs: GPUDevice[] = [
      {
        id: 0,
        name: 'NVIDIA GeForce RTX 4090',
        vendor: 'NVIDIA',
        vram: 24576, // 24GB
        bus: 'PCI-E 4.0 x16',
        supported: true,
        status: 'idle',
        hashrate: 0,
        power: 0,
        temperature: 45
      },
      {
        id: 1,
        name: 'AMD Radeon RX 7900 XTX',
        vendor: 'AMD',
        vram: 24576, // 24GB
        bus: 'PCI-E 4.0 x16',
        supported: true,
        status: 'idle',
        hashrate: 0,
        power: 0,
        temperature: 42
      },
      {
        id: 2,
        name: 'Intel Arc A770',
        vendor: 'Intel',
        vram: 16384, // 16GB
        bus: 'PCI-E 4.0 x16',
        supported: true,
        status: 'idle',
        hashrate: 0,
        power: 0,
        temperature: 38
      }
    ];
    
    // Add detected GPUs to map
    mockGPUs.forEach(gpu => {
      this.gpus.set(gpu.id, gpu);
    });
    
    console.log(`🔍 Detected ${mockGPUs.length} GPUs:`);
    mockGPUs.forEach(gpu => {
      console.log(`  - GPU ${gpu.id}: ${gpu.name} (${gpu.vram}MB VRAM)`);
    });
  }

  private async startGPUMining(gpuId: number, config: MiningConfig): Promise<void> {
    const gpu = this.gpus.get(gpuId);
    if (!gpu) {
      throw new GPUError(`GPU ${gpuId} not found`, 'GPU_NOT_FOUND');
    }
    
    // Create mining command based on GPU vendor
    const command = this.createMiningCommand(gpu, config);
    
    // Start mining process
    const process = spawn(command.executable, command.args, {
      stdio: ['pipe', 'pipe', 'pipe']
    });
    
    this.miningProcesses.set(gpuId, process);
    
    // Handle process output
    process.stdout?.on('data', (data) => {
      this.parseMiningOutput(gpuId, data.toString());
    });
    
    process.stderr?.on('data', (data) => {
      console.error(`GPU ${gpuId} mining error:`, data.toString());
    });
    
    process.on('exit', (code) => {
      console.log(`GPU ${gpuId} mining process exited with code ${code}`);
      this.miningProcesses.delete(gpuId);
      
      // Update GPU status
      if (gpu) {
        this.gpus.set(gpuId, { ...gpu, status: 'idle', hashrate: 0 });
      }
    });
  }

  private createMiningCommand(gpu: GPUDevice, config: MiningConfig): { executable: string; args: string[] } {
    // Create mining command based on GPU vendor and algorithm
    switch (gpu.vendor) {
      case 'NVIDIA':
        return {
          executable: 'xmrig-cuda', // Example CUDA miner
          args: [
            '--algo', config.algorithm,
            '--url', config.pool,
            '--user', config.wallet,
            '--cuda-devices', gpu.id.toString()
          ]
        };
        
      case 'AMD':
        return {
          executable: 'xmrig-opencl', // Example OpenCL miner
          args: [
            '--algo', config.algorithm,
            '--url', config.pool,
            '--user', config.wallet,
            '--opencl-devices', gpu.id.toString()
          ]
        };
        
      case 'Intel':
        return {
          executable: 'xmrig-opencl', // Intel Arc uses OpenCL
          args: [
            '--algo', config.algorithm,
            '--url', config.pool,
            '--user', config.wallet,
            '--opencl-devices', gpu.id.toString()
          ]
        };
        
      default:
        throw new GPUError(`Unsupported GPU vendor: ${gpu.vendor}`, 'UNSUPPORTED_VENDOR');
    }
  }

  private parseMiningOutput(gpuId: number, output: string): void {
    // Parse mining output to extract hashrate and other metrics
    const gpu = this.gpus.get(gpuId);
    if (!gpu) return;
    
    // Mock parsing (in real implementation, this would parse actual miner output)
    const hashrateMatch = output.match(/(\d+\.?\d*)\s*(H|KH|MH|GH)\/s/);
    if (hashrateMatch) {
      let hashrate = parseFloat(hashrateMatch[1]);
      const unit = hashrateMatch[2];
      
      // Convert to MH/s
      switch (unit) {
        case 'H': hashrate /= 1000000; break;
        case 'KH': hashrate /= 1000; break;
        case 'GH': hashrate *= 1000; break;
      }
      
      // Update GPU stats
      this.gpus.set(gpuId, {
        ...gpu,
        hashrate,
        power: this.calculatePowerUsage(gpu, hashrate),
        temperature: 70 + Math.random() * 15, // Mock temperature 70-85°C
        status: 'mining'
      });
    }
  }

  private calculateBenchmarkHashrate(gpu: GPUDevice): number {
    // Calculate expected hashrate based on GPU specifications
    const baseHashrate = {
      'NVIDIA GeForce RTX 4090': 125, // MH/s
      'AMD Radeon RX 7900 XTX': 90,
      'Intel Arc A770': 45
    };
    
    return (baseHashrate[gpu.name as keyof typeof baseHashrate] || 30) * (0.9 + Math.random() * 0.2);
  }

  private calculatePowerUsage(gpu: GPUDevice, hashrate: number): number {
    // Calculate power usage based on hashrate and GPU model
    const powerEfficiency = {
      'NVIDIA GeForce RTX 4090': 0.5, // MH/s per Watt
      'AMD Radeon RX 7900 XTX': 0.4,
      'Intel Arc A770': 0.35
    };
    
    const efficiency = powerEfficiency[gpu.name as keyof typeof powerEfficiency] || 0.3;
    return hashrate / efficiency;
  }

  private selectBestGPUForAcceleration(): GPUDevice | null {
    // Select GPU with best compute capability and lowest current load
    const availableGPUs = Array.from(this.gpus.values())
      .filter(gpu => gpu.supported && gpu.status !== 'mining');
    
    if (availableGPUs.length === 0) return null;
    
    // Prefer NVIDIA for CUDA acceleration, then AMD, then Intel
    const priorities = { 'NVIDIA': 3, 'AMD': 2, 'Intel': 1 };
    
    return availableGPUs.sort((a, b) => {
      const priorityDiff = (priorities[b.vendor as keyof typeof priorities] || 0) - 
                          (priorities[a.vendor as keyof typeof priorities] || 0);
      if (priorityDiff !== 0) return priorityDiff;
      return b.vram - a.vram; // Prefer more VRAM
    })[0];
  }

  private async accelerateRouteCalculation(gpuId: number, data: unknown): Promise<unknown> {
    console.log(`⚡ GPU ${gpuId} accelerating route calculation...`);
    // Mock GPU-accelerated route calculation
    await new Promise(resolve => setTimeout(resolve, 50)); // Faster than CPU
    return { 
      ...(data as object || {}), 
      gpuAccelerated: true, 
      processingTime: 50 
    };
  }

  private async accelerateCryptography(gpuId: number, data: unknown): Promise<unknown> {
    console.log(`⚡ GPU ${gpuId} accelerating cryptographic operations...`);
    // Mock GPU-accelerated cryptography
    await new Promise(resolve => setTimeout(resolve, 20)); // Much faster than CPU
    return { 
      ...(data as object || {}), 
      gpuAccelerated: true, 
      processingTime: 20 
    };
  }

  private async acceleratePaymentVerification(gpuId: number, data: unknown): Promise<unknown> {
    console.log(`⚡ GPU ${gpuId} accelerating payment verification...`);
    // Mock GPU-accelerated payment verification
    await new Promise(resolve => setTimeout(resolve, 30));
    return { 
      ...(data as object || {}), 
      gpuAccelerated: true, 
      processingTime: 30 
    };
  }

  private async cpuFallback(operation: string, data: unknown): Promise<unknown> {
    console.log(`💻 CPU fallback for operation: ${operation}`);
    // Mock CPU processing (slower)
    await new Promise(resolve => setTimeout(resolve, 200));
    return { 
      ...(data as object || {}), 
      gpuAccelerated: false, 
      processingTime: 200 
    };
  }

  private startPerformanceMonitoring(): void {
    // Monitor GPU performance every 10 seconds
    this.performanceInterval = setInterval(() => {
      this.updateGPUStats();
    }, 10000);
  }

  private updateGPUStats(): void {
    // Update GPU statistics (temperature, power, etc.)
    this.gpus.forEach((gpu, id) => {
      if (gpu.status === 'mining') {
        // Simulate fluctuating stats during mining
        const tempVariation = (Math.random() - 0.5) * 5; // ±2.5°C
        const powerVariation = (Math.random() - 0.5) * 20; // ±10W
        
        this.gpus.set(id, {
          ...gpu,
          temperature: Math.max(40, Math.min(90, gpu.temperature + tempVariation)),
          power: Math.max(100, gpu.power + powerVariation)
        });
      }
    });
  }
}
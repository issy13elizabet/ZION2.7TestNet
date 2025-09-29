// 🌈 MULTI-CHAIN BRIDGE MANAGER - Central Hub for Cross-Chain Operations

import { EventEmitter } from 'events';
import { SolanaBridge } from './solana-bridge';
import { StellarBridge } from './stellar-bridge';
import { CardanoBridge } from './cardano-bridge';
import { TronBridge } from './tron-bridge';
import { RealDataManager } from '../modules/real-data-manager';
import { EnhancedDaemonBridge } from '../modules/daemon-bridge';

interface BridgeConfig {
    bridges: {
        solana?: SolanaBridge;
        stellar?: StellarBridge;
        cardano?: CardanoBridge;
        tron?: TronBridge;
    };
    zionCore: {
        realDataManager: RealDataManager;
        daemonBridge: EnhancedDaemonBridge;
    };
}

interface MultiChainStatus {
    bridges: Record<string, BridgeStatus>;
    totalVolume: number;
    activeChains: number;
    rainbowBridgeStatus: string;
}

interface BridgeStatus {
    connected: boolean;
    lastSync: Date;
    blockHeight: number;
    pendingTransactions: number;
    totalVolume: number;
}

interface CrossChainTransfer {
    id: string;
    fromChain: string;
    toChain: string;
    amount: number;
    recipient: string;
    status: 'pending' | 'confirmed' | 'failed';
    lockTxHash?: string;
    mintTxHash?: string;
    timestamp: Date;
}

interface MultiChainMetrics {
    zionCore: {
        blockHeight: number;
        hashRate: number;
        difficulty: number;
        peers: number;
    };
    bridges: Record<string, BridgeMetrics>;
    crossChainVolume: {
        daily: number;
        total: number;
    };
    rainbowBridge: {
        frequency: number;
        status: string;
        dimensionalGateways: number;
    };
}

interface BridgeMetrics {
    connected: boolean;
    latency: number;
    throughput: number;
    errorRate: number;
    lastUpdate: Date;
}

export class MultiChainBridgeManager extends EventEmitter {
    public bridges: Map<string, any> = new Map();
    private config: BridgeConfig;
    private transfers: Map<string, CrossChainTransfer> = new Map();
    private metrics: Map<string, BridgeMetrics> = new Map();
    private isInitialized: boolean = false;

    constructor(config: BridgeConfig) {
        super();
        this.config = config;
    }

    async initializeAllBridges(): Promise<void> {
        console.log('🌈 Initializing Multi-Chain Bridges...');
        
        const bridgeNames = Object.keys(this.config.bridges) as Array<keyof typeof this.config.bridges>;
        
        for (const bridgeName of bridgeNames) {
            const bridge = this.config.bridges[bridgeName];
            if (bridge) {
                try {
                    console.log(`🔄 Initializing ${bridgeName} bridge...`);
                    await bridge.initialize();
                    await this.testBridgeConnection(bridge, bridgeName);
                    
                    this.bridges.set(bridgeName, bridge);
                    this.initializeBridgeMetrics(bridgeName);
                    
                    console.log(`✅ ${bridgeName.toUpperCase()} bridge connected`);
                } catch (error) {
                    console.error(`❌ ${bridgeName.toUpperCase()} bridge failed:`, error);
                    // Continue with other bridges even if one fails
                }
            }
        }

        this.setupBridgeMonitoring();
        this.isInitialized = true;
        
        console.log(`🚀 Multi-Chain Bridge Manager initialized with ${this.bridges.size} bridges`);
    }

    private async testBridgeConnection(bridge: any, bridgeName: string): Promise<void> {
        const startTime = Date.now();
        
        try {
            await bridge.testConnection();
            const latency = Date.now() - startTime;
            
            this.updateBridgeMetrics(bridgeName, {
                connected: true,
                latency,
                throughput: 0,
                errorRate: 0,
                lastUpdate: new Date()
            });
        } catch (error) {
            this.updateBridgeMetrics(bridgeName, {
                connected: false,
                latency: -1,
                throughput: 0,
                errorRate: 1,
                lastUpdate: new Date()
            });
            throw error;
        }
    }

    private initializeBridgeMetrics(bridgeName: string): void {
        this.metrics.set(bridgeName, {
            connected: false,
            latency: 0,
            throughput: 0,
            errorRate: 0,
            lastUpdate: new Date()
        });
    }

    private updateBridgeMetrics(bridgeName: string, metrics: Partial<BridgeMetrics>): void {
        const currentMetrics = this.metrics.get(bridgeName) || {
            connected: false,
            latency: 0,
            throughput: 0,
            errorRate: 0,
            lastUpdate: new Date()
        };
        
        this.metrics.set(bridgeName, { ...currentMetrics, ...metrics });
    }

    private setupBridgeMonitoring(): void {
        // Monitor bridge health every 30 seconds
        setInterval(async () => {
            for (const [bridgeName, bridge] of this.bridges) {
                try {
                    const startTime = Date.now();
                    const health = await bridge.getHealth();
                    const latency = Date.now() - startTime;
                    
                    this.updateBridgeMetrics(bridgeName, {
                        connected: health.connected,
                        latency,
                        lastUpdate: new Date()
                    });
                } catch (error) {
                    console.error(`❌ Bridge health check failed for ${bridgeName}:`, error);
                    this.updateBridgeMetrics(bridgeName, {
                        connected: false,
                        errorRate: 1,
                        lastUpdate: new Date()
                    });
                }
            }
        }, 30000);
    }

    async executeCrossChainTransfer(
        fromChain: string,
        toChain: string,
        amount: number,
        recipient: string
    ): Promise<string> {
        if (!this.isInitialized) {
            throw new Error('Bridge manager not initialized');
        }

        const transferId = this.generateTransferId();
        
        const transfer: CrossChainTransfer = {
            id: transferId,
            fromChain,
            toChain,
            amount,
            recipient,
            status: 'pending',
            timestamp: new Date()
        };

        this.transfers.set(transferId, transfer);

        try {
            console.log(`🌈 Executing cross-chain transfer: ${fromChain} -> ${toChain}`);
            console.log(`   Amount: ${amount}, Recipient: ${recipient}`);

            // Get bridges
            const sourceBridge = this.bridges.get(fromChain);
            const targetBridge = this.bridges.get(toChain);
            
            if (!sourceBridge || !targetBridge) {
                throw new Error(`Bridge not found: ${fromChain} -> ${toChain}`);
            }

            // Step 1: Lock tokens on source chain
            console.log(`🔒 Locking ${amount} tokens on ${fromChain}...`);
            const lockTxHash = await sourceBridge.lockTokens(amount, recipient, transferId);
            
            transfer.lockTxHash = lockTxHash;
            transfer.status = 'pending';
            this.transfers.set(transferId, transfer);
            
            console.log(`✅ Tokens locked on ${fromChain}: ${lockTxHash}`);

            // Step 2: Wait for confirmation
            await this.waitForConfirmation(sourceBridge, lockTxHash);
            
            // Step 3: Mint tokens on target chain
            console.log(`🪙 Minting ${amount} tokens on ${toChain}...`);
            const mintTxHash = await targetBridge.mintTokens(amount, recipient, lockTxHash);
            
            transfer.mintTxHash = mintTxHash;
            transfer.status = 'confirmed';
            this.transfers.set(transferId, transfer);
            
            console.log(`✅ Tokens minted on ${toChain}: ${mintTxHash}`);
            console.log(`🌈 Cross-chain transfer complete: ${transferId}`);

            // Update metrics
            this.updateTransferMetrics(fromChain, toChain, amount);

            // Emit event
            this.emit('crossChainTransfer', transfer);
            
            return transferId;

        } catch (error) {
            console.error(`❌ Cross-chain transfer failed:`, error);
            
            transfer.status = 'failed';
            this.transfers.set(transferId, transfer);
            
            throw error;
        }
    }

    private generateTransferId(): string {
        return `transfer_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    private async waitForConfirmation(bridge: any, txHash: string, maxWait: number = 60000): Promise<void> {
        const startTime = Date.now();
        
        while (Date.now() - startTime < maxWait) {
            try {
                const confirmed = await bridge.isTransactionConfirmed(txHash);
                if (confirmed) {
                    return;
                }
            } catch (error) {
                console.warn(`Warning: Failed to check confirmation for ${txHash}`);
            }
            
            // Wait 5 seconds before checking again
            await new Promise(resolve => setTimeout(resolve, 5000));
        }
        
        throw new Error(`Transaction confirmation timeout: ${txHash}`);
    }

    private updateTransferMetrics(fromChain: string, toChain: string, amount: number): void {
        // Update source chain metrics
        const sourceMetrics = this.metrics.get(fromChain);
        if (sourceMetrics) {
            sourceMetrics.throughput += 1;
            sourceMetrics.lastUpdate = new Date();
        }
        
        // Update target chain metrics
        const targetMetrics = this.metrics.get(toChain);
        if (targetMetrics) {
            targetMetrics.throughput += 1;
            targetMetrics.lastUpdate = new Date();
        }
    }

    async getMultiChainStatus(): Promise<MultiChainStatus> {
        const status: MultiChainStatus = {
            bridges: {},
            totalVolume: 0,
            activeChains: 0,
            rainbowBridgeStatus: 'ACTIVE'
        };

        for (const [chainName, bridge] of this.bridges) {
            try {
                const bridgeStatus = await bridge.getStatus();
                status.bridges[chainName] = bridgeStatus;
                
                if (bridgeStatus.connected) {
                    status.activeChains++;
                }
                
                status.totalVolume += bridgeStatus.totalVolume || 0;
            } catch (error) {
                console.error(`Error getting status for ${chainName}:`, error);
                status.bridges[chainName] = {
                    connected: false,
                    lastSync: new Date(),
                    blockHeight: 0,
                    pendingTransactions: 0,
                    totalVolume: 0
                };
            }
        }

        return status;
    }

    async getMultiChainMetrics(): Promise<MultiChainMetrics> {
        // Get ZION Core metrics
        const zionMetrics = await this.getZionCoreMetrics();
        
        // Get bridge metrics
        const bridgeMetrics: Record<string, BridgeMetrics> = {};
        for (const [chainName, metrics] of this.metrics) {
            bridgeMetrics[chainName] = { ...metrics };
        }
        
        // Calculate cross-chain volume
        const dailyVolume = this.calculateDailyVolume();
        const totalVolume = this.calculateTotalVolume();

        return {
            zionCore: zionMetrics,
            bridges: bridgeMetrics,
            crossChainVolume: {
                daily: dailyVolume,
                total: totalVolume
            },
            rainbowBridge: {
                frequency: 44.44,
                status: 'ACTIVE',
                dimensionalGateways: this.bridges.size
            }
        };
    }

    private async getZionCoreMetrics() {
        try {
            const blockHeight = await this.config.zionCore.daemonBridge.getBlockCount();
            const networkInfo = await this.config.zionCore.daemonBridge.getInfo();
            
            return {
                blockHeight: blockHeight || 0,
                hashRate: networkInfo?.hashrate || 0,
                difficulty: networkInfo?.difficulty || 0,
                peers: networkInfo?.connections || 0
            };
        } catch (error) {
            console.error('Error getting ZION core metrics:', error);
            return {
                blockHeight: 0,
                hashRate: 0,
                difficulty: 0,
                peers: 0
            };
        }
    }

    private calculateDailyVolume(): number {
        const oneDayAgo = Date.now() - (24 * 60 * 60 * 1000);
        let volume = 0;
        
        for (const transfer of this.transfers.values()) {
            if (transfer.timestamp.getTime() > oneDayAgo && transfer.status === 'confirmed') {
                volume += transfer.amount;
            }
        }
        
        return volume;
    }

    private calculateTotalVolume(): number {
        let volume = 0;
        
        for (const transfer of this.transfers.values()) {
            if (transfer.status === 'confirmed') {
                volume += transfer.amount;
            }
        }
        
        return volume;
    }

    async getTransferHistory(limit: number = 100): Promise<CrossChainTransfer[]> {
        const transfers = Array.from(this.transfers.values())
            .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
            .slice(0, limit);
            
        return transfers;
    }

    async getTransferById(transferId: string): Promise<CrossChainTransfer | undefined> {
        return this.transfers.get(transferId);
    }

    async shutdown(): Promise<void> {
        console.log('🌈 Shutting down Multi-Chain Bridge Manager...');
        
        for (const [chainName, bridge] of this.bridges) {
            try {
                console.log(`🔄 Shutting down ${chainName} bridge...`);
                if (bridge.shutdown) {
                    await bridge.shutdown();
                }
                console.log(`✅ ${chainName} bridge shutdown complete`);
            } catch (error) {
                console.error(`❌ Error shutting down ${chainName} bridge:`, error);
            }
        }
        
        this.bridges.clear();
        this.transfers.clear();
        this.metrics.clear();
        
        console.log('✅ Multi-Chain Bridge Manager shutdown complete');
    }
}

export default MultiChainBridgeManager;
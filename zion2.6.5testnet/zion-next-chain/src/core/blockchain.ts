// Minimal blockchain state skeleton
import { Block, BlockHeader, blockId } from './block.js';
import { powHash, meetsTarget } from '../consensus/pow/simple_pow.js';

export interface ChainConfig {
  initialDifficulty: bigint;
}

export class Blockchain {
  private chain: Block[] = [];
  private config: ChainConfig;

  constructor(cfg: ChainConfig) {
    this.config = cfg;
  }

  get height(): number { return this.chain.length; }
  get tip(): Block | null { return this.chain[this.chain.length - 1] || null; }

  addGenesis(genesis: Block) {
    if (this.chain.length > 0) throw new Error('genesis already set');
    this.chain.push(genesis);
  }

  validateAndAdd(block: Block): boolean {
    // Basic checks
    const expectedHeight = this.chain.length;
    if (Number(block.header.height) !== expectedHeight) return false;
    if (expectedHeight > 0) {
      const prev = this.chain[expectedHeight - 1];
      if (block.header.prevHash !== blockId(prev.header)) return false;
    }
    // PoW check placeholder
    const h = powHash(block.header);
    if (!meetsTarget(h, block.header.target)) return false;
    this.chain.push(block);
    return true;
  }
}

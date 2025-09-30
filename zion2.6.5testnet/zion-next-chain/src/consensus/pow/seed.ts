import blake3 from 'blake3';
import { getPowConfig } from '../../config/powConfig.js';

export interface SeedInfo {
  epoch: number;
  seed: string; // hex
}

const cache = new Map<number,string>();

function blakeHex(data: Uint8Array): string {
  return Buffer.from(blake3.hash(data)).toString('hex');
}

export function epochFromHeight(height: number): number {
  const { epochBlocks } = getPowConfig();
  return Math.floor(height / epochBlocks);
}

export function deriveSeed(epoch: number, networkId = 'ZION-NEXT'): string {
  if (cache.has(epoch)) return cache.get(epoch)!;
  let seed: Uint8Array;
  if (epoch === 0) {
    seed = new TextEncoder().encode(`${networkId}-RANDOMX-SEED-0`);
  } else {
    const prev = deriveSeed(epoch - 1, networkId);
    const enc = new TextEncoder().encode(prev + '|' + epoch.toString() + '|' + networkId);
    seed = enc;
  }
  const hex = blakeHex(seed);
  cache.set(epoch, hex);
  return hex;
}

export function getSeedInfo(height: number, networkId = 'ZION-NEXT'): SeedInfo {
  const epoch = epochFromHeight(height);
  return { epoch, seed: deriveSeed(epoch, networkId) };
}

// Test helper to clear seed cache
export function __clearSeedCache() { cache.clear(); }

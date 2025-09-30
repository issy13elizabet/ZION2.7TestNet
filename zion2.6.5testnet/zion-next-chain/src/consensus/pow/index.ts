import { INITIAL_TARGET, meetsTarget } from '../params.js';
import { cosmicDharmaHash } from './cosmic_pow.js';
import blake3 from 'blake3';
import { keccak256 } from 'js-sha3';

export interface PowContext { height: number; epoch?: number; prevSeed?: string; }
export interface PowAlgorithm {
  name: string;
  hash(headerBytes: Uint8Array, nonce: bigint, ctx: PowContext): string;
  verify(headerBytes: Uint8Array, nonce: bigint, target: bigint, ctx: PowContext): boolean;
}

function compositeHash(headerBytes: Uint8Array, nonce: bigint): string {
  const nonceBytes = u64le(nonce);
  const h1 = blake3.hash(headerBytes);
  const h2 = blake3.hash(new Uint8Array([...headerBytes, ...nonceBytes]));
  return keccak256.create().update(new Uint8Array([...h1, ...h2])).hex();
}

function u64le(n: bigint): Uint8Array { const o=new Uint8Array(8); let v=n; for(let i=0;i<8;i++){ o[i]=Number(v & 0xffn); v >>=8n;} return o; }

// Placeholder RandomX stub (returns composite hash with tag). Real implementation will call native addon.
function randomXHash(headerBytes: Uint8Array, nonce: bigint, ctx: PowContext): string {
  const base = compositeHash(headerBytes, nonce);
  return 'rx' + base.slice(2); // tag prefix for differentiation (not final)
}

const compositeAlgo: PowAlgorithm = {
  name: 'composite',
  hash: (h, n) => compositeHash(h, n),
  verify: (h,n,t,ctx) => meetsTarget(compositeHash(h,n), t)
};

const cosmicAlgo: PowAlgorithm = {
  name: 'cosmic',
  hash: (h,n,ctx) => cosmicDharmaHash(h, n, BigInt(ctx.height), new Uint8Array(8), { memoryMiB: 4, rounds: 32 }),
  verify: (h,n,t,ctx) => meetsTarget(cosmicDharmaHash(h, n, BigInt(ctx.height), new Uint8Array(8), { memoryMiB: 4, rounds: 32 }), t)
};

const randomXAlgo: PowAlgorithm = {
  name: 'randomx',
  hash: (h,n,ctx) => randomXHash(h,n,ctx),
  verify: (h,n,t,ctx) => meetsTarget(randomXHash(h,n,ctx), t)
};

export type PowMode = 'COMPOSITE' | 'COSMIC' | 'RANDOMX' | 'HYBRID';

export function selectPow(mode: PowMode, height: number): PowAlgorithm {
  switch (mode) {
    case 'COSMIC': return cosmicAlgo;
    case 'RANDOMX': return randomXAlgo;
    case 'HYBRID':
      // Simple heuristic: first 1000 blocks randomx then cosmic
      return height < 1000 ? randomXAlgo : cosmicAlgo;
    case 'COMPOSITE':
    default: return compositeAlgo;
  }
}

export function powHash(headerBytes: Uint8Array, nonce: bigint, ctx: PowContext, mode: PowMode): string {
  return selectPow(mode, ctx.height).hash(headerBytes, nonce, ctx);
}

export function powVerify(headerBytes: Uint8Array, nonce: bigint, ctx: PowContext, mode: PowMode, target = INITIAL_TARGET): boolean {
  return selectPow(mode, ctx.height).verify(headerBytes, nonce, target, ctx);
}

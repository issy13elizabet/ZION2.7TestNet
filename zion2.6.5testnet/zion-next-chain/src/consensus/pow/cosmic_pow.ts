// Cosmic Dharma PoW - Phase A prototype (NOT FINAL / NOT SECURE)
// Implements minimal subset per COSMIC_DHARMA_POW_DRAFT.md

// NOTE: This is intentionally simplified & uses placeholder hashes.
// Real implementation will replace faux hash functions with blake3 + keccak.

export interface CosmicParams {
  memoryMiB: number; // e.g. 32
  rounds: number;    // e.g. 64
}

export function cosmicDharmaHash(headerBytes: Uint8Array, nonce: bigint, height: bigint, prevSeed: Uint8Array, params: CosmicParams): string {
  const seed = pseudoBlake3(concat(headerBytes, u64le(nonce), u64le(height), prevSeed));
  const words = (params.memoryMiB * 1024 * 1024) / 8;
  const mem = new BigUint64Array(words);
  // Fill
  for (let i = 0; i < words; i++) {
    mem[i] = fnv64(seedBig(seed) ^ BigInt(i));
  }
  let idx = Number(seedBig(seed) & BigInt(words - 1));
  let acc = seedBig(seed);
  for (let r = 0; r < params.rounds; r++) {
    const x = mem[idx];
    acc = (acc ^ rotl64(x, r % 61)) + (acc * 0x9e3779b185ebca87n);
    idx = Number((x ^ acc) & BigInt(words - 1));
  }
  const h1 = pseudoKeccak(u64le(acc));
  const h2 = pseudoKeccak(concat(hexToBytes(h1), seed));
  return h2.slice(0, 64); // hex
}

function seedBig(seed: Uint8Array): bigint {
  let v = 0n;
  for (let i = 0; i < Math.min(8, seed.length); i++) v = (v << 8n) | BigInt(seed[i]);
  return v;
}

function fnv64(x: bigint): bigint {
  let h = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  // Fold x into bytes
  let tmp = x;
  for (let i = 0; i < 8; i++) {
    const b = Number(tmp & 0xffn);
    h ^= BigInt(b);
    h = (h * prime) & 0xffffffffffffffffn;
    tmp >>= 8n;
  }
  return h;
}

function rotl64(x: bigint, n: number): bigint {
  n &= 63;
  return ((x << BigInt(n)) | (x >> BigInt(64 - n))) & 0xffffffffffffffffn;
}

function pseudoBlake3(data: Uint8Array): Uint8Array {
  // Placeholder simple rolling hash
  let a = 0x6d2b79f5n;
  for (const b of data) {
    a ^= BigInt(b);
    a = (a * 0x100000001b3n) & 0xffffffffffffffffn;
    a ^= (a >> 33n);
  }
  return u64le(a);
}

function pseudoKeccak(data: Uint8Array): string {
  // Placeholder: fold to 256 bits using xor & rotations
  const state = new BigUint64Array(4);
  let i = 0;
  for (const b of data) {
    const w = BigInt(b);
    const slot = i & 3;
    state[slot] = rotl64(state[slot] ^ w, (i * 7) % 64);
    i++;
  }
  let hex = '';
  for (const v of state) hex += v.toString(16).padStart(16, '0');
  return hex.padEnd(64, '0');
}

function u64le(x: bigint): Uint8Array {
  const out = new Uint8Array(8);
  let v = x & 0xffffffffffffffffn;
  for (let i = 0; i < 8; i++) {
    out[i] = Number(v & 0xffn);
    v >>= 8n;
  }
  return out;
}

function concat(...arrays: Uint8Array[]): Uint8Array {
  let len = 0;
  for (const a of arrays) len += a.length;
  const out = new Uint8Array(len);
  let o = 0;
  for (const a of arrays) { out.set(a, o); o += a.length; }
  return out;
}

function hexToBytes(hex: string): Uint8Array {
  if (hex.length % 2 !== 0) throw new Error('bad hex');
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = parseInt(hex.substr(i * 2, 2), 16);
  return out;
}

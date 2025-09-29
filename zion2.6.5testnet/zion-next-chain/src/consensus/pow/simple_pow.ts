// Simple PoW placeholder (ZIONHASH v0) - NOT FINAL
// hash = keccak256( blake3(header || nonce) )  (currently stubbed)

import { serializeHeader, BlockHeader } from '../../core/block.js';

export function powHash(header: BlockHeader): string {
  const bytes = serializeHeader(header); // nonce inside
  // TODO: real blake3 + keccak
  // For now: naive mixing -> hex
  let x = 0n;
  for (const b of bytes) x = (x * 1315423911n + BigInt(b)) & 0xffffffffffffffffn;
  // Expand to 32 bytes deterministically
  let hex = x.toString(16).padStart(16, '0');
  while (hex.length < 64) hex = hex + hex.slice(0, 16);
  return hex.slice(0, 64);
}

export function meetsTarget(hashHex: string, target: bigint): boolean {
  // Interpret hashHex as big-endian integer
  const val = BigInt('0x' + hashHex);
  return val <= target;
}

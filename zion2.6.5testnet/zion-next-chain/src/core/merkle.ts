import blake3 from 'blake3';
import { keccak256 } from 'js-sha3';

function hashLeaf(hex: string): Uint8Array {
  const bytes = hexToBytes(hex);
  const h1 = blake3.hash(bytes);
  return blake3.hash(new Uint8Array([...h1, ...bytes]));
}

function hashInternal(a: Uint8Array, b: Uint8Array): Uint8Array {
  const combo = new Uint8Array([...a, ...b]);
  const h1 = blake3.hash(combo);
  return blake3.hash(new Uint8Array([...h1, ...combo]));
}

export function merkleRoot(txids: string[]): string {
  if (txids.length === 0) return '0'.repeat(64);
  let layer: Uint8Array[] = txids.map(hashLeaf);
  while (layer.length > 1) {
    const next: Uint8Array[] = [];
    for (let i=0;i<layer.length;i+=2) {
      const left = layer[i];
      const right = layer[i+1] || layer[i];
      next.push(hashInternal(left, right));
    }
    layer = next;
  }
  // Final keccak overlay for domain separation
  return keccak256.create().update(layer[0]).hex();
}

function hexToBytes(hex: string): Uint8Array {
  if (hex.length % 2) throw new Error('hex');
  const out = new Uint8Array(hex.length/2);
  for (let i=0;i<out.length;i++) out[i]=parseInt(hex.slice(i*2,i*2+2),16);
  return out;
}

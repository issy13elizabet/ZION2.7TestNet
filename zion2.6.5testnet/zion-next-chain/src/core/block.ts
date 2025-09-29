// ZION NEXT CHAIN - Block structures (skeleton)
// See docs/ZION_NEXT_SPEC.md

export interface BlockHeader {
  version: number;
  height: bigint;
  prevHash: string; // hex 64
  merkleRoot: string; // hex 64
  timestamp: bigint;
  target: bigint; // compact or raw difficulty (placeholder: raw)
  nonce: bigint;
}

export interface Block {
  header: BlockHeader;
  transactions: string[]; // tx ids placeholder
}

export function serializeHeader(h: BlockHeader): Uint8Array {
  // Simple placeholder serialization (NOT final, no endianness tweaks yet)
  const parts = [
    numToHex(h.version, 4),
    numToHex(h.height, 16),
    h.prevHash,
    h.merkleRoot,
    numToHex(h.timestamp, 16),
    numToHex(h.target, 16),
    numToHex(h.nonce, 16)
  ];
  return hexToBytes(parts.join(''));
}

export function blockId(header: BlockHeader): string {
  // Placeholder: blake3(headerBytes) hex
  const data = serializeHeader(header);
  // TODO: integrate real blake3 lib; for now simple sha256 fallback
  const hash = cryptoHash(data);
  return hash;
}

function numToHex(n: number | bigint, widthBytes: number): string {
  return BigInt(n).toString(16).padStart(widthBytes * 2, '0');
}

function hexToBytes(hex: string): Uint8Array {
  if (hex.length % 2 !== 0) throw new Error('hex length odd');
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i++) {
    out[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  }
  return out;
}

function cryptoHash(data: Uint8Array): string {
  // Placeholder crypto (NOT secure) - to be replaced with blake3
  let acc = 0n;
  for (const b of data) acc = (acc * 1099511628211n + BigInt(b)) & 0xffffffffffffffffn;
  return acc.toString(16).padStart(16, '0').padEnd(64, '0');
}

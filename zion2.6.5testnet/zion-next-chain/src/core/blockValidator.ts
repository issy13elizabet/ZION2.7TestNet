import { merkleRoot } from './merkle.js';
import { UTXOSet } from '../utxo/utxoSet.js';
import { Transaction, isCoinbase } from '../tx/types.js';
import { txHash } from '../tx/serialize.js';
import { INITIAL_TARGET, meetsTarget } from '../consensus/params.js';
import blake3 from 'blake3';
import { keccak256 } from 'js-sha3';

export interface FullBlock {
  header: {
    previous: string;
    timestamp: number;
    height: number;
    merkleRoot: string;
    nonce: bigint;
    powHash?: string;
  };
  transactions: Transaction[];
}

function powCompose(headerBytes: Uint8Array, nonce: bigint): string {
  const nonceBytes = u64le(nonce);
  const h1 = blake3.hash(headerBytes);
  const h2 = blake3.hash(new Uint8Array([...headerBytes, ...nonceBytes]));
  return keccak256.create().update(new Uint8Array([...h1, ...h2])).hex();
}

function u64le(n: bigint): Uint8Array { const o=new Uint8Array(8); let v=n; for(let i=0;i<8;i++){ o[i]=Number(v & 0xffn); v >>=8n;} return o; }

function serializeHeaderLite(h: FullBlock['header']): Uint8Array {
  const parts: number[] = [];
  const pushHex = (hex: string) => { for (let i=0;i<hex.length;i+=2) parts.push(parseInt(hex.slice(i,i+2),16)); };
  pushHex(h.previous);
  parts.push(...u64le(BigInt(h.timestamp)));
  parts.push(...u64le(BigInt(h.height)));
  pushHex(h.merkleRoot);
  parts.push(...u64le(h.nonce));
  return new Uint8Array(parts);
}

export interface ValidationResult { ok: boolean; error?: string; powHash?: string; }

export function validateBlock(block: FullBlock, utxo: UTXOSet): ValidationResult {
  if (block.transactions.length === 0) return { ok: false, error: 'empty block' };
  if (!isCoinbase(block.transactions[0])) return { ok: false, error: 'first tx not coinbase' };
  for (let i=1;i<block.transactions.length;i++) if (isCoinbase(block.transactions[i])) return { ok: false, error: 'multiple coinbase' };
  const txids = block.transactions.map(tx => tx.txid || txHash(tx));
  const mr = merkleRoot(txids);
  if (mr !== block.header.merkleRoot) return { ok: false, error: 'bad merkle' };
  const headerBytes = serializeHeaderLite(block.header);
  const pow = powCompose(headerBytes, block.header.nonce);
  if (!meetsTarget(pow, INITIAL_TARGET)) return { ok: false, error: 'pow target' };
  // Trial UTXO application (copy)
  const shadow = new UTXOSet();
  // NOTE: This shallow copy does not yet clone state from provided utxo; future improvement for real chain state diff.
  try {
    shadow.applyBlock(block.transactions, block.header.height);
  } catch (e: any) {
    return { ok: false, error: 'utxo apply: ' + e.message };
  }
  return { ok: true, powHash: pow };
}

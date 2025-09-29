import { Transaction, TxInput, TxOutput, MAX_TX_INPUTS, MAX_TX_OUTPUTS } from './types.js';
import blake3 from 'blake3';
import { keccak256 } from 'js-sha3';

function writeU32LE(buf: number[], v: number) { for (let i=0;i<4;i++) buf.push((v >>> (8*i)) & 0xff); }
function writeU64LE(buf: number[], v: bigint) { let x=v; for(let i=0;i<8;i++){ buf.push(Number(x & 0xffn)); x >>=8n; } }

function hexToBytes(hex: string): number[] {
  if (hex.length %2) throw new Error('hex length');
  const out: number[] = [];
  for (let i=0;i<hex.length;i+=2) out.push(parseInt(hex.slice(i,i+2),16));
  return out;
}

function encodeVarInt(n: number): number[] {
  if (n < 0xfd) return [n];
  if (n <= 0xffff) return [0xfd, n & 0xff, (n>>>8)&0xff];
  if (n <= 0xffffffff) return [0xfe, n &0xff,(n>>>8)&0xff,(n>>>16)&0xff,(n>>>24)&0xff];
  throw new Error('varint too large');
}

export function serializeTx(tx: Transaction): Uint8Array {
  if (tx.vin.length > MAX_TX_INPUTS) throw new Error('too many inputs');
  if (tx.vout.length > MAX_TX_OUTPUTS) throw new Error('too many outputs');
  const b: number[] = [];
  writeU32LE(b, tx.version >>> 0);
  b.push(...encodeVarInt(tx.vin.length));
  for (const vin of tx.vin) b.push(...serializeInput(vin));
  b.push(...encodeVarInt(tx.vout.length));
  for (const vout of tx.vout) b.push(...serializeOutput(vout));
  writeU32LE(b, tx.lockTime >>> 0);
  return new Uint8Array(b);
}

function serializeInput(input: TxInput): number[] {
  const out: number[] = [];
  out.push(...hexToBytes(input.prevTxId));
  writeU32LE(out, input.vout >>> 0);
  const script = hexToBytes(input.scriptSig || '');
  out.push(...encodeVarInt(script.length));
  out.push(...script);
  writeU32LE(out, input.sequence >>> 0);
  return out;
}

function serializeOutput(output: TxOutput): number[] {
  const out: number[] = [];
  writeU64LE(out, output.value);
  const script = hexToBytes(output.scriptPubKey || '');
  out.push(...encodeVarInt(script.length));
  out.push(...script);
  return out;
}

export function txHash(tx: Transaction): string {
  const ser = serializeTx(tx);
  const h1 = blake3.hash(ser);
  const h2 = blake3.hash(new Uint8Array([...h1, ...ser]));
  return keccak256.create().update(new Uint8Array([...h1, ...h2])).hex();
}

export function finalizeTx(tx: Transaction): Transaction {
  const copy: Transaction = { ...tx };
  copy.txid = txHash(copy);
  return copy;
}
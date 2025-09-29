import { keccak256 } from 'js-sha3';
import blake3 from 'blake3';
import { INITIAL_TARGET, meetsTarget } from '../consensus/params.js';
import { Transaction } from '../tx/types.js';
import { finalizeTx } from '../tx/serialize.js';

export interface CoinbaseTx { raw: string; txid: string; structured?: Transaction; }
export interface GenesisHeader {
  previous: string;
  timestamp: number;
  height: number;
  merkleRoot: string;
  nonce: bigint;
  powHash?: string;
}

export interface GenesisResult { header: GenesisHeader; coinbase: CoinbaseTx; attempts: bigint; }

function toHex(b: Uint8Array): string { return [...b].map(x=>x.toString(16).padStart(2,'0')).join(''); }
function fromHex(h: string): Uint8Array { if (h.length%2) throw new Error('hex'); const a=new Uint8Array(h.length/2); for(let i=0;i<a.length;i++) a[i]=parseInt(h.substr(i*2,2),16); return a; }
function u64le(n: bigint): Uint8Array { const o=new Uint8Array(8); let v=n; for(let i=0;i<8;i++){ o[i]=Number(v & 0xffn); v >>=8n;} return o; }

function blake3Hash(data: Uint8Array): Uint8Array { return blake3.hash(data); }
function keccakHashHex(data: Uint8Array): string { return keccak256.create().update(data).hex(); }
function powMix(headerBytes: Uint8Array, nonceHex: string): string {
  const h1 = blake3Hash(headerBytes);
  const nonceBytes = fromHex(nonceHex.padStart(16,'0'));
  const h2 = blake3Hash(new Uint8Array([...headerBytes, ...nonceBytes]));
  return keccakHashHex(new Uint8Array([...h1, ...h2]));
}

export function buildCoinbase(allocationAtomic: bigint, foundationAddress: string): CoinbaseTx {
  // Structured coinbase transaction: one special input + one output paying foundation
  const base: Transaction = {
    version: 1,
    vin: [{ prevTxId: '0'.repeat(64), vout: 0xffffffff, scriptSig: '', sequence: 0xffffffff }],
    vout: [{ value: allocationAtomic, scriptPubKey: addressToScriptPubKey(foundationAddress) }],
    lockTime: 0
  };
  const finalized = finalizeTx(base);
  // Maintain legacy raw string field for manifest backward compatibility
  const rawStr = `COINBASE|FOUNDATION|${foundationAddress}|${allocationAtomic.toString()}`;
  const rawBytes = new TextEncoder().encode(rawStr);
  const legacyId = keccakHashHex(blake3Hash(rawBytes));
  return { raw: toHex(rawBytes), txid: finalized.txid || legacyId, structured: finalized };
}

function addressToScriptPubKey(address: string): string {
  // Placeholder: just hex of utf8 address prefixed with simple OP tag (0x51). Real implementation will decode base58/bech32 etc.
  const utf = [...new TextEncoder().encode(address)].map(b=>b.toString(16).padStart(2,'0')).join('');
  return '51' + utf; // OP_TRUE placeholder semantics
}

export function serializeHeader(h: GenesisHeader): Uint8Array {
  const parts: Uint8Array[] = [];
  const pushHex = (hex: string) => { parts.push(fromHex(hex)); };
  pushHex(h.previous);
  parts.push(u64le(BigInt(h.timestamp)));
  parts.push(u64le(BigInt(h.height)));
  pushHex(h.merkleRoot);
  parts.push(u64le(BigInt(h.nonce)));
  const len = parts.reduce((a,p)=>a+p.length,0);
  const out = new Uint8Array(len);
  let o=0; for(const p of parts){ out.set(p,o); o+=p.length; }
  return out;
}

export function mineGenesis(params: { timestamp: number; allocationAtomic: bigint; foundationAddress: string; maxAttempts?: bigint }): GenesisResult {
  const coinbase = buildCoinbase(params.allocationAtomic, params.foundationAddress);
  const headerBase: GenesisHeader = {
    previous: '00'.repeat(32),
    timestamp: params.timestamp,
    height: 0,
    merkleRoot: coinbase.txid, // exact txid, no padding
    nonce: 0n
  };
  let attempts = 0n;
  while (true) {
    const hb = serializeHeader(headerBase);
    const powHash = powMix(hb, headerBase.nonce.toString(16));
    if (meetsTarget(powHash, INITIAL_TARGET)) {
      return { header: { ...headerBase, powHash }, coinbase, attempts };
    }
    headerBase.nonce++;
    attempts++;
    if (params.maxAttempts && attempts >= params.maxAttempts) throw new Error('maxAttempts exceeded');
  }
}

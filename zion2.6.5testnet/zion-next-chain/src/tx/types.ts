export interface TxInput {
  prevTxId: string; // 64 hex (lowercase)
  vout: number;     // uint32
  scriptSig: string; // hex (can be empty)
  sequence: number;  // uint32
}

export interface TxOutput {
  value: bigint;     // atomic units
  scriptPubKey: string; // hex (locks to address or condition)
}

export interface Transaction {
  version: number;
  vin: TxInput[];
  vout: TxOutput[];
  lockTime: number;
  // Computed:
  txid?: string;
}

export interface CoinbaseMeta {
  height: number;
  extra?: string; // hex metadata
}

export function isCoinbase(tx: Transaction): boolean {
  return tx.vin.length === 1 && /^0{64}$/.test(tx.vin[0].prevTxId) && tx.vin[0].vout === 0xffffffff;
}

export const MAX_TX_INPUTS = 2048;
export const MAX_TX_OUTPUTS = 4096;
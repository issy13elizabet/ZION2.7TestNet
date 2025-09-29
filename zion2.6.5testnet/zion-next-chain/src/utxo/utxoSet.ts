import { Transaction, TxOutput, TxInput } from '../tx/types.js';
import { txHash } from '../tx/serialize.js';

export interface UTXOEntry {
  value: bigint;
  scriptPubKey: string;
  height: number;
  coinbase: boolean;
  spent: boolean;
}

export class UTXOSet {
  private map = new Map<string, UTXOEntry>(); // key = txid:index
  private height: number = 0;
  static COINBASE_MATURITY = 60;

  get(key: string): UTXOEntry | undefined { return this.map.get(key); }
  private key(txid: string, idx: number): string { return `${txid}:${idx}`; }

  applyTransaction(tx: Transaction, height: number, isCoinbase = false) {
    const id = tx.txid || txHash(tx);
    // Spend inputs
    if (!isCoinbase) {
      for (const input of tx.vin) {
        const k = this.key(input.prevTxId, input.vout);
        const utxo = this.map.get(k);
        if (!utxo) throw new Error('Missing UTXO ' + k);
        if (utxo.spent) throw new Error('Double spend ' + k);
        if (utxo.coinbase && height - utxo.height < UTXOSet.COINBASE_MATURITY) throw new Error('Coinbase immature');
        utxo.spent = true;
      }
    }
    // Create outputs
    tx.vout.forEach((o, idx) => {
      const k = this.key(id, idx);
      this.map.set(k, { value: o.value, scriptPubKey: o.scriptPubKey, height, coinbase: isCoinbase, spent: false });
    });
  }

  applyBlock(txs: Transaction[], height: number) {
    this.height = height;
    txs.forEach((tx, i) => this.applyTransaction(tx, height, i === 0));
  }

  getBalanceByScript(scriptPubKey: string): bigint {
    let sum = 0n;
    for (const utxo of this.map.values()) if (!utxo.spent && utxo.scriptPubKey === scriptPubKey) sum += utxo.value;
    return sum;
  }
}

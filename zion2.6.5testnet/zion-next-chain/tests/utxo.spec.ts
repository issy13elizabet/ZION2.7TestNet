import { UTXOSet } from '../src/utxo/utxoSet.js';
import { finalizeTx } from '../src/tx/serialize.js';
import { Transaction } from '../src/tx/types.js';

function mkScript(address: string) { return '51' + Buffer.from(address,'utf8').toString('hex'); }

const FOUNDATION = 'FOUND_ADDR_X';
const FOUNDATION_SCRIPT = mkScript(FOUNDATION);

describe('UTXO Set', () => {
  test('applies coinbase and computes balance', () => {
    const coinbase: Transaction = {
      version: 1,
      vin: [{ prevTxId: '0'.repeat(64), vout: 0xffffffff, scriptSig: '', sequence: 0xffffffff }],
      vout: [{ value: 1000n, scriptPubKey: FOUNDATION_SCRIPT }],
      lockTime: 0
    };
    const cb = finalizeTx(coinbase);
    const set = new UTXOSet();
    set.applyBlock([cb], 0);
    expect(set.getBalanceByScript(FOUNDATION_SCRIPT)).toBe(1000n);
  });

  test('spends matured coinbase', () => {
    const set = new UTXOSet();
    const coinbase: Transaction = {
      version: 1,
      vin: [{ prevTxId: '0'.repeat(64), vout: 0xffffffff, scriptSig: '', sequence: 0xffffffff }],
      vout: [{ value: 500n, scriptPubKey: FOUNDATION_SCRIPT }],
      lockTime: 0
    };
    const cb = finalizeTx(coinbase);
    set.applyBlock([cb], 0);
    // Mine maturity blocks
    for (let h=1; h<=UTXOSet.COINBASE_MATURITY; h++) {
      set.applyBlock([], h); // empty blocks for height advance
    }
    const spend: Transaction = {
      version: 1,
      vin: [{ prevTxId: cb.txid!, vout: 0, scriptSig: '', sequence: 0xffffffff }],
      vout: [{ value: 400n, scriptPubKey: FOUNDATION_SCRIPT }],
      lockTime: 0
    };
    const spendFinal = finalizeTx(spend);
    set.applyBlock([spendFinal], UTXOSet.COINBASE_MATURITY + 1);
    expect(set.getBalanceByScript(FOUNDATION_SCRIPT)).toBe(400n);
  });
});

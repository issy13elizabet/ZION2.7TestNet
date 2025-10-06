"use client";
import { useEffect, useState } from 'react';

export default function WalletStatusPage() {
  const [height, setHeight] = useState<any>(null);
  const [balance, setBalance] = useState<any>(null);
  const [error, setError] = useState<string|undefined>();
  const [blocks, setBlocks] = useState<any>({ tip: 0, count: 0, blocks: [] });

  const load = async () => {
    try {
      const [h, b, bl] = await Promise.all([
        fetch('/api/chain/height').then(r => r.json()),
        fetch('/api/wallet/balance').then(r => r.json()),
        fetch('/api/pool/blocks-recent?n=8').then(r => r.json()),
      ]);
      setHeight(h);
      setBalance(b);
      setBlocks(bl);
      setError(undefined);
    } catch (e: any) {
      setError(String(e?.message || e));
    }
  };

  useEffect(() => { load(); const id = setInterval(load, 5000); return () => clearInterval(id); }, []);

  return (
    <div style={{ maxWidth: 720, margin: "40px auto", padding: 16 }}>
      <h1>ZION Wallet Status</h1>
      {error && <div style={{ color: 'red' }}>Error: {error}</div>}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
        <div style={{ border: '1px solid #ddd', borderRadius: 8, padding: 12 }}>
          <h3>Chain height</h3>
          <pre>{JSON.stringify(height, null, 2)}</pre>
        </div>
        <div style={{ border: '1px solid #ddd', borderRadius: 8, padding: 12 }}>
          <h3>Wallet balance</h3>
          <pre>{JSON.stringify(balance, null, 2)}</pre>
        </div>
      </div>
      <div style={{ marginTop: 16, border: '1px solid #ddd', borderRadius: 8, padding: 12 }}>
        <h3>Poslední bloky</h3>
        <pre>{JSON.stringify(blocks, null, 2)}</pre>
      </div>
      <p style={{ marginTop: 16, color: '#666' }}>
        Pozn.: Výplaty se spustí až po coinbase maturitě na chainu. V izolaci je okno ~60 bloků.
      </p>
    </div>
  );
}

"use client";
import React from "react";

export default function PoolTestPage() {
  const [loading, setLoading] = React.useState(false);
  const [result, setResult] = React.useState<any>(null);

  const test = async () => {
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch('/api/pool-test', { cache: 'no-store' });
      const json = await res.json();
      setResult(json);
    } catch (e: any) {
      setResult({ ok: false, error: e?.message || String(e) });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ maxWidth: 720, margin: '40px auto', fontFamily: 'system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, sans-serif' }}>
      <h1>Pool connectivity test</h1>
      <p>Ověř, že vestavěný Stratum pool přijímá spojení a vrací JSON-RPC odpověď na <code>login</code>.</p>
      <button onClick={test} disabled={loading} style={{ padding: '10px 16px', borderRadius: 8, border: '1px solid #ccc' }}>
        {loading ? 'Testuji…' : 'Spustit test'}
      </button>
      {result && (
        <pre style={{ background: '#111', color: '#0f0', padding: 16, marginTop: 20, borderRadius: 8, overflow: 'auto' }}>
{JSON.stringify(result, null, 2)}
        </pre>
      )}
      <p style={{ marginTop: 16, color: '#666' }}>
        Pozn.: Server testuje TCP na {process.env.NEXT_PUBLIC_POOL_HOST || '91.98.122.165'}:{process.env.NEXT_PUBLIC_POOL_PORT || 3333} a odešle 1 řádek s JSON-RPC <code>login</code>.
      </p>
    </div>
  );
}

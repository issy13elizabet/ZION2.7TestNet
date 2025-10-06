"use client";
import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

export default function ExplorerSearchPage() {
  const [q, setQ] = useState('');
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string|undefined>();

  const run = async () => {
    try {
      if (!q) return;
      const res = await fetch(`/api/explorer/search?q=${encodeURIComponent(q)}`, { cache: 'no-store' });
      const json = await res.json();
      setData(json);
      setError(undefined);
    } catch (e: any) {
      setError(String(e?.message || e));
    }
  };

  useEffect(() => {
    const url = new URL(window.location.href);
    const qp = url.searchParams.get('query') || '';
    if (qp) {
      setQ(qp);
      // Auto-run initial search
      (async () => {
        try {
          const res = await fetch(`/api/explorer/search?q=${encodeURIComponent(qp)}`, { cache: 'no-store' });
          const json = await res.json();
          setData(json);
          setError(undefined);
        } catch (e: any) {
          setError(String(e?.message || e));
        }
      })();
    }
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-black text-white p-6">
      <motion.header
        className="relative z-10 mb-6"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="text-3xl font-bold bg-gradient-to-r from-violet-400 to-purple-300 bg-clip-text text-transparent">🔎 Explorer Search</h1>
      </motion.header>
      <div className="flex gap-2 mb-3">
        <input value={q} onChange={e=>setQ(e.target.value)} placeholder="Hash (64 hex) nebo výška" className="flex-1 rounded-lg bg-black/30 border border-purple-500/30 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-purple-400 placeholder:text-purple-300/60"/>
        <button onClick={run} className="px-4 py-2 rounded-lg bg-purple-600 hover:bg-purple-700 text-white">Search</button>
      </div>
      {error && <div className="text-red-400 text-sm mb-2">Error: {error}</div>}
      {data && (
        <div className="bg-black/30 border border-purple-500/30 rounded-lg p-4">
          <pre className="text-sm whitespace-pre-wrap">{JSON.stringify(data, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

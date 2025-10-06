"use client";
import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

export default function BlockDetailPage({ params }: { params: { height: string } }) {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string|undefined>();
  const h = params.height;

  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetch(`/api/explorer/block/height/${encodeURIComponent(h)}`, { cache: 'no-store' });
        const json = await res.json();
        setData(json);
        setError(undefined);
      } catch (e: any) {
        setError(String(e?.message || e));
      }
    };
    load();
  }, [h]);

  const header = data?.header || data?.block_header || {};
  const ts = header.timestamp ? new Date(header.timestamp * 1000).toLocaleString() : '-';

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-black text-white p-6">
      <motion.header 
        className="relative z-10 mb-6"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="text-3xl font-bold bg-gradient-to-r from-violet-400 to-purple-300 bg-clip-text text-transparent">
          🧱 Block #{header.height ?? h}
        </h1>
        <a href="/explorer" className="text-purple-300 hover:text-purple-200">← Back</a>
      </motion.header>

      {error && <div className="text-red-400 text-sm mb-2">Error: {error}</div>}

      <div className="bg-black/30 border border-purple-500/30 rounded-lg p-4">
        <pre className="text-sm whitespace-pre-wrap">{JSON.stringify(header, null, 2)}</pre>
        <div className="mt-3 text-sm text-gray-300">Time: <span className="text-yellow-400">{ts}</span></div>
        <div className="mt-1 text-sm text-gray-300">Prev: {header.prev_hash ? (
          <a href={`/explorer/search?query=${header.prev_hash}`} className="font-mono text-purple-300 underline">{(header.prev_hash || '').slice(0, 24)}…</a>
        ) : '-'}
        </div>
      </div>
    </div>
  );
}

"use client";
import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

type BlockHeader = {
  height: number;
  hash: string;
  timestamp: number;
  reward?: number;
};

export default function ExplorerPage() {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string|undefined>();
  const [limit, setLimit] = useState(10);

  const load = async () => {
    try {
      const res = await fetch(`/api/explorer/blocks?limit=${limit}`, { cache: 'no-store' });
      const json = await res.json();
      setData(json);
      setError(undefined);
    } catch (e: any) {
      setError(String(e?.message || e));
    }
  };

  useEffect(() => { load(); const id = setInterval(load, 5000); return () => clearInterval(id); }, [limit]);

  const blocks: Array<{height:number, header:any, error?:string}> = data?.blocks || [];

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-blue-900 to-black text-white p-6">
      <motion.header 
        className="relative z-10 mb-8"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="text-4xl font-bold bg-gradient-to-r from-violet-400 to-purple-300 bg-clip-text text-transparent">
          🔭 ZION Explorer
        </h1>
        <p className="text-purple-300">Prohlížej poslední bloky, vyhledávej podle hash/výšky</p>
      </motion.header>

      <div className="grid gap-6 md:grid-cols-3">
        <motion.div 
          className="md:col-span-2"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <form action="/explorer/search" className="flex gap-2 mb-4">
            <input name="query" placeholder="Hash (64 hex) nebo výška" className="flex-1 rounded-2xl bg-black/30 border border-purple-500/30 px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-purple-400 placeholder:text-purple-300/60 backdrop-blur-sm"/>
            <button type="submit" className="px-6 py-3 rounded-2xl bg-purple-600 hover:bg-purple-700 text-white font-medium transition-colors">Search</button>
          </form>
          <div className="mb-3 text-sm text-gray-300 flex items-center gap-2">
            <span>Show last</span>
            <input type="number" value={limit} min={5} max={50} onChange={(e)=>setLimit(Math.max(5, Math.min(50, Number(e.target.value)||10)))} className="w-20 rounded bg-black/30 border border-purple-500/30 px-2 py-1"/>
            <span>blocks</span>
          </div>

          {error && <div className="text-red-400 text-sm mb-3 p-3 bg-red-500/10 rounded-2xl border border-red-500/30">Error: {error}</div>}
          <div className="rounded-2xl border border-purple-500/30 overflow-hidden bg-black/30 backdrop-blur-sm">
            <div className="grid grid-cols-4 gap-2 px-4 py-3 bg-purple-900/40 font-semibold text-purple-200">
              <div>Height</div>
              <div>Hash</div>
              <div>Time</div>
              <div>Reward</div>
            </div>
            {blocks.map((b, idx) => {
              const h = b.header;
              const ts = h?.timestamp ? new Date(h.timestamp * 1000).toLocaleString() : '-';
              const rew = typeof h?.reward === 'number' ? h.reward : '-';
              return (
                <a key={idx} href={`/explorer/block/${h?.height ?? b.height}`} className="block hover:bg-purple-900/30 transition-colors">
                  <div className="grid grid-cols-4 gap-2 px-4 py-3 border-t border-purple-500/20">
                    <div className="text-green-400 font-semibold">#{h?.height ?? b.height}</div>
                    <div className="font-mono text-purple-300 text-xs">{(h?.hash || '').slice(0, 16)}…</div>
                    <div className="text-yellow-400 text-sm">{ts}</div>
                    <div className="text-blue-400 text-sm">{rew}</div>
                  </div>
                </a>
              );
            })}
          </div>
        </motion.div>

        <motion.div 
          className="md:col-span-1"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="bg-black/30 border border-purple-500/30 rounded-2xl p-6 backdrop-blur-sm">
            <div className="text-lg font-semibold text-purple-200 mb-3">💡 Tipy</div>
            <ul className="list-disc pl-5 text-sm text-purple-200 space-y-2">
              <li>Vyhledávej hash nebo přesnou výšku</li>
              <li>Klikni na řádek pro detail bloku</li>
              <li>Aktualizace každých 5s</li>
            </ul>
          </div>
        </motion.div>
      </div>
    </div>
  );
}

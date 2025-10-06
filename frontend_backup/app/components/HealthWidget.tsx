'use client';

import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';

type HealthResponse = {
  host: string;
  poolPort: number;
  shimPort: number;
  pool: { ok: boolean; status: number; text: string };
  shim: { ok: boolean; status: number; text: string };
};

function StatusBadge({ ok }: { ok: boolean }) {
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs ${
      ok ? 'bg-green-500/20 text-green-300' : 'bg-red-500/20 text-red-300'
    }`}>
      <span className={`w-2 h-2 rounded-full ${ok ? 'bg-green-400' : 'bg-red-400'}`} />
      {ok ? 'OK' : 'DOWN'}
    </span>
  );
}

export default function HealthWidget({ className = '' }: { className?: string }) {
  const [data, setData] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let stop = false;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch('/api/health', { cache: 'no-store' });
        const json = await res.json();
        if (!stop) setData(json);
      } catch (e: any) {
        if (!stop) setError(e?.message || String(e));
      } finally {
        if (!stop) setLoading(false);
      }
    };
    load();
    const t = setInterval(load, 15000);
    return () => { stop = true; clearInterval(t); };
  }, []);

  const host = data?.host;
  const shimUrl = host ? `http://${host}:${data?.shimPort}` : undefined;
  const poolStratum = data?.pool?.text || '';

  return (
    <motion.div
      className={`bg-black/30 backdrop-blur-sm border border-purple-500/30 rounded-lg p-4 ${className}`}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="text-sm text-gray-400">ZION Services Health</div>
          <div className="text-purple-200 text-xs">
            Host: <span className="font-mono">{host ?? 'n/a'}</span>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="text-sm">
            <div className="text-gray-400">RPC Shim</div>
            <div className="flex items-center gap-2">
              <StatusBadge ok={!!data?.shim?.ok} />
              {shimUrl && (
                <a
                  href={`${shimUrl}/metrics.json`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-xs text-cyan-300 hover:text-cyan-200 underline"
                >
                  metrics.json
                </a>
              )}
            </div>
          </div>

          <div className="text-sm">
            <div className="text-gray-400">Pool (Stratum)</div>
            <div className="flex items-center gap-2">
              {/* We cannot HTTP probe stratum; show URL only */}
              <StatusBadge ok={false} />
              <span className="text-xs text-gray-300 font-mono">{poolStratum}</span>
            </div>
          </div>
        </div>
      </div>

      {loading && (
        <div className="mt-2 text-xs text-gray-400">Loading health…</div>
      )}
      {error && (
        <div className="mt-2 text-xs text-red-400">{error}</div>
      )}
    </motion.div>
  );
}

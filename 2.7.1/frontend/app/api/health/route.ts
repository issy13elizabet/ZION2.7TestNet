import { NextResponse } from 'next/server';

async function probe(url: string, init?: RequestInit) {
  try {
    const res = await fetch(url, { ...init, cache: 'no-store' });
    const text = await res.text();
    return { ok: res.ok, status: res.status, text };
  } catch (e: any) {
    return { ok: false, status: 0, text: String(e?.message || e) };
  }
}

export async function GET() {
  const host = process.env.ZION_HOST || '91.98.122.165';
  const poolPort = Number(process.env.ZION_POOL_PORT || 3333);
  const shimPort = Number(process.env.ZION_SHIM_PORT || 18089);

  const [pool, shim] = await Promise.all([
    // Stratum není HTTP – zkusíme TCP check přes HTTP probe nemožný; vrátíme jen info o socketu přes netcat by nešlo zde.
    // Proto jen placeholder – klient stránka udělá čistý TCP check přes WebSocket nelze; uvedeme URL.
    Promise.resolve({ ok: false, status: 0, text: `stratum://${host}:${poolPort}` }),
    probe(`http://${host}:${shimPort}/`),
  ]);

  return NextResponse.json({ host, poolPort, shimPort, pool, shim });
}

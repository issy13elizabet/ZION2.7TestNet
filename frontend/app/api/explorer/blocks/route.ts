import { NextResponse } from 'next/server';

export async function GET(req: Request) {
  const host = process.env.ZION_HOST || '127.0.0.1';
  const adapterPort = Number(process.env.ZION_ADAPTER_PORT || 18099);
  const url = new URL(req.url);
  const start = url.searchParams.get('start');
  const limit = url.searchParams.get('limit') || '10';
  const order = url.searchParams.get('order') || 'desc';
  const q = new URLSearchParams();
  if (start) q.set('start', start);
  if (limit) q.set('limit', limit);
  if (order) q.set('order', order);
  try {
    const res = await fetch(`http://${host}:${adapterPort}/explorer/blocks?${q.toString()}`, { cache: 'no-store' });
    const json = await res.json();
    return NextResponse.json(json, { status: res.status });
  } catch (e: any) {
    return NextResponse.json({ error: String(e?.message || e) }, { status: 500 });
  }
}

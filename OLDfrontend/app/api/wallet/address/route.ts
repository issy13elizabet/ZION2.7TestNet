import { NextResponse } from 'next/server';

export async function GET() {
  const host = process.env.ZION_HOST || '127.0.0.1';
  const adapterPort = Number(process.env.ZION_ADAPTER_PORT || 18099);
  try {
    const res = await fetch(`http://${host}:${adapterPort}/wallet/address`, { cache: 'no-store' });
    const json = await res.json();
    return NextResponse.json(json, { status: res.status });
  } catch (e: any) {
    return NextResponse.json({ error: String(e?.message || e) }, { status: 500 });
  }
}

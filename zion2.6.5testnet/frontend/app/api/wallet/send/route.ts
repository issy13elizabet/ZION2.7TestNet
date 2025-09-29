import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  const host = process.env.ZION_HOST || '127.0.0.1';
  const adapterPort = Number(process.env.ZION_ADAPTER_PORT || 18099);
  const adapterKey = process.env.ADAPTER_API_KEY || '';

  try {
    const body = await request.json().catch(() => ({}));
    const res = await fetch(`http://${host}:${adapterPort}/wallet/send`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(adapterKey ? { 'x-api-key': adapterKey } : {}),
      },
      cache: 'no-store',
      body: JSON.stringify(body || {}),
    });
    const json = await res.json();
    return NextResponse.json(json, { status: res.status });
  } catch (e: any) {
    return NextResponse.json({ error: String(e?.message || e) }, { status: 500 });
  }
}

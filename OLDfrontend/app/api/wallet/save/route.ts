import { NextResponse } from 'next/server';

export async function POST() {
  const host = process.env.ZION_HOST || '127.0.0.1';
  const adapterPort = Number(process.env.ZION_ADAPTER_PORT || 18099);
  const adapterKey = process.env.ADAPTER_API_KEY || '';
  const adminEnabled = (process.env.ENABLE_WALLET_ADMIN || '').toLowerCase() === 'true';

  if (!adminEnabled) {
    return NextResponse.json({ error: 'admin disabled' }, { status: 403 });
  }

  try {
    const res = await fetch(`http://${host}:${adapterPort}/wallet/save`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(adapterKey ? { 'x-api-key': adapterKey } : {}),
      },
      cache: 'no-store',
      body: JSON.stringify({}),
    });
    const json = await res.json();
    return NextResponse.json(json, { status: res.status });
  } catch (e: any) {
    return NextResponse.json({ error: String(e?.message || e) }, { status: 500 });
  }
}

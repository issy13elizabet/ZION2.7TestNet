import { NextResponse } from 'next/server';

export async function GET(request: Request) {
  const url = new URL(request.url);
  const address = url.searchParams.get('address') || '';
  const host = process.env.ZION_HOST || '127.0.0.1';
  const adapterPort = Number(process.env.ZION_ADAPTER_PORT || 18099);
  const adapterKey = process.env.ADAPTER_API_KEY || '';
  const adminEnabled = (process.env.ENABLE_WALLET_ADMIN || '').toLowerCase() === 'true';

  if (!adminEnabled) {
    return NextResponse.json({ error: 'admin disabled' }, { status: 403 });
  }

  try {
    const res = await fetch(`http://${host}:${adapterPort}/wallet/keys${address ? `?address=${encodeURIComponent(address)}` : ''}` , {
      headers: {
        ...(adapterKey ? { 'x-api-key': adapterKey } : {}),
      },
      cache: 'no-store',
    });
    const json = await res.json();
    return NextResponse.json(json, { status: res.status });
  } catch (e: any) {
    return NextResponse.json({ error: String(e?.message || e) }, { status: 500 });
  }
}

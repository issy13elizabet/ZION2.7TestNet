import { NextResponse } from 'next/server';

export async function GET() {
  const host = process.env.ZION_BACKEND_HOST || '127.0.0.1';
  const port = Number(process.env.ZION_BACKEND_PORT || 8000);

  try {
    // Get wallet addresses and their balances
    const addressesRes = await fetch(`http://${host}:${port}/wallet/addresses`, { cache: 'no-store' });
    const addresses = await addressesRes.json();

    if (!addresses || addresses.length === 0) {
      return NextResponse.json({
        zion: 0,
        btc: 0,
        lightning: 0,
        dharma_score: 0
      });
    }

    // Calculate total balance
    const totalBalance = addresses.reduce((sum: number, addr: any) => sum + (addr.balance || 0), 0);

    return NextResponse.json({
      zion: totalBalance / 100000000, // Convert atomic units to ZION
      btc: 0, // TODO: Integrate BTC wallet
      lightning: 0, // TODO: Integrate Lightning Network
      dharma_score: totalBalance / 1000000000 // Dharma score calculation
    });
  } catch (e: any) {
    console.error('Wallet balance error:', e);
    return NextResponse.json({
      zion: 0,
      btc: 0,
      lightning: 0,
      dharma_score: 0,
      error: String(e?.message || e)
    }, { status: 500 });
  }
}

import { NextResponse } from 'next/server';
import net from 'net';

export const dynamic = 'force-dynamic';

function sendLine(host: string, port: number, line: string, timeoutMs = 2000): Promise<string> {
  return new Promise((resolve, reject) => {
    const socket = new net.Socket();
    let buffer = '';
    let settled = false;

    const cleanup = () => {
      socket.removeAllListeners();
      try { socket.destroy(); } catch {}
    };

    const timer = setTimeout(() => {
      if (!settled) {
        settled = true;
        cleanup();
        reject(new Error('timeout'));
      }
    }, timeoutMs);

    socket.once('error', (err) => {
      if (!settled) {
        settled = true;
        clearTimeout(timer);
        cleanup();
        reject(err);
      }
    });

    socket.connect(port, host, () => {
      socket.setEncoding('utf8');
      socket.on('data', (chunk) => {
        buffer += chunk;
        const idx = buffer.indexOf('\n');
        if (idx !== -1 && !settled) {
          settled = true;
          clearTimeout(timer);
          const line = buffer.slice(0, idx);
          cleanup();
          resolve(line);
        }
      });
      const payload = line.endsWith('\n') ? line : line + '\n';
      socket.write(payload);
    });
  });
}

export async function GET() {
  const host = process.env.NEXT_PUBLIC_POOL_HOST || '91.98.122.165';
  const port = Number(process.env.NEXT_PUBLIC_POOL_PORT || 3333);
  const worker = 'web-test';
  const req = JSON.stringify({
    jsonrpc: '2.0',
    id: 1,
    method: 'login',
    params: { login: worker, pass: 'x', agent: 'zion-web/0.1' },
  });

  try {
    const line = await sendLine(host, port, req, 2500);
    return NextResponse.json({ ok: true, host, port, response: line });
  } catch (err: any) {
    return NextResponse.json({ ok: false, host, port, error: err?.message || String(err) }, { status: 502 });
  }
}

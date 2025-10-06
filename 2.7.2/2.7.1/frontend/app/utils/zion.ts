export function isValidZionAddress(addr: string): boolean {
  // Updated for Z3 prefix (new mainnet) with backward compatibility for aj (legacy)
  // Z3: new mainnet prefix, aj: legacy prefix still accepted
  return /^(Z3|aj)[123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz]{93}$/.test(addr);
}

export function buildXmrigCommand(opts: {
  host: string;
  port?: number;
  address: string;
  algo?: string;
  rigId?: string;
  tls?: boolean;
}): string {
  const { host, port = 3333, address, algo = 'rx/0', rigId = 'WEB', tls = false } = opts;
  const url = `${tls ? 'stratum+ssl' : 'stratum+tcp'}://${host}:${port}`;
  return [
    'xmrig',
    `--url ${url}`,
    `--algo ${algo}`,
    `--user ${address}`,
    '--pass x',
    '--keepalive',
    `--rig-id ${rigId}`,
    '--donate-level 0',
  ].join(' ');
}

# ZION v2.6 – Shrnutí auditu a změn (2025-09-25)

Tento dokument sumarizuje audit, bezpečnostní úpravy a operace provedené při přechodu na v2.6.

## Bezpečnost a hardening
- Frontend/SSR: hlavičky (HSTS, X-Content-Type-Options, X-Frame-Options, Referrer-Policy), CSP řízená env pro `connect-src`.
- Adapter (Express): helmet, CORS whitelist z env, API klíč povinný v prod, rate limiting.
- Daemon: RPC bind a CORS pouze přes parametry/env, defaultně bez veřejné expozice.
- Docker/Compose: RPC interní, veřejně jen P2P; pinned images; secrets přes env.
- Submodul: `zion-cryptonote` vendored, bez git submodule.

## Operace a nasazení
- Vytvořen čistý repozitář v2.6 (sanitizovaný export), `.gitignore` a `.gitattributes` pro Windows/Linux.
- Přidán rychlý SSH deploy skript s:
  - vytvořením minimálního `.env` na serveru,
  - odstraněním starých kontejnerů, `--remove-orphans`,
  - readiness čekací smyčkou přes interní RPC,
  - systemd službou `zion.service`.
- Nasazení na 91.98.122.165: node OK, P2P 18080 OK; pool port 3333 kolize (řešit uvolněním portu či přemapováním, viz níže).

## Doporučení
- Pool port: přemapovat na 3334 a upravit UFW pravidla, nebo uvolnit 3333.
- Git LFS: přidat pro velké binární soubory (např. `*.mp4`).

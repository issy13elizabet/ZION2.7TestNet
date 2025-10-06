# Monitor + Proxy + Frontend CI Update (2025-09-21)

This update adds:

- scripts/monitor-shim.sh — lightweight monitor for the RPC shim via proxy or direct endpoint. Detects height stalls and can POST to a webhook.
- scripts/ryzen-up.sh — now optionally starts the nginx reverse proxy (docker/compose.proxy.yml) and prints quick proxy checks.
- .github/workflows/frontend-ci.yml — builds and lints the Next.js frontend on pushes/PRs that touch `frontend/**`.

## How to use

- Bring up on Ryzen including proxy:
  - Run `scripts/ryzen-up.sh` on the server. If `docker/compose.proxy.yml` is present, proxy exposes port 8080.
- Monitor the chain height:
  - Local proxy: `./scripts/monitor-shim.sh --url http://localhost:8080 --interval 30 --stall-min 5`
  - Direct shim: `URL=http://91.98.122.165:18089 ./scripts/monitor-shim.sh`
  - Optional alerting: set `WEBHOOK_URL` to an HTTP endpoint to receive JSON alerts.
- CI: Frontend builds automatically on GitHub Actions when frontend files change.

## Notes

- Proxy routes:
  - `/healthz` — basic health
  - `/shim/` — RPC shim (Monero-like), metrics at `/shim/metrics.json`
  - `/adapter/` — explorer/wallet adapter API
- Rate limit remains ~10 rps/IP with burst 20 (nginx). Adjust in `docker/nginx/nginx.conf` as needed.

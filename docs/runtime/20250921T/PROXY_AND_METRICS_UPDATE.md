# Reverse Proxy & Shim Metrics Update

Date: 2025-09-21

## Summary
- Added Nginx reverse proxy with basic per-IP rate limiting (~10 rps, burst 20) and unified entrypoint on port 8080.
- Enhanced RPC shim with health (`/`) and metrics (`/metrics.json`) endpoints for monitoring/integrators.
- Updated CMC whitelist documentation to recommend proxy usage and document endpoints.

## Files Changed
- `docker/compose.proxy.yml` — proxy service definition.
- `docker/nginx/nginx.conf` — proxy config with rate limiting and routing to shim/adapter.
- `adapters/zion-rpc-shim/server.js` — added `/` and `/metrics.json` endpoints.
- `docs/COINMARKETCAP_WHITELIST.md` — documented proxy endpoints and rate limits.

## How to Run (on Ryzen)
```bash
# ensure network exists
docker network create zion-seeds || true

# start proxy (requires shim and adapter services on the same network)
docker compose -f docker/compose.proxy.yml up -d

# quick checks
curl -s http://localhost:8080/healthz
curl -s http://localhost:8080/shim/
curl -s http://localhost:8080/shim/metrics.json | jq '.height, .uptimeSec'
```

## Notes
- Proxy forwards to `zion-rpc-shim:18089` and `zion-wallet-adapter:18099` via `/shim/` and `/adapter/` paths.
- Keep ports 18089/18099 reachable within the docker network; only 8080 needs to be public.
- For higher traffic needs, adjust `limit_req_zone` in `docker/nginx/nginx.conf`.

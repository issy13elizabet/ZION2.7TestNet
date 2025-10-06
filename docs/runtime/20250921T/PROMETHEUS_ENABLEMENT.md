# Prometheus Enablement (2025-09-21)

We added a scrape-ready Prometheus integration for the Zion RPC shim.

What’s new:
- Shim now exposes `/metrics` in Prometheus plaintext format (in addition to `/metrics.json`).
- Nginx proxy routes:
  - `/shim/metrics` → plaintext metrics (no rate limit) for Prometheus
  - `/shim/metrics.json` → JSON metrics (rate-limited)
- Docker compose for Prometheus: `docker/compose.prometheus.yml`
- Prometheus config: `docker/prometheus/prometheus.yml` with two jobs (direct and proxy).
- `scripts/ryzen-up.sh` optionally starts Prometheus on port 9090 if compose file is present.

How to run:
- On the server, run `scripts/ryzen-up.sh` (proxy and Prometheus start if compose files exist).
- Open Prometheus UI: http://<host>:9090 and query `zion_shim_last_height` or `zion_shim_gbt_requests_total`.

Notes:
- Default scrape interval is 15s. Adjust in `docker/prometheus/prometheus.yml` as needed.
- Keep nginx `/shim/metrics` unthrottled for Prometheus; other endpoints remain rate-limited.

# dApp Wallet Upgrade (2025-09-21)

What changed:
- Adapter: new endpoints `/wallet/address`, `/wallet/history`, `/wallet/validate`, Prometheus `/metrics`; stricter send validation; basic route metrics.
- Frontend: Wallet page now shows own address (QR), balance, history and includes a simple send form.
- API: New proxy routes in Next.js under `/api/wallet/address` and `/api/wallet/history`.

How to try:
- Start stack on the server (ryzen-up) and run frontend locally or remotely.
- Open `/wallet` to see address/QR, balances, last ~20 txs, and a send form.
- Adapter metrics available at `http://<host>:18099/metrics` or via proxy `/adapter/metrics`.

Notes:
- Address/history depend on wallet RPC support; if unsupported, UI degrades gracefully.
- Send form is minimal; consider allowlist/limits for production.

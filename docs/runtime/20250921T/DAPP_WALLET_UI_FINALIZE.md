# DApp Wallet UI Finalize – 2025-09-21

Scope:
- Add admin actions to wallet UI (Generate new, Save, Export keys)
- Route calls via internal Next.js API with x-api-key when configured
- UX polish: busy states, inline validation, confirm dialog for key export

Changes:
- frontend/app/wallet/page.tsx: UI actions + confirm modal, better messaging, refresh after send
- frontend/app/api/wallet/{create,save,keys,send}/route.ts: proxy routes with admin gating and x-api-key
- frontend/README.md: docs for ENABLE_WALLET_ADMIN / NEXT_PUBLIC_ENABLE_WALLET_ADMIN, ADAPTER_API_KEY

Security:
- Admin routes require ENABLE_WALLET_ADMIN=true on server, UI buttons gated by NEXT_PUBLIC_ENABLE_WALLET_ADMIN=true
- API key set on server-side only; never exposed in client code

Validation:
- Typecheck/lint: no errors in changed files
- Quick smoke: UI components render conditions valid; send uses internal API and refreshes balance/history on success

Next:
- Optional: password-encrypted key export; address selection for multi-account wallets
- Consider e2e tests for wallet-adapter proxy routes

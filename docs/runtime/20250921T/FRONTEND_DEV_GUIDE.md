# Frontend Dev Guide (macOS against Ryzen backend)

Date: 2025-09-21

## Quick Start

```bash
bash scripts/frontend-dev.sh --open
```

- This generates `frontend/.env.local` pointing to the Ryzen backend defaults:
  - Host: 91.98.122.165
  - Stratum: 3333 (info only in UI)
  - Shim: 18089
  - Wallet Adapter: 18099
- Installs dependencies if missing and runs Next.js at http://localhost:3000.

## Environment Overrides

You can override defaults:

```bash
bash scripts/frontend-dev.sh \
  --host 203.0.113.10 \
  --pool-port 3333 \
  --shim-port 18089 \
  --adapter-port 18099 \
  --open
```

## Related Scripts
- `scripts/macos-clean-frontend-only.sh` — cleans all Zion Docker resources on macOS.
- `scripts/ryzen-up.sh` (run on Ryzen) — builds images, generates compose override, and starts the full backend stack.

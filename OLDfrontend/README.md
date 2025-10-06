# ZION Frontend (Next.js)

Minimal skeleton wired to the Amenti Library manifest in `../data/amenti/library.json`.

## Run

```bash
cd frontend
npm install
npm run dev
# open http://localhost:3000
```

Pages:
- `/` – Home
- `/amenti` – Amenti Library (reads via `/api/amenti`)
- `/wallet` – Jednoduchá peněženka (QR vlastní adresy, balance, historie, odeslání; nově i Generate/Save/Export)

## Pool connectivity test

Testovací endpoint pro ověření Stratum poolu je dostupný na `/amenti/pool-test`.

Volitelná konfigurace host/port přes env proměnné:

```bash
export NEXT_PUBLIC_POOL_HOST=91.98.122.165
export NEXT_PUBLIC_POOL_PORT=3333
npm run dev
# open http://localhost:3000/amenti/pool-test
```

API route `/api/pool-test` naváže TCP spojení na pool a odešle JSON-RPC `login` s workerem `web-test`.

## Wallet admin akce (Generate/Save/Export)

Frontend obsahuje interní API routy, které přeposílají požadavky do `wallet-adapter` a přidávají volitelný API klíč. Pro povolení administrativních akcí (vytvoření nové adresy, uložení wallet souboru, export privátních klíčů) nastav na serveru i UI:

```bash
export ENABLE_WALLET_ADMIN=true       # povolí /api/wallet/{create,save,keys}
export NEXT_PUBLIC_ENABLE_WALLET_ADMIN=true  # zobrazí tlačítka v UI na /wallet
export ZION_HOST=127.0.0.1            # nebo IP/hostname backendu
export ZION_ADAPTER_PORT=18099        # port wallet-adapteru
export ADAPTER_API_KEY=your-secret    # musí odpovídat ADAPTER_API_KEY na wallet-adapteru (pokud je na něm nastaven)
```

Routy:
- POST `/api/wallet/create` → zavolá `POST /wallet/create_address` na adapteru
- POST `/api/wallet/save` → zavolá `POST /wallet/save` na adapteru
- GET `/api/wallet/keys[?address=...]` → zavolá `GET /wallet/keys` na adapteru a vrátí JSON s klíči (klient nabídne stažení)
- POST `/api/wallet/send` → přepošle na `POST /wallet/send` (také přidá `x-api-key`, pokud je nastaven)

Poznámka: Pokud `ENABLE_WALLET_ADMIN` není `true`, admin routy vrací 403.

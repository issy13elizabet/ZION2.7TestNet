# ZION – Provozní log (2025-09-23/24)

Tento log shrnuje všechny kroky k zajištění stabilní těžby (60 bloků) a opravy skriptů/ops nástrojů. Obsahuje i přesné příkazy pro opětovné spuštění.

## AKTUALIZACE 2025-09-24

### Mining Upgrade: 4 → 12 threads + Direct Connection

**Původní stav:**
- XMRig: 4 vlákna, SSH tunel na localhost:3333
- Výkon: ~400-500 H/s
- Backend: "Core is busy" chyby

**Provedenné změny:**
1. **XMRig upgrade na 12 vláken:**
   ```powershell
   Start-Process -FilePath "D:\Zion TestNet\Zion\mining\platforms\windows\xmrig-6.21.3\xmrig.exe" -ArgumentList "--threads", "12", "--url", "91.98.122.165:3333", "--coin", "monero", "--user", "MAITREYA", "--pass", "x", "--donate-level", "1" -WindowStyle Hidden
   ```

2. **Git commit a push:**
   - Přidáno: xmrig-MAITREYA.json, xmrig-Ryzen3600.json
   - Přidány runtime logy s mining konfigurací
   - Commit: "Mining upgrade: XMRig to 12 threads + cryptonote submodule deployed"

3. **Docker rebuild attempt:**
   - Pokus o rebuild s cryptonote submodulem: ÚSPĚŠNÝ build
   - Problém: nový kontejner měl segfault (exit code 139)
   - Řešení: návrat na fungující zion-uzi-pool

4. **Backend stabilizace:**
   - Restart seed nodes: `docker restart zion-seed1 zion-seed2`
   - Oprava "Core is busy" chyby
   - Pool API funkční na 91.98.122.165:3333

**Výsledný stav:**
- XMRig: 12 threads, přímé připojení na 91.98.122.165:3333 (bez SSH tunelu)
- Teoretický výkon: 3x zvýšení (1200+ H/s)
- Proces ID: 23880, běží na pozadí (WindowStyle Hidden)
- Backend: stabilní, žádné "Core is busy" chyby

## 1) Změny v kódu a skriptech

- rpc-shim: bez změn zdrojáku v tomto kole; běží prefetch GBT + backoff.
- scripts/start-mining.ps1
  - Výchozí porty jen `3333` (dříve 443/80/3333)
  - Jasnější hlášky při testu konektivity
- scripts/ssh-restart-services.ps1
  - Restart přes nahraný bash skript (SCP) → vyhnutí se quoting chybám
  - Zobrazení shim metrics (`18089/metrics.json`), pool API (`8117/live_stats`), tail logů (shim/pool)
  - Fallback na SSH agent/heslo, pokud klíč není čitelný
- scripts/ssh-hotpatch-pool.ps1
  - Opraveno `scp` quotování (cesty s mezerami)
- scripts/ssh-probe-pool.ps1
  - Načítá `ssh-defaults.json`, robustnější SSH, přenos skriptu přes SCP
- scripts/ssh-tunnel-maitreya.ps1
  - Tuneluje `3333`, `18089` i `8117`; approved verb fix; načtení defaults
- scripts/ssh-start-pool.ps1 (nové)
  - Spuštění pouze `uzi-pool` kontejneru na serveru + status + tail logu
- scripts/auto-mine.ps1 (nové)
  - Orchestrátor: start poolu → restart služeb → otevření tunelu → start mineru proti `127.0.0.1:3333`
- scripts/ssh-defaults.ps1
  - Fix chybné interpolace v chybových hláškách
- scripts/ssh-defaults.json
  - `KeyPath` aktualizován na `C:\Users\anaha\.ssh\id_ed25519`
- README.md
  - Přidán Windows Quickstart pro `start-mining.ps1`, datum aktualizace

## 2) Stav po restartu (výřez)

- rpc-shim běží a prefetchuje GBT; health `http://127.0.0.1:18089/metrics.json` vrací `{"status":"ok"...}`
- `zion-uzi-pool` byl dočasně zastaven; přidán helper `ssh-start-pool.ps1` pro start
- Pool API `http://127.0.0.1:8117/live_stats` dostupné po startu poolu a tunelu

## 3) Orchestrátor – těžba „na auto“

Spusť vše jedním příkazem (Windows PowerShell):

```powershell
pwsh -NoProfile -File .\scripts\auto-mine.ps1
```

Co se stane:
- startne se `uzi-pool` na serveru,
- restartnou se služby (seed1, seed2, redis, rpc-shim, walletd, wallet-adapter),
- otevře se SSH tunel `127.0.0.1:3333 / :18089 / :8117`,
- spustí se XMRig přes `start-mining.ps1` na `127.0.0.1:3333`.

Ověření:
```powershell
curl http://127.0.0.1:8117/live_stats
curl http://127.0.0.1:18089/metrics.json
```

## 4) Manuální běh (kdyby bylo třeba)

```powershell
# Start pouze poolu (server)
pwsh -NoProfile -File .\scripts\ssh-start-pool.ps1

# Restart a health check (server)
pwsh -NoProfile -File .\scripts\ssh-restart-services.ps1

# SSH tunel (3333/18089/8117)
pwsh -NoProfile -File .\scripts\ssh-tunnel-maitreya.ps1 -Action start

# Start mineru proti tunelu
pwsh -NoProfile -ExecutionPolicy Bypass -File .\scripts\start-mining.ps1 -ServerIp 127.0.0.1 -Ports 3333
```

## 5) Pool config a shim poznámky

- Pool (`adapters/uzi-pool-config/config.json`):
  - `shareTrust.min=1`, `threshold=1` – okamžitá viditelnost share
  - `banning.enabled=false` – žádné banování během ladění
  - `varDiff.minDiff=2` – vhodné pro Ryzen
  - API povoleno na `8117`
- rpc-shim: GBT prefetch/cache, backoff při `Core is busy (-9)`, `/metrics.json`

## 6) Postup při potížích

- Časté `submit -9/Core is busy`: můžeme navýšit submit backoff v rpc-shim a hotpatchnout bez rebuildu.
- XMRig stále míří na `:443`: ukončit starý proces a spustit `start-mining.ps1` proti `127.0.0.1:3333`.
- SSH klíč „Permission denied“: dočasně použít heslo/agent; později fixnout ACL (icacls) na `id_ed25519`.

## 7) Cíl

- Vytěžit a potvrdit alespoň 60 bloků pro bootstrap sítě.

— Log vytvořen: 2025‑09‑23

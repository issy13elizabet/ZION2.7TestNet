# Migration z Legacy 2.6 na 2.6.5

## Souhrn migrace

Datum: 29. září 2025  
Stav: **Dokončena základní segregace, probíhá integrace zion-core**

## Co bylo přesunuto do legacy/

### ✅ Dokončeno - Legacy segregace

V původním repo (`/media/maitreya/ZION1/`) byla vytvořena `legacy/` struktura:

#### 1. `legacy/miners/` 
- `zion-miner-1.3.0/` - Původní miner verze
- `zion-miner-1.4.0/` - Poslední standalone verze
- `zion-multi-platform/` - Experimentální multi-platform build

#### 2. `legacy/experiments/`
- `build-*` adresáře - Různé build experimenty
- `CMakeLists-*.txt` - Hybridní/minimal/simple/xmrig varianty
- `zion-cryptonote/` - Původní CryptoNote implementace
- `xmrig-*` - XMRig konfigurační experimenty

#### 3. `legacy/configs/`
- `docker-compose-*.yml` - Bootstrap/production varianty
- `test-*.json`, `test-*.sh` - Legacy test soubory
- `*.bat` - Windows skripty

#### 4. `legacy/docker-variants/`
- Celá původní `docker/` složka (~30 Dockerfile variant)
- Compose přepisování, komplexní entrypointy
- Duplicitní konfigurace

### ⚠️ Varování v README souborech

Každá legacy složka má README s:
- Vysvětlením proč je zastaralá
- Odkazem na nové řešení v `zion2.6.5testnet/`
- Instrukcemi pro rollback (`v2.6-backup-final` tag)

## Co zůstává v původním repo

### Aktivní komponenty (k integraci)
- `zion-core/` - **TOHLE POTŘEBUJEME INTEGROVAT** ⭐
- `frontend/` - Již zkopírováno do nového repo
- `docs/` - Již zkopírováno do nového repo  
- `scripts/` - Utility skripty (některé relevantní)
- `config/` - Network konfigurace (některé relevantní)

### Archive/backup komponenty
- `Zion-2.6-final-backup.tar.gz` - Kompletní backup
- `MIGRATED.md` - Migration log
- `VERSION` - Single source versioning
- `.git/` - Git historie s backup tagy

## Stav nového repo (zion2.6.5testnet/)

### ✅ Hotové komponenty

1. **Základní struktura**
   - `VERSION` (2.6.5)
   - `docker-compose.yml` (core, frontend, miner)
   - `.env.example` pro konfigurace

2. **Core skeleton**
   - `core/src/server.ts` - HTTP API + Stratum bootstrap
   - `core/src/pool/StratumServer.ts` - Embedded mining pool
   - Environment variable support (PORT, STRATUM_PORT, INITIAL_DIFFICULTY)

3. **Miner** 
   - C++ sources zkopírované z `zion-miner-1.4.0/`
   - CMakeLists.txt pro unified build
   - Docker build support

4. **Frontend**
   - Next.js aplikace zkopírovaná
   - Package.json aktualizován
   - Docker build ready

5. **Infrastruktura**
   - GitHub Actions CI (core build + miner build)
   - Docker orchestrace (3 služby, jednoduché)
   - Network config placeholders

6. **Dokumentace**
   - `SIMPLE_MIGRATION.md` - Souhrn současného stavu
   - `DOCKER_GUIDE.md` - Docker usage guide
   - `docs/ARCHITECTURE.md` - Kompletní architektura

### 🔄 Probíhající - Zion-core integrace

**Aktuální priorita**: Integrace `zion-core/` do nové struktury

Zion-core obsahuje:
- Blockchain logiku (consensus, blocks, transactions)
- RPC server implementaci
- P2P networking
- Wallet API
- Mining koordinaci

**Plán integrace**:
1. Analyzovat `zion-core/` strukturu a závislosti
2. Rozdělit na moduly: blockchain/, rpc/, p2p/, wallet/
3. Integrovat do `core/src/` alongside pool/ modulem
4. Aktualizovat TypeScript konfigurace a dependencies
5. Testovat kompatibilitu s existing Stratum serverem

### ❌ Chybí implementovat

1. **Real blockchain state**
   - Import consensus logiky z zion-core
   - Block template generation místo random bytes
   - Chain synchronization

2. **Share validation**
   - Target calculation a verification
   - Duplicate share detection
   - Mining statistics tracking

3. **Difficulty adjustment**
   - Vardiff implementace
   - `mining.set_difficulty` updates
   - Per-connection difficulty management

4. **Security hardening**
   - Input validation
   - Rate limiting
   - Authentication middleware

## Rollback strategie

### Kompletní rollback
```bash
cd /media/maitreya/ZION1
git checkout v2.6-backup-final
# nebo
tar -xzf Zion-2.6-final-backup.tar.gz
```

### Částečný rollback komponent
```bash
# Obnovit specifický komponent z legacy/
cp -r legacy/miners/zion-miner-1.4.0 ./
cp -r legacy/docker-variants/docker ./
```

## Následující kroky

### Priorita 1: Zion-core integrace
1. **Analyzovat zion-core struktur** - identifikovat klíčové moduly
2. **Rozdělit na čisté moduly** - blockchain, rpc, p2p, wallet
3. **Integrovat do core/src/** - zachovat Stratum server
4. **Aktualizovat dependencies** - TypeScript, Node modules
5. **Testovat build pipeline** - ujistit se že CI projde

### Priorita 2: Funkční validace  
1. **Implementovat share validation** - real target checking
2. **Block template generation** - real blockchain headers
3. **Chain sync testing** - peer discovery a synchronizace

### Priorita 3: Production readiness
1. **Security review** - input validation, auth, rate limiting  
2. **Performance testing** - load testing pool serveru
3. **Documentation** - deployment guides, troubleshooting

## Metriky úspěchu

### Současné (skeleton fáze)
- ✅ Docker compose up bez chyb
- ✅ Health endpoint responds (200 OK)
- ✅ Stratum server accepts connections
- ✅ CI build passes (core + miner)
- ✅ Environment variables fungují

### Cílové (post zion-core integrace)
- 🔄 Real mining jobs generation
- 🔄 Share validation + reject neplatných  
- 🔄 Blockchain sync s peers
- 🔄 RPC endpoints functional
- 🔄 Wallet operations working

### Production ready
- ❌ Performance: 1000+ shares/sec
- ❌ Latency: <100ms job broadcast
- ❌ Security: Penetration testing passed
- ❌ Reliability: 99.9% uptime capability
- ❌ Monitoring: Comprehensive metrics/alerting

## Technické dluhy

### Legacy cleanup (post-migration)
- [ ] Smazat zastaralé test soubory z root
- [ ] Consolidovat dokumentaci (remove duplicates)
- [ ] Archive session logs do docs/sessions/
- [ ] Update README.md s odkazy na nové repo

### Type safety improvements
- [ ] Instalovat @types/node místo lightweight shims
- [ ] Stricter TypeScript config (noImplicitAny, etc.)
- [ ] ESLint setup pro code quality
- [ ] Prettier pro code formatting

### Monitoring/observability
- [ ] Structured logging (Winston/Pino)
- [ ] Prometheus metrics export
- [ ] Health checks rozšířit (DB, peers, sync status)
- [ ] Error tracking (Sentry integration)

## Závěr

Legacy segregace je **dokončena** - všechny zastaralé komponenty jsou bezpečně archivované v `legacy/` s jasnou dokumentací.

**Další krok**: Integrace `zion-core/` do nové struktury pro získání real blockchain functionality.

Nové repo má solidní základ s Docker orchestrací, CI/CD, a Stratum server placeholderem. Potřebujeme teď přidat "mozek" z original zion-core implementace.
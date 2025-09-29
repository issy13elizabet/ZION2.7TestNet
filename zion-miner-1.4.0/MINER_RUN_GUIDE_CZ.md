# ZION Miner – Průvodce během na noexec (Linux)

Tento dokument popisuje proč binárka na některých systémech končí chybou `exit code 126` a jak miner správně spustit v prostředí, kde je cílový disk připojen s příznakem `noexec` (typicky např. USB flash / VFAT, některé NAS mounty nebo zabezpečené oddíly).

---
## 1. Projev chyby
Při pokusu spustit miner přímo z adresáře repozitáře:
```
./zion-miner
bash: ./zion-miner: Permission denied
# nebo
Process exited with code 126
```
I když jsou práva správně (chmod +x), jádro odmítne spustit binárku z filesystému, který má mount flag `noexec`.

### Jak ověřit
```
mount | grep ZION1   # nebo název zařízení
```
Pokud výpis obsahuje `noexec`, spadá do této situace.

---
## 2. Řešení: Kopírování na exec FS (např. /tmp)
Adresář `/tmp` (tmpfs) je typicky připojen s možností spouštění binárek. Postup:
```
cd zion-miner-1.4.0/build
make -j$(nproc)
cp zion-miner /tmp/zion-miner
cd /tmp
./zion-miner --protocol=cryptonote --pool 91.98.122.165:3333 --wallet <ADRESA> --cpu-only
```
> Pozn: Parametr `--cpu-only` je vhodný pro rychlý test bez inicializace GPU.

### Jednořádkový helper
```
(cd zion-miner-1.4.0/build && make -j$(nproc) && cp zion-miner /tmp/zion-miner) && /tmp/zion-miner --protocol=cryptonote --pool 91.98.122.165:3333 --wallet <ADRESA> --cpu-only
```

---
## 3. Automatizace skriptem
Vytvořte skript `run-local.sh` (volitelné):
```
#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "$0")" && pwd)
BUILD_DIR="$ROOT_DIR/zion-miner-1.4.0/build"
BIN="zion-miner"
if [ ! -d "$BUILD_DIR" ]; then
  echo "Chybí build/ – spusť nejprve cmake nebo existuje jiná verze." >&2
  exit 1
fi
make -C "$BUILD_DIR" -j"$(nproc)"
cp "$BUILD_DIR/$BIN" /tmp/$BIN
exec /tmp/$BIN "$@"
```
Pak:
```
chmod +x run-local.sh
./run-local.sh --protocol=cryptonote --pool 91.98.122.165:3333 --wallet <ADRESA>
```

---
## 4. Bezpečnostní poznámky
- `/tmp` je sdílené místo – pokud běží více uživatelů, můžete kolidovat se jménem. Případně použijte `/tmp/$USER-zion-miner`.
- Po rebootu se obsah `/tmp` smaže – je nutné znovu kopírovat.
- Neprovádějte těžbu jako `root` bez nutnosti.

---
## 5. Debug tipy
| Symptom | Akce |
|---------|------|
| Stále 126 | Ověř `ls -l /tmp/zion-miner` a `file /tmp/zion-miner` (není poškozeno) |
| Segfault po spuštění | Spusť `ldd /tmp/zion-miner` – chybějící knihovny (OpenSSL, blake3, atd.) |
| 0 H/s | Čeká na první job – zkus `--protocol=cryptonote` + ověř spojení a firewall |
| Bez shares dlouho | Přidej low target debug (až implementováno) nebo ověř algoritmus (musí být cosmic) |

---
## 6. Klávesové zkratky (v1.4.0+)
| Klávesa | Funkce |
|---------|--------|
| q | Ukončit miner |
| s | Zap/Vyp periodický výpis statistik |
| d | Detaily (rozšířené info) |
| b | Brief mód (kompaktní výpis) |
| g | Zap/Vyp GPU těžbu (pokud dostupná) |
| o | Cyklické přepnutí algoritmu (cosmic → blake3 → keccak → cosmic) – debug módy neposílají shares |
| r | Reset počítadel (hashes, shares) |
| c | Clear screen |
| v | Verbose režim stratum (RAW logy) |
| h / ? | Nápověda |

---
## 7. Doporučený workflow při vývoji
1. Editace zdrojáků v repu (i na noexec FS nevadí).
2. Build do `build/`.
3. Kopie binárky na exec FS (`/tmp`).
4. Spuštění s parametry.
5. Tail / grep `stratum-raw.log` při debugování (verbose ON).
6. Opakuj.

---
## 8. Možné alternativy k /tmp
| Cíl | Poznámka |
|-----|----------|
| `~/.local/bin` | Pokud je mount bez `noexec` a v PATH |
| Vlastní tmpfs mount | `sudo mount -t tmpfs -o size=256M tmpfs ~/zion-run` |
| Docker kontejner | Izolované prostředí, lze spustit "FROM ubuntu:..." + build uvnitř |

---
## 9. FAQ
**Proč kompilace proběhne ale spuštění ne?**  Kompilace jen zapisuje soubor; spouštění potřebuje exec flag na filesystemu.

**Můžu remountnout disk s exec?** Pokud máš root a není to bezpečnostní politika. Např. `sudo mount -o remount,exec /media/XYZ`. Dejte pozor na rizika.

**Je rozdíl výkonu běhu z /tmp?** Ne, spíše mírně pozitivní (tmpfs v RAM). Hashrate nebude negativně ovlivněn.

---
## 10. Další kroky
- Implementace `--force-low-target` pro snadné generování test shares (TODO).
- Normalizovaná prezentace difficulty (částečně hotová – job-level výpočet).
- Refaktor Big256 helper do jedné hlavičky.

---
*Verze dokumentu:* 1.0 / 2025-09-29

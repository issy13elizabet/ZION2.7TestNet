# Zion Network 2.6.5 Testnet (Skeleton)

Prvni minimalni skeleton nove struktury. Obsahuje pouze:
- VERSION (2.6.5)
- core (Express skeleton /healthz)

Dalsi kroky:
1. Prenos realne logiky (blockchain, rpc, pool) do `core/`.
2. Integrace mineru 1.4.0 do `miner/`.
3. Pridani frontend aplikace.
4. Docker a bootstrap skripty.

## Zion Next Chain (Experimental)
Prototyp noveho chainu je v adresari `zion-next-chain/`.

### Genesis Tooling
Script `scripts/genesis.ts` generuje deterministicky genesis blok s velkou foundation alokaci (1B ZION) a hleda nonce splnujici jednoduchy prefix target.

Aktualni placeholder PoW: kompozitni hash blake3 + keccak256 (keccak(blake3(header) || blake3(header||nonce))). Cilem je pouze nalezt hash se zacatkem `TARGET_PREFIX` (momentálne velmi jednoduche pro rychlou iteraci).

Vystupni artefakty (adresar `zion-next-chain/genesis/`):
- `genesis.json` manifest (header, coinbase, powHash, allocation, target_prefix)
- `header.hex` serializovana hlavicka (JSON zobrazeni)
- `coinbase.hex` raw hex coinbase transakce

Prikazy (spoustet v `zion-next-chain/`):
- Genesis mining: `npm run genesis`
- Overeni artefaktu: `npm run verify:genesis`

Determinismus: Pri stejnem timestampu, alokaci a target prefixu je nalezeny nonce reprodukovatelny (linearni hledani od 0). Jakakoliv zmena serializace nebo hashing pipeline zmeni vysledek -> je nutne znovu spustit generate + verify.

Planovane kroky:
- Nahradit prefix check realnym ciselny target/difficulty.
- Integrace dalsich fazi Cosmic Dharma PoW.
- Finalizace merkle root vypoctu (aktualne trivialni) a transakcniho formatu.

Poznamka: Soucasny kod je experimentalni a neni urcen pro produkcni pouziti.

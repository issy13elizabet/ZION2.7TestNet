# Wallet key management (pool + dev)

Do NOT commit real private keys.

Plánované změny:
- Pool fee wallet (0.33%) — vytvořit novou peněženku a nastavit její adresu v `config.json` (`poolServer.poolAddress`).
- Dev/Core dev donation (2%) — vytvořit novou peněženku pro dev tým a nastavit adresu v odpovídajícím poli (viz níže).

Poznámka k adresám pro donation:
- Uzi-Pool konfigurace používá `poolServer.poolAddress` pro těžbu a payouty. Donation adresy jsou řešené v `blockUnlocker` (viz zdroj Uzi-Pool). Pokud implementace vyžaduje explicitní pole (např. `devAddress`), je nutné je přidat do kódu poolu nebo jeho config parseru.

## Z3 prefix migrace (2025-09)

- Nové adresy začínají na `Z3...`. Staré legacy `aj...` zůstávají akceptované (příjem).
- Validátory byly aktualizovány: pool (`coins/zion.json`) i adapter (`wallet-adapter`).
- V nasazení NEukládej adresy do repozitáře. Využij proměnné prostředí v `docker compose`:

Proměnné (pro službu `uzi-pool`):
- `POOL_ADDRESS` — hlavní pool peněženka (příjem block rewardů)
- `DEV_ADDRESS` — dev donation peněženka
- `CORE_DEV_ADDRESS` — core-dev donation peněženka

Entry point poolu tyto hodnoty při startu zapíše do `/app/config.json`.

Bezpečné uložení privátních klíčů:
- Použijte offline generování řešené mimo tento repozitář.
- Uložte do password manageru (KeePass/1Password) a do zabezpečeného trezoru (např. HashiCorp Vault) na serveru.
- V CI/CD nepřenášet jako prosté proměnné, raději injektovat runtime tajemství přes vault/agent.

Po vytvoření adres:
- Aktualizujte `config.json` poolu s novými adresami.
- Redeploy pomocí `docker compose`.

# Denní log – 2025-09-25

- Inicializace repozitáře v2.6, push na GitHub.
- Úprava `deploy-ssh.sh`: nový repo URL, auto `.env`, cleanup kontejnerů, readiness, systemd.
- Rychlé nasazení přes SSH s heslem na 91.98.122.165 – node běží, P2P 18080 OK, pool má kolizi portu 3333.
- Plán: oprava kolize poolu (3333 -> 3334 nebo uvolnit), smoke testy adapteru, Git LFS pro velké soubory.

## Pokračování – oprava poolu a ověření
- Identifikováno: server stále používal compose s mapováním 3333:3333 a starý pool image; současně běžel docker-proxy na 3333.
- Řešení: změna `docker-compose.prod.yml` pro službu `zion-pool` na Node stratum stub (`./pool`) s mapováním host `3334` -> container `3333`.
- Oprava YAML (PowerShell here-string zdvojoval uvozovky) a re-deploy na serveru (`git reset --hard origin/main`).
- UFW: otevřen port 3334/TCP.
- Výsledek: `zion-pool` nasazen, publikuje `0.0.0.0:3334->3333/tcp`; `ss -tuln` potvrzuje LISTEN na 3334. Miner endpoint: `stratum+tcp://91.98.122.165:3334`.

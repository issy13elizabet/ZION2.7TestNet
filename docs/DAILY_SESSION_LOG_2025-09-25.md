# Denní log – 2025-09-25

- Inicializace repozitáře v2.6, push na GitHub.
- Úprava `deploy-ssh.sh`: nový repo URL, auto `.env`, cleanup kontejnerů, readiness, systemd.
- Rychlé nasazení přes SSH s heslem na 91.98.122.165 – node běží, P2P 18080 OK, pool má kolizi portu 3333.
- Plán: oprava kolize poolu (3333 -> 3334 nebo uvolnit), smoke testy adapteru, Git LFS pro velké soubory.

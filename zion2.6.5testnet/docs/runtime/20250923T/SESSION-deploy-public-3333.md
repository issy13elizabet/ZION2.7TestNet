# SESSION: Deploy veřejného Stratum 3333 (2025-09-23)

## Kontext a stav
- Shim přepnut na seed1-only (stabilizace GBT), uzi-pool publikuje `0.0.0.0:3333` lokálně.
- Ryzen miner přepnut na novou miner peněženku `Z321vz...Y5HcFU`.
- 91.98.122.165:3333 nešlo zvenku, protože lokální host je v LAN (`192.168.0.107`), bez NAT forwardingu.

## Cíl
- Zpřístupnit Stratum na `91.98.122.165:3333` pro externí těžaře.

## Zvolený postup
1) Nasadit pool přímo na veřejný server `91.98.122.165` (Docker, stable):
   - scripts/ssh-redeploy-pool.ps1 použije repo `Maitreya-ZionNet/Zion-v2.5-Testnet` a větev `main`.
   - Ověření: `Test-NetConnection 91.98.122.165 -Port 3333` a krátký `xmrig --url=91.98.122.165:3333` test.
2) Alternativy: NAT forward nebo reversní SSH tunel (nepreferováno pro produkci).

## Poznámky k bezpečnosti
- `.env` drží adresy (pool/dev/core-dev) mimo git. Dev adresy zatím prázdné.
- Firewall na serveru otevřít 3333/TCP; pokud běží UFW, skript to zkusí povolit.

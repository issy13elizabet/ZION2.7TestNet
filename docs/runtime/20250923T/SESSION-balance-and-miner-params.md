# SESSION: Balance a parametry mineru (2025-09-23)

## Wallet balance (wallet-adapter)
- availableBalance: `0`
- lockedAmount: `109860976246611` (≈ `109,860.976246611` ZION)
- Pozn.: Coinbase odměny se odemykají po block maturity (u nás 60 bloků, ~2 hodiny).

## XMRig parametry (Ryzen)
- pool: `zion-uzi-pool:3333` (z kontejneru) / `host.docker.internal:3333` (pro docker run)
- algo: `rx/0`
- user (wallet): `Z321y5Ugios4VrzgpbGkvoBrmA3QeVC2xao6kY3SDSkA7J8toEUktYzBAYyqQ8SJvP4kAaz1APCBEGRuYWersbCR33tetBoXxg`
- pass: `x`
- rig-id: `RYZEN`
- keepalive: `true`
- tls: `false`

## Příkazy (PowerShell)
- Tail logů:
```powershell
docker logs --tail=80 -f zion-xmrig-ryzen
```
- Spuštění vlastní instance (CLI varianta):
```powershell
docker run --rm -d --name xmrig-local `
  --cpus 12 `
  zion:xmrig `
  --url=host.docker.internal:3333 `
  --algo=rx/0 `
  --user=Z321y5Ugios4VrzgpbGkvoBrmA3QeVC2xao6kY3SDSkA7J8toEUktYzBAYyqQ8SJvP4kAaz1APCBEGRuYWersbCR33tetBoXxg `
  --pass=x `
  --rig-id=MYPC `
  --keepalive
```
- Stop:
```powershell
docker rm -f xmrig-local
```

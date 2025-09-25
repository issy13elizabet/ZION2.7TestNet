# Windows miner run (autodetect XMRig + 443)

Date: 2025-09-23
Host: MAITREYA (Ryzen 5 3600)

Summary:
- Added Start-Mining.ps1 with wallet autodetect and port probing
- Enhanced XMRig path autodetection to use repo-bundled binary at mining/platforms/windows/xmrig-6.21.3/xmrig.exe
- Port checks: 443 reachable, 80/3333 refused from network
- XMRig launched via 443 → observed "read error: end of file" while core reports "getblocktemplate: Core is busy"
- Pool and rpc-shim healthy; ziond returns -9 busy; mining should activate once templates are available

Details:
- Config generated: Zion/mining/xmrig-Ryzen3600.json
- XMRig args: --config <cfg> --asm ryzen --cpu-priority 3
- Server 91.98.122.165:443 is forwarded to pool 3333 via socat; UFW allows 443/80/3333

Next:
- Keep miner running; once getblocktemplate stops returning busy, stratum handshake will complete
- Optionally adjust pool to keep TCP open even when busy (to avoid EOF churn)

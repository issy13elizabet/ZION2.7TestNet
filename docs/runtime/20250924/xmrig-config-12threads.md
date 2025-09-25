# XMRig Configuration - 2025-09-24

## Active Mining Configuration

### Command Line Parameters:
```bash
--threads 12
--url 91.98.122.165:3333
--coin monero
--user MAITREYA
--pass x
--donate-level 1
```

### System Information:
- CPU: AMD Ryzen 5 3600 (12 threads available)
- OS: Windows 11
- XMRig Version: 6.21.3
- Algorithm: RandomX

### Network Configuration:
- Pool Server: 91.98.122.165:3333
- Connection: Direct (no SSH tunnel)
- Stratum Protocol: stratum+tcp

### Process Information:
- Process ID: 23880
- Memory Usage: ~14MB
- CPU Usage: Variable (mining workload)
- Window: Hidden (background process)

### Historical Performance:
- Previous: 4 threads → ~400-500 H/s
- Current: 12 threads → Expected 1200+ H/s
- Improvement: 3x theoretical increase

### Docker Backend:
- Pool Container: zion-uzi-pool (stable)
- Backend Nodes: zion-seed1, zion-seed2 (healthy)
- RPC Shim: functioning, no "Core is busy" errors

### Monitoring Commands:
```powershell
# Process status
Get-Process | Where-Object {$_.ProcessName -eq "xmrig"} | Select-Object ProcessName, Id, CPU, WorkingSet

# Network connectivity test
Test-NetConnection -ComputerName 91.98.122.165 -Port 3333

# Pool backend logs
ssh -o StrictHostKeyChecking=no root@91.98.122.165 "docker logs zion-uzi-pool --tail 20"

# Stop mining (if needed)
Get-Process | Where-Object {$_.ProcessName -eq "xmrig"} | Stop-Process -Force

# Restart mining with same config
Start-Process -FilePath "D:\Zion TestNet\Zion\mining\platforms\windows\xmrig-6.21.3\xmrig.exe" -ArgumentList "--threads", "12", "--url", "91.98.122.165:3333", "--coin", "monero", "--user", "MAITREYA", "--pass", "x", "--donate-level", "1" -WindowStyle Hidden
```
@echo off
echo ===============================================
echo ZION CORE v2.5.0 - MULTI-ALGORITHM GPU MINING
echo ===============================================
echo.
echo 🚀 Starting ZION Core Multi-Algo Server...
start "ZION-MULTI-CORE" /D "D:\Zion TestNet\Zion\zion-core" node dist/server.js

timeout /t 5 /nobreak > nul

echo 🎯 Available Mining Algorithms:
echo   - RandomX    (port 3333) - Monero-compatible
echo   - KawPow     (port 3334) - Ravencoin-compatible  
echo   - Ethash     (port 3335) - Ethereum Classic
echo   - CryptoNight(port 3336) - ZION native
echo.

echo 🧠 Starting Multi-Algo Bridge with SRBMiner-Multi...
start "ZION-MULTI-BRIDGE" python "D:\Zion TestNet\Zion\mining\zion-multi-algo-bridge.py"

timeout /t 3 /nobreak > nul

echo 💻 Starting CPU Mining (Backup/Fallback)...
start "ZION-CPU-BACKUP" "D:\Zion TestNet\Zion\mining\xmrig-6.21.3\xmrig.exe" --cpu-threads 6 -o localhost:3333 -u Z321vzLnnRu15AF82Kwx5DVZAX8rf1jZdDThTFZERRwKV23rXUqcV5u5mvT45ZDpW71kG3zEGuM5F2n3uytvAx5G9jFSY5HcFU -p cpu-backup-ryzen --no-color --print-time 30

echo.
echo ===============================================
echo MULTI-ALGO MINING STATUS:
echo - GPU: AMD Radeon RX 5600 XT (SRBMiner-Multi v2.9.7)
echo - CPU: AMD Ryzen 5 3600 (6 threads backup)
echo - Algorithms: RandomX, KawPow, Octopus, Ergo, Ethash, CryptoNight
echo - Bridge: Intelligent profitability switching
echo - Server: ZION Core TypeScript v2.5.0
echo ===============================================
echo.
echo Press any key to stop all mining processes...
pause > nul

echo Stopping all mining processes...
taskkill /F /FI "WindowTitle eq ZION-*" 2>nul
taskkill /F /IM "python.exe" /FI "WindowTitle eq ZION-*" 2>nul
echo Multi-algo mining stopped.
pause
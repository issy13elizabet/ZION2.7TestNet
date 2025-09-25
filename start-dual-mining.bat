@echo off
echo ===============================================
echo ZION CORE v2.5.0 - DUAL CPU + GPU MINING
echo ===============================================
echo.
echo 🚀 Starting ZION Core TypeScript Server...
start "ZION-CORE" /D "D:\Zion TestNet\Zion\zion-core" node dist/server.js

timeout /t 3 /nobreak > nul

echo 💻 Starting CPU Mining (12 threads)...
start "ZION-CPU" "D:\Zion TestNet\Zion\mining\xmrig-6.21.3\xmrig.exe" --threads 12 -o localhost:3333 -u CPU-MINER-RYZEN -p cpu-12threads --no-color --print-time 10

timeout /t 2 /nobreak > nul

echo 🎮 Starting GPU Mining (AMD RX 5600 XT)...
start "ZION-GPU" "D:\Zion TestNet\Zion\mining\xmrig-6.21.3\xmrig.exe" --config "D:\Zion TestNet\Zion\mining\zion-gpu-config.json"

echo.
echo ===============================================
echo MINING STATUS:
echo - CPU: AMD Ryzen 5 3600 (12 threads)  
echo - GPU: AMD Radeon RX 5600 XT (OpenCL)
echo - Server: ZION Core TypeScript v2.5.0
echo - Pool: localhost:3333 (Stratum)
echo ===============================================
echo.
echo Press any key to stop all mining processes...
pause > nul

echo Stopping all mining processes...
taskkill /F /FI "WindowTitle eq ZION-*" 2>nul
echo Mining stopped.
pause
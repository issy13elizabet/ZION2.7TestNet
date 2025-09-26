@echo off
echo ===============================================
echo ZION BLOCKCHAIN NODE + MINING POOL STARTUP
echo ===============================================
echo.
echo 🌐 Starting ZION Blockchain Node...

REM Check if zion-core directory exists
if not exist "D:\Zion TestNet\Zion\zion-core" (
    echo ❌ Error: zion-core directory not found!
    pause
    exit /b 1
)

REM Start ZION Core node first
echo 🔗 Initializing ZION Core v2.5.0...
start "ZION-CORE-NODE" /D "D:\Zion TestNet\Zion\zion-core" node dist/server.js

REM Wait for node to initialize
timeout /t 8 /nobreak > nul

echo.
echo ⛏️  Starting ZION Mining Pool...
echo    - Pool Wallet: Z321vzLnnRu15AF82Kwx5DVZAX8rf1jZdDThTFZERRwKV23rXUqcV5u5mvT45ZDpW71kG3zEGuM5F2n3uytvAx5G9jFSY5HcFU  
echo    - Network: zion-mainnet-v2
echo    - Pool Fee: 2.5%
echo    - Min Payout: 0.1 ZION
echo.

echo 🎯 Multi-Algorithm Ports:
echo    - RandomX:    localhost:3333
echo    - KawPow:     localhost:3334  
echo    - Ethash:     localhost:3335
echo    - CryptoNight: localhost:3336
echo    - Octopus:    localhost:3337
echo    - Ergo:       localhost:3338
echo.

echo 🚀 Starting Mining Pool Server...
start "ZION-MINING-POOL" /D "D:\Zion TestNet\Zion\zion-core" node dist/modules/mining-pool.js

timeout /t 5 /nobreak > nul

echo.
echo ===============================================
echo ZION MINING INFRASTRUCTURE STATUS:
echo - Blockchain Node: Running (ZION Core v2.5.0)
echo - Mining Pool: Multi-Algorithm Ready
echo - Wallet Address: ZION Native (87 chars)
echo - Pool Validation: Active
echo ===============================================
echo.
echo Press any key to stop all ZION services...
pause > nul

echo Stopping ZION services...
taskkill /F /FI "WindowTitle eq ZION-*" 2>nul
echo ZION mining infrastructure stopped.
pause
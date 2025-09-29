@echo off
REM ZION GPU Mining with SRBMiner-Multi (Windows)
REM Universal GPU miner for NVIDIA and AMD cards

echo 🚀 ZION GPU Mining Setup
echo ========================

REM Check if SRBMiner exists
set SRBMINER_PATH=""
if exist "SRBMiner-MULTI.exe" set SRBMINER_PATH="SRBMiner-MULTI.exe"
if exist "SRBMiner-Multi\SRBMiner-MULTI.exe" set SRBMINER_PATH="SRBMiner-Multi\SRBMiner-MULTI.exe"
if exist "C:\ZionMining\SRBMiner-Multi\SRBMiner-MULTI.exe" set SRBMINER_PATH="C:\ZionMining\SRBMiner-Multi\SRBMiner-MULTI.exe"

if %SRBMINER_PATH%=="" (
    echo ❌ SRBMiner-MULTI not found!
    echo 💡 Please download from: https://github.com/doktor83/SRBMiner-Multi/releases
    echo    Or run: setup-gpu-mining.bat
    pause
    exit /b 1
)

REM Check pool connectivity
echo 🔗 Testing mining pool connection...
powershell -Command "Test-NetConnection -ComputerName localhost -Port 3333 -InformationLevel Quiet" >nul 2>&1
if errorlevel 1 (
    echo ❌ Mining pool not reachable on localhost:3333
    echo 💡 Make sure ZION mining pool is running
    pause
    exit /b 1
)

echo ✅ Mining pool connection OK
echo.

REM Detect GPU type
echo 🔍 Detecting GPU hardware...
for /f "tokens=*" %%i in ('powershell -Command "(Get-WmiObject Win32_VideoController | Where-Object {$_.Name -like '*NVIDIA*' -or $_.Name -like '*AMD*' -or $_.Name -like '*Radeon*'}).Name"') do (
    echo 🖥️  Found GPU: %%i
    
    REM NVIDIA GPUs
    echo %%i | findstr /i "NVIDIA RTX GTX" >nul
    if not errorlevel 1 (
        echo 💚 NVIDIA GPU detected - Using optimized settings
        goto :nvidia_mining
    )
    
    REM AMD GPUs  
    echo %%i | findstr /i "AMD Radeon RX" >nul
    if not errorlevel 1 (
        echo 🔴 AMD GPU detected - Using optimized settings
        goto :amd_mining
    )
)

echo ⚠️  GPU auto-detection failed - Using generic settings
goto :generic_mining

:nvidia_mining
echo 🚀 Starting NVIDIA GPU mining...
%SRBMINER_PATH% ^
    --algorithm randomx ^
    --pool localhost:3333 ^
    --wallet Z321nirFfsdGcAE8Loe1vzcZS9ztUvsTCYsYKYsncA63QqjqQyMnLiedvZSnniUsfE93Zdvu5tpkvC2qNVpDf4ot9q1UJUBMap ^
    --password nvidia-gpu-rig ^
    --worker gpu-nvidia-01 ^
    --gpu-id 0 ^
    --gpu-threads 16 ^
    --gpu-worksize 8 ^
    --gpu-intensity 20 ^
    --cpu-threads 0 ^
    --disable-cpu ^
    --log-file logs\srbminer-nvidia.log ^
    --api-enable --api-port 21555
goto :end

:amd_mining
echo 🚀 Starting AMD GPU mining...
%SRBMINER_PATH% ^
    --algorithm randomx ^
    --pool localhost:3333 ^
    --wallet Z321nirFfsdGcAE8Loe1vzcZS9ztUvsTCYsYKYsncA63QqjqQyMnLiedvZSnniUsfE93Zdvu5tpkvC2qNVpDf4ot9q1UJUBMap ^
    --password amd-gpu-rig ^
    --worker gpu-amd-01 ^
    --gpu-id 0 ^
    --gpu-threads 18 ^
    --gpu-worksize 256 ^
    --gpu-intensity 25 ^
    --cpu-threads 0 ^
    --disable-cpu ^
    --log-file logs\srbminer-amd.log ^
    --api-enable --api-port 21555
goto :end

:generic_mining
echo 🚀 Starting generic GPU mining...
%SRBMINER_PATH% ^
    --algorithm randomx ^
    --pool localhost:3333 ^
    --wallet Z321nirFfsdGcAE8Loe1vzcZS9ztUvsTCYsYKYsncA63QqjqQyMnLiedvZSnniUsfE93Zdvu5tpkvC2qNVpDf4ot9q1UJUBMap ^
    --password generic-gpu-rig ^
    --worker gpu-generic-01 ^
    --gpu-id 0 ^
    --gpu-threads 16 ^
    --gpu-worksize 16 ^
    --gpu-intensity 20 ^
    --cpu-threads 0 ^
    --disable-cpu ^
    --log-file logs\srbminer-generic.log ^
    --api-enable --api-port 21555

:end
echo.
echo 🎯 GPU Mining started!
echo 📊 Monitor at: http://localhost:21555
echo 📈 Pool stats: http://localhost:8080/mining/stats
echo.
pause
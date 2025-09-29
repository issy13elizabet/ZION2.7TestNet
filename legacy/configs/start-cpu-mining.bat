@echo off
REM ZION CPU Mining with XMRig (Windows)
REM Usage: start-cpu-mining.bat

echo 🚀 Starting ZION CPU Mining
echo ===========================

REM Check config file
if not exist "xmrig-zion-cpu.json" (
    echo ❌ Config file xmrig-zion-cpu.json not found!
    pause
    exit /b 1
)

REM Check pool connectivity
echo 🔗 Testing pool connection...
powershell -Command "Test-NetConnection -ComputerName localhost -Port 3333 -InformationLevel Quiet" >nul 2>&1
if errorlevel 1 (
    echo ❌ Mining pool not reachable on localhost:3333
    echo 💡 Make sure ZION bootstrap stack is running:
    echo    docker-compose -p zion-bootstrap -f docker-compose-bootstrap.yml up -d
    pause
    exit /b 1
)

echo ✅ Pool connection OK
echo 💎 Starting CPU mining with config: xmrig-zion-cpu.json
echo 📊 API available on: http://localhost:16000
echo.

REM Try different XMRig locations
if exist "xmrig.exe" (
    xmrig.exe --config=xmrig-zion-cpu.json
) else if exist "xmrig\xmrig.exe" (
    xmrig\xmrig.exe --config=xmrig-zion-cpu.json
) else (
    echo 📥 XMRig not found. Using Docker fallback...
    docker run --rm -v "%cd%:/mnt" --network host xmrig/xmrig:latest --config=/mnt/xmrig-zion-cpu.json
)
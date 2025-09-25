@echo off
REM ZION Mining Startup Script - Windows
REM Spouští XMRig pro těžení ZION blockchainu na Windows

echo 🪟 ZION Mining for Windows
echo Genesis Block Hash: d763b61e4e542a6973c8f649deb228e116bcf3ee099cec92be33efe288829ae1
echo Mining Address: ajmqontZjiVUmtNjQu1RNUYq1RZgd5EDodX3qgjcaTMoMzG8EkG4bVPgLhEgudBoH82fQU1iZVw6XPfddKWAHDdA3x92ToH4uo
echo Pool: 91.98.122.165:3334 (Stratum)
echo RPC Shim (health): http://91.98.122.165:18089/metrics.json
echo.

REM Check if shim health endpoint is accessible (daemon RPC may not be exposed)
echo Checking shim health endpoint...
curl -s http://91.98.122.165:18089/metrics.json >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo ✅ Shim is accessible on 18089
) else (
    echo ⚠️ Shim is NOT accessible; mining will continue via Stratum :3333
)

echo 🔥 Starting XMRig for Windows...
cd /d "%~dp0platforms\windows\xmrig-6.21.3"
REM If config file exists, use it; otherwise fall back to direct CLI
set "WALLET=ajmqontZjiVUmtNjQu1RNUYq1RZgd5EDodX3qgjcaTMoMzG8EkG4bVPgLhEgudBoH82fQU1iZVw6XPfddKWAHDdA3x92ToH4uo"
set "RIG=MAITREYA-Ryzen-1"
if exist config-zion.json (
    echo Using config-zion.json
    start "XMRig ZION" /min xmrig.exe --config=config-zion.json -t 12
 ) else (
    echo config-zion.json not found, using CLI fallback
    start "XMRig ZION" /min xmrig.exe -o 91.98.122.165:3334 -a rx/0 -u %WALLET% -p %RIG% --rig-id %RIG% -t 12 --donate-level 1 --print-time 60 --retry-pause 5 --retries 5 --log-file xmrig-3334-public.log
 )

echo Miner launched in background (minimized). Logs: xmrig-3334-public.log

REM Do not pause; exit to leave miner running in background
exit /b 0
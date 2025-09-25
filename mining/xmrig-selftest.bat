@echo off
REM XMRig short self-test on Windows (non-intrusive)
setlocal ENABLEDELAYEDEXPANSION

set "WD=%~dp0platforms\windows\xmrig-6.21.3"
set "EXE=%WD%\xmrig.exe"
set "LOG=%WD%\xmrig-stress-%RANDOM%.log"

if not exist "%EXE%" (
  echo XMRig executable not found at %EXE%
  exit /b 1
)

echo Starting XMRig self-test (2 threads) -> %LOG%
start "XMRIG_STRESS" /min "%EXE%" --stress -a rx/0 -t 2 --print-time 5 --log-file "%LOG%"

REM Let it run briefly
timeout /t 12 /nobreak >nul 2>&1

echo --- Self-test last lines ---
powershell -NoProfile -Command "if (Test-Path '%LOG%') { Get-Content '%LOG%' -Tail 30 } else { 'Self-test log not found.' }"
echo --- end ---

REM Kill only the stress window by its title so we don't touch the main miner
taskkill /f /fi "WINDOWTITLE eq XMRIG_STRESS" >nul 2>&1

endlocal
exit /b 0

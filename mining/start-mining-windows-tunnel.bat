@echo off
REM ZION Mining Startup Script with SSH Tunnel - Windows
REM Opens a local SSH tunnel 127.0.0.1:3334 -> 91.98.122.165:3334 and starts XMRig against the tunnel

setlocal ENABLEDELAYEDEXPANSION

set POOL_HOST=91.98.122.165
set POOL_PORT=3334
set LOCAL_HOST=127.0.0.1
set LOCAL_PORT=3334
set WALLET=ajmqontZjiVUmtNjQu1RNUYq1RZgd5EDodX3qgjcaTMoMzG8EkG4bVPgLhEgudBoH82fQU1iZVw6XPfddKWAHDdA3x92ToH4uo
set RIG=MAITREYA-Ryzen-1
set LOGFILE=xmrig-3334-tunnel.log
set THREADS=10
set SSH_USER=root
REM If you have a dedicated key, set SSH_KEY to its full path (without extension if using agent); autodetect will try common defaults.
set SSH_KEY=
set LOCAL_ZION_SSH=%LOCALAPPDATA%\Zion\ssh
set TUNNEL_VISIBLE=0

REM --- SSH key autodetect (ed25519, then rsa) ---
if not defined SSH_KEY (
    if exist "%LOCAL_ZION_SSH%\id_ed25519" set SSH_KEY=%LOCAL_ZION_SSH%\id_ed25519
)
if not defined SSH_KEY (
	if exist "%USERPROFILE%\.ssh\id_ed25519" set SSH_KEY=%USERPROFILE%\.ssh\id_ed25519
)
if not defined SSH_KEY (
	if exist "%USERPROFILE%\.ssh\id_rsa" set SSH_KEY=%USERPROFILE%\.ssh\id_rsa
)

REM If no SSH key resolved, prepare to show visible window for password entry
if not defined SSH_KEY (
    set TUNNEL_VISIBLE=1
)

REM Build SSH options: prefer publickey, but allow password fallback for convenience
set SSH_OPTS=-o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -o PreferredAuthentications=publickey,password -o PasswordAuthentication=yes -o KbdInteractiveAuthentication=yes
if defined SSH_KEY (
	if exist "%SSH_KEY%" (
		set SSH_OPTS=%SSH_OPTS% -i "%SSH_KEY%"
		set TUNNEL_VISIBLE=0
	) else (
		echo ⚠️ SSH key path set but not found: %SSH_KEY%
		echo    Please verify the file exists or unset SSH_KEY to use autodetect.
		set TUNNEL_VISIBLE=1
	)
)

echo 🧵 Starting SSH tunnel %LOCAL_HOST%:%LOCAL_PORT% -> %POOL_HOST%:%POOL_PORT% ...
REM Start SSH tunnel: visible window if password entry may be needed, otherwise minimized
if "%TUNNEL_VISIBLE%"=="1" (
	start "ZION Tunnel :%LOCAL_PORT%" cmd /k "ssh %SSH_OPTS% -L %LOCAL_HOST%:%LOCAL_PORT%:127.0.0.1:%POOL_PORT% %SSH_USER%@%POOL_HOST% -N"
) else (
	start "ZION Tunnel :%LOCAL_PORT%" /min cmd /c "ssh %SSH_OPTS% -L %LOCAL_HOST%:%LOCAL_PORT%:127.0.0.1:%POOL_PORT% %SSH_USER%@%POOL_HOST% -N"
)

REM Give the tunnel a moment to initialize and verify it is running
set TUNNEL_OK=0
for /L %%i in (1,1,10) do (
	timeout /t 1 /nobreak >nul 2>&1
	for /f "tokens=*" %%p in ('tasklist /FI "IMAGENAME eq ssh.exe" ^| find /I "ssh.exe"') do set TUNNEL_OK=1
	if !TUNNEL_OK! EQU 1 goto :TUNNEL_READY
)

if !TUNNEL_OK! NEQ 1 (
	echo ❌ SSH tunnel failed to start (likely missing/unauthorized SSH key). Aborting miner start.
	echo    Tip: PowerShell -File scripts\ssh-key-setup.ps1 -Host %POOL_HOST% -User %SSH_USER%
	goto :END
)

:TUNNEL_READY

echo 🚀 Launching XMRig against %LOCAL_HOST%:%LOCAL_PORT% with %THREADS% threads ...
pushd "%~dp0platforms\windows\xmrig-6.21.3"

REM Run XMRig with explicit CLI to target the tunnel and set wallet/rig, start minimized in separate window
start "XMRig ZION Tunnel" /min xmrig.exe -o %LOCAL_HOST%:%LOCAL_PORT% -a rx/0 -u %WALLET% -p %RIG% --rig-id %RIG% --donate-level 1 --print-time 60 --retry-pause 5 --retries 5 -t %THREADS% --log-file %LOGFILE%

echo.
echo Miner launched in background. Log: %CD%\%LOGFILE%
echo If tunnel failed due to missing SSH key, create one and install it on the server.
popd

:END
endlocal

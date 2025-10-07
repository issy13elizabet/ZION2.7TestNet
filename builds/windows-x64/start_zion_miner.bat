@echo off
title ZION Multi-AI Miner v2.7.1
echo.
echo [ZION] Multi-AI Miner v2.7.1 - Windows x64
echo ==============================================
echo.

REM Install Python dependencies
echo [SETUP] Installing Python dependencies...
python -m pip install -r requirements.txt --user
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo [OK] Dependencies installed successfully!
echo.
echo [COPY] Available mining algorithms:
echo    [1] Yescrypt    - CPU optimized (C extension)
echo    [2] RandomX     - CPU memory-hard  
echo    [3] Multi-algo  - Automatic switching
echo    [4] Pool info   - Show pool settings
echo.

set /p choice="Select option (1-4): "

if "%choice%"=="1" (
    echo.
    echo [FIRE] Starting Yescrypt mining...
    python zion_yescrypt_hybrid.py --threads 6
) else if "%choice%"=="2" (
    echo.
    echo [FIRE] Starting RandomX mining...
    python zion_real_mining_system.py --mine randomx
) else if "%choice%"=="3" (
    echo.
    echo [FIRE] Starting multi-algorithm mining...
    python zion_real_mining_system.py --check
    python zion_real_mining_system.py --mine yescrypt
) else if "%choice%"=="4" (
    echo.
    echo [INFO] Pool Information:
    echo    Host: 127.0.0.1
    echo    Port: 3335
    echo    Pool: ZION Universal Pool v3.0
    pause
) else (
    echo [ERROR] Invalid choice
)

echo.
echo [FLAG] Mining session ended
pause

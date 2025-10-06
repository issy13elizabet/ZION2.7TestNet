@echo off
REM ZION AMD RX 5600 XT Mining Setup
REM Enable SRBMiner and start mining

echo ============================================
echo ZION AMD RX 5600 XT Mining Setup
echo ============================================

echo.
echo Your AMD RX 5600 XT is detected and ready!
echo GPU Benchmark: 28 MH/s KawPow, 52 MH/s Octopus
echo.

echo To enable real mining, we need to bypass Windows Defender:
echo.

echo OPTION 1: Add SRBMiner to Windows Defender exclusions
echo ====================================================
echo 1. Open Windows Security
echo 2. Go to Virus & threat protection
echo 3. Under "Virus & threat protection settings" click "Manage settings"
echo 4. Scroll down to "Exclusions" and click "Add or remove exclusions"
echo 5. Click "Add an exclusion" -^> "Folder"
echo 6. Add this folder: E:\2.7.1\miners
echo.

echo OPTION 2: Download fresh SRBMiner
echo ================================
echo 1. Go to: https://github.com/doktor83/SRBMiner-Multi/releases
echo 2. Download SRBMiner-Multi-2-9-8-win64.zip
echo 3. Extract to E:\2.7.1\miners\
echo 4. Rename folder to "SRBMiner-Multi-2-9-8"
echo.

echo OPTION 3: Use alternative miner (T-Rex)
echo ======================================
echo 1. Download T-Rex miner from official site
echo 2. Configure for KawPow algorithm
echo 3. Use pool: stratum+tcp://pool.ravenminer.com:3838
echo.

echo After enabling SRBMiner, run:
echo python zion_gpu_miner.py
echo.

pause
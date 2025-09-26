@echo off
echo ===============================================
echo ZION SRBMiner-Multi Octopus GPU Mining Test
echo ===============================================
echo.
echo 🐙 Starting Octopus mining with SRBMiner-Multi...
echo    - Algorithm: Octopus (Conflux-compatible)
echo    - GPU: AMD Radeon RX 5600 XT
echo    - Pool: localhost:3337
echo.

"D:\Zion TestNet\Zion\mining\SRBMiner-Multi-2-9-7\SRBMiner-MULTI.exe" --config "D:\Zion TestNet\Zion\mining\srb-octopus-config.json"

echo.
echo Octopus mining stopped. Press any key to exit...
pause > nul
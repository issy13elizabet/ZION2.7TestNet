@echo off
echo ===============================================
echo ZION SRBMiner-Multi KawPow GPU Mining Test
echo ===============================================
echo.
echo 🎮 Starting KawPow mining with SRBMiner-Multi...
echo    - Algorithm: KawPow (Ravencoin-compatible)
echo    - GPU: AMD Radeon RX 5600 XT
echo    - Pool: localhost:3334
echo.

"D:\Zion TestNet\Zion\mining\SRBMiner-Multi-2-9-7\SRBMiner-MULTI.exe" --config "D:\Zion TestNet\Zion\mining\srb-kawpow-config.json"

echo.
echo KawPow mining stopped. Press any key to exit...
pause > nul
@echo off
echo ===============================================
echo ZION SRBMiner-Multi Ergo GPU Mining Test
echo ===============================================
echo.
echo 💎 Starting Ergo mining with SRBMiner-Multi...
echo    - Algorithm: Autolykos2 (Ergo-compatible)
echo    - GPU: AMD Radeon RX 5600 XT  
echo    - Pool: localhost:3338
echo.

"D:\Zion TestNet\Zion\mining\SRBMiner-Multi-2-9-7\SRBMiner-MULTI.exe" --config "D:\Zion TestNet\Zion\mining\srb-ergo-config.json"

echo.
echo Ergo mining stopped. Press any key to exit...
pause > nul
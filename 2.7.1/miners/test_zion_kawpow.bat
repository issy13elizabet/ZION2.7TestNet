@echo off
cd %~dp0
echo Testing ZION KawPow GPU Mining with SRBMiner-MULTI
echo ===================================================

REM Set GPU environment variables for better performance
setx GPU_MAX_HEAP_SIZE 100
setx GPU_MAX_USE_SYNC_OBJECTS 1
setx GPU_SINGLE_ALLOC_PERCENT 100
setx GPU_MAX_ALLOC_PERCENT 100
setx GPU_MAX_SINGLE_ALLOC_PERCENT 100

REM Test KawPow algorithm (used by Ravencoin, similar to what ZION could use)
REM Using a test pool - replace with actual ZION pool when available
SRBMiner-MULTI.exe --disable-cpu --algorithm kawpow --pool stratum.ravenminer.com:3838 --wallet test_wallet_address --password x

pause
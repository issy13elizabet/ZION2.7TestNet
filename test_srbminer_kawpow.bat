@echo off
:: SRBMiner KawPow test pro ZION Universal Pool
:: Test multi-algo podpory - KawPow algoritmus

cd "e:\2.7.1\miners\SRBMiner-Multi-latest\SRBMiner-Multi-2-9-8\"

echo ========================================
echo ZION Universal Pool - KawPow Test
echo ========================================
echo Pool: 127.0.0.1:3333
echo Algorithm: KawPow
echo Address: ZION_KAWPOW_TEST_ADDRESS_1234567890AB
echo ========================================

SRBMiner-MULTI.exe ^
  --algorithm kawpow ^
  --pool 127.0.0.1:3333 ^
  --wallet ZION_KAWPOW_TEST_ADDRESS_1234567890AB ^
  --password x ^
  --gpu-intensity 20 ^
  --disable-cpu ^
  --log-file kawpow_test.log

pause
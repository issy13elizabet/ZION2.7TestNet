:: This is an example you can edit and use
:: There are numerous parameters you can set, please check Help and Examples folder

@echo off
cd %~dp0
cls

SRBMiner-MULTI.exe --algorithm phihash --pool eu.neuropool.net:10110 --wallet phicoin-wallet
pause
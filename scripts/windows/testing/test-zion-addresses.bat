@echo off
echo ===============================================
echo ZION MINING POOL - Address Validation Test
echo ===============================================
echo.
echo Testing ZION address validation...
echo.

echo ✅ Valid ZION address:
echo Z321vzLnnRu15AF82Kwx5DVZAX8rf1jZdDThTFZERRwKV23rXUqcV5u5mvT45ZDpW71kG3zEGuM5F2n3uytvAx5G9jFSY5HcFU
echo.

echo ❌ Invalid addresses (should be rejected):
echo - Short: Z123456
echo - Wrong prefix: X321vzLnnRu15AF82Kwx5DVZAX8rf1jZdDThTFZERRwKV23rXUqcV5u5mvT45ZDpW71kG3zEGuM5F2n3uytvAx5G9jFSY5HcFU  
echo - Empty: ""
echo - Bitcoin: bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh
echo.

echo 🎯 Pool Configuration:
echo - Pool Wallet: Z321vzLnnRu15AF82Kwx5DVZAX8rf1jZdDThTFZERRwKV23rXUqcV5u5mvT45ZDpW71kG3zEGuM5F2n3uytvAx5G9jFSY5HcFU
echo - Pool Fee: 2.5%%
echo - Min Payout: 0.1 ZION  
echo - Network: zion-mainnet-v2
echo.

echo Starting ZION Pool validation test...
cd "D:\Zion TestNet\Zion\zion-core"
node -e "
const pool = new (require('./dist/modules/mining-pool.js').MiningPool)();

// Test addresses
const validAddr = 'Z321vzLnnRu15AF82Kwx5DVZAX8rf1jZdDThTFZERRwKV23rXUqcV5u5mvT45ZDpW71kG3zEGuM5F2n3uytvAx5G9jFSY5HcFU';
const invalidAddrs = [
  'Z123456',
  'X321vzLnnRu15AF82Kwx5DVZAX8rf1jZdDThTFZERRwKV23rXUqcV5u5mvT45ZDpW71kG3zEGuM5F2n3uytvAx5G9jFSY5HcFU',
  '',
  'bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh'
];

console.log('✅ Valid address test:', pool.validateZionAddress ? pool.validateZionAddress(validAddr) : 'Function not available');
console.log('❌ Invalid addresses test:');
invalidAddrs.forEach((addr, i) => {
  const result = pool.validateZionAddress ? pool.validateZionAddress(addr) : false;
  console.log(\`  \${i+1}. \${addr.substring(0,20)}... → \${result}\`);
});
"

echo.
echo Address validation test completed.
pause
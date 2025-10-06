// Test real data connection from frontend
async function testRealDataConnection() {
  try {
    console.log('🔍 Testing real data connection...');
    
    const response = await fetch('http://localhost:8001/api/zion-2-7-stats', {
      method: 'GET',
      mode: 'cors',
      headers: {
        'Content-Type': 'application/json',
      }
    });
    
    const data = await response.json();
    console.log('✅ Real Data Retrieved:', data);
    
    // Test specific values
    console.log('📊 Block Height:', data.data.blockchain.block_height);
    console.log('💰 Total Supply:', data.data.blockchain.total_supply.toLocaleString());
    console.log('⭐ Consciousness Level:', data.data.blockchain.consciousness_level);
    console.log('⛏️ Mining Algorithm:', data.data.mining.algorithm);
    console.log('👛 Wallet Addresses:', data.data.wallet.addresses_count);
    
  } catch (error) {
    console.error('❌ Real data connection failed:', error);
  }
}

// Run test
testRealDataConnection();
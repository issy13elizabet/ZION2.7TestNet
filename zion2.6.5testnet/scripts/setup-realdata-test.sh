#!/bin/bash

# ZION 2.6.5 Real Data Integration Test Setup
# This script sets up environment for testing Phase 3 real data components

set -e

echo "🧪 ZION 2.6.5 Real Data Integration Test Setup"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if we're in the right directory
if [[ ! -f "core/package.json" ]] || [[ ! -f "docker-compose.yml" ]]; then
    log_error "Please run this script from the zion2.6.5testnet root directory"
    exit 1
fi

log_info "Current directory: $(pwd)"

# Step 1: Environment Setup
log_info "Step 1: Setting up environment variables..."

if [[ ! -f ".env" ]]; then
    log_info "Creating .env from .env.example..."
    cp .env.example .env
    log_success ".env file created"
else
    log_warning ".env file already exists, backing up as .env.backup"
    cp .env .env.backup
fi

# Add real data specific configurations
log_info "Configuring real data integration settings..."

cat >> .env << 'EOF'

# ==================================================
# Phase 3 Real Data Testing Configuration
# ==================================================

# Enable all real data features
EXTERNAL_DAEMON_ENABLED=true
REALDATA_API_ENABLED=true
ENHANCED_POOL_ENABLED=true
RANDOMX_VALIDATION_ENABLED=true

# Development testing settings
DEBUG_ENABLED=true
VERBOSE_BRIDGE_LOGGING=true
METRICS_LOGGING_ENABLED=true
MOCK_DATA_FALLBACK=true

# Test-specific settings
BRIDGE_TIMEOUT_MS=2000
REALDATA_SYNC_INTERVAL=3000
TEMPLATE_WALLET=Z3_TEST_POOL_ADDRESS_PLACEHOLDER

EOF

log_success "Real data configuration added to .env"

# Step 2: Install dependencies
log_info "Step 2: Installing dependencies..."

cd core
if [[ -f "package.json" ]]; then
    log_info "Installing Node.js dependencies..."
    npm install
    log_success "Dependencies installed"
else
    log_error "core/package.json not found"
    exit 1
fi
cd ..

# Step 3: TypeScript compilation check
log_info "Step 3: Checking TypeScript compilation..."

cd core
if npm run build 2>/dev/null; then
    log_success "TypeScript compilation successful"
else
    log_warning "TypeScript compilation has errors (expected due to missing type dependencies)"
    log_info "Installing additional type dependencies..."
    npm install --save-dev @types/node @types/express
    if npm run build 2>/dev/null; then
        log_success "TypeScript compilation successful after installing types"
    else
        log_warning "TypeScript compilation still has issues - will proceed with runtime testing"
    fi
fi
cd ..

# Step 4: Real Data Components Verification
log_info "Step 4: Verifying real data components..."

COMPONENTS=(
    "core/src/modules/daemon-bridge.ts"
    "core/src/modules/randomx-validator.ts" 
    "core/src/modules/real-data-manager.ts"
    "core/src/modules/enhanced-mining-pool.ts"
    "core/src/api/real-data-api.ts"
)

for component in "${COMPONENTS[@]}"; do
    if [[ -f "$component" ]]; then
        log_success "✅ $component"
    else
        log_error "❌ $component - MISSING"
        exit 1
    fi
done

log_success "All real data components verified"

# Step 5: Create test daemon mock
log_info "Step 5: Setting up test daemon mock..."

mkdir -p test-utils

cat > test-utils/mock-daemon.js << 'EOF'
#!/usr/bin/env node

// Mock ZION legacy daemon for testing real data integration
const express = require('express');
const app = express();
const port = 18081;

app.use(express.json());

let mockHeight = 12345;
let mockDifficulty = 1000000;

// Mock RPC responses
const mockResponses = {
  get_info: () => ({
    height: mockHeight,
    difficulty: mockDifficulty,
    wide_difficulty: mockDifficulty,
    tx_count: Math.floor(Math.random() * 1000000),
    tx_pool_size: Math.floor(Math.random() * 50),
    status: 'OK',
    top_block_hash: '1234567890abcdef'.repeat(4),
    timestamp: Math.floor(Date.now() / 1000) - Math.floor(Math.random() * 300)
  }),
  
  get_block_template: () => ({
    blocktemplate_blob: '0100' + mockHeight.toString(16).padStart(8, '0') + '0'.repeat(150),
    difficulty: mockDifficulty,
    height: mockHeight,
    status: 'OK',
    seed_hash: 'abcdef1234567890'.repeat(4),
    prev_hash: 'fedcba0987654321'.repeat(4)
  }),
  
  get_block: ({ height }) => ({
    blob: '0100' + height.toString(16).padStart(8, '0') + '0'.repeat(150),
    block_header: {
      height,
      hash: 'block_hash_' + height.toString(16).padStart(16, '0').repeat(2),
      timestamp: Math.floor(Date.now() / 1000) - (mockHeight - height) * 60,
      difficulty: mockDifficulty
    },
    tx_hashes: [],
    status: 'OK'
  }),
  
  get_connections: () => ({
    connections: [
      { ip: '192.168.1.100', port: 8080, state: 'active' },
      { ip: '10.0.0.50', port: 8080, state: 'active' }
    ]
  }),
  
  get_tx_pool: () => ({
    transactions: Array.from({length: Math.floor(Math.random() * 10)}, (_, i) => ({
      id_hash: 'tx_' + i.toString(16).padStart(16, '0').repeat(2),
      fee: Math.floor(Math.random() * 1000000),
      blob_size: Math.floor(Math.random() * 2000),
      receive_time: Math.floor(Date.now() / 1000) - Math.floor(Math.random() * 3600)
    }))
  }),
  
  submit_block: () => {
    mockHeight++;
    return { status: 'OK', block_hash: 'submitted_block_' + Date.now().toString(16) };
  }
};

// JSON-RPC endpoint
app.post('/json_rpc', (req, res) => {
  const { method, params } = req.body;
  
  console.log(`[mock-daemon] RPC call: ${method}`);
  
  if (mockResponses[method]) {
    const result = mockResponses[method](params || {});
    res.json({
      jsonrpc: '2.0',
      id: req.body.id,
      result
    });
  } else {
    res.status(404).json({
      jsonrpc: '2.0',
      id: req.body.id,
      error: { code: -1, message: `Method ${method} not found` }
    });
  }
});

app.listen(port, () => {
  console.log(`🔧 Mock ZION daemon running on port ${port}`);
  console.log(`   Height: ${mockHeight}, Difficulty: ${mockDifficulty}`);
  console.log('   Available RPC methods:');
  Object.keys(mockResponses).forEach(method => {
    console.log(`     - ${method}`);
  });
});
EOF

chmod +x test-utils/mock-daemon.js
log_success "Mock daemon created at test-utils/mock-daemon.js"

# Step 6: Create test runner script
log_info "Step 6: Creating test runner script..."

cat > test-utils/run-integration-test.sh << 'EOF'
#!/bin/bash

# Integration test runner for ZION 2.6.5 real data components

echo "🧪 Starting ZION 2.6.5 Real Data Integration Test"
echo "=================================================="

# Start mock daemon in background
echo "Starting mock daemon..."
node test-utils/mock-daemon.js &
DAEMON_PID=$!

# Wait for daemon to start
sleep 2

# Test daemon connectivity
echo "Testing daemon connectivity..."
curl -s -X POST http://localhost:18081/json_rpc \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"get_info","params":{}}' | jq

if [[ $? -eq 0 ]]; then
    echo "✅ Mock daemon is responding"
else
    echo "❌ Mock daemon not responding"
    kill $DAEMON_PID 2>/dev/null
    exit 1
fi

# Start ZION core with real data integration
echo "Starting ZION core..."
cd core
npm start &
CORE_PID=$!

# Wait for core to start
sleep 5

# Test real data API endpoints
echo "Testing real data API endpoints..."

ENDPOINTS=(
    "http://localhost:8602/api/realdata/bridge/health"
    "http://localhost:8602/api/realdata/bridge/status"
    "http://localhost:8602/api/realdata/blockchain/info"
    "http://localhost:8602/api/realdata/monitoring/sync-metrics"
)

for endpoint in "${ENDPOINTS[@]}"; do
    echo "Testing: $endpoint"
    if curl -s "$endpoint" | jq '.status' | grep -q "success"; then
        echo "✅ $endpoint"
    else
        echo "❌ $endpoint"
    fi
done

# Cleanup
echo "Cleaning up..."
kill $CORE_PID 2>/dev/null
kill $DAEMON_PID 2>/dev/null

echo "🎉 Integration test completed"
EOF

chmod +x test-utils/run-integration-test.sh
log_success "Integration test runner created"

# Step 7: Summary
echo ""
log_success "🎉 ZION 2.6.5 Real Data Integration Test Setup Complete!"
echo ""
echo "Next steps:"
echo "1. Start mock daemon:    node test-utils/mock-daemon.js"
echo "2. Start ZION core:      cd core && npm start"  
echo "3. Run integration test: ./test-utils/run-integration-test.sh"
echo ""
echo "Real Data API will be available at: http://localhost:8602/api/realdata/"
echo ""
echo "Key endpoints to test:"
echo "  - GET /api/realdata/bridge/health"
echo "  - GET /api/realdata/blockchain/info" 
echo "  - GET /api/realdata/mining/enhanced-stats"
echo "  - GET /api/realdata/monitoring/sync-metrics"
echo ""

log_info "Configuration files created:"
log_info "  - .env (with real data settings)"
log_info "  - test-utils/mock-daemon.js"
log_info "  - test-utils/run-integration-test.sh"
echo ""

log_warning "Note: Some TypeScript compilation warnings are expected"
log_warning "Real RandomX validation requires actual RandomX library integration"
log_info "Current setup uses placeholder validation for testing"
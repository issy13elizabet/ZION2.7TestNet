#!/bin/bash
# ZION Migration Script: 2.7 → 2.7.1
# Complete migration to clean 2.7.1 implementation

echo "🔄 ZION Migration: 2.7 → 2.7.1"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SOURCE_DIR="/Volumes/Zion/2.7"
TARGET_DIR="/Volumes/Zion/2.7.1"
BACKUP_DIR="/Volumes/Zion/backup_2.7_$(date +%Y%m%d_%H%M%S)"

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if source exists
if [ ! -d "$SOURCE_DIR" ]; then
    print_error "Source directory $SOURCE_DIR does not exist"
    exit 1
fi

# Check if target exists
if [ ! -d "$TARGET_DIR" ]; then
    print_error "Target directory $TARGET_DIR does not exist"
    exit 1
fi

print_warning "Creating backup of current 2.7 installation..."
mkdir -p "$BACKUP_DIR"

# Backup important files
cp -r "$SOURCE_DIR/core" "$BACKUP_DIR/" 2>/dev/null || true
cp -r "$SOURCE_DIR/mining" "$BACKUP_DIR/" 2>/dev/null || true
cp -r "$SOURCE_DIR/data" "$BACKUP_DIR/" 2>/dev/null || true
cp "$SOURCE_DIR/zion_integrated_cli.py" "$BACKUP_DIR/" 2>/dev/null || true

print_status "Backup created at $BACKUP_DIR"

# Migrate blockchain data if it exists
if [ -d "$SOURCE_DIR/data" ]; then
    print_warning "Migrating blockchain data..."

    # Copy optimized blockchain data
    if [ -d "$SOURCE_DIR/data/optimized" ]; then
        mkdir -p "$TARGET_DIR/data"
        cp -r "$SOURCE_DIR/data/optimized" "$TARGET_DIR/data/"
        print_status "Migrated optimized blockchain data"
    fi

    # Copy legacy blocks if no optimized data exists
    if [ ! -d "$TARGET_DIR/data/optimized" ] && [ -d "$SOURCE_DIR/data/blocks" ]; then
        mkdir -p "$TARGET_DIR/data/blocks"
        cp "$SOURCE_DIR/data/blocks/"*.json "$TARGET_DIR/data/blocks/" 2>/dev/null || true
        print_status "Migrated legacy blockchain data"
    fi
fi

# Install 2.7.1 dependencies
print_warning "Installing 2.7.1 dependencies..."
cd "$TARGET_DIR"
pip3 install -r requirements.txt --quiet

if [ $? -eq 0 ]; then
    print_status "Dependencies installed"
else
    print_error "Failed to install dependencies"
    exit 1
fi

# Run 2.7.1 tests
print_warning "Running 2.7.1 test suite..."
python3 tests/run_tests.py > /tmp/zion_migration_test.log 2>&1

if [ $? -eq 0 ]; then
    print_status "All tests passed"
else
    print_error "Some tests failed. Check /tmp/zion_migration_test.log"
    exit 1
fi

# Test basic functionality
print_warning "Testing basic functionality..."
python3 zion_cli.py info > /tmp/zion_info.log 2>&1

if [ $? -eq 0 ]; then
    print_status "Basic functionality test passed"
else
    print_error "Basic functionality test failed"
    exit 1
fi

# Show migration results
echo ""
echo "📊 Migration Results:"
echo "====================="
echo "Source: $SOURCE_DIR"
echo "Target: $TARGET_DIR"
echo "Backup: $BACKUP_DIR"
echo ""

# Show 2.7.1 info
echo "🌟 ZION 2.7.1 Status:"
echo "---------------------"
python3 zion_cli.py info

echo ""
print_status "Migration completed successfully!"
echo ""
echo "🚀 Next Steps:"
echo "=============="
echo "1. Start ZION 2.7.1: cd $TARGET_DIR && ./start.sh"
echo "2. Test mining: python3 zion_cli.py mine --address your_address"
echo "3. Benchmark algorithms: python3 zion_cli.py algorithms benchmark"
echo ""
echo "📚 Documentation: See $TARGET_DIR/README.md"
echo "🔧 CLI Help: python3 zion_cli.py --help"
echo ""
echo "⚠️  Original 2.7 installation backed up at: $BACKUP_DIR"
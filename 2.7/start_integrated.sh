#!/bin/bash
# ZION 2.7 + 2.7.1 Integrated System Startup

echo "🌟 ZION 2.7 + 2.7.1 Integrated System"
echo "======================================"

cd "$(dirname "$0")"

# Show system info
echo "📊 System Information:"
python zion_integrated_cli.py info

echo ""
echo "🔧 Available Commands:"
echo "  python zion_integrated_cli.py algorithms list      # List algorithms"
echo "  python zion_integrated_cli.py algorithms benchmark # Performance test"
echo "  python zion_integrated_cli.py test                 # Integration test"
echo "  python zion_integrated_cli.py mine --address ADDR  # Start mining"
echo ""

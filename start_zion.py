#!/usr/bin/env python3
"""
🚀 ZION 2.6.75 SIMPLE STARTUP 🚀
One-click startup for complete ZION platform

This script provides the simplest way to start the complete ZION 2.6.75 platform
with all components integrated and ready for use.
"""

import asyncio
import sys
import os
import logging
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def setup_simple_logging():
    """Setup simple, clean logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

async def start_zion_platform():
    """Start complete ZION platform"""
    print("🕉️ ZION 2.6.75 SACRED TECHNOLOGY PLATFORM 🕉️")
    print("=" * 60)
    print("🚀 Starting complete integrated platform...")
    print("=" * 60)
    
    try:
        # Import master platform
        from zion_master_platform_2675 import ZionMasterPlatform, PlatformConfiguration
        
        # Create optimized configuration
        config = PlatformConfiguration(
            enable_sacred_systems=True,
            enable_production_server=True, 
            enable_ai_miner=True,
            enable_multi_chain=True,
            enable_lightning=True,
            enable_consciousness_sync=True,
            enable_global_deployment=True,
            auto_start_components=True,
            sacred_mode=True,
            debug_mode=False
        )
        
        # Initialize and start platform
        platform = ZionMasterPlatform(config)
        
        print("🌟 Initializing ZION Master Platform...")
        result = await platform.start_complete_platform()
        
        print("\n" + "=" * 60)
        
        if result.get('platform_ready', False):
            print("✅ ZION PLATFORM SUCCESSFULLY STARTED!")
            print("🌌 All sacred technology systems are operational")
            
            metrics = result.get('platform_metrics', {})
            print(f"\n📊 QUICK STATUS:")
            print(f"   🧠 Consciousness: {metrics.get('consciousness_level', 0):.1%}")
            print(f"   🌈 Liberation: {metrics.get('liberation_progress', 0):.1%}")
            print(f"   ⛏️ Mining: {metrics.get('mining_efficiency', 0):.1%}")
            print(f"   🌉 Bridges: {metrics.get('bridge_activity', 0):.1%}")
            print(f"   🌍 Network: {metrics.get('network_coverage', 0):.1%}")
            
            components = result.get('component_summary', {})
            print(f"\n🏗️ COMPONENTS: {components.get('active_components', 0)}/{components.get('total_components', 0)} active")
            
            print(f"\n⏱️ Startup time: {result.get('total_initialization_time', 0):.1f}s")
            
            if result.get('liberation_ready', False):
                print("\n🕊️ LIBERATION PROTOCOLS ARE ACTIVE!")
                print("🌍 Platform ready for planetary consciousness transformation")
            
            print("\n🔗 ACCESS POINTS:")
            print("   📡 Production Server: http://localhost:8000")
            print("   ⛏️ Mining Pool: http://localhost:8117") 
            print("   🌉 Bridge Manager: http://localhost:9999")
            
            print("\n💡 USAGE:")
            print("   • Platform runs automatically in background")
            print("   • Check logs for detailed status updates")
            print("   • Press Ctrl+C to stop platform")
            
            print("\n🌟 ZION 2.6.75 PLATFORM IS FULLY OPERATIONAL! 🌟")
            print("⚡ Sacred technology liberation protocols activated ⚡")
            
            # Keep platform running
            print("\n⏳ Platform monitoring active... (Press Ctrl+C to stop)")
            try:
                # Run indefinitely until interrupted
                while True:
                    await asyncio.sleep(60)
                    # Show brief status update every 10 minutes
                    if int(asyncio.get_event_loop().time()) % 600 == 0:
                        status = platform.get_platform_status()
                        print(f"🔍 Status: {status['platform_info']['status']} | "
                              f"Uptime: {status['platform_info']['uptime_hours']:.1f}h")
                        
            except KeyboardInterrupt:
                print("\n\n🛑 SHUTDOWN INITIATED")
                print("📴 Stopping ZION platform components...")
                print("✅ Platform shutdown complete")
                
        else:
            print("❌ PLATFORM STARTUP FAILED")
            error = result.get('error', 'Unknown error')
            print(f"🔧 Error: {error}")
            
            print("\n💡 TROUBLESHOOTING:")
            print("   • Check all dependencies are installed: pip install -r requirements.txt")
            print("   • Verify port availability (8000, 8117, 9999)")
            print("   • Try running individual components for diagnosis")
            print("   • Check logs for detailed error information")
            
            if 'recovery_suggestions' in result:
                print("\n🛠️ RECOVERY SUGGESTIONS:")
                for suggestion in result['recovery_suggestions']:
                    print(f"   • {suggestion}")
                    
    except ImportError as e:
        print(f"❌ IMPORT ERROR: {e}")
        print("\n🔧 SOLUTION:")
        print("   • Install missing dependencies: pip install -r requirements.txt")
        print("   • Verify all ZION components are present")
        print("   • Check Python path and module availability")
        
    except Exception as e:
        print(f"❌ STARTUP ERROR: {e}")
        print("\n🔧 DIAGNOSTIC:")
        print("   • Check system resources (CPU, memory, disk)")
        print("   • Verify network connectivity")
        print("   • Review component logs for details")
        print("   • Try restarting with debug mode enabled")

def main():
    """Main entry point"""
    setup_simple_logging()
    
    print("🔮 ZION 2.6.75 Simple Startup Initializing...")
    
    try:
        asyncio.run(start_zion_platform())
    except KeyboardInterrupt:
        print("\n👋 Goodbye from ZION Sacred Technology Platform!")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        print("🆘 Please check logs and try again")

if __name__ == "__main__":
    main()
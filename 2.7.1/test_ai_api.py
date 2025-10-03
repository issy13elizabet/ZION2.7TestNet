#!/usr/bin/env python3
"""
ZION AI API Test Script
Test AI orchestrator integration with FastAPI
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_ai_api():
    """Test AI API endpoints"""
    print("🧪 Testing ZION AI API Integration")
    print("=" * 50)

    try:
        from api import app, AI_AVAILABLE, ai_orchestrator
        print("✅ API loaded successfully")
        print(f"📊 Total routes: {len(app.routes)}")

        # Test AI availability
        print(f"🤖 AI Available: {AI_AVAILABLE}")

        if AI_AVAILABLE and ai_orchestrator:
            # Test orchestrator status
            status = ai_orchestrator.get_status()
            print(f"📊 AI Components loaded: {status['total_components']}")

            # Test sacred mining
            mining_result = ai_orchestrator.perform_sacred_mining()
            print(f"⛏️ Sacred Mining: AI boost {mining_result['ai_boost']:.2f}x")

            # Test unified analysis
            analysis = ai_orchestrator.perform_unified_ai_analysis()
            print(f"🔍 AI Analysis: Consensus {analysis['consensus_score']:.2f}")

            # Test resource monitoring
            resources = ai_orchestrator.get_resource_usage()
            print(f"⚙️ CPU Usage: {resources.get('cpu_usage', 0):.1f}%")

        # Test AI endpoints manually (without TestClient)
        print("\n🌐 Testing AI endpoint availability...")
        print("   (Note: Full HTTP testing requires 'pip install httpx')")

        # Check that endpoints are registered
        ai_routes = [route for route in app.routes if hasattr(route, 'path') and 'ai' in route.path]
        print(f"   🤖 AI routes registered: {len(ai_routes)}")
        for route in ai_routes:
            print(f"      - {route.path}")

        health_route = [route for route in app.routes if hasattr(route, 'path') and route.path == '/health']
        print(f"   🏥 Health route registered: {len(health_route) > 0}")

        print("\n✅ All tests completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ai_api()
    sys.exit(0 if success else 1)
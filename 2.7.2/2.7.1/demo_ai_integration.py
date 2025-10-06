#!/usr/bin/env python3
"""
ZION AI API Demo
Demonstration of AI orchestrator integration with blockchain API
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def demo_ai_integration():
    """Demonstrate AI-blockchain integration"""
    print("🚀 ZION AI-Blockchain Integration Demo")
    print("=" * 50)
    print("JAI RAM SITA HANUMAN - ON THE STAR 🌟")
    print()

    try:
        # Load API and AI orchestrator
        from api import app, AI_AVAILABLE, ai_orchestrator
        print("✅ API and AI Orchestrator loaded")

        if not AI_AVAILABLE or ai_orchestrator is None:
            print("❌ AI not available")
            return False

        # Show AI status
        status = ai_orchestrator.get_status()
        print("🤖 AI System Status:")
        print(f"   Components: {status['total_components']}")
        print(f"   Active: {status['active_components']}")
        print(f"   Component List: {', '.join(status['components'])}")
        print()

        # Demonstrate sacred mining
        print("⛏️ Sacred Mining with AI Support:")
        mining_result = ai_orchestrator.perform_sacred_mining({
            'block_hash': 'demo_block_' + str(hash('demo'))[:8],
            'mining_power': 100.0,
            'difficulty': 10000
        })
        print(f"   Block Hash: {mining_result['block_hash'][:16]}...")
        print(f"   Mining Power: {mining_result['mining_power']:.1f}")
        print(f"   AI Boost: {mining_result['ai_boost']:.2f}x")
        print(f"   AI Contribution: {mining_result['ai_contribution']:.1f}%")
        print(f"   Divine Validation: {'✅' if mining_result['divine_validation'] else '❌'}")
        print(f"   AI Components Used: {len(mining_result['ai_components_used'])}")
        print()

        # Demonstrate unified AI analysis
        print("🔍 Unified AI Analysis:")
        analysis = ai_orchestrator.perform_unified_ai_analysis({
            'market_data': {'price': 100, 'volume': 1000},
            'blockchain_metrics': {'hashrate': 500, 'difficulty': 5000}
        })
        print(f"   Consensus Score: {analysis['consensus_score']:.2f}")
        print(f"   Divine Validation: {'✅' if analysis['divine_validation'] else '❌'}")
        print(f"   Analyses Performed: {analysis['total_analyses']}")
        if analysis['analyses']:
            print("   Analysis Results:")
            for key, result in analysis['analyses'].items():
                if isinstance(result, dict) and 'confidence' in result:
                    print(f"     {key}: {result.get('confidence', 0):.2f} confidence")
        print()

        # Demonstrate resource management
        print("⚙️ Resource Management:")
        resources = ai_orchestrator.get_resource_usage()
        optimization = ai_orchestrator.optimize_resources()
        print(f"   CPU Usage: {resources.get('cpu_usage', 0):.1f}%")
        print(f"   Memory Usage: {resources.get('memory_usage', 0):.1f}%")
        print(f"   Active Components: {optimization.get('active_components_count', 0)}")
        if optimization.get('optimizations'):
            print("   Recommendations:")
            for rec in optimization['optimizations']:
                print(f"     - {rec}")
        else:
            print("   ✅ Resources optimized")
        print()

        # Show API endpoints
        print("🌐 Available API Endpoints:")
        ai_endpoints = [
            "/ai/status - Get AI orchestrator status",
            "/ai/sacred-mining - Perform sacred mining with AI",
            "/ai/analysis - Get unified AI analysis",
            "/ai/resources - Get resource usage and optimization",
            "/health - System health check (includes AI status)"
        ]
        for endpoint in ai_endpoints:
            print(f"   {endpoint}")
        print()

        print("✅ AI-Blockchain Integration Demo Completed!")
        print("🚀 Ready for production deployment")
        print()
        print("To start the API server:")
        print("  python run_api.py")
        print()
        print("API will be available at: http://localhost:8001")
        print("Swagger docs at: http://localhost:8001/docs")

        return True

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = demo_ai_integration()
    sys.exit(0 if success else 1)
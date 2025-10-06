#!/usr/bin/env python3
"""
ZION 2.7.1 Humanitarian Distribution System Demo
Demonstrates 10% automatic distribution to humanitarian projects
"""

import asyncio
import sys
import os
from decimal import Decimal

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from mining.humanitarian_distribution import get_humanitarian_distributor


async def demo_humanitarian_distribution():
    """Demo humanitarian distribution system"""
    print("🌟 ZION 2.7.1 Humanitarian Distribution System Demo 🌟")
    print("=" * 70)
    
    # Get distributor instance
    distributor = get_humanitarian_distributor()
    
    # Show current configuration
    print("\n📋 Current Configuration:")
    distributor.print_configuration()
    
    # Simulate mining rewards
    print("\n🎯 Simulation: Mining Rewards Distribution")
    print("-" * 50)
    
    test_rewards = [
        Decimal('100'),    # Small reward
        Decimal('500'),    # Medium reward  
        Decimal('1000'),   # Large reward
        Decimal('2500'),   # Very large reward
    ]
    
    for i, reward in enumerate(test_rewards, 1):
        print(f"\n🔸 Block {i}: Mining Reward = {reward} ZION")
        
        # Calculate distribution
        distribution = distributor.calculate_distribution(reward)
        
        # Show breakdown
        humanitarian_total = reward * distributor.humanitarian_percentage
        miner_reward = reward - humanitarian_total
        
        print(f"  👤 Miner receives: {miner_reward} ZION ({90}%)")
        print(f"  🌍 Humanitarian fund: {humanitarian_total} ZION ({10}%)")
        print("  📊 Project distributions:")
        
        for project_id, amount in distribution.items():
            project = next(p for p in distributor.projects if p.id == project_id)
            percentage_of_total = (amount / reward) * 100
            print(f"    • {project.name}: {amount} ZION ({percentage_of_total:.1f}%)")
        
        # Process actual distribution
        print("  🚀 Processing distribution...")
        report = await distributor.distribute_rewards(reward, 1000 + i)
        print(f"  ✅ Distribution complete: {report['total_distributed']} ZION distributed")
    
    # Show final statistics
    print("\n📈 Final Project Statistics:")
    print("-" * 50)
    stats = distributor.get_project_statistics()
    
    for project_id, project_stats in stats.items():
        if project_stats['active']:
            print(f"\n🏛️  {project_stats['name']}")
            print(f"   💰 Total received: {project_stats['total_received']} ZION")
            print(f"   📅 Last payout: {project_stats['last_payout'] or 'Never'}")
            print(f"   🎯 Allocation: {project_stats['percentage']}% of humanitarian fund")
    
    # Calculate totals
    total_distributed = sum(Decimal(stats[pid]['total_received']) for pid in stats if stats[pid]['active'])
    total_mined = sum(test_rewards)
    
    print(f"\n📊 Summary:")
    print(f"   🎯 Total mined: {total_mined} ZION")
    print(f"   🌍 Total to humanitarian projects: {total_distributed} ZION")
    print(f"   👤 Total to miners: {total_mined - total_distributed} ZION")
    print(f"   📈 Humanitarian percentage: {(total_distributed / total_mined) * 100:.1f}%")
    
    # Environmental impact calculation
    print(f"\n🌱 Estimated Environmental Impact:")
    print(f"   🌲 Trees potentially planted: {int(total_distributed * Decimal('10'))} trees")
    print(f"   🌊 Ocean plastic removed: {float(total_distributed * Decimal('0.5')):.1f} kg")
    print(f"   ❤️  People helped: {int(total_distributed * Decimal('50'))} individuals")
    print(f"   🚀 Space research funded: ${float(total_distributed * Decimal('100')):,.2f}")
    print(f"   🕉️  Sacred spaces developed: {float(total_distributed * Decimal('0.1')):.1f} m²")


async def demo_project_management():
    """Demo project management features"""
    print("\n\n🔧 Project Management Demo")
    print("=" * 40)
    
    distributor = get_humanitarian_distributor()
    
    # Test project updates
    print("\n📝 Testing project updates...")
    
    # Temporarily disable a project
    success = distributor.update_project('space_program', active=False)
    print(f"Disabled space program: {'✅' if success else '❌'}")
    
    # Update project percentage
    success = distributor.update_project('forest_restoration', percentage=25.0)
    print(f"Updated forest restoration to 25%: {'✅' if success else '❌'}")
    
    # Show updated distribution
    print("\n📊 Updated distribution (1000 ZION reward):")
    distribution = distributor.calculate_distribution(Decimal('1000'))
    
    for project_id, amount in distribution.items():
        project = next((p for p in distributor.projects if p.id == project_id), None)
        if project:
            status = "✅ Active" if project.active else "❌ Inactive"
            print(f"  • {project.name}: {amount} ZION ({status})")
    
    # Restore original settings
    distributor.update_project('space_program', active=True)
    distributor.update_project('forest_restoration', percentage=20.0)
    print("\n🔄 Restored original settings")


def demo_wallets():
    """Display humanitarian wallet addresses"""
    print("\n\n💳 Humanitarian Project Wallets")
    print("=" * 50)
    
    distributor = get_humanitarian_distributor()
    
    for project in distributor.projects:
        if project.active:
            print(f"\n🏛️  {project.name}")
            print(f"   📧 Wallet: {project.wallet_address}")
            print(f"   📝 {project.description}")
            
    print("\n💡 Note: These are demo wallet addresses.")
    print("   In production, these would be real wallet addresses")
    print("   controlled by the respective humanitarian organizations.")


async def main():
    """Main demo function"""
    try:
        await demo_humanitarian_distribution()
        await demo_project_management()
        demo_wallets()
        
        print("\n\n🎉 Demo Complete!")
        print("=" * 30)
        print("The ZION 2.7.1 humanitarian distribution system")
        print("automatically distributes 10% of all mining rewards")
        print("to 5 global humanitarian and environmental projects.")
        print("\nThis creates a positive impact mining ecosystem")
        print("where every block mined helps make the world better! 🌍✨")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
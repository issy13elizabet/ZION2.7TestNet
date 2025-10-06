#!/usr/bin/env python3
"""
ZION 2.7.1 Humanitarian Distribution System
Automatic 10% distribution of mining rewards to humanitarian projects
"""

import json
import logging
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import os


@dataclass
class HumanitarianProject:
    """Humanitarian project configuration"""
    id: str
    name: str
    description: str
    wallet_address: str
    percentage: float  # Percentage of the 10% humanitarian fund
    active: bool = True
    total_received: Decimal = Decimal('0')
    last_payout: Optional[datetime] = None
    
    def __post_init__(self):
        if isinstance(self.total_received, (int, float, str)):
            self.total_received = Decimal(str(self.total_received))


class HumanitarianDistributor:
    """
    Manages automatic distribution of 10% mining rewards to humanitarian projects
    """
    
    def __init__(self, config_file: str = "humanitarian_config.json"):
        self.config_file = config_file
        self.humanitarian_percentage = Decimal('0.10')  # 10% of all rewards
        self.logger = self._setup_logging()
        self.projects = self._load_projects()
        
        # Verify percentages add up to 100%
        self._validate_project_percentages()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for humanitarian distribution"""
        logger = logging.getLogger('humanitarian_distribution')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s [HUMANITARIAN] %(levelname)s: %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_projects(self) -> List[HumanitarianProject]:
        """Load humanitarian projects configuration"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                projects = []
                for project_data in data.get('projects', []):
                    project = HumanitarianProject(**project_data)
                    projects.append(project)
                
                return projects
            except Exception as e:
                self.logger.error(f"Failed to load humanitarian config: {e}")
        
        # Return default projects if config doesn't exist
        return self._create_default_projects()
    
    def _create_default_projects(self) -> List[HumanitarianProject]:
        """Create default humanitarian projects"""
        default_projects = [
            HumanitarianProject(
                id="forest_restoration",
                name="🌲 Zalesňování pralesů",
                description="Obnova tropických pralesů a ochrana biodiverzity",
                wallet_address="ZION1ForestRestoration2024HumanitarianProject",
                percentage=20.0  # 20% of humanitarian fund (2% of total)
            ),
            HumanitarianProject(
                id="ocean_cleanup",
                name="🌊 Vyčištění oceánů",
                description="Odstranění plastů z oceánů a ochrana mořského života",
                wallet_address="ZION1OceanCleanup2024EnvironmentalProtection",
                percentage=20.0  # 20% of humanitarian fund (2% of total)
            ),
            HumanitarianProject(
                id="humanitarian_aid",
                name="❤️ Humanitární pomoc",
                description="Pomoc potřebným komunitám po celém světě",
                wallet_address="ZION1HumanitarianAid2024GlobalCommunitySupport",
                percentage=20.0  # 20% of humanitarian fund (2% of total)
            ),
            HumanitarianProject(
                id="space_program",
                name="🚀 Space program",
                description="Výzkum vesmíru a technologický rozvoj pro lidstvo",
                wallet_address="ZION1SpaceProgram2024CosmicExplorationFund",
                percentage=20.0  # 20% of humanitarian fund (2% of total)
            ),
            HumanitarianProject(
                id="dharma_development",
                name="🕉️ Dharma vývoj",
                description="Zahrada v Portugalsku s Triple pyramid a La Palma Dharma Temple se stromem Bodhi",
                wallet_address="ZION1DharmaDevelopment2024SacredGardenPortugal",
                percentage=20.0  # 20% of humanitarian fund (2% of total)
            )
        ]
        
        # Save default configuration
        self._save_config()
        return default_projects
    
    def _validate_project_percentages(self):
        """Validate that project percentages add up to 100%"""
        total_percentage = sum(project.percentage for project in self.projects if project.active)
        
        if abs(total_percentage - 100.0) > 0.01:  # Allow small floating point errors
            self.logger.warning(
                f"Project percentages sum to {total_percentage}%, not 100%. "
                "This may cause distribution issues."
            )
    
    def calculate_distribution(self, mining_reward: Decimal) -> Dict[str, Decimal]:
        """
        Calculate humanitarian distribution from mining reward
        
        Args:
            mining_reward: Total mining reward amount
            
        Returns:
            Dictionary mapping project IDs to distribution amounts
        """
        # Calculate total humanitarian fund (10% of mining reward)
        humanitarian_fund = mining_reward * self.humanitarian_percentage
        
        # Distribute among active projects
        distribution = {}
        active_projects = [p for p in self.projects if p.active]
        
        for project in active_projects:
            project_amount = humanitarian_fund * Decimal(str(project.percentage / 100.0))
            # Round down to avoid over-distribution due to floating point precision
            project_amount = project_amount.quantize(Decimal('0.00000001'), rounding=ROUND_DOWN)
            distribution[project.id] = project_amount
        
        return distribution
    
    async def distribute_rewards(self, mining_reward: Decimal, block_height: int) -> Dict[str, Dict]:
        """
        Distribute mining rewards to humanitarian projects
        
        Args:
            mining_reward: Total mining reward
            block_height: Block height for logging
            
        Returns:
            Distribution report
        """
        distribution = self.calculate_distribution(mining_reward)
        total_distributed = Decimal('0')
        distribution_report = {
            'block_height': block_height,
            'total_reward': str(mining_reward),
            'humanitarian_fund': str(mining_reward * self.humanitarian_percentage),
            'distributions': {},
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Process distributions
        for project in self.projects:
            if not project.active or project.id not in distribution:
                continue
            
            amount = distribution[project.id]
            if amount > 0:
                # Update project statistics
                project.total_received += amount
                project.last_payout = datetime.now(timezone.utc)
                total_distributed += amount
                
                # Log distribution
                self.logger.info(
                    f"💰 Distributed {amount} ZION to {project.name} "
                    f"(Block {block_height})"
                )
                
                # Add to report
                distribution_report['distributions'][project.id] = {
                    'name': project.name,
                    'amount': str(amount),
                    'wallet_address': project.wallet_address,
                    'percentage': project.percentage,
                    'total_received': str(project.total_received)
                }
        
        # Save updated statistics
        self._save_config()
        
        # Log summary
        self.logger.info(
            f"🌟 Humanitarian distribution complete: {total_distributed} ZION "
            f"distributed to {len(distribution)} projects (Block {block_height})"
        )
        
        distribution_report['total_distributed'] = str(total_distributed)
        return distribution_report
    
    def get_project_statistics(self) -> Dict[str, Dict]:
        """Get statistics for all projects"""
        stats = {}
        for project in self.projects:
            stats[project.id] = {
                'name': project.name,
                'description': project.description,
                'wallet_address': project.wallet_address,
                'percentage': project.percentage,
                'active': project.active,
                'total_received': str(project.total_received),
                'last_payout': project.last_payout.isoformat() if project.last_payout else None
            }
        return stats
    
    def update_project(self, project_id: str, **kwargs) -> bool:
        """Update project configuration"""
        for project in self.projects:
            if project.id == project_id:
                for key, value in kwargs.items():
                    if hasattr(project, key):
                        setattr(project, key, value)
                
                self._save_config()
                self.logger.info(f"Updated project {project_id}: {kwargs}")
                return True
        
        self.logger.error(f"Project {project_id} not found")
        return False
    
    def add_project(self, project: HumanitarianProject) -> bool:
        """Add new humanitarian project"""
        # Check if project ID already exists
        if any(p.id == project.id for p in self.projects):
            self.logger.error(f"Project with ID {project.id} already exists")
            return False
        
        self.projects.append(project)
        self._save_config()
        self.logger.info(f"Added new project: {project.name}")
        return True
    
    def remove_project(self, project_id: str) -> bool:
        """Remove humanitarian project"""
        for i, project in enumerate(self.projects):
            if project.id == project_id:
                removed_project = self.projects.pop(i)
                self._save_config()
                self.logger.info(f"Removed project: {removed_project.name}")
                return True
        
        self.logger.error(f"Project {project_id} not found")
        return False
    
    def _save_config(self):
        """Save configuration to file"""
        try:
            config_data = {
                'humanitarian_percentage': float(self.humanitarian_percentage),
                'projects': []
            }
            
            for project in self.projects:
                project_data = asdict(project)
                # Convert Decimal to string for JSON serialization
                project_data['total_received'] = str(project.total_received)
                # Convert datetime to ISO string
                if project.last_payout:
                    project_data['last_payout'] = project.last_payout.isoformat()
                config_data['projects'].append(project_data)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Saved humanitarian configuration to {self.config_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to save humanitarian config: {e}")
    
    def print_configuration(self):
        """Print current humanitarian configuration"""
        print("🌟 ZION Humanitarian Distribution Configuration 🌟")
        print("=" * 60)
        print(f"Total humanitarian percentage: {self.humanitarian_percentage * 100}%")
        print("\nActive Projects:")
        
        for project in self.projects:
            if project.active:
                status = "✅ Active"
                total_pct = float(self.humanitarian_percentage) * project.percentage / 100.0 * 100
                print(f"\n  {project.name}")
                print(f"    Description: {project.description}")
                print(f"    Percentage: {project.percentage}% of humanitarian fund ({total_pct:.1f}% of total)")
                print(f"    Wallet: {project.wallet_address}")
                print(f"    Total received: {project.total_received} ZION")
                print(f"    Status: {status}")
        
        print("\nInactive Projects:")
        for project in self.projects:
            if not project.active:
                print(f"  ❌ {project.name} (Inactive)")
        
        print("\n" + "=" * 60)


# Global instance
_humanitarian_distributor = None

def get_humanitarian_distributor() -> HumanitarianDistributor:
    """Get global humanitarian distributor instance"""
    global _humanitarian_distributor
    if _humanitarian_distributor is None:
        _humanitarian_distributor = HumanitarianDistributor()
    return _humanitarian_distributor


# CLI interface
if __name__ == "__main__":
    import asyncio
    
    async def main():
        distributor = get_humanitarian_distributor()
        
        print("ZION 2.7.1 Humanitarian Distribution System")
        print("=" * 50)
        
        # Show configuration
        distributor.print_configuration()
        
        # Example distribution
        print("\n🧪 Example Distribution (1000 ZION reward):")
        reward = Decimal('1000')
        distribution = distributor.calculate_distribution(reward)
        
        for project_id, amount in distribution.items():
            project = next(p for p in distributor.projects if p.id == project_id)
            print(f"  {project.name}: {amount} ZION")
        
        print(f"\nTotal humanitarian distribution: {sum(distribution.values())} ZION")
        print(f"Remaining for miner: {reward - sum(distribution.values())} ZION")
    
    asyncio.run(main())
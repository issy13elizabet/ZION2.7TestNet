#!/usr/bin/env python3
"""
🌟 ZION Sacred Mining Git Integration 🌟

Automatické Git commit logování mining sessions, hashrate statistiky
a sacred bonusy pro kompletní sledování výkonu ZION Sacred Mining Protocol

Features:
- Automatické commit mining milestones  
- Session statistics tracking
- Consciousness points logging
- Sacred geometry bonus calculations
- Multi-miner support (CPU, GPU, ZION)
- Blockchain interaction logging
"""

import os
import sys
import git
import json
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import logging

@dataclass
class MiningSessionLog:
    """Log struktura pro mining session"""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    
    # Miner information
    miner_type: str = "unknown"
    worker_name: str = "unknown"
    wallet_address: str = ""
    
    # Performance statistics
    total_shares: int = 0
    accepted_shares: int = 0
    rejected_shares: int = 0
    blocks_found: int = 0
    average_hashrate: float = 0.0
    peak_hashrate: float = 0.0
    
    # Sacred metrics
    consciousness_points: float = 0.0
    sacred_level: str = "initiate"
    sacred_multiplier: float = 1.0
    quantum_coherence: float = 0.0
    
    # Session metrics
    uptime_hours: float = 0.0
    reconnect_count: int = 0
    difficulty_adjustments: int = 0
    
    def get_duration_hours(self) -> float:
        """Vrací délku session v hodinách"""
        end = self.end_time or time.time()
        return (end - self.start_time) / 3600.0
    
    def get_shares_per_hour(self) -> float:
        """Vrací průměr shares za hodinu"""
        duration = self.get_duration_hours()
        return self.total_shares / duration if duration > 0 else 0.0
    
    def get_efficiency_percentage(self) -> float:
        """Vrací efektivnost v procentech"""
        if self.total_shares == 0:
            return 0.0
        return (self.accepted_shares / self.total_shares) * 100.0

class ZionMiningGitIntegration:
    """
    🌟 ZION Sacred Mining Git Integration System
    
    Automaticky commituje mining statistiky, milestones a sacred bonusy
    do Git repository pro kompletní sledování mining výkonu.
    """
    
    def __init__(self, repo_path: str, branch: str = "main"):
        self.repo_path = repo_path
        self.branch = branch
        self.logger = logging.getLogger("ZionMiningGit")
        
        # Git repository setup
        try:
            self.repo = git.Repo(repo_path)
            self.git_client = self.repo.git
            
            # Create mining logs directory if not exists
            self.logs_dir = os.path.join(repo_path, "mining_logs")
            os.makedirs(self.logs_dir, exist_ok=True)
            
            # Create sacred statistics directory
            self.sacred_stats_dir = os.path.join(repo_path, "sacred_stats") 
            os.makedirs(self.sacred_stats_dir, exist_ok=True)
            
            self.logger.info(f"🌟 ZION Mining Git Integration initialized: {repo_path}")
            
        except git.InvalidGitRepositoryError:
            self.logger.error(f"Invalid Git repository: {repo_path}")
            raise
        except Exception as e:
            self.logger.error(f"Git initialization error: {e}")
            raise
    
    def commit_mining_session_start(self, session_log: MiningSessionLog) -> bool:
        """
        Commitne začátek mining session
        """
        try:
            # Create session log file
            session_file = os.path.join(
                self.logs_dir, 
                f"session_{session_log.session_id}_{int(session_log.start_time)}.json"
            )
            
            with open(session_file, 'w') as f:
                json.dump(asdict(session_log), f, indent=2)
            
            # Git commit
            timestamp = datetime.fromtimestamp(session_log.start_time, timezone.utc)
            commit_message = f"""🌟 ZION Sacred Mining Session Started

Session ID: {session_log.session_id}
Miner Type: {session_log.miner_type}
Worker: {session_log.worker_name}
Sacred Level: {session_log.sacred_level}
Sacred Multiplier: {session_log.sacred_multiplier}x
Started: {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}

Sacred Mining Protocol - Session Initialization Complete ✨"""

            self.git_client.add(session_file)
            self.git_client.commit("-m", commit_message)
            
            self.logger.info(f"✅ Mining session start committed: {session_log.session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to commit session start: {e}")
            return False
    
    def commit_mining_milestone(self, session_log: MiningSessionLog, milestone_type: str, 
                               milestone_value: int) -> bool:
        """
        Commitne mining milestone (např. 100 shares, block found, atd.)
        """
        try:
            # Update session log file
            session_file = os.path.join(
                self.logs_dir, 
                f"session_{session_log.session_id}_{int(session_log.start_time)}.json"
            )
            
            with open(session_file, 'w') as f:
                json.dump(asdict(session_log), f, indent=2)
            
            # Create milestone log
            milestone_file = os.path.join(
                self.logs_dir,
                f"milestone_{session_log.session_id}_{milestone_type}_{milestone_value}.json"
            )
            
            milestone_data = {
                "session_id": session_log.session_id,
                "milestone_type": milestone_type,
                "milestone_value": milestone_value,
                "timestamp": time.time(),
                "session_stats": asdict(session_log),
                "sacred_bonus_earned": self._calculate_milestone_bonus(milestone_type, milestone_value),
                "consciousness_level": session_log.consciousness_points
            }
            
            with open(milestone_file, 'w') as f:
                json.dump(milestone_data, f, indent=2)
            
            # Git commit with detailed message
            timestamp = datetime.now(timezone.utc)
            commit_message = f"""🎯 ZION Sacred Mining Milestone Achieved!

Milestone: {milestone_type.upper()} - {milestone_value}
Session: {session_log.session_id}
Worker: {session_log.worker_name} ({session_log.miner_type})

📊 Session Statistics:
├─ Total Shares: {session_log.total_shares}
├─ Accepted: {session_log.accepted_shares}
├─ Efficiency: {session_log.get_efficiency_percentage():.1f}%
├─ Avg Hashrate: {session_log.average_hashrate:.2f} H/s

🧠 Sacred Metrics:
├─ Consciousness Points: {session_log.consciousness_points:.2f}
├─ Sacred Level: {session_log.sacred_level}
├─ Sacred Multiplier: {session_log.sacred_multiplier}x
├─ Quantum Coherence: {session_log.quantum_coherence:.3f}

🕐 Session Info:
├─ Uptime: {session_log.get_duration_hours():.1f} hours
├─ Shares/Hour: {session_log.get_shares_per_hour():.1f}
├─ Reconnects: {session_log.reconnect_count}

Achieved: {timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
Sacred Mining Protocol - Milestone Blessed ⭐"""

            self.git_client.add([session_file, milestone_file])
            self.git_client.commit("-m", commit_message)
            
            self.logger.info(f"🎯 Milestone committed: {milestone_type} - {milestone_value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to commit milestone: {e}")
            return False
    
    def commit_session_summary(self, session_log: MiningSessionLog) -> bool:
        """
        Commitne finální session summary na konci mining session
        """
        try:
            session_log.end_time = time.time()
            
            # Final session log update
            session_file = os.path.join(
                self.logs_dir, 
                f"session_{session_log.session_id}_{int(session_log.start_time)}.json"
            )
            
            with open(session_file, 'w') as f:
                json.dump(asdict(session_log), f, indent=2)
            
            # Create session summary
            summary_file = os.path.join(
                self.sacred_stats_dir,
                f"summary_{session_log.session_id}_{int(session_log.end_time)}.json"
            )
            
            summary_data = {
                "session_summary": asdict(session_log),
                "performance_analysis": self._analyze_session_performance(session_log),
                "sacred_achievements": self._analyze_sacred_achievements(session_log),
                "recommendations": self._generate_recommendations(session_log)
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
            
            # Final commit message
            duration = session_log.get_duration_hours()
            efficiency = session_log.get_efficiency_percentage()
            
            commit_message = f"""🏁 ZION Sacred Mining Session Complete

═══════════════════════════════════════════════════════════════════
🌟 SESSION SUMMARY: {session_log.session_id} 🌟
═══════════════════════════════════════════════════════════════════

👤 Miner Information:
├─ Worker: {session_log.worker_name}
├─ Type: {session_log.miner_type}
├─ Wallet: {session_log.wallet_address[:15]}...

📈 Performance Results:
├─ Duration: {duration:.2f} hours
├─ Total Shares: {session_log.total_shares}
├─ Accepted: {session_log.accepted_shares} ({efficiency:.1f}%)
├─ Rejected: {session_log.rejected_shares}
├─ Blocks Found: {session_log.blocks_found} 🎉
├─ Average Hashrate: {session_log.average_hashrate:.2f} H/s
├─ Peak Hashrate: {session_log.peak_hashrate:.2f} H/s
├─ Shares/Hour: {session_log.get_shares_per_hour():.1f}

🧠 Sacred Achievements:
├─ Consciousness Points: {session_log.consciousness_points:.2f}
├─ Sacred Level: {session_log.sacred_level.upper()}
├─ Sacred Multiplier: {session_log.sacred_multiplier}x
├─ Quantum Coherence: {session_log.quantum_coherence:.3f}

🔧 Technical Stats:
├─ Reconnections: {session_log.reconnect_count}
├─ Difficulty Adjustments: {session_log.difficulty_adjustments}
├─ Uptime: {(duration / 24):.1f} days

═══════════════════════════════════════════════════════════════════
Sacred Mining Protocol - Session Blessed and Complete ✨
Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
═══════════════════════════════════════════════════════════════════"""

            self.git_client.add([session_file, summary_file])
            self.git_client.commit("-m", commit_message)
            
            self.logger.info(f"🏁 Session summary committed: {session_log.session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to commit session summary: {e}")
            return False
    
    def commit_pool_statistics(self, pool_stats: Dict) -> bool:
        """
        Commitne celkové pool statistiky
        """
        try:
            timestamp = time.time()
            stats_file = os.path.join(
                self.sacred_stats_dir,
                f"pool_stats_{int(timestamp)}.json"
            )
            
            with open(stats_file, 'w') as f:
                json.dump(pool_stats, f, indent=2)
            
            commit_message = f"""📊 ZION Sacred Pool Statistics Update

Pool Statistics Snapshot:
├─ Active Miners: {pool_stats.get('miners_online', 0)}
├─ Total Sessions: {pool_stats.get('unique_miners', 0)}
├─ Total Shares: {pool_stats.get('total_shares', 0)}
├─ Blocks Found: {pool_stats.get('total_blocks', 0)}
├─ Total Consciousness: {pool_stats.get('total_consciousness', 0):.2f}
├─ Sacred Masters: {pool_stats.get('sacred_masters', 0)}
├─ Ascended Beings: {pool_stats.get('ascended_beings', 0)}

Pool Uptime: {(time.time() - pool_stats.get('session_start', time.time())) / 3600:.1f} hours
Generated: {datetime.fromtimestamp(timestamp, timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}

Sacred Mining Pool - Statistics Archive 📈"""

            self.git_client.add(stats_file)
            self.git_client.commit("-m", commit_message)
            
            self.logger.info("📊 Pool statistics committed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to commit pool statistics: {e}")
            return False
    
    def _calculate_milestone_bonus(self, milestone_type: str, milestone_value: int) -> float:
        """Vypočítá sacred bonus za milestone"""
        bonuses = {
            "shares": milestone_value * 0.1,
            "blocks": milestone_value * 108.0,  # Sacred number 108
            "reconnect": milestone_value * 5.0,
            "difficulty": milestone_value * 2.0
        }
        return bonuses.get(milestone_type, 1.0)
    
    def _analyze_session_performance(self, session_log: MiningSessionLog) -> Dict:
        """Analyzuje výkon mining session"""
        duration = session_log.get_duration_hours()
        
        return {
            "efficiency_rating": "excellent" if session_log.get_efficiency_percentage() > 95 else "good",
            "duration_rating": "marathon" if duration > 24 else "standard",
            "hashrate_stability": "stable" if session_log.peak_hashrate / max(session_log.average_hashrate, 1) < 2 else "variable",
            "reconnect_stability": "excellent" if session_log.reconnect_count < 3 else "needs_improvement"
        }
    
    def _analyze_sacred_achievements(self, session_log: MiningSessionLog) -> Dict:
        """Analyzuje sacred achievements"""
        return {
            "consciousness_growth": session_log.consciousness_points,
            "sacred_level_achieved": session_log.sacred_level,
            "quantum_coherence_level": session_log.quantum_coherence,
            "sacred_bonus_earned": session_log.consciousness_points * session_log.sacred_multiplier
        }
    
    def _generate_recommendations(self, session_log: MiningSessionLog) -> List[str]:
        """Generuje doporučení pro zlepšení mining"""
        recommendations = []
        
        if session_log.get_efficiency_percentage() < 90:
            recommendations.append("🔧 Consider checking network stability for better share acceptance")
        
        if session_log.reconnect_count > 5:
            recommendations.append("🌐 Investigate connection stability - too many reconnects")
        
        if session_log.consciousness_points < 100:
            recommendations.append("🧠 Focus on longer sessions to increase consciousness points")
        
        if session_log.sacred_level == "initiate":
            recommendations.append("⭐ Upgrade to ZION Sacred Miner for enhanced bonuses")
        
        return recommendations

# ═══════════════════════════════════════════════════════════════════════════════════════════
# 🌟 EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test ZION Mining Git Integration
    git_integration = ZionMiningGitIntegration("/media/maitreya/ZION1")
    
    # Create test session log
    test_session = MiningSessionLog(
        session_id="test_sacred_001",
        start_time=time.time(),
        miner_type="zion-sacred",
        worker_name="ZION-SACRED-TEST",
        wallet_address="Z32f72f93c095d78fc8a2fe01c0f97fd4a7f6d1bcd9b251f73b18b5625be654e84",
        total_shares=250,
        accepted_shares=248,
        rejected_shares=2,
        consciousness_points=156.7,
        sacred_level="master",
        sacred_multiplier=2.618,
        quantum_coherence=0.618
    )
    
    # Test commits
    git_integration.commit_mining_session_start(test_session)
    print("🌟 ZION Mining Git Integration test complete!")
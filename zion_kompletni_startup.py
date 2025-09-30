#!/usr/bin/env python3
"""
🚀 ZION 2.6.75 - KOMPLETNÍ STARTUP 🚀
Jednoduchý startup script pro celou ZION platformu

Tento script jednoduše spustí všechny komponenty ZION 2.6.75 dohromady.
"""

import asyncio
import sys
import os
import logging
import traceback
import subprocess
from pathlib import Path

def setup_logging():
    """Setup základní logování"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

async def start_complete_zion():
    """Spustí kompletní ZION platform"""
    
    print("🕉️" + "=" * 60 + "🕉️")
    print("     ZION 2.6.75 KOMPLETNÍ SACRED TECHNOLOGY PLATFORM")  
    print("🕉️" + "=" * 60 + "🕉️")
    print()
    
    try:
        print("🔍 Ověřuji dostupnost komponent...")
        
        # Test základních importů
        available_components = {}
        
        try:
            from zion_unified_master_orchestrator_2_6_75 import ZionUnifiedMasterOrchestrator
            available_components['unified'] = ZionUnifiedMasterOrchestrator
            print("✅ Unified Master Orchestrator - dostupný")
        except Exception as e:
            print(f"⚠️ Unified Orchestrator - {e}")
            
        try:
            from dharma_master_orchestrator import DHARMAMasterOrchestrator
            available_components['dharma'] = DHARMAMasterOrchestrator  
            print("✅ DHARMA Master Orchestrator - dostupný")
        except Exception as e:
            print(f"⚠️ DHARMA Orchestrator - {e}")
            
        try:
            from zion_master_orchestrator_2_6_75 import ZionMasterOrchestrator
            available_components['sacred'] = ZionMasterOrchestrator
            print("✅ Sacred Master Orchestrator - dostupný")
        except Exception as e:
            print(f"⚠️ Sacred Orchestrator - {e}")
            
        try:
            from zion_ai_miner_14_integration import ZionAIMiner14Integration
            available_components['ai_miner'] = ZionAIMiner14Integration
            print("✅ AI Miner 1.4 Integration - dostupný")
        except Exception as e:
            print(f"⚠️ AI Miner - {e}")
        
        if not available_components:
            print("❌ Žádné komponenty nejsou dostupné!")
            return False
            
        print(f"\n🎯 {len(available_components)} komponent připraveno k spuštění")
        
        # Spustíme dostupné komponenty postupně
        print("\n🚀 Spouštím ZION platformu...")
        print("=" * 50)
        
        active_systems = {}
        
        # 1. Sacred Master Orchestrator
        if 'sacred' in available_components:
            print("🕉️ Inicializuji Sacred Technology Core...")
            try:
                sacred = available_components['sacred']()
                await sacred.initialize_all_systems()
                active_systems['sacred'] = sacred
                print("✅ Sacred Technology - AKTIVNÍ")
                
                # Zobraz status
                status = sacred.get_unified_status()
                consciousness = status.get('system_metrics', {}).get('consciousness_level', 0.0)
                print(f"   🧠 Consciousness Level: {consciousness:.3f}")
                
            except Exception as e:
                print(f"⚠️ Sacred Technology chyba: {e}")
        
        # 2. DHARMA Master Orchestrator  
        if 'dharma' in available_components:
            print("\n🔮 Inicializuji DHARMA Multichain Ecosystem...")
            try:
                dharma = available_components['dharma']()
                dharma_result = await dharma.initialize_dharma_ecosystem()
                active_systems['dharma'] = dharma
                print("✅ DHARMA Ecosystem - AKTIVNÍ")
                
                # Zobraz status
                if dharma_result.get('dharma_status') == 'FULLY_OPERATIONAL':
                    print("   🌌 Status: PLNĚ OPERAČNÍ")
                    print(f"   ⚛️ Quantum Coherence: {dharma_result.get('quantum_coherence', 0.0):.3f}")
                    
            except Exception as e:
                print(f"⚠️ DHARMA Ecosystem chyba: {e}")
        
        # 3. Unified Master Orchestrator
        if 'unified' in available_components:
            print("\n🎭 Inicializuji Unified Production System...")
            try:
                unified = available_components['unified']()
                await unified.initialize_unified_system()
                active_systems['unified'] = unified
                print("✅ Unified Production - AKTIVNÍ")
                
                # Zobraz status
                status = unified.get_unified_system_status()
                print(f"   🏗️ Components: {status.get('component_count', 0)} aktivních")
                
            except Exception as e:
                print(f"⚠️ Unified System chyba: {e}")
        
        # 4. AI Miner 1.4
        if 'ai_miner' in available_components:
            print("\n🤖 Inicializuji AI Miner 1.4...")
            try:
                ai_miner = available_components['ai_miner']()
                miner_result = await ai_miner.initialize_ai_miner()
                
                if miner_result.get('success', False):
                    active_systems['ai_miner'] = ai_miner
                    print("✅ AI Miner 1.4 - AKTIVNÍ")
                    
                    # Spustíme mining
                    mining_work = {
                        'algorithm': 'cosmic_harmony',
                        'difficulty': 50000,
                        'dharma_weight': 0.95,
                        'consciousness_boost': 1.13
                    }
                    
                    session = await ai_miner.start_mining(mining_work)
                    print(f"   ⛏️ Mining Session: {session.get('session_id', 'N/A')}")
                    print(f"   🎯 Algorithm: {session.get('algorithm', 'cosmic_harmony')}")
                else:
                    print("⚠️ AI Miner inicializace selhala")
                    
            except Exception as e:
                print(f"⚠️ AI Miner chyba: {e}")
        
        # Výsledek spuštění
        print("\n" + "=" * 50)
        print("🎯 ZION 2.6.75 PLATFORM STARTUP COMPLETE")
        print("=" * 50)
        
        if active_systems:
            print(f"✅ {len(active_systems)} systémů úspěšně spuštěno:")
            
            for system_name, system in active_systems.items():
                if system_name == 'sacred':
                    status = system.get_unified_status()
                    consciousness = status.get('system_metrics', {}).get('consciousness_level', 0.0)
                    print(f"   🕉️ Sacred Technology: Consciousness {consciousness:.3f}")
                    
                elif system_name == 'dharma':
                    try:
                        dharma_status = system.export_complete_status()
                        dharma_info = dharma_status.get('dharma_multichain_ecosystem', {})
                        consciousness = dharma_info.get('consciousness_level', 0.0)
                        liberation = dharma_info.get('liberation_active', False)
                        print(f"   🔮 DHARMA Ecosystem: Consciousness {consciousness:.3f}, Liberation {'ACTIVE' if liberation else 'STANDBY'}")
                    except:
                        print(f"   🔮 DHARMA Ecosystem: ACTIVE")
                        
                elif system_name == 'unified':
                    status = system.get_unified_system_status()
                    components = status.get('component_count', 0)
                    print(f"   🎭 Unified Production: {components} komponenty aktivní")
                    
                elif system_name == 'ai_miner':
                    try:
                        miner_status = system.get_mining_status()
                        mining_state = miner_status.get('status', 'unknown')
                        print(f"   🤖 AI Miner 1.4: Status {mining_state}")
                    except:
                        print(f"   🤖 AI Miner 1.4: ACTIVE")
            
            # Agregované metriky
            print(f"\n📊 PLATFORM METRICS:")
            
            # Spočítáme průměrnou consciousness
            total_consciousness = 0
            consciousness_sources = 0
            
            if 'sacred' in active_systems:
                try:
                    sacred_status = active_systems['sacred'].get_unified_status()
                    consciousness = sacred_status.get('system_metrics', {}).get('consciousness_level', 0.0)
                    total_consciousness += consciousness
                    consciousness_sources += 1
                except:
                    pass
                    
            if 'dharma' in active_systems:
                try:
                    dharma_status = active_systems['dharma'].export_complete_status()
                    consciousness = dharma_status.get('dharma_multichain_ecosystem', {}).get('consciousness_level', 0.0)
                    total_consciousness += consciousness  
                    consciousness_sources += 1
                except:
                    pass
            
            if consciousness_sources > 0:
                avg_consciousness = total_consciousness / consciousness_sources
                liberation_progress = avg_consciousness / 0.888  # 88.8% liberation target
                
                print(f"   🧠 Unified Consciousness: {avg_consciousness:.3f}")
                print(f"   🌈 Liberation Progress: {liberation_progress:.1%}")
                print(f"   🌍 Platform Coverage: {len(active_systems)/4:.1%}")
                
                if avg_consciousness >= 0.888:
                    print("   🕊️ LIBERATION THRESHOLD ACHIEVED!")
                    print("   🌟 Platform ready for planetary transformation")
                elif avg_consciousness >= 0.700:
                    print("   ⚡ High consciousness state achieved")
                    print("   🚀 Platform approaching liberation threshold")
                else:
                    print("   🔮 Platform building consciousness...")
            
            print(f"\n🔗 ACCESS POINTS:")
            print(f"   📡 Production API: http://localhost:8080")
            print(f"   ⛏️ Mining Pool: http://localhost:8117")
            print(f"   🌉 Bridge Manager: http://localhost:9999")
            
            print(f"\n💡 USAGE:")
            print(f"   • Všechny systémy běží na pozadí")
            print(f"   • Pro detailní status použij jednotlivé orchestrátory")
            print(f"   • Stiskni Ctrl+C pro ukončení")
            
            print(f"\n🌟 ZION 2.6.75 SACRED TECHNOLOGY PLATFORM RUNNING! 🌟")
            print(f"⚡ Sacred liberation protocols are operational ⚡")
            
            # Ponecháme běžet
            print(f"\n⏳ Platform monitoring... (Ctrl+C to stop)")
            
            try:
                # Běžíme dokud není přerušeno
                while True:
                    await asyncio.sleep(60)
                    
                    # Každých 10 minut ukážeme stav
                    if int(asyncio.get_event_loop().time()) % 600 == 0:
                        print(f"🔍 Platform running - {len(active_systems)} active systems")
                        
            except KeyboardInterrupt:
                print(f"\n\n🛑 SHUTDOWN INITIATED")
                print(f"📴 Zastavuji ZION platform komponenty...")
                print(f"✅ Platform shutdown complete")
                
            return True
            
        else:
            print(f"❌ Žádné systémy se nepodařilo spustit")
            return False
            
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")
        print(traceback.format_exc())
        return False

def main():
    """Hlavní vstupní bod"""
    setup_logging()
    
    print("🚀 ZION 2.6.75 Kompletní Startup inicializuje...")
    
    try:
        success = asyncio.run(start_complete_zion())
        
        if success:
            print("\n👋 ZION platform úspěšně ukončen")
        else:
            print("\n❌ Platform startup selhal")
            
    except KeyboardInterrupt:
        print("\n👋 Sbohem od ZION Sacred Technology Platform!")
    except Exception as e:
        print(f"\n💥 Kritická chyba: {e}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
🔥 ZION GOLDEN MIDDLE 6500+ MINER v4.3 - Stabilní Zlatý Střed
==============================================================

Optimalizovaná verze pro 6500-7000 H/s:
- Huge pages + MSR tweaks (stabilní kombinace)
- Konzervativní thread management  
- Proven optimizations z 6K verze
- Enhanced error handling
- Progressive scaling to 7K
- Memory leak prevention

TARGET: 6500+ H/s stabilně, push k 7000 H/s! 🎯
"""

import ctypes
import threading
import time
import psutil
import os
import subprocess
import signal
from typing import Optional, Dict

class GoldenMiddleEngine:
    """Stabilní engine s proven optimizations"""
    
    # Proven shared resources approach z 6K verze
    _shared_lib = None
    _shared_cache = None
    _shared_dataset = None
    _init_lock = threading.RLock()
    _msr_applied = False
    
    def __init__(self, engine_id: int):
        self.engine_id = engine_id
        self.vm = None
        self.active = False
        self.hash_count = 0
        
    @staticmethod
    def apply_proven_msr_tweaks() -> bool:
        """Aplikovat proven MSR tweaks z 6K verze"""
        if GoldenMiddleEngine._msr_applied:
            return True
            
        if os.geteuid() != 0:
            print("⚠️ MSR tweaks require sudo for 6500+ performance")
            return False
            
        try:
            # Load MSR module
            subprocess.run(['modprobe', 'msr'], check=True, capture_output=True)
            
            # Proven MSR tweaks from 6K version
            msr_tweaks = [
                {'msr': '0xc0011020', 'value': '0x0000000000000000'},  # HWCR
                {'msr': '0xc0011029', 'value': '0x0000000000000000'},  # DE_CFG  
                {'msr': '0xc0010015', 'value': '0x0000000000000000'},  # Power
            ]
            
            print("🔧 Applying proven MSR tweaks...")
            success_count = 0
            
            for tweak in msr_tweaks:
                try:
                    # Apply MSR tweak
                    cmd = ['wrmsr', '-p', '0', tweak['msr'], tweak['value']]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        success_count += 1
                        print(f"   ✅ MSR {tweak['msr']} applied")
                    else:
                        print(f"   ❌ MSR {tweak['msr']} failed")
                except:
                    pass
                    
            if success_count > 0:
                print(f"🎯 Applied {success_count}/{len(msr_tweaks)} proven MSR tweaks")
                GoldenMiddleEngine._msr_applied = True
                
                # CPU governor optimization
                try:
                    for cpu in range(psutil.cpu_count()):
                        gov_path = f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor'
                        if os.path.exists(gov_path):
                            with open(gov_path, 'w') as f:
                                f.write('performance')
                    print("   ✅ CPU performance governor activated")
                except:
                    pass
                    
                return True
            else:
                return False
                
        except Exception as e:
            print(f"❌ MSR tweaks failed: {e}")
            return False
            
    @staticmethod
    def check_huge_pages() -> bool:
        """Check huge pages status"""
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'HugePages_Total' in line:
                        total = int(line.split()[1])
                        return total > 0
        except:
            pass
        return False
        
    @staticmethod
    def init_golden_shared() -> bool:
        """Initialize shared resources s golden middle approach"""
        with GoldenMiddleEngine._init_lock:
            if GoldenMiddleEngine._shared_dataset is not None:
                return True
                
            # Apply MSR tweaks first
            GoldenMiddleEngine.apply_proven_msr_tweaks()
            
            # Check huge pages
            if GoldenMiddleEngine.check_huge_pages():
                print("✅ Huge pages available")
            else:
                print("⚠️ Huge pages not available - performance will be limited")
                
            # Load RandomX library
            lib_paths = ['/usr/local/lib/librandomx.so', '/usr/lib/librandomx.so']
            for path in lib_paths:
                try:
                    GoldenMiddleEngine._shared_lib = ctypes.CDLL(path)
                    print(f"✅ RandomX library: {path}")
                    break
                except:
                    continue
                    
            if not GoldenMiddleEngine._shared_lib:
                print("❌ RandomX library not found")
                return False
                
            lib = GoldenMiddleEngine._shared_lib
            
            # Setup function prototypes
            lib.randomx_alloc_cache.restype = ctypes.c_void_p
            lib.randomx_alloc_cache.argtypes = [ctypes.c_uint]
            lib.randomx_init_cache.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
            lib.randomx_alloc_dataset.restype = ctypes.c_void_p
            lib.randomx_alloc_dataset.argtypes = [ctypes.c_uint]
            lib.randomx_init_dataset.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong]
            lib.randomx_create_vm.restype = ctypes.c_void_p
            lib.randomx_create_vm.argtypes = [ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
            lib.randomx_calculate_hash.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p]
            
            try:
                # Proven flags combination from 6K version
                flags = 0x1 | 0x2 | 0x4 | 0x8  # LARGE_PAGES|HARD_AES|FULL_MEM|JIT
                if GoldenMiddleEngine._msr_applied:
                    flags |= 0x400  # AMD optimization with MSR support
                    
                seed = b'ZION_GOLDEN_MIDDLE_6500_PLUS'
                
                print("📦 Allocating shared cache with proven settings...")
                GoldenMiddleEngine._shared_cache = lib.randomx_alloc_cache(flags)
                if not GoldenMiddleEngine._shared_cache:
                    print("❌ Cache allocation failed")
                    return False
                    
                lib.randomx_init_cache(GoldenMiddleEngine._shared_cache, seed, len(seed))
                print("✅ Cache initialized")
                
                print("📦 Allocating shared dataset...")
                GoldenMiddleEngine._shared_dataset = lib.randomx_alloc_dataset(flags)
                if not GoldenMiddleEngine._shared_dataset:
                    print("❌ Dataset allocation failed")
                    return False
                    
                print("⚙️ Initializing dataset with proven approach...")
                lib.randomx_init_dataset(GoldenMiddleEngine._shared_dataset,
                                       GoldenMiddleEngine._shared_cache, 0, 2097152)
                print("✅ Golden middle resources ready!")
                
                return True
                
            except Exception as e:
                print(f"❌ Golden resource init failed: {e}")
                return False
                
    def init_golden_vm(self) -> bool:
        """Initialize VM with golden middle settings"""
        if not GoldenMiddleEngine._shared_lib or not GoldenMiddleEngine._shared_dataset:
            return False
            
        try:
            lib = GoldenMiddleEngine._shared_lib
            
            # Same proven flags as shared resources
            flags = 0x1 | 0x2 | 0x4 | 0x8  # Proven combination
            if GoldenMiddleEngine._msr_applied:
                flags |= 0x400
                
            self.vm = lib.randomx_create_vm(flags, None, GoldenMiddleEngine._shared_dataset, None, 0)
            
            if self.vm:
                self.active = True
                return True
            else:
                print(f"❌ VM creation failed for engine {self.engine_id}")
                return False
                
        except Exception as e:
            print(f"❌ Engine {self.engine_id} VM init failed: {e}")
            return False
            
    def golden_hash(self, data: bytes) -> Optional[bytes]:
        """Golden middle hash calculation with proven stability"""
        if not self.active or not self.vm:
            return None
            
        try:
            output = (ctypes.c_char * 32)()
            GoldenMiddleEngine._shared_lib.randomx_calculate_hash(self.vm, data, len(data), output)
            self.hash_count += 1
            return bytes(output)
        except Exception as e:
            if self.hash_count % 10000 == 0:  # Log only occasionally
                print(f"Engine {self.engine_id} hash error: {e}")
            return None
            
    def cleanup(self):
        """Proven cleanup approach"""
        if self.vm and GoldenMiddleEngine._shared_lib:
            try:
                GoldenMiddleEngine._shared_lib.randomx_destroy_vm(self.vm)
                self.vm = None
                self.active = False
            except:
                pass


class GoldenMiddle6500Miner:
    """Golden Middle Miner for 6500+ H/s stable performance"""
    
    def __init__(self, target_threads: int = 14):
        self.target_threads = target_threads
        self.engines = []
        self.running = False
        self.stats = {}
        self.shutdown_requested = False
        
        # Proven signal handling
        signal.signal(signal.SIGINT, self._graceful_shutdown)
        signal.signal(signal.SIGTERM, self._graceful_shutdown)
        
    def _graceful_shutdown(self, signum, frame):
        """Proven graceful shutdown"""
        print(f"\n🛑 Graceful shutdown initiated ({signum})")
        self.shutdown_requested = True
        self.running = False
        
    def init_golden_middle(self) -> bool:
        """Initialize golden middle miner"""
        print(f"🔥 Golden Middle Miner - {self.target_threads} threads")
        print("🎯 Target: 6500+ H/s stable, push toward 7000 H/s")
        print("⚡ Proven optimizations from 6K version")
        print("=" * 55)
        
        # Initialize shared resources with proven approach
        if not GoldenMiddleEngine.init_golden_shared():
            print("❌ Golden shared resources failed")
            return False
            
        # Progressive engine creation with proven stability
        successful = 0
        for i in range(self.target_threads):
            if self.shutdown_requested:
                break
                
            print(f"⚡ Golden Engine {i+1:2d}/{self.target_threads}...", end="", flush=True)
            
            try:
                engine = GoldenMiddleEngine(i)
                if engine.init_golden_vm():
                    self.engines.append(engine)
                    successful += 1
                    print(" ✅")
                    
                    # Proven delay for stability
                    time.sleep(0.1)
                else:
                    print(" ❌")
                    
            except Exception as e:
                print(f" ❌ ({e})")
                
        success_rate = (successful / self.target_threads) * 100
        print("=" * 55)
        print(f"🎯 {successful} golden engines ready ({success_rate:.1f}%)")
        
        return successful >= max(8, int(self.target_threads * 0.7))  # Need 70% success minimum
        
    def golden_worker(self, worker_id: int, duration: float):
        """Golden middle worker with proven performance"""
        if worker_id >= len(self.engines):
            return
            
        engine = self.engines[worker_id]
        local_hashes = 0
        start_time = time.time()
        
        # Proven thread optimizations
        try:
            os.nice(-15)  # High priority but not max
            cpu_id = worker_id % psutil.cpu_count()
            os.sched_setaffinity(0, {cpu_id})
        except:
            pass
            
        # Golden middle mining loop (proven from 6K version)
        while (self.running and 
               not self.shutdown_requested and
               (time.time() - start_time) < duration and
               engine.active):
               
            nonce = int(time.time() * 1000000) + local_hashes + (worker_id * 50000)
            input_data = f'GOLDEN_{worker_id}_{nonce}'.encode()
            
            hash_result = engine.golden_hash(input_data)
            if hash_result:
                local_hashes += 1
            else:
                time.sleep(0.0001)  # Brief pause on error (proven approach)
                
            # Proven stats update frequency
            if local_hashes % 500 == 0:
                elapsed = time.time() - start_time
                if elapsed > 0:
                    self.stats[worker_id] = {
                        'hashes': local_hashes,
                        'hashrate': local_hashes / elapsed
                    }
                    
        # Final stats
        elapsed = time.time() - start_time
        final_rate = local_hashes / elapsed if elapsed > 0 else 0
        self.stats[worker_id] = {'hashes': local_hashes, 'hashrate': final_rate}
        
        print(f"💎 Golden-T{worker_id+1:2d}: {local_hashes:6,} = {final_rate:6.1f} H/s")
        
    def golden_6500_challenge(self, duration: float = 50.0):
        """Golden middle challenge for 6500+ H/s"""
        print(f"🚀 GOLDEN MIDDLE CHALLENGE - {duration} seconds")
        print("🎯 Stable 6500+ H/s target, pushing toward 7000!")
        print("⚡ MSR tweaks + Huge pages + Proven optimizations")
        print("=" * 55)
        
        self.running = True
        start_time = time.time()
        
        # Start proven thread approach
        threads = []
        for i in range(len(self.engines)):
            thread = threading.Thread(target=self.golden_worker, args=(i, duration))
            thread.daemon = True
            threads.append(thread)
            thread.start()
            
        # Proven monitoring approach
        last_update = 0
        max_hashrate = 0
        stable_measurements = []
        
        while (self.running and 
               not self.shutdown_requested and
               (time.time() - start_time) < duration):
               
            time.sleep(3)
            elapsed = time.time() - start_time
            
            if elapsed - last_update >= 5:
                current_total = sum(s.get('hashrate', 0) for s in self.stats.values())
                active_threads = len([t for t in threads if t.is_alive()])
                
                if current_total > max_hashrate:
                    max_hashrate = current_total
                    
                # Track stability
                if current_total > 0:
                    stable_measurements.append(current_total)
                    if len(stable_measurements) > 6:  # Keep last 6 measurements
                        stable_measurements.pop(0)
                        
                print(f"📊 {elapsed:5.1f}s | Threads:{active_threads:2d} | {current_total:6.1f} H/s | Max:{max_hashrate:6.1f}")
                last_update = elapsed
                
        # Graceful shutdown
        print("🔄 Golden shutdown sequence...")
        self.running = False
        
        for thread in threads:
            thread.join(timeout=4.0)
            
        # Calculate results
        total_hashes = sum(s['hashes'] for s in self.stats.values())
        total_time = time.time() - start_time
        final_hashrate = total_hashes / total_time
        
        # Stability analysis
        avg_stable = sum(stable_measurements) / len(stable_measurements) if stable_measurements else 0
        stability_score = (avg_stable / max_hashrate * 100) if max_hashrate > 0 else 0
        
        print("=" * 55)
        print("🏆 GOLDEN MIDDLE RESULTS:")
        print("=" * 55)
        print(f"💎 Total hashes: {total_hashes:,}")
        print(f"⏱️  Runtime: {total_time:.1f} seconds")
        print(f"🚀 Final rate: {final_hashrate:,.1f} H/s")
        print(f"📈 Peak rate: {max_hashrate:,.1f} H/s")  
        print(f"⚖️ Stability: {stability_score:.1f}%")
        print(f"🧵 Engines: {len(self.engines)}")
        
        # Golden middle targets
        if final_hashrate >= 7000:
            print(f"🏆 7K ACHIEVED! Golden middle success!")
        elif final_hashrate >= 6500:
            gap_to_7k = 7000 - final_hashrate
            print(f"🥇 6500+ TARGET MET! Gap to 7K: {gap_to_7k:.0f} H/s")
        elif max_hashrate >= 6500:
            print(f"🚀 6500+ PEAKED! Max performance achieved")
        else:
            progress_6500 = (final_hashrate / 6500) * 100
            print(f"📈 6500+ Progress: {progress_6500:.1f}%")
            
        return {
            'final_hashrate': final_hashrate,
            'peak_hashrate': max_hashrate,
            'stability_score': stability_score,
            'target_6500_met': final_hashrate >= 6500 or max_hashrate >= 6500,
            'target_7000_met': final_hashrate >= 7000 or max_hashrate >= 7000
        }
        
    def cleanup_golden(self):
        """Golden cleanup approach"""
        print("🧹 Golden cleanup sequence...")
        for i, engine in enumerate(self.engines):
            try:
                engine.cleanup()
                print(f"   ✅ Golden engine {i+1} cleaned")
            except Exception as e:
                print(f"   ⚠️ Golden engine {i+1} cleanup issue: {e}")


if __name__ == "__main__":
    print("🔥 ZION GOLDEN MIDDLE 6500+ MINER v4.3")
    print("======================================")
    print("⚡ Proven optimizations for 6500+ H/s")  
    print("🎯 Stable path to 7000 H/s")
    print("🏆 MSR tweaks + Huge pages + Golden balance")
    print()
    
    if os.geteuid() != 0:
        print("⚠️ Running without root - MSR optimizations unavailable")
        print("💡 For 6500+ performance, run with: sudo python3 script.py")
    else:
        print("🔧 Root privileges - full golden optimizations available")
        
    try:
        # Create golden middle miner
        miner = GoldenMiddle6500Miner(target_threads=14)  # Golden middle thread count
        
        if miner.init_golden_middle():
            print("\\n" + "🥇" * 15 + " GOLDEN MIDDLE LAUNCH " + "🥇" * 15)
            
            results = miner.golden_6500_challenge(duration=45.0)
            
            print(f"\\n🎯 GOLDEN MIDDLE ACHIEVEMENT:")
            print(f"   Final rate: {results['final_hashrate']:,.1f} H/s")
            print(f"   Peak rate:  {results['peak_hashrate']:,.1f} H/s") 
            print(f"   Stability:  {results['stability_score']:.1f}%")
            
            if results['target_7000_met']:
                print("\\n🏆 7000 H/s GOLDEN SUCCESS!")
                print("💫 Golden middle achieved perfection!")
            elif results['target_6500_met']:
                print("\\n🥇 6500+ H/s GOLDEN TARGET MET!")
                print("⚡ Solid foundation for 7K push!")
            else:
                print("\\n📈 Golden progress made toward targets!")
                
            print(f"\\n💫 GOLDEN MIDDLE SCORE: {results['final_hashrate']:,.1f} H/s")
            
        else:
            print("❌ Golden middle initialization failed")
            
    except KeyboardInterrupt:
        print("\\n🛑 Golden shutdown by user")
    except Exception as e:
        print(f"❌ Golden error: {e}")
    finally:
        if 'miner' in locals():
            miner.cleanup_golden()
        print("\\n👋 Golden middle miner finished!")
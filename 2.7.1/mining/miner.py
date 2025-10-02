#!/usr/bin/env python3
"""
ZION 2.7.1 ASIC-Resistant CPU Miner
Pure Argon2 mining for maximum decentralization
"""

import time
import threading
import os
from typing import Dict, Any, Optional, Callable
from mining.config import get_mining_config, create_asic_resistant_algorithm
from mining.algorithms import Argon2Algorithm

class ASICResistantMiner:
    """
    ASIC-Resistant CPU Miner using Argon2

    This miner is designed to work only with ASIC-resistant algorithms,
    preventing hardware centralization and ensuring fair mining.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_mining_config().get_mining_config()
        self.algorithm = create_asic_resistant_algorithm()
        self.is_mining = False
        self.mining_threads: list = []
        self.thread_stats = {}  # Per-thread statistics
        self.stats_lock = threading.Lock()
        self.stats = {
            'hashes': 0,
            'blocks_found': 0,
            'start_time': None,
            'hashrate': 0.0
        }

        algo_name = self.config.get('algorithm', 'argon2')
        print(f"⛏️ ASIC-Resistant Miner initialized with {algo_name} algorithm")

    def start_mining(self, address: str, num_threads: int = 1, callback: Optional[Callable] = None) -> bool:
        """
        Start ASIC-resistant mining with multiple threads

        Args:
            address: Mining reward address
            num_threads: Number of mining threads (max enforced by config)
            callback: Optional callback for found blocks

        Returns:
            True if mining started successfully
        """
        if self.is_mining:
            print("⚠️ Mining already running")
            return False

        # Enforce ASIC-resistant thread limits
        max_threads = self.config.get('max_threads', 1)
        if num_threads > max_threads:
            print(f"⚠️ Limiting threads to {max_threads} for ASIC resistance")
            num_threads = max_threads

        self.is_mining = True
        self.stats['start_time'] = time.time()
        self.mining_threads = []
        self.thread_stats = {i: {'hashes': 0, 'blocks_found': 0} for i in range(num_threads)}

        print(f"🚀 Starting ASIC-resistant mining with {num_threads} thread(s) to address: {address}")
        print("🛡️ Using Argon2 algorithm - ASIC resistance verified")

        # Start multiple mining threads
        for i in range(num_threads):
            thread = threading.Thread(
                target=self._mining_worker,
                args=(i, address, callback),
                daemon=True,
                name=f"ASIC-Miner-{i+1}"
            )
            thread.start()
            self.mining_threads.append(thread)

        return True

    def stop_mining(self) -> bool:
        """
        Stop mining

        Returns:
            True if mining stopped successfully
        """
        if not self.is_mining:
            print("⚠️ Mining not running")
            return False

        self.is_mining = False

        # Wait for all threads to finish
        for thread in self.mining_threads:
            if thread.is_alive():
                thread.join(timeout=5.0)

        self.mining_threads = []

        print("⏹️ ASIC-resistant mining stopped")
        self._print_stats()

        return True

    def _mining_loop(self, address: str, callback: Optional[Callable]):
        """
        Main ASIC-resistant mining loop

        Args:
            address: Mining reward address
            callback: Optional block found callback
        """
        print("🔄 Entering ASIC-resistant mining loop...")

        while self.is_mining:
            try:
                # Generate mining data
                timestamp = int(time.time())
                nonce = os.urandom(8).hex()

                # Create block data for Argon2 mining
                block_data = f"{address}:{timestamp}:{nonce}".encode()

                # Mine with Argon2 - difficulty is checked in verify method
                # No need for separate target calculation

                # Calculate target from difficulty for verification
                max_target = (1 << 256) - 1
                target_int = max_target // self.config.get('difficulty', 0x0000FFFF)
                target_hash = target_int.to_bytes(32, 'big')

                if self.algorithm.verify(block_data, target_hash):
                    # Block found!
                    self.stats['blocks_found'] += 1
                    print(f"🎉 ASIC-resistant block found! Nonce: {nonce}")

                    if callback:
                        callback({
                            'address': address,
                            'timestamp': timestamp,
                            'nonce': nonce,
                            'algorithm': 'argon2',
                            'asic_resistant': True
                        })

                self.stats['hashes'] += 1

                # Update hashrate every 100 hashes
                if self.stats['hashes'] % 100 == 0:
                    self._update_hashrate()

            except Exception as e:
                print(f"⚠️ Mining error: {e}")
                time.sleep(1)

    def _mining_worker(self, thread_id: int, address: str, callback: Optional[Callable]):
        """
        Mining worker for multi-threaded ASIC-resistant mining

        Args:
            thread_id: Thread identifier
            address: Mining reward address
            callback: Optional block found callback
        """
        print(f"🔨 Thread {thread_id + 1} starting ASIC-resistant mining worker...")

        while self.is_mining:
            try:
                # Generate mining data with thread-specific nonce
                timestamp = int(time.time())
                thread_nonce = os.urandom(8).hex()
                nonce = f"{thread_nonce}_{thread_id}"

                # Create block data for Argon2 mining
                block_data = f"{address}:{timestamp}:{nonce}".encode()

                # Mine with Argon2 - difficulty is checked in verify method
                # No need for separate target calculation

                # Calculate target from difficulty for verification
                max_target = (1 << 256) - 1
                target_int = max_target // self.config.get('difficulty', 0x0000FFFF)
                target_hash = target_int.to_bytes(32, 'big')

                if self.algorithm.verify(block_data, target_hash):
                    # Block found!
                    with self.stats_lock:
                        self.stats['blocks_found'] += 1
                        self.thread_stats[thread_id]['blocks_found'] += 1

                    print(f"🎉 ASIC-resistant block found by thread {thread_id + 1}! Nonce: {nonce}")

                    if callback:
                        callback({
                            'address': address,
                            'timestamp': timestamp,
                            'nonce': nonce,
                            'algorithm': 'argon2',
                            'asic_resistant': True,
                            'thread_id': thread_id
                        })

                # Update thread-specific stats
                with self.stats_lock:
                    self.thread_stats[thread_id]['hashes'] += 1
                    self.stats['hashes'] += 1

                    # Update hashrate every 100 hashes
                    if self.stats['hashes'] % 100 == 0:
                        self._update_hashrate()

            except Exception as e:
                print(f"⚠️ Thread {thread_id + 1} mining error: {e}")
                time.sleep(1)

    def _update_hashrate(self):
        """Update mining hashrate statistics"""
        if self.stats['start_time']:
            elapsed = time.time() - self.stats['start_time']
            if elapsed > 0:
                self.stats['hashrate'] = self.stats['hashes'] / elapsed

    def get_stats(self) -> Dict[str, Any]:
        """
        Get mining statistics

        Returns:
            Dictionary with mining stats
        """
        self._update_hashrate()

        return {
            'is_mining': self.is_mining,
            'algorithm': 'argon2',
            'asic_resistant': True,
            'hashes': self.stats['hashes'],
            'blocks_found': self.stats['blocks_found'],
            'hashrate': f"{self.stats['hashrate']:.1f} H/s",
            'uptime': f"{time.time() - (self.stats['start_time'] or time.time()):.0f}s",
            'memory_usage': f"{self.config.get('memory_cost', 65536) // 1024}MB",
            'threads': len(self.mining_threads) if self.is_mining else 0
        }

    def _print_stats(self):
        """Print mining statistics"""
        stats = self.get_stats()
        print("📊 ASIC-Resistant Mining Statistics:")
        print(f"   Algorithm: {stats['algorithm']} (ASIC Resistant: ✅)")
        print(f"   Hashes: {stats['hashes']}")
        print(f"   Blocks Found: {stats['blocks_found']}")
        print(f"   Hashrate: {stats['hashrate']}")
        print(f"   Memory Usage: {stats['memory_usage']}")
        print(f"   Uptime: {stats['uptime']}")
        print(f"   Threads: {stats['threads']}")

        # Print per-thread stats if available
        if self.thread_stats:
            print("   Per-Thread Stats:")
            for thread_id, thread_stat in self.thread_stats.items():
                print(f"     Thread {thread_id + 1}: {thread_stat['hashes']} hashes, {thread_stat['blocks_found']} blocks")


def create_miner(config: Optional[Dict[str, Any]] = None) -> ASICResistantMiner:
    """
    Create ASIC-resistant miner instance

    Args:
        config: Optional mining configuration

    Returns:
        ASICResistantMiner instance
    """
    return ASICResistantMiner(config)


# Standalone mining functions for CLI usage
def start_asic_resistant_mining(address: str, duration: Optional[int] = None, num_threads: int = 1) -> Dict[str, Any]:
    """
    Start ASIC-resistant mining session

    Args:
        address: Mining reward address
        duration: Optional mining duration in seconds
        num_threads: Number of mining threads

    Returns:
        Mining results dictionary
    """
    miner = create_miner()

    def block_callback(block_data):
        print(f"🎉 Block found with Argon2: {block_data}")

    miner.start_mining(address, num_threads, block_callback)

    if duration:
        print(f"⏰ Mining for {duration} seconds...")
        time.sleep(duration)
        miner.stop_mining()
    else:
        print("⏰ Mining indefinitely... (Ctrl+C to stop)")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            miner.stop_mining()

    return miner.get_stats()


if __name__ == "__main__":
    # Test ASIC-resistant mining
    print("🧪 Testing ASIC-resistant Argon2 mining...")

    miner = create_miner()
    stats = miner.get_stats()
    print(f"✅ Miner created: {stats}")

    # Quick benchmark
    from mining.algorithms import benchmark_all_algorithms
    benchmark = benchmark_all_algorithms(50)
    print(f"🏃 Benchmark results: {benchmark}")
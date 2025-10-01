#!/usr/bin/env python3
"""
ZION Virtual Memory Implementation
Based on XMRig VirtualMemory patterns for proper RandomX cache allocation
"""
import os
import sys
import mmap
import ctypes
from ctypes import *
import subprocess

class ZionVirtualMemory:
    """
    ZION Virtual Memory allocator following XMRig patterns
    Implements huge pages allocation with proper fallback strategy
    """
    
    def __init__(self):
        self.huge_page_size = 2 * 1024 * 1024  # 2MB default
        self.one_gb_page_size = 1024 * 1024 * 1024  # 1GB
        self.allocated_memory = {}
        self.setup_huge_pages()
    
    def setup_huge_pages(self):
        """Setup huge pages - based on XMRig osInit"""
        try:
            # Check current huge pages
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
                if 'HugePages_Free' in meminfo:
                    for line in meminfo.split('\n'):
                        if 'HugePages_Free' in line:
                            free_pages = int(line.split()[1])
                            print(f"[ZION-VM] Available huge pages: {free_pages}")
                            break
            
            # Try to allocate more if needed
            if free_pages < 10:
                print("[ZION-VM] Allocating more huge pages...")
                try:
                    subprocess.run(['sudo', 'sysctl', '-w', 'vm.nr_hugepages=128'], 
                                 capture_output=True, check=True)
                    print("[ZION-VM] Huge pages allocated successfully")
                except Exception as e:
                    print(f"[ZION-VM] Warning: Could not allocate huge pages: {e}")
        except Exception as e:
            print(f"[ZION-VM] Warning: Huge pages setup failed: {e}")
    
    def allocate_large_pages_memory(self, size):
        """
        Allocate large pages memory - following XMRig VirtualMemory_unix.cpp pattern
        Uses mmap with MAP_HUGETLB | MAP_POPULATE flags
        """
        try:
            # Align size to huge page boundary
            aligned_size = self.align_to_huge_page_size(size)
            print(f"[ZION-VM] Allocating {aligned_size} bytes with huge pages")
            
            # Use mmap with huge pages flags (following XMRig pattern)
            # MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE
            flags = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
            
            # Try to add MAP_HUGETLB equivalent
            MAP_HUGETLB = 0x40000
            MAP_POPULATE = 0x8000
            
            flags |= MAP_HUGETLB | MAP_POPULATE
            
            # Create memory mapping
            mem = mmap.mmap(-1, aligned_size, 
                           prot=mmap.PROT_READ | mmap.PROT_WRITE,
                           flags=flags)
            
            if mem:
                print(f"[ZION-VM] Successfully allocated {aligned_size} bytes with huge pages")
                self.allocated_memory[id(mem)] = (mem, aligned_size)
                return mem
            
        except Exception as e:
            print(f"[ZION-VM] Huge pages allocation failed: {e}")
        
        return None
    
    def allocate_executable_memory(self, size, huge_pages=True):
        """
        Allocate executable memory - following XMRig pattern
        """
        try:
            aligned_size = self.align_to_huge_page_size(size) if huge_pages else size
            
            flags = mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
            if huge_pages:
                flags |= 0x40000  # MAP_HUGETLB
            
            # Allocate with execute permissions
            mem = mmap.mmap(-1, aligned_size,
                           prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
                           flags=flags)
            
            if mem:
                print(f"[ZION-VM] Allocated executable memory: {aligned_size} bytes")
                self.allocated_memory[id(mem)] = (mem, aligned_size)
                return mem
                
        except Exception as e:
            print(f"[ZION-VM] Executable memory allocation failed: {e}")
        
        return None
    
    def allocate_randomx_cache(self, cache_size):
        """
        Allocate RandomX cache memory with proper alignment
        Following XMRig RandomX integration pattern
        """
        print(f"[ZION-VM] Allocating RandomX cache: {cache_size} bytes")
        
        # Try huge pages first (XMRig pattern)
        mem = self.allocate_large_pages_memory(cache_size)
        if mem:
            print("[ZION-VM] RandomX cache allocated with huge pages")
            return mem
        
        # Fallback to regular memory
        try:
            aligned_size = self.align_size(cache_size, 64)  # 64-byte alignment
            mem = mmap.mmap(-1, aligned_size,
                           prot=mmap.PROT_READ | mmap.PROT_WRITE,
                           flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
            
            if mem:
                print(f"[ZION-VM] RandomX cache allocated with regular memory: {aligned_size} bytes")
                self.allocated_memory[id(mem)] = (mem, aligned_size)
                return mem
        except Exception as e:
            print(f"[ZION-VM] RandomX cache allocation failed: {e}")
        
        return None
    
    def free_memory(self, mem):
        """Free allocated memory"""
        try:
            mem_id = id(mem)
            if mem_id in self.allocated_memory:
                stored_mem, size = self.allocated_memory[mem_id]
                stored_mem.close()
                del self.allocated_memory[mem_id]
                print(f"[ZION-VM] Freed {size} bytes")
                return True
        except Exception as e:
            print(f"[ZION-VM] Memory free failed: {e}")
        return False
    
    def align_to_huge_page_size(self, size):
        """Align size to huge page boundary - following XMRig pattern"""
        return self.align_size(size, self.huge_page_size)
    
    def align_size(self, size, alignment):
        """Generic size alignment"""
        return ((size - 1) // alignment + 1) * alignment
    
    def get_memory_info(self):
        """Get memory allocation info"""
        return {
            'allocated_blocks': len(self.allocated_memory),
            'total_allocated': sum(size for _, size in self.allocated_memory.values()),
            'huge_page_size': self.huge_page_size
        }

# Global instance
zion_vm = ZionVirtualMemory()

def allocate_randomx_memory(size):
    """Helper function for RandomX cache allocation"""
    return zion_vm.allocate_randomx_cache(size)

def free_randomx_memory(mem):
    """Helper function for RandomX memory cleanup"""
    return zion_vm.free_memory(mem)

if __name__ == "__main__":
    # Test memory allocation
    print("=== ZION Virtual Memory Test ===")
    
    # Test RandomX cache allocation (typical size)
    cache_size = 2097152  # 2MB RandomX cache
    cache_mem = allocate_randomx_memory(cache_size)
    
    if cache_mem:
        print("✓ RandomX cache allocation successful")
        info = zion_vm.get_memory_info()
        print(f"Memory info: {info}")
        
        # Test cleanup
        if free_randomx_memory(cache_mem):
            print("✓ Memory cleanup successful")
        else:
            print("✗ Memory cleanup failed")
    else:
        print("✗ RandomX cache allocation failed")
    
    print("=== Test Complete ===")
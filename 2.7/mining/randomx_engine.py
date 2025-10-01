"""
Minimal RandomX Engine Wrapper (TestNet 2.7)
Falls back to sha256 if librandomx.so unavailable.
NO simulations - explicit fallback only.
"""
from __future__ import annotations
import ctypes, hashlib, os, logging, time
from typing import Optional

logger = logging.getLogger(__name__)

LIB_CANDIDATES = [
    './librandomx.so', '../librandomx.so', '/usr/local/lib/librandomx.so', '/usr/lib/librandomx.so'
]

class RandomXEngine:
    def __init__(self):
        self.lib = None
        self.cache = None
        self.vm = None
        self.seed_key: Optional[bytes] = None
        self.initialized = False
        self.fallback = False
        self._try_load()

    def _try_load(self):
        for path in LIB_CANDIDATES:
            if os.path.exists(path):
                try:
                    self.lib = ctypes.CDLL(path)
                    logger.info(f"Loaded RandomX library: {path}")
                    return
                except OSError as e:
                    logger.warning(f"Failed to load {path}: {e}")
        logger.warning("RandomX library not found - using SHA256 fallback")
        self.fallback = True

    def init(self, seed: bytes) -> bool:
        self.seed_key = seed
        if self.fallback:
            self.initialized = True
            return True
        # Minimal function presence check
        required = ['randomx_alloc_cache','randomx_init_cache','randomx_create_vm','randomx_vm_set_cache','randomx_calculate_hash']
        for fn in required:
            if not hasattr(self.lib, fn):
                logger.error(f"Missing function {fn}, switching to fallback")
                self.fallback = True
                self.initialized = True
                return True
        # Allocate + init cache only (no full dataset phase 1)
        alloc_cache = self.lib.randomx_alloc_cache
        alloc_cache.restype = ctypes.c_void_p
        self.cache = alloc_cache(ctypes.c_uint32(0))
        init_cache = self.lib.randomx_init_cache
        init_cache(self.cache, ctypes.c_void_p(ctypes.addressof(ctypes.create_string_buffer(seed))), len(seed))
        create_vm = self.lib.randomx_create_vm
        create_vm.restype = ctypes.c_void_p
        self.vm = create_vm(ctypes.c_uint32(0), self.cache, None)
        if not self.vm:
            logger.error("Failed to create VM, fallback engaged")
            self.fallback = True
        self.initialized = True
        return True

    def hash(self, data: bytes) -> bytes:
        if not self.initialized:
            raise RuntimeError("RandomXEngine not initialized")
        if self.fallback:
            return hashlib.sha256(self.seed_key + data).digest()
        out = (ctypes.c_ubyte * 32)()
        calc = self.lib.randomx_calculate_hash
        calc(self.vm, data, len(data), out)
        return bytes(out)

if __name__ == '__main__':
    rx = RandomXEngine()
    rx.init(b'ZION_2_7_SEED')
    h = rx.hash(b'test')
    print(f"hash={h.hex()}")

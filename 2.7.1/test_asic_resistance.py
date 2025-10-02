#!/usr/bin/env python3
"""
ASIC Resistance Enforcement Test
Verifies SHA256 is blocked and Argon2 is allowed
"""

print("🧪 Testing ASIC Resistance Enforcement...")

from mining.algorithms import AlgorithmFactory

# Test 1: SHA256 should be blocked
print("\n1. Testing SHA256 block...")
try:
    algo = AlgorithmFactory.create_algorithm('sha256', {})
    print("❌ FAILED: SHA256 algorithm allowed (ASIC vulnerability!)")
except ValueError as e:
    print("✅ SUCCESS: SHA256 blocked -", str(e))

# Test 2: Argon2 should be allowed
print("\n2. Testing Argon2 allowance...")
try:
    algo = AlgorithmFactory.create_algorithm('argon2', {
        'time_cost': 2,
        'memory_cost': 65536,
        'parallelism': 1,
        'hash_len': 32
    })
    print("✅ SUCCESS: Argon2 algorithm allowed (ASIC resistant)")
except Exception as e:
    print("❌ FAILED: Argon2 not working -", e)

# Test 3: ASIC resistance verification
print("\n3. Testing ASIC resistance verification...")
from mining.algorithms import verify_asic_resistance

sha256_resistant = verify_asic_resistance('sha256')
argon2_resistant = verify_asic_resistance('argon2')

if not sha256_resistant and argon2_resistant:
    print("✅ SUCCESS: ASIC resistance properly enforced")
else:
    print("❌ FAILED: ASIC resistance verification incorrect")

print("\n🎉 ASIC Resistance Test Complete!")
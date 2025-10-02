#!/usr/bin/env python3
"""Simple wallet format validator for Zion addresses.

Usage:
  ./validate_wallet_format.py <address>

Exit codes:
 0 = valid pattern (basic checks passed)
 1 = invalid format / pattern mismatch
 2 = usage error

Checks performed:
 - Prefix 'Z3'
 - Length between 90 and 120 chars (adjust if spec stabilizes)
 - Allowed Base58-like charset (no 0 O I l)
"""
import re
import sys

BASE58_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]+$")

MIN_LEN = 90
MAX_LEN = 120
PREFIX = "Z3"

def validate(addr: str) -> bool:
    if not addr or not isinstance(addr, str):
        return False
    if not addr.startswith(PREFIX):
        return False
    if not (MIN_LEN <= len(addr) <= MAX_LEN):
        return False
    if not BASE58_RE.match(addr):
        return False
    # TODO: Add checksum / integrated address decoding when spec available
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: validate_wallet_format.py <address>", file=sys.stderr)
        sys.exit(2)
    address = sys.argv[1].strip()
    if validate(address):
        print("VALID", address)
        sys.exit(0)
    else:
        print("INVALID", address, file=sys.stderr)
        sys.exit(1)

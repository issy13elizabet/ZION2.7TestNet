#!/usr/bin/env python3
"""Zion Address Decode / Inspect Tool

Purpose:
  - Decode (future) Zion Base58 addresses into their binary components
  - Validate prefix (currently 'Z3') and length
  - Placeholder for checksum extraction (TBD in ADDRESS_SPEC)

Usage:
  ./tools/address_decode.py Z3....ADDRESS

Exit codes:
  0 = valid basic structure
  1 = invalid / error

Note: This is intentionally minimal and does NOT yet implement full
Base58 multi-field decoding (public spend/view keys) until the final
encoding layout is confirmed. It focuses on early bootstrap validation.
"""
import sys
import re
from typing import NamedTuple

BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
BASE58_RE = re.compile(r'^[123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz]+$')

PREFIX = "Z3"  # Current working human prefix
MIN_LEN = 60     # Temporarily relaxed while formats stabilize
MAX_LEN = 140

class AddressInfo(NamedTuple):
    address: str
    length: int
    prefix_valid: bool
    charset_valid: bool
    checksum_valid: bool
    notes: str

def basic_check(addr: str) -> AddressInfo:
    notes = []
    prefix_valid = addr.startswith(PREFIX)
    if not prefix_valid:
        notes.append(f"prefix != {PREFIX}")
    if not (MIN_LEN <= len(addr) <= MAX_LEN):
        notes.append(f"length {len(addr)} outside [{MIN_LEN},{MAX_LEN}]")
    charset_valid = bool(BASE58_RE.match(addr))
    if not charset_valid:
        notes.append("non-Base58 characters present")
    # Placeholder checksum logic: future design may append e.g. 4 bytes b58
    checksum_valid = True  # Accept all for now
    if not checksum_valid:
        notes.append("checksum mismatch")
    return AddressInfo(
        address=addr,
        length=len(addr),
        prefix_valid=prefix_valid,
        charset_valid=charset_valid,
        checksum_valid=checksum_valid,
        notes='; '.join(notes) if notes else 'ok'
    )

def main():
    if len(sys.argv) < 2:
        print("Usage: address_decode.py <ADDRESS>", file=sys.stderr)
        sys.exit(1)
    addr = sys.argv[1].strip()
    info = basic_check(addr)
    # Machine friendly output (could add --json flag later)
    print("address:", info.address)
    print("length:", info.length)
    print("prefix_valid:", info.prefix_valid)
    print("charset_valid:", info.charset_valid)
    print("checksum_valid:", info.checksum_valid)
    print("notes:", info.notes)
    ok = info.prefix_valid and info.charset_valid
    sys.exit(0 if ok else 1)

if __name__ == '__main__':
    main()

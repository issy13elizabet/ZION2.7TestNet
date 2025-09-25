# Address Prefix Update (2025-09-21)

Context:
- Requested vanity-like mainnet addresses to start with `Z3...` instead of older `Z1...`/`aj...` forms.

Changes in zion-cryptonote core (branch feature/address-prefix-z3):
- Set CRYPTONOTE_PUBLIC_ADDRESS_BASE58_PREFIX = 0x433F (yields `Z3...` standard addresses in Base58)
- Kept backward-compat: accept legacy prefix 0x5A49 (`aj...` start) when parsing addresses.

Impact:
- Newly created standard addresses will begin with `Z3...`.
- Existing wallets/addresses with the legacy prefix remain valid for receiving/spend.
- No chain re-init required; this is purely display/validation for address encoding.

Build/Deploy on Ryzen host:
- Pull submodule branch:
  - In submodule `zion-cryptonote`: `git fetch && git checkout feature/address-prefix-z3`
  - Rebuild docker images or native binaries as per existing pipeline.
- Parent repo pointer updated to this submodule commit.

Verification:
- Generate a new address via wallet CLI or walletd RPC; confirm it begins with `Z3`.
- Ensure sending to old `aj...` address is still accepted by node/wallet.

Notes:
- If you prefer `Z1` specifically, we can tune the prefix constant accordingly; `0x433F` is chosen for stable `Z3..` head across keyspace.

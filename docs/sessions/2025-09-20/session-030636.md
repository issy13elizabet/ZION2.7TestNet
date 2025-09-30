# Session Log – 2025-09-20 030636Z

> POOL 3333 – GBT unblocked, shim cache, pending miner auth

Meta
- Time (UTC): 2025-09-20T03:06:36Z
- Branch: master
- Remote: origin	https://github.com/Yose144/Zion.git (fetch)
- Host: Darwin Yose--MacBook-Pro.local 24.6.0 Darwin Kernel Version 24.6.0: Mon Jul 14 11:30:34 PDT 2025; root:xnu-11417.140.69~1/RELEASE_ARM64_T8103 arm64
- PWD: /Users/yose/Zion

Git status


Last commits


Notes
- Nasazeno: seed1/seed2 rebuild s allowBusyCore pro getblocktemplate; rpc-shim s GBT cache, backoff, rotate; uzi-pool blockRefreshInterval=30s.\n- Stav: pool bezi na 3333, lastGbt height=1; XMRig test (na serveru) stale stará verze 2.4.2 -> reset spojeni a invalid address.\n- Dalsi krok: prepnout XMRig na 6.21.3 a pouzit platnou ZION adresu (shodnou s poolAddress).\n- Vedlejsi: zvazit validaci adresy v uzi coin def., prip. povolit test bez plne validace pro dev.\n- Logy: zion-uzi-pool (OK), zion-rpc-shim (OK), seed1/2 healthy.


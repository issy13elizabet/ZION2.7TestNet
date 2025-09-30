# Changelog

All notable changes to this project will be documented in this file.

The format is inspired by Keep a Changelog and uses semantic-ish versioning adapted to miner evolution.

## [1.3.0] - 2025-09-28
### Added
- Benchmark metrics: average, best, standard deviation, baseline delta (%), toggle via `[b]`.
- GPU mining scaffolding with device detection and algorithm switch placeholder `[g]` / `[o]`.
- Extended RandomX initialization flags: huge pages, JIT, secure mode, full-mem dataset with graceful fallback.
- Thread affinity option `--pin-threads` for more consistent CPU hashrate on NUMA / multi-CCX systems.
- Hybrid share submission queue: lock-free ring buffer + fallback mutex queue.
- New CLI flags for RandomX tuning (`--rx-hugepages`, `--rx-jit`, `--rx-secure`, `--rx-full-mem`).

### Changed
- Optimized CPU hashing loop: buffer reuse, batched nonce hashing (128), minimized atomic ops, fast hex encoder.
- UI banner and controls updated (new toggles + clearer stats panel layout).
- Build system: optional LTO/IPO + `-march=native` controlled via CMake options.

### Performance
- Reduced per-hash overhead (string formatting & atomics) leading to higher effective H/s especially on >8 core systems.
- Lower contention on share submission under high valid-share scenarios.

### Notes
- GPU hashing currently simulated; real kernel integration targeted for 1.4.0.
- Baseline metric initializes after first measurement window.
- Huge pages/JIT auto-fallback prevents crashes on unsupported environments.

## [1.2.0] - 2025-09-27
### Added
- Real RandomX + Stratum integration (subscribe, authorize, job, submit share).
- Per-job seed reinitialization for RandomX dataset.
- Share acceptance / rejection tracking & UI table columns (FOUND / ACCEPT / REJECT).
- Stats aggregator collecting per-thread hashrate & recent share events.
- Interactive XMRig-style console UI with key controls: `[s]` stats toggle, `[h]` detail toggle, `[g]` GPU toggle, `[o]` algorithm cycle, `[q]` quit.

### Changed
- Removed simulated accept/reject logic (all shares now real according to returned job & target mask).
- Refactored mining core separation from legacy simulation path.

### Notes
- GPU path still placeholder; CPU RandomX baseline established for future optimization.

## [1.1.0] - 2025-09-26
### Added
- Initial RandomX core integration (prototype) & basic UI improvements.

## [1.0.0] - 2025-09-25
### Added
- Initial project structure & placeholder mining loop.

---

Future roadmap highlights:
- 1.4.0: Real GPU kernels (CUDA/OpenCL), precise difficulty normalization, adaptive autotuning.
- 1.5.0: Multi-algorithm pluggable pipeline + remote monitoring API.

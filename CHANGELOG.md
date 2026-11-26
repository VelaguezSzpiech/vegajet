# Changelog

## v0.1.1 (2026-11-27)

A source release. Measurements were taken on 3× Instinct MI50 32 GB running Ornith-122B; see the README's
three-card scorecard for the conditions.

### Added
- Three-card GPU execution (`--tp 3`) with bandwidth-proportional shards: a TP=3 N-way BAR0 AllReduce, the
  speculative-verify epilogue prepush (`VEGAJET_TP3_EP`, on by default) and dispatch-thread pinning
  (`VEGAJET_DISPATCH_CPUS`) with hardware preflight checks.
- A persistent GQA KV-quant super kernel (`VEGAJET_KV_QUANT=q8_0|q4_0|turbo3`). `q8_0` runs at the speed of the
  unquantized cache at short context (173.7 tok/s).
- Independent batched decode rows (`VEGAJET_GPU_BATCH=2..8`, `--bench-mode batch`): rows agree, and row 0 matches
  raw decode 129/129. Aggregate throughput is 1.23× raw at B=4.
- Diagnostics: `VEGAJET_SUPER_PROFILE=2` (critical-path stage stamps over all workgroups), `VEGAJET_VERIFY_PROFILE`,
  `VEGAJET_MEGA_PROFILE`, per-verifier `spec_cycle_ms` breakdowns, `VEGAJET_RAW_TOKEN_REFERENCE` (token references
  that persist across engine restarts) and `VEGAJET_BENCH_CONTEXT`.
- Experimental, off by default: `VEGAJET_AR_FUSED`, `VEGAJET_TOP8_WG0`, `VEGAJET_GDN_VSHARD`, `VEGAJET_MEGA`,
  `VEGAJET_VERIFY_N3`, `VEGAJET_SUB4_DOT8` / MiLo / `VEGAJET_TCC_XOR`, `VEGAJET_SPEC_ACCEPTMOE`,
  `VEGAJET_SPEC_EVICT`, `VEGAJET_OFFLOAD_EXPERT_FRAC`, and an opt-in `_x` super-kernel module for experiments. The
  default super kernels are unchanged.

### Changed
- On by default: the zero-tax suffix cascade (Hybrid 338.9 -> ~368 tok/s), the LumoTree GDN verifier
  (-> ~382 tok/s) and `VERIFY_FAST` (-> 390.5 tok/s). All three match raw decode 129/129 on repeated code.
- Version string: `vegajet 0.1.1`.

### Fixed
- Warm-engine variants rebuild options from the post-environment load options. Before this fix,
  `VEGAJET_GDN_VERIFY=lumotree` page-faulted on unallocated ring states.
- AcceptMoE keeps exact routing for 2-8-row ring batches and for prompt prefill again.
- Batched decode rows: fixed row-logit indexing and GDN state ordering. Before the fix, rows could disagree.
- The super-kernel sense counter wraps below the flag bits.

### Measured and left off
The README lists results for `AR_FUSED`, `AR_TILE_PIPE`, `TOP8_WG0`, `MEGA`, `SUB4_DOT8=q3`, `SPEC_ACCEPTMOE`,
`VERIFY_N3` and the KV-quant formats.

## v0.1.0-alpha.1

First public alpha: two MI50 32 GB cards (`--tp 2`). See `release/ALPHA_NOTES.md`.

<!-- SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 -->
# VegaJet Fast-Build, Hot-Reload & Layered Testing Framework

VegaJet uses a permanent 3-tier testing workflow backed by modular `.hsaco` compilation, zero-reload executable code-object hot-reloading, a 1-layer golden hardware fixture, and a standardized multi-agent handoff contract.

## 1. Modular `.hsaco` Build Layout

The Ornith GPU kernel suite is split into focused `.hip` modules sharing inline device primitives in `ops/kernels/vj_ornith_common.hiph`:

- `ops/kernels/vj_ornith_ar.hip` (`vj_ornith_ar_hsaco`): token embedding, RMSNorm+Q8, and Push-Only Dual-BAR0 P2P AllReduce kernels
- `ops/kernels/vj_ornith_gemm.hip` (`vj_ornith_gemm_hsaco`): asymmetric `VJ_Q4_A8` GEMV/GEMM kernels
- `ops/kernels/vj_ornith_attn.hip` (`vj_ornith_attn_hsaco`): GDN recurrence and GQA FlashAttention kernels
- `ops/kernels/vj_ornith_moe.hip` (`vj_ornith_moe_hsaco`): MoE router, SwiGLU, merged/grouped IQ2_S & IQ3_XXS kernels
- `ops/kernels/vj_ornith_moe_m1.hip` (`vj_ornith_moe_m1_hsaco`): fused `M=1` MoE front/back work-stealing kernels
- `ops/kernels/vj_ornith_vocab.hip` (`vj_ornith_vocab_hsaco`): TP-sharded LM head argmax/logits and MTP norm-concat kernels

CMake auto-enables `ccache` when installed and supports `-G Ninja` alongside Unix Makefiles. Every `<name>_hsaco` target compiles ONLY that single module and runs `vj_hsaco_gate` (0 B spill, VGPR class ceilings, `<24,576 B` code size) and `vj_isa_scan` (`lds_dword=0`).

## 2. Three-Tier Testing Workflow

### Tier 0 — CPU Math Oracle (ASan / UBSan)
Before touching GPU kernels, validate new quantization layouts, routing policies, eviction controllers, or arithmetic invariants in CPU unit tests under `build-cpu` (`ctest --test-dir build-cpu -R <test_name>`).

### Tier 1 — 5-Second Single-Module GPU Gate (`tools/vj_dev.sh`)
Capture a reusable 2-layer golden fixture (Layer 0 GDN+MoE + Layer 3 GQA for `M = 1, 5, 16, 128, 512`) once into an external gitignored path:
```bash
./build-gpu/vegajet bench "$VEGAJET_MODEL_PATH" --dump-golden-layers "$VEGAJET_GOLDEN_DIR"
```
Then iterate in ~5 seconds per kernel edit:
```bash
tools/vj_dev.sh vj_ornith_moe_m1 --stage moe --m 1,5,16
```
`vj_dev.sh` compiles ONLY the target `.hsaco`, enforces `vj_hsaco_gate` + `vj_isa_scan`, executes `vegajet_layer_test` against `$VEGAJET_GOLDEN_DIR` with `s_memrealtime()` timing and 6-block RAS ECC checks (`0 UE / 0 DE`, `TMem < 65 C`), and prints a 6-line status + JSON scorecard.

### Tier 2 — Warm 48-Layer Hot-Reload & `--sweep` (`tools/vj_bench_warm.sh`)
Eliminate cold 42 GiB model reloads during multi-variant sweeps and iterative kernel tuning:

- **One-shot multi-variant A/B sweep in a single process**:
  ```bash
  ./build-gpu/vegajet bench "$VEGAJET_MODEL_PATH" --draft "$VEGAJET_DRAFT_PATH" \
      --sweep "--gu-grp-verify --down-grp-verify 1 ; VEGAJET_MOE_LRU=32 VEGAJET_EVICT_ADAPT=1"
  ```
- **Persistent warm interactive session with `.hsaco` hot-reload**:
  ```bash
  mkfifo /tmp/vj.fifo
  ./build-gpu/vegajet bench "$VEGAJET_MODEL_PATH" --interactive-fifo /tmp/vj.fifo > /tmp/vj_warm.log 2>&1 &
  VEGAJET_WARM_LOG=/tmp/vj_warm.log tools/vj_bench_warm.sh /tmp/vj.fifo reload vj_ornith_moe_m1
  VEGAJET_WARM_LOG=/tmp/vj_warm.log tools/vj_bench_warm.sh /tmp/vj.fifo run --bench-mode raw
  VEGAJET_WARM_LOG=/tmp/vj_warm.log tools/vj_bench_warm.sh /tmp/vj.fifo parity
  ```
`Runtime::reload_hsaco()` drains all AQL queues, validates & stages the updated `.hsaco` (re-running the MessagePack `private_segment_fixed_size == 0` gate), and swaps executable code BOs atomically without touching the resident weight/KV/scratch VRAM arena.

### Tier 3 — Once-Per-Commit Full Gate (`tools/vj_commit_gate.sh`)
Run strictly once right before `git commit`:
```bash
tools/vj_commit_gate.sh
```
Builds `build-gpu` + `build-cpu`, runs full `ctest --output-on-failure` on both, runs `tools/privacy_audit.sh`, executes `vegajet doctor`, and verifies `0 UE / 0 DE` across all 6 RAS ECC blocks.

## 3. Multi-Agent `HANDOFF.json` Contract

Sidecar workspaces publish a 10-line `HANDOFF.json` at their workspace root:
```json
{
  "status": "READY",
  "base_sha": "<short-or-full-commit-sha>",
  "patches": ["patches/feature_a.patch"],
  "files": ["core/new_module.hpp", "core/new_module.cpp"],
  "gates_passed": ["build-cpu", "ctest", "privacy_audit"],
  "summary": "One-line description of verified changes"
}
```
Validate and apply with:
```bash
tools/apply_handoff.sh <sidecar_dir> --dry-run --require-base-sha "$(git rev-parse --short HEAD)"
tools/apply_handoff.sh <sidecar_dir>
```
`status` must be `"READY"` (otherwise `apply_handoff.sh` exits non-zero without touching the working tree).

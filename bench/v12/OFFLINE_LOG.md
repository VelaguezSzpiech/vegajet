# v1.2 offline verification log (VM without GPU)

| Date | Commit | Check | Result |
|---|---|---|---|
| 2026-11-26 | b741ae4 (main) | `cmake -DVEGAJET_CPU_ONLY=ON` + `ctest` | 100% tests passed, 0 failed out of 90 (vegajet_layer_test skipped: needs GPU) |
| 2026-11-26 | b741ae4 (main) | GPU build, TheRock ROCm 7.12.0a20260311 gfx906 (HIP 7.12.60610), `vj_hsaco_gate` + `vj_isa_scan` | 22/22 code objects PASS, 0 B spill |
| 2026-11-26 | b741ae4 (main) | same, ROCm 7.14.0a20260612 (for reference) | 2 FAIL: `ornith_gemm_q4a8_tcc_kernel` 75 > 73 VGPR waiver; `vj_vm_ffn_block` 12 B private segment. Compiler-version drift, not a code change. Use 7.12. |

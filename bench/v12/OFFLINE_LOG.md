# v1.2 offline verification log (VM without GPU)

| Date | Commit | Check | Result |
|---|---|---|---|
| 2026-11-26 | b741ae4 (main) | `cmake -DVEGAJET_CPU_ONLY=ON` + `ctest` | 100% tests passed, 0 failed out of 90 (vegajet_layer_test skipped: needs GPU) |

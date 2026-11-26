# v1.2 review remediation — v12/fixes

This worktree starts at release/v1.2 `3bc0015` (review findings were pinned to `5becd47`). No GPU is available in the VM. CPU tests and a gfx906 compilation/resource gate are not GPU execution, token parity, throughput, or release approval.

## Disposition

| Finding | Status | Implementation commit | Change / remaining work |
| --- | --- | --- | --- |
| B1 | Fixed | `8a3b189` | Latch faults before releasing pinned fault storage. |
| B2 | Fixed | `16f529a` | Explicit 56/32/32-byte super-kernel arguments with trailing null fault pointers in the layer harness; host-compiled ABI assertions. |
| B3 | Fixed | `b96674f` | Allocate each logical collective's epoch and parity once, outside the rank loop. |
| B4 | Fixed | `eae4d56` | Sticky typed failure unwinds all generation drivers before callbacks, commits and catch-up; stops AQL queues and propagates serving worker failure to the requesting thread. Validate argmax before LRU repacking. Check direct super-profile dispatches. Shared cleanup owns pinned storage even in dump modes. |
| B5 | Fixed | `1fd1bc3` | Enforce KV spans and actual row/chunk/head scratch capacity before split-attention launches; overflow-safe host arithmetic. Oversized requests fail instead of falling back silently. |
| M1 | Fixed | `9266dc1` | Estimate conditional acceptance only from reached decisions: trials = accepted + first rejection, if any. |
| M2 | Not fixed | — | Complete proposal/verification/catch-up cost instrumentation and a warmed plain-decode decision require a broader controller/driver change, not a small safe correction. Do not advertise C7 as an end-to-end throughput optimizer. |
| M3 | Fixed (scope/reject alternative) | `b0a927c` | Reject D13 on the active fused super path, report actual persistent-kernel launches and reject zero-launch runs. Test BOTH D13 and its standalone baseline with `VEGAJET_ENABLE_SUPER=0`; this does not establish a win against fused decoding. |
| M4 | Fixed | `9e879a0` | All-off loader never opens experimental super objects. Explicitly requested variants remain required; conflicting variants fail. |
| M5 | Fixed (synchronous alternative) | `f6ec6e4` | Complete each batch before shared token/hidden staging is reused. Scalar fallback completes each token before reusing slot zero. This can reduce S2 overlap, but cannot alter all-off scheduling. |
| M6 | Partially fixed | `f50229e` | Clear every inherited VEGAJET_V12_* variable per invocation, record selected environment, preserve actual failure status, fail the runner on warmup/timed failures, and match standalone D13 controls. Separately identified main/all-off binaries and automated complete token-ID comparison are still missing; the runner explicitly disclaims this evidence. |
| M7 | Fixed | `9664983` | Batched EH launches M workgroups along X, matching blockIdx.x row indexing. |
| N1–N3 | Not changed | — | Minor findings were outside this blocker/small-major repair scope. |

## Tests and evidence boundary

- `test_v12_dispatch`: epoch/parity allocation, KV and scratch boundaries including integer overflow, experimental module selection, EH row mapping, and sticky exception behavior preventing simulated post-failure driver work.
- `test_v12_fix_contracts`: CPU source contracts for GPU-only orchestration/lifetime/ABI call sites, plus explicit mock-CLI tests of runner isolation and failure exit status. These are source contracts/mocks, not GPU fault injection or kernel execution.
- `test_spec_governor`: identical conditional acceptance at proposal depths 1/4/8/15.
- Final CPU log: `build-cpu/final-ctest.log`; final GPU compilation log: `build-gpu/final-build.log`. Both are produced after the final commit; inspect the logs/final task response for observed results.

Required final commands (all in this worktree, `-j2`):

```sh
cmake -S . -B build-cpu -G Ninja -DVEGAJET_CPU_ONLY=ON
cmake --build build-cpu -j2
ctest --test-dir build-cpu -j2 --output-on-failure
cmake -S . -B build-gpu -G Ninja -DVEGAJET_CPU_ONLY=OFF -DVEGAJET_ROCM_PATH=/data/tools/therock712/lib/python3.14/site-packages/_rocm_sdk_devel
cmake --build build-gpu -j2 -- -k 0
```

Still required on 2× MI50: injected fault tests across raw/suffix/MTP/DFlash/serve and early dump modes; S2 scalar-vs-batched KV comparison with 64 distinct prompt IDs and partial final batches; split-attention red zones at 16×60 (accepted) and 16×61/64 (rejected); healthy all-off/main complete token and speed comparison. For the S2 checks, enable `VEGAJET_V12_BATCH_CATCHUP=1` with the same `--draft <MTP model> --mtp-k 4` configuration; add `--spec-flash --ctx 16384` for the scratch rejection boundary. Run `vegajet_layer_test --stage super_m1 --m 1` against the existing golden fixtures for the ABI repair. No runtime or performance result is claimed here.

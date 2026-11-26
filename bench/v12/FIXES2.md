# v1.2 second-review fixes

CPU tests and ROCm 7.12 gfx906 resource gates are not GPU execution or token parity evidence.

## R2-B1 — asymmetric experts + batched MTP

Startup rejects `VEGAJET_V12_ASYM_EXPERTS=1` together with
`VEGAJET_V12_BATCH_CATCHUP=1` before initializing the runtime or uploading weights.
The batched Down ABI remains 512-only. Either toggle alone is unchanged; both default off.
`test_v12_dispatch` covers all four flag combinations.

## R2-B2 — fused verify norm/quant

`VEGAJET_V12_VERIFY_FUSED_NQ` remains default off. The fused path now requires
known wave64 kernel metadata and enough actual GPU CU/register/LDS/descriptor-wave
capacity for all 128 256-thread WGs. Unknown/insufficient metadata falls back to
the existing two-dispatch verify+norm path independently per rank. Sense allocation
is independent of whether a rank falls back.

The barrier returns a WG-uniform failure on timeout, skips norm/quant, and writes
the existing C1 host-visible fault ABI (kind=barrier, barrier=5, stage=1, epoch=sense).
The existing sticky driver check aborts before tokens/callbacks/state commits.
C1's writer is shared with super kernels without changing its healthy behavior.
CPU tests cover residency selection and GPU-only source contracts, not timeout execution.
On MI50, compare full token IDs and recurrent state for the toggle off/on across
accepted and rejected speculative batches. Inject a missing barrier arrival in an
instrumented build and require a device-fault error and no token callback/commit.

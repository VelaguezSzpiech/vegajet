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

## RT-B1 — readiness epoch reuse

Use the review's correctness-first host-completion alternative: before the next
slice of one collective, submit both ranks and wait for BOTH queues to drain.
A failed submission/completion/timeout becomes a sticky run failure. Maximum
producer lead at readiness-word reuse is zero slices. Payload and flag layout
and the kernels are unchanged. No extra submission/wait is added for a single
slice (FP16 <=128 rows, INT8 <=512); multi-slice prefill has a new host round trip
per non-final slice and may be slower. This correctness repair is always on.

CPU coverage checks boundaries including partial slices/overflow, and enumerates
all publication/observation schedules for two and three slices: the old protocol
has deadlocked states; the completion-gated model has none. This is not GPU proof.

### GPU instructions (2 x MI50)

From this worktree, set `MODEL` to the supported Ornith GGUF. Use a prompt that
tokenizes to at least 1024 IDs (verify the reported prompt-token count):

```sh
PROMPT=$(python3 -c 'print("The quick brown fox jumps over the lazy dog. " * 256)')
for chunk in 128 256 512 1024; do
  VEGAJET_ENABLE_SUPER=0 timeout 1800 ./build-gpu/vegajet run "$MODEL" --tp 2 \
    --hsaco-dir ./build-gpu/hsaco --ctx 4096 --prompt "$PROMPT" --n-predict 32 \
    --no-fast-prefill --prefill-chunk "$chunk" --ar-wire fp16 \
    --dump-hidden "rt-b1-fp16-$chunk.hidden" > "rt-b1-fp16-$chunk.log" 2>&1 || exit 1
done
for chunk in 512 1024; do
  VEGAJET_ENABLE_SUPER=0 timeout 1800 ./build-gpu/vegajet run "$MODEL" --tp 2 \
    --hsaco-dir ./build-gpu/hsaco --ctx 4096 --prompt "$PROMPT" --n-predict 32 \
    --no-fast-prefill --prefill-chunk "$chunk" --ar-wire int8 \
    --dump-hidden "rt-b1-int8-$chunk.hidden" > "rt-b1-int8-$chunk.log" 2>&1 || exit 1
done
```

Run repeatedly with all inherited VEGAJET_V12_* variables cleared. Require every
run to finish without a timeout/device error and no token/hidden corruption.
Compare chunked results to the same wire mode's single-slice reference; use the
project's existing FP numerical tolerances for differing batch GEMM shapes, not
a cross-wire bit-identity assertion. Record prefill timings: multi-slice latency
can increase; single-slice decode scheduling must not gain host waits.
For the exact adversarial case, in a temporary instrumented GPU build delay rank
B after publishing readiness but before observing its peer's epoch; run batches
of 256 and 384 FP16 rows (two/three slices) through the runner/harness. Confirm A
cannot publish its next slice until B has consumed the prior one. An intentionally
non-returning B must surface a host completion timeout, not return output tokens.
The CPU schedule regression covers that delayed-B ordering without hardware.

## P-M3 — expert payload validation

All 48 trunk layers and the active optional MTP layer are validated before runtime
initialization and before expert payload mapping/reads. The GPU runner explicitly
requires D=3072, n_ff_exp=1024, n_expert=256 for its fixed-layout expert kernels.
Gate/Up must be IQ2_S with 984-byte rows; Down must be IQ3_XXS with 392-byte rows.
Require exact [input, output, expert] axes (or equivalent flattened 2-D rows),
including a unit unused fourth axis and exact payload byte count. MTP's generic
uploader accepts registered decodable/block-aligned types, with the same shape
checks, rather than needlessly constraining its existing source quantization.
Errors include the tensor name and expected layout/type. No toggle: malformed or
unsupported input fails closed; healthy model bytes and tokens are unchanged.

`test_ornith_expert_layout` covers canonical/flat layouts, wrong/missing types,
small declared tensors, n_ff_exp disagreement, wrong/swapped axes with equal byte
counts, extra dimensions, and overflow-sized axes. CPU source contracts pin the
check before runtime initialization and payload access; no GPU is needed to run
these validators. On MI50, mutate one expert descriptor in a copy of the model
(type, n_ff_exp, axis, or tiny extent) and require the named error without GPU
allocation/upload. The unchanged healthy file must still load and match tokens.

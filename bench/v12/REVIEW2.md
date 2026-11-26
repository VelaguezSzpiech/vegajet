# VegaJet v1.2 — group-2 review

## Scope and verdict

**Do not enable the affected combinations for release: there are two new blockers.** D1 leaves batched MTP on the old rank/layout contract, and S5's new grid barrier silently proceeds on timeout. S5 also changes the disabled path, and the D6 toggle is bypassed by the normal monolithic decode route.

Reviewed the 20 non-documentation files in `git diff 5becd47..847607f`: `16ea7f3` (D6), `0352189` (S4), `310b4a8` (S5), `ebfd421` (D1), and `847607f` (S3). Read `bench/v12/REVIEW.md` to exclude its findings. The checked-out HEAD was `3bc0015`; its only change after `847607f` is that earlier review, so source line numbers below are also valid at the requested endpoint. No source files were edited and no commit was made.

This is a static source/ABI/layout review, supplemented by read-only inspection of existing HSACOs and execution of existing CPU reference-test binaries. **No GPU execution, token parity, GPU bit-exactness, timeout injection, or throughput is established.** Existing build products were not rebuilt or independently provenance-verified; their metadata is evidence about those files, not a guarantee about every compiler/build.

## Findings

### R2-1 — BLOCKER — D1 changes MTP storage per rank but leaves batched catch-up using the maximum width and the 512-only Down kernel

**Files/lines:** `models/ornith/ornith_gpu.cpp:1097–1100,1566–1613,1847–1848,1967,4010–4035`; `ops/kernels/vj_ornith_moe.hip:855–925`.

**Introduced by:** `ebfd421`. Trigger: `VEGAJET_V12_ASYM_EXPERTS=1`, an unequal split, MTP present, and `VEGAJET_V12_BATCH_CATCHUP=1` selecting `step_mtp_batch`.

**Quotes:**

```cpp
// Capacity planning uses the larger shard; launches use the rank's width.
    const uint32_t local_ff = std::max(ff_size[0], ff_size[1]);
```

But the `step_mtp_batch` expert loop has no per-rank shadow:

```cpp
for (size_t g = 0; g < 2; ++g) {
    const auto& ml = mtp[g];
    ...
    SwiKA ska{..., M, local_ff};
    ...
    GuQ4KA gka{..., M, local_ff, ml.w_exp_gate.row_stride_bytes};
    ...
    struct DnQ4KA { uint64_t wdq, wddm, mq, mds, tids, tw, shd, out; uint32_t M, D, stride; };
    ...
    VJ_CHECK(rt.enqueue_kernel(g, *K[g].moe_down_q4a8, ...));
}
```

The selected legacy Down kernel contains:

```cpp
sm_mq[threadIdx.x] = reinterpret_cast<const uint4*>(mid_q8 + static_cast<uint64_t>(m) * 4096u)[threadIdx.x];
if (threadIdx.x < 128u) sm_mds[threadIdx.x] = mid_ds[static_cast<uint64_t>(m) * 128u + threadIdx.x];
...
const uint32_t* dm    = wd_dm + erow * 16ULL;
```

**Why:** D1 correctly changes MTP Gate/Up uploads to `256 * ff_size[g]` rows and Down to `ff_size[g]` columns, and allocates rank-specific scratch. It updates scalar `step_mtp`, but not `step_mtp_batch`. For the documented 480/544 split, both batch launches use 544:

* Rank 0's Gate/Up buffers contain 122,880 rows. Expert 255 is addressed as rows `[138720,139264)`, beyond that allocation; even ordinary expert IDs select the wrong rows. Shared SwiGLU also misidentifies its Gate/Up boundary and token stride.
* `mtp_moe_ds_va` is allocated for 16 tokens, eight experts, and 15 blocks on rank 0: 15,360 bytes. The 544-wide batch kernel writes 17 blocks: 17,408 bytes at M=16, **2,048 bytes beyond this allocation**. This bound does not depend on the larger general-purpose prefill workspace.
* Both ranks still run a Down consumer hard-coded to 512, including a 16-word scale-row stride instead of the uploaded 15 or 17. Fixing only the batch loop's width does not fix Down.

The existing S2 synchronization blocker can mask this defect; it is not repeated here. Fixing S2's epochs alone would expose a still-invalid D1+S2 data path. The asymmetry CPU test exercises slices/plans, not this launch-argument composition.

**Concrete fix:** Add `const uint32_t local_ff = ff_size[g]` to the batch expert loop and mirror the scalar MTP Down dispatch: when asymmetric, pass the actual width in the extended kernarg and select `moe_down_asym_q4`; otherwise preserve the original kernarg/kernel. Audit every batch producer/consumer stride together. Until that is implemented, reject this toggle combination before allocation/submission or explicitly fall back to scalar catch-up. Add a CPU launch-plan test for both ranks at 480/544, reverse the split, exercise M=2 and M=16 and expert 255, and assert every buffer range and Down scale stride.

### R2-2 — BLOCKER — S5's new grid barrier fails open on timeout and consumes potentially unwritten output

**Files/lines:** `ops/kernels/vj_spec_gdn.hip:285–304,446–466`; host launch at `models/ornith/ornith_gpu.cpp:2701–2716`.

**Introduced by:** `310b4a8`. Trigger: `VEGAJET_V12_VERIFY_FUSED_NQ=1` with `--spec-gdn`, if not all 128 workgroups reach the barrier before a wait expires.

**Quote:**

```cpp
for (uint32_t spin = 0u; spin < 5000000u; ++spin) {
    if (ld_glc_u32(&bar[16]) == sense) break;
    __builtin_amdgcn_s_sleep(2);
}
...
grid_barrier_wbinvl1(nq_bar, gridDim.x, sense);
...
const float fo0 = __builtin_nontemporal_load(ob + j0);
```

**Why:** Exhausting the loop is indistinguishable from observing release. The helper returns `void`, sets no fault, and all callers proceed into norm/quant. That phase remaps workgroups to different heads/tokens and can read another workgroup's `o_out` before it has been produced. The result is stale/uninitialized INT8 activations, not a clean error. Delayed arrivals can also leave the shared arrival count/release sense inconsistent with the next invocation.

The finite loop means this is not an unconditional infinite-spin assertion. Under insufficient residency it can instead produce a very long stall followed by **silent computation with incomplete inputs**. Other delays or an injected missing participant have the same fail-open outcome. This is a new barrier implementation and new failure path, separate from the super-kernel fault-propagation findings in REVIEW.md.

**Residency assessment:** The inspected fused HSACO reports 49 VGPRs, 76 SGPRs, 20,736 bytes LDS and zero private bytes. Using the repository's gfx906 occupancy calculation gives three 256-thread workgroups/CU, or 180 slots on a 60-CU MI50: the 128-workgroup grid fits. Thus **I did not find a deterministic residency deadlock for that binary on the stipulated otherwise-available MI50**. Nevertheless, the host checks only symbol/scratch availability; the comment and `__launch_bounds__(256,3)` are not a run-time residency/admission check and do not make timeout fall-through correct.

**Concrete fix:** Use a checked barrier that publishes a device fault on timeout and returns a workgroup-uniform success flag; skip norm/quant and subsequent state commit on failure, and make the host report a fatal error before emitting tokens. Check occupancy using the loaded kernel's actual VGPR/SGPR/LDS metadata and detected CU count, falling back to the existing two-dispatch path if the full grid cannot be admitted. Validate the exact grid shape and add reduced-spin/missing-participant GPU fault injection when hardware is available. The CPU arithmetic reference cannot test barrier safety.

### R2-3 — MAJOR — S5 allocates and clears its new barrier even when its toggle is off

**Files/lines:** `models/ornith/ornith_gpu.cpp:1938–1946,2063–2071`.

**Introduced by:** `310b4a8`. Trigger: existing `--spec-gdn` mode, with **all group-2 toggles unset or zero**.

**Quotes:**

```cpp
if (opt.spec_gdn) {
    ...
    S[g].spec_nq_bar_va = alloc(256u, 256);
    S[g].spec_gdn_rec_kv_va = alloc(..., 4096);
```

```cpp
if (S[g].spec_nq_bar_va) zero_buf(S[g].spec_nq_bar_va, 256u);
```

**Why:** Neither allocation nor reset is guarded by `VEGAJET_V12_VERIFY_FUSED_NQ`. The preceding `spec_gdn_o_va` allocation ends on a 4-KiB boundary, so inserting this 256-byte object before the next 4-KiB-aligned allocation advances the following workspace by **4,096 bytes per rank**, not merely 256. It also adds a GPU memset to every reset. Disabled execution therefore has a different allocation footprint, addresses, and reset dispatch stream, including a possible changed OOM outcome at the allocation boundary. This violates the stated disabled-path isolation requirement. It is **not** evidence of a healthy-run logit/token mismatch; none is asserted.

**Concrete fix:** Resolve the toggle before workspace construction, allocate/clear this scratch only when the fused feature is enabled and usable, and otherwise leave the VA zero. If in-process sweep enablement is supported, arrange explicit opt-in workspace reservation rather than charging disabled runs implicitly. Compare allocation records and reset dispatches against `5becd47` for both ordinary mode and `--spec-gdn` with every new toggle off.

### R2-4 — MAJOR — D6 is bypassed by the normal M=1 super-kernel route

**Files/lines:** `models/ornith/ornith_gpu.cpp:938–941,2489–2498,2514–2540,2652–2668`; default options in `models/ornith/ornith_gpu.hpp:34–36,44–45,58`.

**Introduced by:** `16ea7f3`.

**Quotes:**

```cpp
if (v12_gdn_splitv()) {
    K[g].gdn_conv_recur_ct = need_k("ornith_gdn_conv_recur_colthread_quant_m1_kernel");
}
...
const kfd::LoadedKernel* const cr_k =
    (K[g].gdn_conv_recur_ct != nullptr) ? K[g].gdn_conv_recur_ct : K[g].gdn_conv_recur_sv;
```

The earlier execution branch instead selects:

```cpp
const bool use_m1_super = (!asym_experts && ab_enable_super && ... && M == 1u && ...);
...
if (use_m1_super) {
    ...
    VJ_CHECK(rt.enqueue_kernel(g, *K[g].m1_gdn_super, ...));
}
```

**Why:** The D6 toggle substitutes only the standalone ConvRec kernel. Normal M=1 decode with the default fusion options satisfies `use_m1_super`; that route runs the integrated GDN implementation and never uses `gdn_conv_recur_ct`. Turning on D6 alone does not select the advertised column-thread implementation for that common baseline. Stage profiling disables the super route, so profiling can appear to exercise D6 while the normal throughput pass does not. This is a routing/measurement defect, not an arithmetic mismatch or an off-path regression.

**Concrete fix:** Either add and select a D6 super-kernel variant, or explicitly restrict D6 to standalone mode and reject/warn on an otherwise-super run. For standalone A/B testing disable the super route in **both** runs and report the actual selected kernel. Merely disabling the entire super route in the toggle-on run would confound the comparison with loss of fusion. Add host-only route-selection tests for default decode, profiled decode, and explicit standalone decode.

## Coverage, negative findings, and evidence limits

### Default-path audit

* **D1 sharder:** `core/sharder.cpp:47–53` returns the legacy two equal ranges before consulting bandwidth when disabled. For the supported 1024-wide experts this remains 512/512. The new Gate/Up source formula `expert * hp.n_ff_exp + ff_begin[g]` reduces to the old `expert * (2 * 512) + g * 512`; no off-path offset defect was found.
* **D1 repack/runtime:** The asym Down routines in `core/q4_dot8_repack.cpp` are additions, not rewrites of the legacy repacker. `ornith_gpu.cpp:1493–1505` enters the old Down repack branch when disabled. `core/kfd_runtime.cpp:1016–1019` loads `vj_moe_asym.hsaco` only for exact `"1"`. Both the host and loader use the same exact-string test. With equal widths, the rewritten promotion/LRU byte formulas and per-rank strides reduce to their old values. Scalar MTP preserves its legacy kernarg in the disabled branch. I found no D1 off-path numerical/layout regression in these changes.
* **S4 routing:** The new branch at `ornith_gpu.cpp:2415–2420` is wholly conditional on `v12_narrow_verify`. When false, branch order and launch arguments for prefill, m8, m12, m16 and TCC remain unchanged. It also leaves M=1 alone. On-path caveats: earlier prefill choices still win, width 512 is deliberately excluded, and the condition is not restricted to `use_ring`; small prefill batches can use it when enabled. These are not disabled-path regressions.
* **D6:** Disabled resolution retains the old standalone kernel; the new kernel is required via `need_k` only inside the enabled branch. No disabled arithmetic change was found. The enabled route problem is R2-4.
* **S5:** Disabled verify dispatch selection remains the old two-kernel pair, but allocation/reset isolation fails as described in R2-3.
* **S3:** `tree_verify_requested` requires exact `"1"`. Both the CPU-only stub and real `run_ornith_gpu` return an unsuccessful result with the explicit unavailable message. The real guard at `ornith_gpu.cpp:827–830` precedes preflight, model opening, tokenization, runtime initialization and allocations. There is no fallback claiming to be a tree verifier. Unset, `0`, `10` and `1extra` do not activate it. Its incompleteness is explicitly documented, not an additional defect.

### D1 layout/promotion/LRU audit

The 32-column Down slice uses `sb=(col_begin+b*32)/256` and `ib=((col_begin+b*32)%256)/32`; it stores the original subscale and superblock float scale for each 32 weights. This avoids incorrect 256-column rounding at splits such as 480/544. The row-range source pointer and destination plane indexing agree for chunked repacks. Existing CPU testing passed all 62 rank-specific slices, including padding/sentinels.

Gate/Up source indexing uses the full 1024-row expert pitch before selecting each rank's range; uploaded rows then use the rank-local pitch. Raw IQ2, Q3 and DOT8 promotion paths use local widths. LRU storage has rank-specific layouts and strides, and both the repack producer and cached-GU consumer use that rank's width while sharing only logical expert/slot identities. The max-width variable is appropriate for conservative capacity estimates, **not** the batch launch use in R2-1. The new generic Down kernels size LDS through the maximum supported 992-column shard; the s224 work assignment strides by 224 over both routed and shared work. No additional definite offset/coverage defect was found in those paths. This does not establish GPU reduction parity or an acceptable performance tradeoff.

### D6 arithmetic and register headroom

The new kernel preserves each lane's 16-row FMA chain, explicitly sums the eight group partials in ascending order (rather than a different butterfly), and keeps the old normalization/quantization order. The reference test has an order-sensitive negative control and passed its three bitwise stage checks; 13 of 24 alternative butterfly controls differed. No specific source-level bit-exactness counterexample was found.

The inspected standalone kernels report:

| Kernel | VGPR | SGPR | LDS bytes | Private bytes |
|---|---:|---:|---:|---:|
| Original ConvRec split-V | 40 | 48 | 2,304 | 0 |
| New column-thread ConvRec | 64 | 48 | 1,320 | 0 |
| S5 fused verify + norm/quant | 49 | 76 | 20,736 | 0 |
| D1 asymmetric s224 back | 63 | 58 | 13,504 | 0 |

D6 is **exactly at** the declared 64-VGPR tiled limit: there is no VGPR-count headroom, although this artifact has no spill. Keep the gate hard, rerun it for the exact deployed compiler/build, and inspect ISA when checking arithmetic parity. The reference test does not execute HIP code or establish compiler-generated bit-exactness; reduced LDS is not measured speed evidence.

### S3 host planning

Reviewed parent-before-child validation, sibling uniqueness, ancestor masks, convolution histories, greedy path acceptance, capacity/overflow checks, and serial KV compaction. Destination rows never exceed their increasing source rows, so the specified ascending serial copies do not overwrite future sources. Metadata is reconstructed and validated rather than trusted blindly. No additional concrete host-plan defect was found. GPU attention, branched GDN state and KV commit remain deliberately unavailable.

### Read-only observations

Executed the existing `build-cpu/` binaries directly:

* `test_asym_experts`: PASS; default 512/512, measured 480/544, 62 Down slices exact.
* `test_gdn_recur_colthread_ref`: three bit-exact stage checks; 13/24 butterfly controls differ.
* `test_gemv_q4a8_mn_ref`: 6,480 row outputs identical across the CPU order models; 4,232 split-K controls differ.
* `test_spec_gdn_ref`: all 38 checks passed. Its test sizes are M=1,5,8,12, **not 13–16**, and both norm/quant traversal models call the same arithmetic helper. It does not exercise grid synchronization.
* `test_spec_tree`: PASS for 720 layouts and the unavailable-toggle guard.

Read existing code-object metadata with `/usr/bin/llvm-readobj-18 --notes`; the existing `build-gpu/vj_hsaco_gate` also passed the inspected spec-GDN, attention and asymmetric-MoE objects (the attention object has its pre-existing WY-prefill waiver). These checks do not close R2-1/R2-2 or prove the all-off token/speed acceptance gate.

## BLOCKERS only

1. **R2-1 — D1 + batched MTP catch-up uses the maximum shard width on both ranks and the legacy 512-only Down kernel** (`models/ornith/ornith_gpu.cpp:1099–1100,4010–4035`). Unequal splits produce wrong expert/scale addressing and out-of-bounds accesses; adapt the entire batch ABI/layout or reject the combination.
2. **R2-2 — S5's new grid barrier silently falls through on timeout and normalizes potentially unwritten output** (`ops/kernels/vj_spec_gdn.hip:285–304,446–466`). Add checked, fatal timeout propagation and a residency-validated fallback before enabling it.

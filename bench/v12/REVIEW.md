# VegaJet v1.2 code review

## Verdict and evidence boundary

**Do not release this revision.** The all-off benchmark has a use-after-free, three GPU-test launch sites still use the old super-kernel ABI, and the new failure path does not reliably stop token production or downstream GPU work. S2 also has a deterministic two-rank deadlock and an unchecked scratch-capacity overflow.

Reviewed `git diff b741ae4..release/v1.2` and surrounding callers, runtime packet construction, kernels, and tests. Reviewed head: `5becd47591617f3850161d9aac305d1ab951f0e7`; base: `b741ae48599cdfc2a21fa0211b532bd68865fda8`. Line numbers below refer to that head. During final checks another actor advanced the branch/worktree to `847607f7dade71c4445377de396369d678923f13`. Final source checks therefore use `git show 5becd47591617f3850161d9aac305d1ab951f0e7:<file>`; this report does not certify the later, concurrently added commits. Findings include unchanged callers broken by this diff, explicitly identified where applicable.

Executed, without source changes:

```text
cmake -S /data/src/vegajet -B /data/src/vegajet/build-review -DVEGAJET_CPU_ONLY=ON
cmake --build /data/src/vegajet/build-review -j2
ctest --test-dir /data/src/vegajet/build-review --output-on-failure
```

Observed: `100% tests passed, 0 tests failed out of 96`; `vegajet_layer_test` was **skipped**. Total reported test time: 75.49 seconds. `git diff --check b741ae4..release/v1.2` produced no diagnostics. These results do **not** exercise GPU launch ABI, cache coherence, two-rank progress, GPU token parity, or speed. Small Python calculations used below are source-level counterexamples, not GPU execution results.

Severity convention: BLOCKER = release-stopping memory-safety, guaranteed hang, ABI break, or breach of the mandatory fail-closed contract. MAJOR = material correctness, measurement, or feature-functionality defect. MINOR = narrower diagnostic/test/observability defect. An opt-in feature is not exempt from BLOCKER classification when enabling it guarantees a hang or permits memory corruption.

## BLOCKER findings

### B1 — The normal all-off benchmark reads the fault buffer after freeing it

**Location:** `models/ornith/ornith_gpu.cpp:5841–5864`; supporting code `:2196–2199`, `core/kfd_runtime.cpp:636–652`.

**Quote:**

```cpp
rt.free_buffer(io_buf);
...
device_fault_check("run end");
...
if (device_faulted) return res;
```

**Why wrong:** `device_fault_check` dereferences `io_fault_host`, a pointer into `io_buf`. `free_buffer` frees or unmaps `host_ptr` and clears the buffer object, not that captured pointer. After the parity and telemetry gates pass, the fault check unconditionally reads that dangling pointer. This affects the ordinary end-of-benchmark path with every opt-in toggle off, and the common teardown reached after the serve hook returns. Depending on allocation backing/reuse, the result can be a crash, an invented fault, or undefined behavior that happens to appear successful. Main did not make this read.

**Concrete fix:** Inspect/latch the fault while the allocation is still live, then release buffers through a common cleanup path. Never invoke a reader of `io_fault_host` after cleanup. Add a teardown test that keeps fault checking before buffer destruction, and exercise a healthy GPU benchmark through its normal return.

### B2 — C1 changed the kernel ABI but missed all three layer-test launch structs

**Location:** `tests/vegajet_layer_test.cpp:739–742`, `:784–786`, `:820–822` (unchanged callers broken by this diff); changed signatures in `ops/kernels/vj_ornith_super.hip:723–734`, `:1012–1018`, `:1298–1303`; packet construction in `core/kfd_runtime.cpp:1433–1469`.

**Quote:**

```cpp
struct KA { uint64_t desc, ov; uint32_t pep, oep, ci, co, sh, ur, ss, gs; };
struct KA { uint64_t desc; uint32_t pep, oep, p0, ss; };
struct KA { uint64_t desc; uint32_t aep, oep, ss; };
// Kernel signatures now append uint64_t fault_va.
const size_t hidden_off = align_up(kernarg_bytes, 8ULL);
```

**Why wrong:** These callers still pass the old explicit-argument sizes: GDN 48 bytes and GQA/MoE 24 bytes, instead of 56/32/32. This is not merely an omitted optional null argument. The runtime writes hidden launch fields immediately after the caller's byte count; the new kernel interprets the first eight hidden bytes as `fault_va` and reads its actual hidden arguments eight bytes too late. For a 240-workgroup launch, packing the runtime's first hidden words gives the bogus fault pointer `0x1000000f0`. Grid-dependent barrier and work mapping can also be wrong even before a timeout attempts to write that address. The only relevant GPU test was skipped by the CPU suite.

**Concrete fix:** Append an explicitly initialized `uint64_t fault` to each test struct, preferably passing a real checked fault slot; at minimum pass zero for synthetic tests. Centralize these ABI structs in a shared host header and assert both sizes and field offsets. Verify every production, profiler, microbenchmark, and layer-test launch, plus hidden-argument placement. The production/profiler/microbenchmark structs in `ornith_gpu.cpp` were updated; these three were not.

### B3 — S2 assigns different AllReduce epochs and ping-pong slots to the two ranks

**Location:** `models/ornith/ornith_gpu.cpp:3831–3848` and `:3879–3896`; supporting wait in `ops/kernels/vj_ornith_common.hiph:290–296`.

**Quote:**

```cpp
for (size_t g = 0; g < 2; ++g) {
    const size_t peer = 1u - g;
    const uint32_t ep = ar_epoch++;
    const uint32_t pp = ep & 1u;
    ... // enqueue p2p_ar_rms with ep and pp
}
```

**Why wrong:** The same bug exists in both batched catch-up AllReduces. Rank 0 publishes epoch E and waits for E; rank 1 publishes E+1 and waits for E+1. Each receives the other's different epoch. Their payload ping-pong addresses also disagree. The standalone AllReduce kernel uses an unbounded exact-epoch wait, so the first affected exchange cannot complete. The host eventually times out in `wait_idle` and hits `VJ_CHECK`/`abort`, not the promised controlled error. The old scalar MTP helper correctly increments the epoch **outside** its rank loop at `:3588–3591`.

**Concrete fix:** Allocate one epoch and one parity per logical collective, outside each two-rank loop, and use them unchanged on both ranks. Add a CPU launch-descriptor test asserting matching peer epochs/payload slots and a real two-rank M>1 catch-up test. Do not infer S2 functionality from the CPU suite.

### B4 — C1/C3 return success-shaped `-1` predictions; callers still publish them and dispatch drafter work

**Location:** `models/ornith/ornith_gpu.cpp:2214–2230`, `:3157–3158`, `:3203–3211`; consumers `:3410–3455`, `:3490–3553`, `:4044–4114`, `:5663–5712`, `:5860–5865`; unguarded drafter entry `:3570–3573`.

**Quote:**

```cpp
if (device_faulted) return std::vector<int32_t>(M, -1);
...
return std::vector<int32_t>(M, -1); // invalid embedding input
...
gen.push_back(preds[0]);
...
gen.push_back(preds[acc]);
```

**Why wrong:** A nonempty vector still denotes a successful prediction to all these callers. Raw/suffix callbacks can receive token `-1`; benchmark loops count sentinel tokens and continue; speculative callers still commit recurrent state and run drafter catch-up. In the MTP path a failed verifier's `gen.back() == -1` can become the next `step_mtp` input. That function neither checks the sticky fault nor validates the token before embedding; `ornith_mtp_eh_norm_cat_q8_kernel` casts the token to `uint64_t` and addresses `emb_qs + tok * D` (`vj_ornith_vocab.hip:574–576`). Thus a new fail-closed condition can lead to further invalid GPU memory access.

There is a second consequence for C3 specifically: it only sets `res.error`, not `device_faulted`. After correcting B1, the normal return still executes `res.ok = true` whenever the fault slot is clean, even when a C3 error was recorded. Error text alone does not keep this function on an error return as its comment claims.

**Concrete fix:** Return an explicit step status/result (or a controlled exception caught at the request boundary), and terminate every generation/verification loop immediately on failure, before callbacks, state commit, or drafter dispatch. Check input tokens and the sticky failure at all embedding/drafter entry points. Set success only when no error is present. Preserve the first diagnostic, drain/destroy queues safely, and return the documented error/exit status. Test fault and invalid-argmax injection in raw, suffix, MTP, DFlash2, and serve, not merely the host fault-record helpers.

### B5 — S2 does not bound attention scratch by its actual chunk count

**Location:** `models/ornith/ornith_gpu.cpp:1869–1871`, `:3810–3821`; addressing in `ops/kernels/vj_spec_flash.hip:109`, `:192–196`.

**Quote:**

```cpp
S[g].spec_fl_part_o_va = alloc(kMaxSpecBatch * 16ULL * 60ULL * 256ULL * 4ULL, 4096);
...
if (opt.spec_flash && K[g].spec_flash_part) {
    const uint32_t span = pos0 + M;
    const uint32_t chunk_len = (span <= 1920u) ? 32u : ((span <= 3840u) ? 64u : ((span <= 7680u) ? 128u : 256u));
    const uint32_t n_chunks  = std::max<uint32_t>(1u, (span + chunk_len - 1u) / chunk_len);
    // M and n_chunks are passed without a capacity check.
}
```

**Why wrong:** Allocation reserves `kMaxSpecBatch * 60` row-chunk units per head, but the kernel's flattened index is `(m * n_q_local + h) * n_chunks + c`. When S2 is combined with `--spec-flash`, it uses this split-attention path for M-row batches. Without `--spec-flash`, the code does have an ordinary-attention fallback; this finding is specifically about the enabled split-attention combination. At M=16 and span=16384, `nc=64`: required row-chunk capacity is 1024 versus the allocated 960. The writes run past both partial-output/statistics allocations into adjacent VRAM. Larger contexts can exceed capacity with smaller batches too. A handoff instruction to test short contexts is not an executable guard, and the configured context limit is not restricted here to a safe capacity. This remains a defect after fixing B3.

**Concrete fix:** Size scratch from the maximum supported context and actual row/head/chunk dimensions, or enforce `M * nc <= kMaxSpecBatch * 60` and fall back to a safe scalar/chunked implementation before launch. Check multiplication/overflow and KV position bounds together. Add boundary descriptor tests and GPU red-zone tests at the first exceeding context/batch combination.

## MAJOR findings

### M1 — The new C7 acceptance estimator feeds a prefix fraction into a conditional-probability model

**Location:** `core/spec_policy.cpp:150–160`, `:218–219`; reward model `:111–125`.

**Quote:**

```cpp
return std::clamp(static_cast<double>(acc) / static_cast<double>(prop), ...);
std::vector<double> cond(cap, alpha());
const EvictDecision ev = plan_evict(cond, cm, ..., 0.0);
```

**Why wrong:** `accepted` is an accepted **prefix length**, not the number of successes from `proposed` independent verification decisions. Tokens beyond the first rejection were never conditional acceptance trials. `plan_evict` then multiplies these already-censored ratios as if they were per-position conditional probabilities, penalizing deep proposals twice. A source-level calculation with true conditional acceptance 0.9 and K=8 gives expected accepted length 5.12579511. This estimator converges to 0.64072438875 and predicts reward 2.7327244883 instead of 6.12579511. Merely changing proposal depth changes the estimated quality of the same drafter, creating feedback toward shallow depths.

**Concrete fix:** Estimate a conditional acceptance hazard from actual reached decisions (for a homogeneous model, successes `a`, trials `a + (a < k)`), or retain per-position prefix-survival statistics and compute expected accepted tokens directly. Test identical underlying acceptance at multiple proposal depths, not just monotonic reactions to arbitrary ratios.

### M2 — C7 chooses depth before drafting but optimizes as if draft and catch-up were free

**Location:** `core/spec_policy.cpp:218–219`; call sites `models/ornith/ornith_gpu.cpp:4018–4021`, `:5385–5402`.

**Quote:**

```cpp
plan_evict(cond, cm, std::min(cfg_.k_min, cap), 0.0);
```

**Why wrong:** Zero draft cost is appropriate when evicting an already-produced draft whose cost is sunk. Here `next_depth` is called **before** the MTP/DFlash2 proposal is generated and determines how much work to do. It only sees measured verifier latency. MTP runs a sequential step per draft token and then catch-up; those costs vary with the choice. The optimizer can select a verifier-efficient but end-to-end slower depth. It also cannot select plain decoding when all speculative choices lose because the default minimum is one and zero is reserved for “cold.” Measured verifier timing alone does not establish the advertised throughput optimum.

**Concrete fix:** Measure/model proposal and synchronization cost per candidate depth and minimize complete cycle cost per expected emitted token. Distinguish a warmed plain-decode decision from “no measurement yet.” Add tests where verifier timings are held fixed and increasing draft cost moves the optimum toward shallow/plain decoding.

### M3 — D13 is bypassed by the normal raw super-kernel path

**Location:** `models/ornith/ornith_gpu.cpp:2303–2315`, `:2399–2407`, `:2415–2480`.

**Quote:**

```cpp
if (K[g].gemv_q4_w32p != nullptr && n_tiles >= pwgs && barrier) { ... }
...
if (use_m1_super) {
    ... // GDN/GQA and MoE super-kernel launches, then final AR
} else {
    ... // standalone per-layer kernels
}
```

**Why wrong:** The new persistent kernel is selected only inside the standalone `enq_gemm_q4` helper. On the intended Ornith M=1 configuration with super-kernels available, the step takes `use_m1_super`, launches fused kernels, and skips the standalone-layer branch entirely. `VEGAJET_V12_DENSE_PERSIST=1` does not alter this eligibility. Consequently the raw D13 command in the handoff/runner can measure no changed GEMV at all. The embedded `gemv_q4a8_persistent` in the super source is a different, pre-existing implementation, not the new atomic tile queue.

**Concrete fix:** Integrate the intended scheduling into the active super-kernel path, or explicitly scope/reject D13 when that path is active and benchmark it under documented standalone-path settings. Log/assert the count of actual D13 launches so a no-op lever cannot be credited with a speedup. Do not silently disable all fusion and assume that proves the proposed optimization.

### M4 — All-off startup loads and gates unused D5/D8 code objects

**Location:** `core/kfd_runtime.cpp:1005–1013`; Ornith call site `models/ornith/ornith_gpu.cpp:857–858`.

**Quote:**

```cpp
"vj_ornith_super_vol.hsaco",
"vj_ornith_super_iq2sreg.hsaco",
...
if (fs::exists(p, ec)) {
    if (!load_code_object(p.string(), err)) return false;
}
```

**Why wrong:** The directory loader, also called by the Ornith runner before variant selection, processes these variants regardless of whether either toggle is enabled. Missing files are correctly optional, but an existing stale/corrupt/incompatible experimental object causes all-off startup to fail, and valid unused objects consume code allocations and gate/load work. Main never examined those files. This breaks off-state isolation; it is **not** a claim that absent variant files break startup or that suffixed symbols overwrite baseline symbols.

**Concrete fix:** Resolve requested variant settings before loading and only load/gate the selected experimental object. Keep the empty-suffix all-off path independent of experimental artifacts. Test all-off startup with each optional variant absent and malformed, and requested-variant startup with it absent. Continue rejecting a requested unavailable variant rather than silently falling back.

### M5 — S2 reuses the same host token slots before queued batches consume them

**Location:** `models/ornith/ornith_gpu.cpp:3761–3762`, `:3794–3796`, `:3941–3971`; related deferred ingest at `:4001–4012`.

**Quote:**

```cpp
for (uint32_t j = 0; j < M; ++j) io_toks_host[j] = toks[j];
...
(void) step_mtp_batch(bt, pos + static_cast<uint32_t>(b0), bh0, bh1, false);
...
VJ_CHECK(rt.submit_batch(0) && rt.submit_batch(1));
```

**Why wrong:** Every batched EH/embedding launch receives the same `io_toks_va` base. Prompt ingestion queues multiple batches of up to 16 rows inside a 64-token group with `do_submit=false`, overwriting slots `[0,M)` on every call, then submits only after the group. For a 64-token group with distinct IDs, all four M=16 launches can read the last 16 IDs at four different positions, rather than their own prompt slices. A release fence publishes the latest writes; it does not snapshot per-dispatch input memory. Even if a queue implementation begins processing packets early, that would make it a host/device race, not fix the lifetime. The scalar path uses distinct `j-j0` token slots for those queued operations. Hidden-state rows can be right while token embeddings and resulting drafter KV are wrong. This remains after the epoch deadlock is fixed.

**Concrete fix:** Reserve nonoverlapping token slots for every outstanding sub-batch and pass its token-base offset in the EH kernargs, retaining them until queue completion. Alternatively submit and wait before reusing the shared slots. Apply the same lifetime rule to deferred multi-run `mtp_ingest`. Add a 64-distinct-token prompt test that inspects each queued token pointer/range and compares scalar versus batched drafter KV.

### M6 — The A/B runner does not actually clear the other v1.2 toggles

**Location:** `bench/v12/run.sh:24–26`, `:47–55`.

**Quote:**

```sh
env $envs "$BIN" bench ...
run baseline_raw "VEGAJET_V12_NONE=1" $RAW
```

**Why wrong:** `VEGAJET_V12_NONE` has no implementation in the engine. `env` inherits existing variables; it does not clear other toggles. An exported `VEGAJET_V12_VOCAB_ILP=1`, for example, remains active in the purported baseline and in every other lever run. The same applies to the inherited persistent environment in repeated experiments. The script also does not itself build/run main or automatically compare raw token sequences, so its baseline labels cannot establish the branch's all-off/main parity requirement.

**Concrete fix:** Explicitly unset every opt-in variable for each command, then enable only the lever under test. Record the effective environment and selected kernel names. Use separately identified main/all-off binaries and compare complete emitted token IDs. Treat process failure and mismatch as runner failures. Separately, capture the command status before `!` inversion: the current `FAILED (exit $?)` in the `then` arm reports the inverted status, not the failed engine's status.

### M7 — S2 launches EH embedding/normalization with M in the wrong grid dimension

**Location:** `models/ornith/ornith_gpu.cpp:3793–3796`; kernel `ops/kernels/vj_ornith_vocab.hip:566–574`.

**Quote:**

```cpp
VJ_CHECK(rt.enqueue_kernel(g, *K[g].mtp_eh_cat, 256u, M, 1, 256, 1, 1, &ecka, sizeof(ecka), true));
// ornith_mtp_eh_norm_cat_q8_kernel:
const uint32_t m = blockIdx.x;
if (m >= M) return;
```

**Why wrong:** With 256 threads per workgroup this launch creates one X workgroup and M Y workgroups. The kernel ignores Y and derives its row solely from `blockIdx.x`. All M workgroups therefore process row zero and write the same row-zero outputs; rows 1 through M-1 are never produced. The subsequent M-row EH projection consumes stale/uninitialized normalized concatenation rows and corrupts drafter hidden state/KV. Scalar M=1 did not expose the dimension mismatch. This remains after fixing the independent epoch and token-buffer defects.

**Concrete fix:** Launch this particular kernel as `M * 256u, 1, 1` with the same 256-thread block, or deliberately change its indexing and update every caller consistently. Audit each batched kernel's row mapping separately; do not assume that the GEMM kernels' Y-row convention applies to EH. Add a distinct-input-row CPU launch-mapping test and GPU per-row comparison against scalar EH output.

## MINOR findings

### N1 — Concurrent C1 writers can produce a mixed fault record

**Location:** `ops/kernels/vj_ornith_super.hip:154–160`; callers `:183`, `:240`, `:381`.

**Quote:**

```cpp
rec[1] = want;
rec[0] = (kVjFaultTag << 24) | (kind << 16) | (bar_id << 8) | stage;
```

**Why wrong:** Every failing workgroup's thread zero may write the same rank slot. These are unarbitrated volatile stores, not a single-writer publication protocol. A mailbox waiter and the other workgroups' grid-barrier wait can report different awaited epochs to the same slot. Writer A can store epoch A, writer B epoch B and identity B, then writer A identity A: the host reports identity A with epoch B. Cascading failures can also overwrite the original cause. This does not by itself erase the fault tag; after a successful system-scope queue drain the host has a completion synchronization point, so a missing acquire fence alone is not the finding.

**Concrete fix:** Atomically claim the empty slot once, let only that writer publish the complete record, and use a distinct complete/valid marker with the required device-to-host ordering. Preserve the first failure. Test interleaved publishers and verify identity/epoch coherence, not only host-side packing.

### N2 — D3's test claims a lowest-ID tie rule that neither reduction implements

**Location:** `ops/kernels/vj_ornith_vocab.hip:156–169`; `tests/test_decode_vocab_ilp_argmax.cpp:125–135`, `:165–167`.

**Quote:**

```cpp
if (acc > best_v) { best_v = acc; best_i = ...; }
if (sm_v[w] > bv) { bv = sm_v[w]; bi = sm_i[w]; }
// Test reports: "exact ties bit-identical, lowest row wins"
```

**Why wrong:** The result is first in wave/reduction order, not globally smallest token ID. With grid=7, equal maxima at rows 1 and 28, WG 0's wave 0 owns row 28 while wave 1 owns row 1; the strict-`>` wave reduction keeps 28. A source-level emulation returned 28. The all-zero tie test happens to put the smallest ID in the first lane and cannot expose this. **The baseline reduction has the same behavior**, so this is a false test/contract claim, not evidence that D3 introduced this tie ordering or that all-off tie behavior changed. D3's separate FP accumulation reassociation can still change near-tie winners, as the handoff acknowledges.

**Concrete fix:** Add sparse ties across waves, workgroups, and ranks and document/assert the actual baseline ordering for this release. If lowest-ID ordering is required later, use a consistent `(value, inverse-id)` comparator at every reduction level, version that semantic change, and do not silently alter the mandatory all-off baseline.

### N3 — PD2 can lose a final send failure from cancellation/error accounting

**Location:** `core/server.cpp:771–779`, `:796–802`, `:934–964`, `:1008`.

**Quote:**

```cpp
void finish() {
    if (!th_.joinable()) return;
    ...
    th_.join();
}
...
if (!send_blocking(fd_, batch)) {
    ...
    aborted_ = true;
    ...
}
...
if (gone || stopping) {
    ++n_cancelled;
}
...
pump.finish();  // flush queued SSE bytes before the socket goes back to the I/O thread
```

**Why wrong:** A push acknowledges enqueueing, not delivery. The writer can fail on queued content after the final generation callback but before the final drain. `finish()` returns no status, and the worker never transfers `aborted_` into `gone` before accounting, so failed content delivery can be classified as a successful, uncancelled response. The long-running disconnect test catches failure on a later callback and does not cover this window. Ignoring a terminal-frame send result already existed in the inline path; the new asynchronous gap additionally includes content previously counted as successfully pushed.

**Concrete fix:** Return a delivery result from `finish()` and fold failures into cancellation/error accounting before recording completion. Add a disconnect test specifically during final drain. Preserve FIFO ownership and joining before the descriptor is handed back to the I/O thread.

## Focus-area conclusions and unresolved hardware evidence

- **Default path:** B1 and B2 are definite all-off regressions; B4 breaks mandatory always-on error semantics. Empty-suffix kernel resolution selects the baseline names. The D3/D13 branch choices themselves remain off by default. PD2 is intentionally default-on and PD3 intentionally changes unsupported-request validation; those documented exceptions should not be confused with optional lever activation. No claim of ±1% speed parity is possible from this VM.
- **ABI inventory:** All four super entry points append `fault_va`; GDN/GQA/MoE production structs, the stage profiler, and microbenchmark launch structs in `ornith_gpu.cpp` carry the new member and size assertions. MoE needs padding before the 64-bit member, yielding 32 bytes, not 28. The separate layer-test callers are B2. Synthetic microbenchmarks deliberately pass null fault addresses and therefore do not verify fault-record delivery.
- **D5/D8 variant loading:** All four exported entry points use `VJ_KNAME`, and suffixes `_vol`/`_iq2sreg` match build definitions and resolution. Requesting both is explicitly rejected at `ornith_gpu.cpp:870–873`; it is **not** silently resolved to one. Requested missing variants produce an error. Remaining isolation defect: M4. The handoff's blanket “combine all kept levers” instruction must respect this implemented mutual exclusion.
- **D5 coherence:** `grid_barrier_wbinvl1` substitutes `buffer_wbinvl1_vol` at `vj_ornith_super.hip:242–246`, while cross-stage activation/KV loads remain ordinary loads, e.g. `:1094`, `:1168`, `:1186–1189`, `:1364–1367`. The source comment at `:197–220` explicitly acknowledges cached consumers and bases safety on a first-touch argument; that argument is not an executable coherence check. However, ordinary loads alone are not a complete stale-line counterexample: first-touch properties, actual compiler ISA, allocation aliasing and dispatch acquire invalidation matter. **This review has not proved a concrete stale-line execution or certified the variant as coherent.** Keep it off; retain full invalidation unless an explicit per-stage cache/coherence proof and generated-ISA audit justify the weaker instruction. The required GPU parity/soak evidence is absent. Do not mislabel CPU tests or ISA spill gates as a coherence test.
- **D13 queue reset:** A missing per-launch reset is **not** present in this revision. `vj_ornith_gemm.hip:291–297` resets both `q[0]` and `q[1]` in the last-arriving workgroup. The host initializes the queue during reset and selects the persistent path only for `barrier == true` (`ornith_gpu.cpp:2303–2305`), preventing two launches sharing that queue from overlapping. All workgroups stop claiming before incrementing the arrival count; the next dispatch waits for completion. The CPU tile test is sequential emulation, not a GPU ordering test. The actual default-path usability issue is M3.
- **PD2 lifetime/shutdown:** Queue ownership is by value, queue/counters are mutex-protected, and the writer is joined before the worker queues `Done`. `stop()` joins workers before closing their sockets (`server.cpp:1102–1113`); it does **not** shut down busy sockets to interrupt sends. Thus shutdown waits for the writer's current send/queued drain or its configured send timeout. The pump itself does not observe `stopping` while draining. I did not find a supported descriptor-use-after-close or queue data race, but prompt shutdown under saturated backpressure is not established. The existing tests exercise byte order, a modest slow reader, and mid-generation disconnect; they do not cover a saturated-queue stop or final-drain failure. N3 identifies the concrete late-failure accounting hole. Add an explicit abort/deadline policy and a saturated-queue shutdown test before claiming bounded stop latency.
- **S2 KV/positions:** Batched prep passes M and pos0 and flash attention uses row-specific causal length `pos0 + m + 1`; there is no demonstrated generic off-by-one position bug. Copying selected hidden rows into contiguous MTP scratch before EH projection is consistent with that batching scheme. This does not rescue B3/B5 or the token-buffer lifetime bug M5. M-row numeric equivalence and rejection/rewind behavior still require hardware tests.
- **C7:** Measurement gathering is toggle-guarded, so M1/M2 concern the enabled governor, not all-off prediction changes. CPU governor tests currently validate self-consistency of the model, not the correctness of its stochastic/cycle-cost assumptions.
- **Error resource handling:** C1 detection occurs only after successful queue drain; a real host wait timeout still hits `VJ_CHECK`/`abort` before structured fault handling. C1 spin-budget exhaustion must be tested separately from an unresponsive GPU/queue. Error propagation must not free pinned buffers while queued GPU work can still touch them. B1/B3/B4 are concrete failures in the presently implemented paths; no safe recovery is established by the host-only fault-format tests.

Only this review document was written in the source tree; build/test outputs were produced under `build-review`. No source fixes, commits, or pushes were made.

## BLOCKERS only

1. **B1 — `models/ornith/ornith_gpu.cpp:5841–5864`: healthy all-off benchmark reads fault storage after `io_buf` is freed.**
2. **B2 — `tests/vegajet_layer_test.cpp:739–742,784–786,820–822`: three super-kernel callers retain the old ABI, corrupting `fault_va` and hidden launch arguments.**
3. **B3 — `models/ornith/ornith_gpu.cpp:3831–3848,3879–3896`: S2 increments the collective epoch per rank, guaranteeing a two-rank mailbox deadlock.**
4. **B4 — `models/ornith/ornith_gpu.cpp:2214–2230,3157–3158,3203–3211` and prediction consumers: C1/C3 errors become `-1` tokens while callbacks/state commits/drafter GPU work continue; C3 can ultimately report success.**
5. **B5 — `models/ornith/ornith_gpu.cpp:3810–3821`: S2 with `--spec-flash` launches attention beyond its fixed scratch capacity at supported large context/batch combinations, permitting VRAM corruption.**

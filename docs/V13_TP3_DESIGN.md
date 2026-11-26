# VegaJet v1.3: modular TP=1..N execution (TP=3 first)

> Scope rule (owner): build modular mechanics that work for any TP size (1..200). Placement policy, tuning and performance targets are the owner's decisions; the planner only proposes defaults that the owner can override.

## Decision and evidence boundary

**Immediate release goal: fast three-MI50 decode, both raw M=1 and hybrid MTP/speculative decode.** Do not wait for pipeline parallelism, continuous batching, or a universal N-way executor before shipping it. Keep the fast TP=2 path as a reference and regression target. Phase 1 below includes the entire TP=3 path, not merely a planner or standalone collective.

Source inspected: `/data/src/vegajet`, `release/v1.2`, commit `269e3b96ca0be27d53ca167d9b1c981e2fbf62e1`. All source locations below refer to this snapshot; concurrent fixes in other worktrees may move them. Repository status was clean before this work. No tracked source is changed by this design. During final verification the branch advanced to `13739410ebd64c42845900ad791da1881ccab75f` through parser-hardening commits from other work; the cited executor/runtime/sharder/collective sources were unchanged. Source-reference validation checked 108 cited ranges across 22 files without a missing file or out-of-range line.

**Meaning of “any model and size”:** the parallelism machinery must consume model semantics and tensor layouts, not Ornith dimensions or a table of special cases per GPU count. A model adapter must still implement its architecture's operators correctly; a generic sharder does not implement an unknown architecture. Every supported architecture should receive a legal TP=3 plan or an explicit capability/capacity error, never silently run two cards. Oversized supported models need host-backed weight streaming, including dense weights, not only MoE experts. Finite GPU and host memory still impose limits; no design can promise arbitrary capacity or resident-model speed for a streaming model. The highest-performance configuration is resident weights on three verified P2P-connected cards.

This is a source-grounded design, not GPU verification. This VM has no MI50. Existing CPU tests were exercised (section 9); throughput estimates below are conditional arithmetic, not measured TP=3 results. All new flags in this document are **proposed** unless explicitly placed in the “available today” command block.

## 1. What is reusable, and what is not already implemented

* `core/sharder.cpp:10-104`: largest-remainder apportionment, aligned row splits, attention and GDN descriptors already accept a variable-length bandwidth vector. Reuse the integer apportionment, but repair the attention ownership contract: Q and KV are currently split independently, while GPU attention uses a local Q/KV ratio. GDN currently assigns whole K groups and their value heads together.
* `core/tp3_plan.hpp:14-103`, `core/tp3_plan.cpp`: CPU TP=3 planner, checks, and collective models exist. Its default 31.7/34.1/34.1 shares are a reference profile, **not a measured law for every three-card system**. The cost model is not an implementation of the TP3 header's full-partial all-to-all.
* `tools/vj_tp3_preflight.py`: useful GGUF inventory, block-alignment checks, DOT8 layout estimates, and balanced/coarse/whole-expert alternatives. `:370-393` calibrates one qwen35moe case against an older two-card runner and otherwise requires explicit overhead. Its output is not proof that an arbitrary GPU executor supports that model.
* `ops/kernels/vj_ar_tp3_alltoall.hiph:9-17,124-244`: actual three-rank collective source with parity banks, canonical rank-order summation, bounded polling and sticky failure. It is fixed to D=3072, one 64-thread workgroup, three ranks. `tests/test_ar_tp3_alltoall.cpp` is a CPU protocol/wire test; inclusion there is **not** an installed KFD GPU benchmark or a wired model executor.
* `core/rig_planner.cpp:172-187,199-201,237-307`: repack/capacity/offload estimator, but assumes evenly divided tensors, clamps GPU count to eight, and does not execute a placement. Its minimum 2 GiB reserve is useful policy, not an exact memory account.
* `core/moe_offload.*`: reusable async reads, staging arena, pinning and residency bookkeeping. It is not connected to the Ornith launch path and currently has two-card bookkeeping. The file-backed IO/staging engine is **not already a complete pinned-host expert cache or GPU refill pipeline**.
* `core/kfd_runtime.cpp:293-318,495-568`: enumerates devices and supports per-owner VRAM P2P mailboxes. It chooses peer transport if any P2P link exists and falls back to host userptr after failure. That is insufficient for a strict three-card decode group.
* `core/cont_batch.hpp`: CPU scheduling/cache API, not a multi-replica server executor. Reuse later rather than putting batching on the TP=3 critical path.

## 2. Hard two-rank assumptions: migration inventory

The table covers the live Ornith executor, its draft/verify/diagnostic paths, shared runtime and related collective kernels. Repeated rank-loop sites are indexed in Appendix A so replacing only the main decode loop cannot be mistaken for completion. Two parity buffers, two statistics per attention fragment, half-byte quant packing, two projections (gate/up), or two collectives per layer are **not** rank counts.

| Area and exact source | Assumption / required replacement |
|---|---|
| `models/ornith/ornith_gpu.cpp:866-880,899-902` | Accepts `num_gpus() >= 2` but instantiates `KernelTable K[2]` and resolves only two ranks. A third enumerated GPU is not automatically used. Require selected group membership equal to the plan, not a lower bound. |
| `models/ornith/ornith_gpu.cpp:1072-1107` | Two-entry BW vector; `voc_rows[2]`, `voc_off[2]`, `ff_size[2]`, `ff_begin[2]`; exactly two FF ranges; max over two; paired diagnostics. Generate all from per-rank manifests. |
| `models/ornith/ornith_gpu.cpp:1109-1138,1197-1204,1227,1275` | Two GDN row/head maps; two embedding/head/FR buffers; FR shortlist divided by two; two layer arrays. Use ragged maps, not `rank * width`. |
| `models/ornith/ornith_gpu.cpp:1292-1297,1356-1358,1383-1393,1485-1505` | Legacy expert GU rows use `g*512`; Down uses 512 input columns and paired expert slicing; GQA Q and KV splits assumed compatible. Use coupled head maps and explicit local FF width in every repack. |
| `models/ornith/ornith_gpu.cpp:1559-1573,1629,1758-1761` | `mtp[2]`, two GU row lists, `df_peer=1-df_gpu`, model/shape-specific MTP widths and fixed memory preflight. Draft placement must be its own plan with a target-group contract. |
| `models/ornith/ornith_gpu.cpp:1778-1815,1832-1839,1873-1919,1954-1980` | Paired argmax and DFlash top-k host/VA slices, two fault slots, AR timestamp kernels and `S[2]`; rank-0/else selection aliases every additional rank to rank 1's IO. Two LRU layouts and `[2][48]` expert refill arrays. Fixed 32 local GDN V heads / 16 Q-head scratch are TP=2-derived capacities, not safe dimensions for other models. Allocate from each rank's actual state and max verification width. |
| `models/ornith/ornith_gpu.cpp:1817-1823,3924-3949,4185-4195,5408-5440` | Hidden taps are deliberately on GPU1; MTP batch eligibility and hidden saves/catch-up explicitly name K[0]/K[1] and hs0/hs1; DFlash merges only two vocab outputs. Make tap rank an explicit plan choice and include all target-head shards in candidate reduction. |
| `models/ornith/ornith_gpu.cpp:2000-2083` | Two-rank LRU initialization and state reset; fixed `[36][32][128][128]` GDN allocation/reset sizes; paired completion. Reset must cover the selected rank group and ownership map. |
| `models/ornith/ornith_gpu.cpp:2090-2224` | Descriptor creation computes `peer=1-g`, one RX frame and one peer flag/TX address; GDN/GQA/MoE descriptor vectors have Ornith layer counts. Replace communication ABI with group descriptor and layer manifest. |
| `models/ornith/ornith_gpu.cpp:2255-2262,2291-2302,2330-2353` | Two-rank profiler, fault scans and fault-slot reset; snapshots/reset paths use two ranks. Any third-rank fault must invalidate the whole group before emitting a token. |
| `models/ornith/ornith_gpu.cpp:2465-2504` | `enq_p2p_ar_rms` takes two norm pointers, iterates two ranks and chooses a single peer. Create one collective operation with rank-indexed norm/partial/output pointers. |
| `models/ornith/ornith_gpu.cpp:2511-2517,2532-2625` | Fast superkernel rejects asymmetry and requires `local_ff==512`, D=3072; kernel availability checked only on K[0]; dispatch/wait only ranks 0 and 1. This is a **performance blocker** after adding a third rank, not merely a correctness detail. |
| `models/ornith/ornith_gpu.cpp:2657-3236` | Standalone GDN/attention/shared-expert/routed-expert loops launch two ranks; prepush uses `(1-g)` at 2794, 2889, 2964, 3048, 3229, 3236; 512-wide expert fast paths. Generalize both fused and unfused paths. |
| `models/ornith/ornith_gpu.cpp:3290-3311,3354-3398,3470-3490,3514-3537` | Vocab output, completion, golden/logit dump, tensor audits and IO candidates cover two ranks. Replace paired argmax/logit merge with all-rank reduction and deterministic global-ID ties. |
| `models/ornith/ornith_gpu.cpp:3769-3885,3924-3926,3973-4086` | MTP step and batch use two ranks; paired hidden-state vectors `hs0/hs1`; paired eligibility and reductions. Use rank-indexed hidden-state references or explicit single-rank materialization. |
| `models/ornith/ornith_gpu.cpp:4145-4156,4641-4760,4882-4913,5096-5187,5401-5415,5764-5790` | MTP hidden-state saves, stage/AR telemetry and benchmarks still depend on a two-card group. Generalize output schemas and make timings critical-path metrics, not a sum of rank times. |
| `models/ornith/ornith_gpu.cpp:6067-6213` | `run_ornith_ar_bench` explicitly requires exactly two cards; all benchmark buffers, peers, launches and result checking are paired. A new TP3 microbench is required; `--bench-ar --gpu all` is not one today. |
| `models/ornith/ornith_gpu.cpp:6319-6320,6430-6431,6546-6547,6767-6768,6798-6799,6862-6863,6884-6885,6916-6917,6935-6936,6974-6975,7088-7089` | Ancillary GEMV benchmark code is intentionally rank-0-only, not evidence of distributed support. Label these tests as local and iterate selected ranks when comparing bandwidth; do not infer group success from them. |
| `models/ornith/ornith_gpu.cpp:7137-7145` | `make_ornith_gpu_session` requires exactly two cards and uses fixed budget arithmetic. Serving must accept the same manifest as bench; otherwise TP3 benchmark support will not ship as a usable engine. |
| `ops/kernels/vj_ornith_ar.hip:81-129`, `ops/kernels/vj_ornith_common.hiph:237-350` | One peer payload/flag in fused AR ABI; sums local partial plus one peer. Modify every FP16, INT8, prepush and diagnostic variant, or explicitly gate unsupported variants off. |
| `ops/kernels/vj_ornith_super_desc.hpp:11-166`, `ops/kernels/vj_ornith_super.hip:348-480,653,1529-1537,1842-1883` | Single peer fields and single-peer wait/merge; expert addressing masks `&511` and strides `exp_id*512`, `g_p+512`. Descriptor `local_ff` existing does not make those instructions generic. |
| `ops/kernels/vj_ornith_gemm.hip:21,117-133,161,270-285,490,566-599`, `ops/kernels/vj_quant_ops.hiph:1369-1400,1422-1504,1521-1592,1611-1686` | Dense producer/prepush helpers take one peer TX pointer. Updating the consumer AR alone leaves the third destination unpublished. Extend the prepush abstraction or disable it in favor of the complete generic collective. |
| `ops/kernels/vj_ornith_moe_m1.hip:37,248-318,480,625-703,898`, `ops/kernels/vj_moe_dot8.hip:71-118`, `ops/kernels/vj_moe_asym.hip:53,207-222,242-291` | Routed/shared MoE producers and fused router paths likewise have a single peer TX (and sometimes fused two-source AR). Even the asymmetric-width kernel is not automatically a three-peer producer. |
| `ops/kernels/vj_moe_down_grouped.hiph:2,34-38` | Verification/prefill grouped Down fixes local input width at 512. Generalize its pack/stride/block counts or gate it off for 320/352 shards; this is relevant to hybrid verify, not only prefill. |
| `ops/kernels/vegajet_megakernel_vm.hip:34-44,244-281` | Alternate VM AR descriptor is also one-peer and sums exactly two contributions. Keep disabled for TP3 until it shares the new collective contract. |
| `core/moe_offload.cpp:649-650,782-798,829-832`, `core/moe_offload.hpp:230-233` | Clamps cards to two, clears only card_keys_[0]/[1], stores two per-card vectors. Replace with group-sized containers, and give residency entries group/rank identity. |
| `cli/bench.cpp:246-259` | Rejects GPU executor `--tp != 2`, although the planner accepts other counts. Expose TP3 only after capability checks; remove this guard in the same release as actual execution. |

Additional N-card limits (not “two” literals): `core/kfd_runtime.hpp:94-104` has `kMaxGpus=4` and fixed mailbox layout; `core/kfd_runtime.cpp:509-510,538-542` reserves space for four but allocates for `devices_.size()`; `core/kfd_runtime.hpp:256` uses a 32-bit touched mask. `core/rig_planner.cpp:201` clamps to eight. For 20+ cards, make mappings/mailboxes group-local and replace physical-device masks with vectors or a checked dynamic bitset; do not just raise a constant. For Phase 1, explicitly reject more than four devices in a runtime whose allocation is still fixed, and instantiate only the selected three-card group.

## 3. Phase 1A — model-agnostic TP=3 planning contract

Proposed entry point: `--parallel auto --objective latency|throughput --tp 3`, with `--tp 3` a hard requested group size and `auto` free to recommend TP1/TP2 instead. `--strict-p2p` is the default for the fast TP3 path. Keep CLI selection keyed by full PCI domain:BDF, using the v1.2 selection work rather than writing a competing selector. Ordinal rank is an internal plan index, never persistent device identity.

### 3.1 Required descriptors

Introduce a serializable, versioned `ExecutionPlan`, produced before weight allocations:

* `ModelGraph`: ordered operators/layers; hidden/residual dimensions; quant types and storage strides; complete tensor roles (embedding, head, router, shared/routed experts, attention and recurrence); legal reduction locations; tied tensor identities; architecture semantics and reference implementation.
* `TensorShard`: global shape, axis, **logical index map**, physical padded shape/strides, storage-format ID, packing block/alignment, rank ownership or replication, input/output layout, and bytes. An output-channel shard of Gate/Up and the corresponding input-channel shard of Down share one index map. Every real element belongs exactly once unless explicitly replicated.
* `AttentionShard`: global Q IDs, global KV IDs, Q-to-local-KV lookup, RoPE convention and head dimensions. A replicated KV head must have complete projection/cache ownership on every consumer rank. No integer `n_q_local/n_kv_local` shortcut unless that invariant is verified.
* `RecurrenceShard`: global V-head IDs, V-to-K-group lookup, Q/K projections and conv channels needed by those heads, state shape/dtype and state lifetime. Copy/restore and speculative commit use this map too.
* `CollectiveDesc`: group ID, ordered BDFs, local rank, participants, D/M/dtype, canonical reduction order, mailbox addresses/strides, epoch space, status and completion signal. Tensor layout must specify replicated input versus partial sum; residual is added **once after reduction**, never once per peer.
* `KernelCapabilities`: supported formats, dimensions and alignments, bounds/tails, rank count, collective mode, fused epilogues, VGPR/LDS/occupancy constraints. A generic correct path is mandatory for supported semantic operators; tuned kernels are selected by capabilities, not model-name conditionals.
* `StateBudget`: weights **after** repack; duplicate embedding/head or tied sharing; replicated KV and routers; per-rank KV for requested context; GDN state and speculative snapshots; draft model/cache; scratch; IO; mailboxes; allocator reserve and peak conversion/upload workspace. Budget each rank, not total bytes divided by three.

Plan validation rejects NaN/nonpositive BW, duplicate devices, uncovered/overlapping indices, incompatible producer/consumer maps, unsupported quant tails, zero-width kernel launches, integer overflows and memory overflow. Tiny dimensions may leave a rank with zero local work; that rank must still participate in required collective epochs, using a zero contribution. Prefer TP1 automatically for small models if collective overhead dominates, but an explicit TP3 correctness mode must remain legal through replication/zero-work ranks.

### 3.2 Uneven sharding: concrete examples and critical corrections

Initial weights are measured **kernel-effective HBM bandwidth**, at intended power caps and clocks; PCIe width is not HBM bandwidth. Adjust with measured operator timing and communication contention. Use `shard::apportion` on legal units, then a small discrete minimax search over neighboring unit allocations to minimize `max_r completion_time(r)`. Plan generation is automatic; no hand-written TP3 dimensions.

For normalized reference shares 0.317:0.341:0.341:

| Quantity | Legal allocation | Consequence |
|---|---|---|
| Expert FF=1024, 32-row unit | **320 / 352 / 352**; equal shares give 352/352/320 | Fastest rank's critical local width is 352 versus TP2's 512. Bounds and output strides must use each local width. Never use 341/341/342 if the packer requires 32. |
| FF=1024 with unmodified 256-column Down packing | **256 / 512 / 256** under reference shares | No slowest-rank MoE width reduction versus TP2. A successful three-card allocation alone may give no speedup. |
| 32 **global** GDN V heads, unrestricted one-head unit | **10 / 11 / 11** | Legal only with explicit K-group dependency mapping/replication. If grouped in sets of two over 16 K groups, the existing planner instead gives K=5/6/5 and V=10/12/10. |
| 64 global GDN V heads and 16 K groups | Existing grouped plan: K=5/6/5, V=20/24/20. Fine-grained alternative: V=20/22/22 | The current runner's repeated “32” is a **per-rank TP2 allocation**, not reliable evidence of 32 global heads. Read `hp.ssm_v_heads` and `hp.ssm_groups` from the GGUF. Do not accidentally halve the model. |
| GQA with 4 KV heads | Whole-group default: KV=1/2/1, Q assigned with those groups | Safe but coarse; one rank may own half the Q work. Independent Q=5/6/5 and KV=1/2/1 is **not generally correct**. For 16 Q and 4 KV, e.g. Q IDs 0..4,5..10,11..15 require KV sets {0,1}, {1,2}, {2,3}; map each Q explicitly. |
| MQA with 1 KV head | Replicate that KV projection/cache on all Q-owning ranks | Avoid empty K/V kernels; account duplicate KV memory and updates. |
| Vocab=248320, 32-row unit | **78784 / 84768 / 84768** | Exclude any pad IDs from top-k, use global offsets/maps. Merge only local top-k for greedy/sparse selection, rather than transferring full logits. Exact dense softmax sampling needs global max/sum and correct distributed selection. |

**GDN detail:** `ornith_gpu.cpp:1118-1123` constructs V IDs as `grp*ssm_groups+hk`, not necessarily a contiguous interval. Preserve that model ordering. Whole-K-group allocation is a correctness baseline. Then compare finer V partitioning with replicated small Q/K/conv dependencies against its extra projection bytes: it can remove the 24-versus-20 head bottleneck. State belongs to V heads; no replication of the large recurrent state is required just to share K inputs.

**Quant-layout blocker:** an IQ3_XXS source block spans 256 columns; 320/352 splits cannot be obtained by blindly slicing compressed source bytes. Reuse the existing asymmetric 32-column Down repack (`ornith_gpu.cpp:1292-1297,1356-1358`), prove its reconstruction against decoded source values and use its actual padded `Dot8RowLayout` sizes. Prefer a lossless expansion/relayout of source quant values into a legal local layout. If that is unavailable, use a correct source-decode/tiled kernel or the coarse 256-column split first. Requantization to IQ4_NL/Q4_0 changes numerics and must be an explicit quality mode, not silently presented as exact sharding. `vj_tp3_preflight.py:407-415,464-468` already exposes this distinction.

**Model independence beyond Ornith:** dense FFN uses the same coupled Gate/Up/Down maps; standard GQA/MQA uses explicit head maps; MoE uses all selected experts' intermediate partitions on every TP rank initially. Add an adapter for different recurrence/attention semantics (e.g. latent-attention cache) rather than pretending it is GQA. Whole-expert placement/dispatch is an optional costed expert-parallel variant, not the TP3 default: M=1 top-k load skew and variable traffic can cost more than balanced split-FF. Existing `ops::ModelDesc` is a useful seed but needs these layout and semantic details.

### 3.3 Memory and capacity

Minimize **time**, not even resident bytes. For M=1 MoE, charge selected expert bytes per token, while reserving all resident expert weights; for dense models charge every participating layer's weight bytes. Cache lengths, draft cache, and rollback headroom are first-class constraints.

For a model that does not fit on three cards, propose `--capacity host` (opt-in until mature): resident norm/router/dense hot tensors where possible; a NUMA-local host backing store and per-rank pinned staging/VRAM LRU for experts; generic tensor/tile streaming for oversized dense tensors. Double-buffer H2D refill and GEMV, pin only bounded windows if pinning the full backing store is impractical, account alignment and simultaneous buffers, and do not evict until completion signals retire every consumer. Unknown routing makes expert misses largely exposed; only measured prefetch hits may be subtracted. Streaming a dense model can cost roughly streamed-bytes / effective host-to-GPU bandwidth per token and is not a TP3 speed claim. If host RAM, staging, or minimum one-tile working set cannot fit, fail with required bytes and a smaller-context/format recommendation. The contract must support this path in Phase 1; the fastest resident mode must not be blocked waiting for aggressive offload optimizations.

## 4. Phase 1B — strict topology and three-rank all-reduce

### 4.1 Qualify the actual three-card group

Discover full BDF, KFD GPU ID, NUMA node, PCIe ancestors, negotiated link width/speed **under load**, VRAM/available arena, and directional peer mapping. Measure six directed peer pushes and simultaneous traffic. Record shared upstream/root/chipset link capacity, not just six nominal edge bandwidths.

A chipset-attached third card can be bottlenecked by a shared uplink, ACS routing, IOMMU policy or missing P2P even when KFD lists some peer links. `total_p2p > 0` in the current runtime does not prove a complete group. Allocate/map only group BOs to the required members and run a verified push test on **every required directed edge**. Preserve successful mapping sets through partial failures; never free a BO while any GPU can still access it.

Strict TP3 behavior: unsupported mapping, failed correctness probe or missing edge => abort before model launch, name the BDF pair and reason. **No silent host-pinned/userptr fallback.** Explicit `--collective host-debug` may be added later for diagnosis/correctness and must report degraded transport. An auto latency planner may recommend a proven two-card group instead; an explicit `--tp 3` must not quietly do that. Do not automatically modify ACS/IOMMU, power limits, firmware or sysfs policies.

### 4.2 First implementation: reuse the TP3 header, do not invent a protocol

Instantiate three owner-local inboxes, with two parity banks and three source frames per inbox. Every source pushes its full packed partial to the two peers and its local frame; each rank polls **only its local memory**, then sums source ranks 0,1,2 in canonical order. No peer read, remote atomic, or CPU barrier in the steady-state token loop. This is full-partial all-to-all, **not** a reduce-scatter/all-gather ring. The header's weighted partition changes tile visit order (30/33/33), not tensor ownership or how much data each source sends.

Wire sizes at D=3072:

* FP16 payload = 3072×2 = 6144 B; frame including padded epoch = `ceil((6144+4)/64)×64` = **6208 B**.
* INT8 block32 wire = 96×34 = 3264 B; padded frame = **3328 B**.
* Per destination inbox = `2 banks × 3 sources × frame_bytes`; per rank remote traffic = `2×frame_bytes` each collective, plus protocol effects. INT8 is not the default: TP2 README reports **17.30 µs INT8 versus 7.75 µs FP16**, so fewer bytes alone do not establish a win.

**Mapping must be reconciled before declaring this wired.** The header requires final epoch lines UC (`:11-13`) yet embeds epochs in the same frame as payload. The runtime currently uses separate COH-WC payload and UC flag allocations (`kfd_runtime.cpp:500-508`); merely passing its payload VA to the header does not establish UC epoch semantics. First hardware harness can conservatively map the entire small inbox UC and validate the literal header. Fast integration should split source/parity readiness records into separate UC flag BOs, retain COH-WC payload BOs and the tested epoch state machine, with an explicitly versioned collective ABI and fresh CPU/ISA tests. Cache-line alignment alone does not change a page's memory type. Verify actual gfx906 mapping/fence behavior; comments saying NC/UC/WC are not proof.

Epoch rules: start at 1, consecutive, no wrap; after error/timeout, permanently fail the group and discard outputs; reset only after **all queues are quiescent**. Reuse two banks only with the header's ordering proof: publication of e+2 requires consumption of e+1, which establishes the slowest producer has consumed e. Separate target/draft/verify collectives by communicator/epoch space. Concurrent operations cannot share an inbox just because they use a different layer number. New vectorized/fused code must preserve publication-after-all-writers-release, and bounded timeout status from every rank.

Numerics: the header rounds every source (including local) onto the wire and sums in the same rank order, avoiding rank-dependent local-unrounded results. This differs from some existing two-rank implementations. Establish CPU wire oracle and per-layer/logit error envelopes first; exact bitwise identity to a differently partitioned TP2 FP16 reduction is not promised. Greedy speculative equivalence is to the **same TP3 target engine and numeric mode**, with mismatches to TP2 separately investigated.

### 4.3 General collective ABI and decode latency goal

Parameterize D, rank count, strides, tails and dtype; provide rank-count templates for 1/2/3/4 as code-generation choices, not model-specific plans. TP1 bypasses communication but still applies residual/RMS/quant. For small M=1 partials prefer direct push all-to-all; for larger verification batches and prefill compare reduce-scatter/all-gather ring. Ring correctness supports uneven chunk lengths and zero tail chunks.

Simple candidate cost:

`T_direct ≈ L_launch + L_publish/poll + T_pack + max(shared-edge bytes / measured-edge BW) + T_local_reduce`.

`T_ring ≈ 2(p−1)·alpha + 2(p−1)/p · payload_bytes/B_ring + T_pack/reduce` (plus contention, launch/fusion effects and nonuniform chunks).

Use the graph of shared links for bytes, not `min(edge BW)` in isolation. A three-card ring needs four latency-bearing phases; direct all-to-all has one publish/consume rendezvous but twice the per-rank remote full-partial volume. Small decode messages favor direct; calibrate, do not force ring just because it scales to many ranks.

Budget versus the measured TP2 **7.75 µs/AR**: engineering target **9–10 µs median** for fused FP16 TP3 at D=3072, **12 µs** conservative planning point; record p95/p99 and skew. These are targets to test, not guarantees. At the README's 6.31 GB/s verified pair push, `2×6208 / (6.31×1000)` = **1.968 µs** of serialized outgoing wire time alone; arbitration, local read/pack, fences and launch dominate the rest. A chipset link changes that bound. A slow unoptimized one-wave header may miss the budget: use it as the correctness seed, then vectorize packet stores/loads and overlap production with push. Preserve its fail-stop protocol.

## 5. Phases 1C–1D — executor and superkernel changes

1. Make one `RankContext` per plan participant (weights, kernel registry, state, queues, IO/status), plus a `GroupContext` and immutable per-layer descriptors. Replace every `[2]`, paired norm/hidden argument, two-rank loop and wait listed in section 2/Appendix A. Resolve kernel capabilities on **all** participants.
2. Keep hidden/residual activations replicated within a TP group. Column-parallel mixer/FF front ends consume replicated input; row-parallel output projections produce partial hidden vectors; one all-reduce per mixer and one per FF/MoE output preserves the existing graph. For Ornith this is **48×2=96 collective events per token**, not 48. Do not count the same fused collective twice in profiles.
3. Build a correct unfused TP3 path first and compare intermediate tensors/state to the oracle. No fixed 48/36/12 layer loops, D=3072 limit, 32-local-head capacities, 16-Q-head scratch, expert count 256 or top-k 8 in the general path. Optimized Ornith kernels may retain guarded shape specializations, with a general fallback for other supported models.
4. **Restore the fast fused path before calling TP3 performance complete.** Current `use_m1_super` disables itself for asymmetric experts, so simply enabling three queues could make decode slower. Generalize `local_ff` through GU indexing, Down strides, tail masks and intermediate buffers; remove `&511`, `exp_id*512` and implicit 32-local-V-head state strides. Use explicit Q→KV and V→K maps in all recurrence/attention helpers.
5. Replace single-peer descriptor fields with a compact pointer to `CollectiveDesc`. Producer workgroups prepush their own output stripes to both peers. Each writer drains its own stores; a **local** grid rendezvous proves all stripes are visible before publication. The next superkernel consumes readiness from all sources, reduces, adds residual and runs RMS/quant once. Do not add a host sync or three remote atomics per layer.
6. A local persistent-kernel barrier expects the actually resident/launched workgroups on that **GPU**, not `3×workgroups` and not all ranks. Preserve occupancy bounds on gfx906: no deadlock from oversubscribing a global barrier. Ranks with uneven widths still have every launched WG reach every barrier; “no output rows” does not permit early return before a barrier. Validate all `super_wgs` choices against LDS/VGPR/private-segment metadata rather than assuming TP2's 240 is safe.
7. Keep mailbox publication epoch and local grid-barrier sense distinct. Move every timeout/fault into a sticky group fault; the host checks all ranks before publishing any token. A kernel timeout is not a completion fence; quiesce or abort dispatches before reusing/freeing state.
8. Vocab: local top-k with global IDs, deterministic tie handling, then small device/host merge. Broadcast chosen token to all ranks. Host merge may remain initially, but profile its serial time; full-logit gather remains an explicit diagnostic path only.
9. Bench and `make_ornith_gpu_session` must share the same planner, packer and manifest. Serve TP3 as a single sequence/engine first; replica scheduling comes later.

## 6. Phase 1E — MTP/drafter placement and hybrid speculative decode

Optimize **committed target tokens per wall second**, not proposed draft tokens. Raw M=1 and speculative results are separate scorecard columns; no prefill or DP throughput substitution.

### 6.1 Placement candidates on exactly three cards

* **Spread integrated MTP over all three ranks:** simplest reference implementation. Reuse target layouts where weights/state are shared, but MTP's own layer dimensions are read independently. It adds collective latency for each draft step; tiny MTP workloads may not amortize it. Share only tensors whose ownership maps actually agree. No assumptions that blk.48 is the drafter for all models.
* **MTP on one selected rank:** default latency candidate when full drafter plus KV/state and required shared head/embedding fit without target eviction. The target hidden state after its final AR is already replicated, so its local hidden input often costs no transfer. Gather/materialize the full MTP weights and any head it needs; a single rank cannot use only its old sharded vocabulary and claim global sampling. FR shortlist may accelerate **draft** selection only, with global-ID mapping; target verification remains full-vocab. Choose rank by incremental critical-path time and headroom, not “always GPU0.” The rank carrying the drafter may receive less target weight work if measured contention makes that optimal.
* **Separate small external drafter on one rank:** same capacity rule, plus target hidden-tap transfer if required (e.g. DFlash2). For taps that are true replicated activations, use the local copy; for sharded tensors gather the specified layout. No `df_peer=1-df_gpu`. A heavyweight draft model can be sharded across a subgroup or all three if it genuinely reduces total cycle time.
* **Two-card target + one-card dedicated draft** remains a separately measured alternative if the target fits TP2. It may beat TP3 hybrid through lower interference, but it is not TP3 and must be labeled `target_tp=2,draft_tp=1`. An explicit TP3 request cannot silently select it.

Draft and target usually depend on each other's results; do not assume perfect overlap. Same-rank drafting contends for HBM/compute and capacity. Prioritize target queues; overlap only work whose inputs/state are established and whose outputs can be discarded safely. Distinct queues need explicit signals and retirement before scratch reuse.

### 6.2 Verification and rollback contract

Plan candidates K=1..supported bound, tree width and batch/tile sizes from actual capabilities. Preserve MTP catch-up after rejection. Verification token batches use the same per-rank maps and a batch-capable collective, not sequential M=1 repacks by default. Reject unsupported tree shapes explicitly.

Each request has committed token position and transaction ID. For GQA, provisional cache rows are valid only for the transaction; commit the accepted prefix and invalidate the remainder. For GDN, maintain rank-local snapshots/deltas/ring slots sized by that rank's V heads and verification window; rollback every affected recurrence and conv history to the same accepted prefix. Broadcast one commit decision to all target ranks and the drafter. Do not reuse a parity bank, KV span or snapshot while a verification kernel still references it. Forced-rejection, zero-acceptance and all-acceptance paths are mandatory tests.

Use:

`T_cycle = T_draft(K, placement) + T_verify(K+1, TP3) + T_catchup/rollback + T_transfer + T_schedule − measured_valid_overlap`.

`T_spec/token = E[T_cycle] / E[committed_tokens_per_cycle]` (measure this ratio over complete cycles, not the mean of individual cycle ratios).

Collect acceptance histograms by prompt/model/context and auto-disable speculation if its measured effective rate is below raw decode after warmup with hysteresis. Example **only to show arithmetic**, not a forecast: four 0.40 ms draft steps + 7.00 ms verify + 0.40 ms overhead = 9.00 ms/cycle; 3.2 committed tokens/cycle gives 2.8125 ms/token = 355.56 tok/s. None of those timing/acceptance inputs has been measured here; the README's speculative scores must not be scaled by 3/2 to produce a fake TP3 forecast.

## 7. Decode cost model and conditional TP3 token budget

For each layer/operator and rank, estimate:

`T_local = max(bytes_read / BW_effective(format,shape,power), ops / compute_effective) + launch/nonoverlapped_fusion_cost`.

For each synchronized segment use `max_r T_local(r)`; add measured non-overlapped collective, sampling and offload terms. Serial replicated operators are not divided by TP. Use current context length for attention bytes and active-expert expectation/distribution for MoE. Include memory-format expansion and instruction throughput, not just on-disk GGUF size. A new model gets conservative estimates plus an optional short calibration; cached calibration is keyed by BDF/topology, clocks/power, kernel hash, wire type, shape/context and quant format.

Optimize raw latency by minimizing estimated full-token critical path under per-rank capacity and strict topology constraints. Throughput optimization later compares replica service rates under queueing/latency SLO; it is a different objective. Profile launch/barrier/publish/wait/pack separately so reducing work on a fast rank does not masquerade as a communication improvement.

### 7.1 Performance targets

No performance estimate is given here. The owner sets the targets and the execution strategy; this design only provides the mechanics (sharding, collectives, topology, executor) and the measurement hooks.

## 8. Later phases: automatic 1..20+ placement without per-count tuning

These remain short follow-on releases; they are not blockers for resident TP3 decode.

### Phase 2 — TP1/offload and TP4; generic small-group release

Flags: proposed `--tp 1|4`, `--capacity resident|host`. TP1 turns AR into a local epilogue and wires host-backed expert staging through `MoeOffload`; complete dense paging if not yet enabled outside TP3. TP4 reuses rank-sized maps and direct/ring ABI, not a new model executor. Preserve independently runnable TP2. CPU gates: 1/2/3/4-way partition conservation, head maps, quant tails, capacity/overflow/zero shards and offload eviction/IO-failure state machines. GPU gates: TP1 offload misses/hits and recovery, TP2 regression, TP4 all directed edges and correctness/performance. Small resident models should auto-select TP1 where launch/AR dominates.

### Phase 3 — PP for capacity, prefill improvements, 5+ cards

Flag: proposed `--pipeline auto|P`. Enumerate topology-local tensor groups with TP in {1,2,3,4}; partition **contiguous layers** across stages using dynamic programming with each stage's exact peak memory and estimated compute. Prefer uniform TP per replica initially; permit heterogeneous stage TP via an explicit boundary layout (so 5 cards can be a 3+2 pipeline), not by pretending every placement is a single TP×PP product.

After the last row-parallel reduction in a stage, hidden state is replicated within that group. Push one `[M,D]` activation (and any required residual/tap side tensors) from a chosen boundary sender into the next group's leader-local receive buffer, then broadcast within that group; alternatively paired sends can be costed if they reduce a bottleneck. Use credits, transaction/sequence ID, token range, shape/dtype and monotonic epoch; bounded buffering, explicit completion and push-only P2P where valid. Different hidden layout/dimensions require an explicit conversion, not reinterpretation.

KV/GDN/conv/rollback state remains on the stage owning its layers, sharded by that stage's TP map. Route one commit/rollback decision to **all** stages. Only hidden states/metadata cross a stage boundary, not the whole KV cache per token. Logits from the final stage select the next token and return the token/position to stage zero. On one autoregressive stream, stages execute sequentially per token: latency is approximately sum of stage times plus handoffs; PP does not magically create single-stream parallelism. Fill the pipeline only across independent requests or legal verification/prefill microbatches. CPU gates: layer coverage, minimax stage partitioning, boundary format validation, credit/backpressure and distributed rollback simulation. GPU gates: cross-group P2P, PP-vs-unpartitioned logits/state, cancel/fault tests, bubble and bandwidth measurements.

### Phase 4 — DP replicas and serve scheduling

Flag: proposed `--replicas auto|R --objective throughput`. One independent engine/runtime group per replica, disjoint GPU membership and mailbox namespaces. An admission controller routes by queue + estimated remaining work and memory, with request/prefix affinity where useful. `core/cont_batch` can manage slots/KV within each replica only after the engine supplies correct per-sequence state; never share Ornith's mutable singleton state across requests. A generation cannot migrate replicas midstream without explicit state transfer or prompt replay. Cancellation drains its in-flight work before releasing cache; a replica fault quarantines that group, not the entire service. CPU gates: placement uniqueness, bounded queues, fairness, cancellation and affinity tests. GPU gates: simultaneous replicas, latency SLO under load, no cross-request state contamination, measured aggregate committed tok/s.

### Placement defaults and policy by card count

There is no model-independent “correct TP for N”; use the measured cost and exact fit. Let G be the fastest fitting legal group (1..4 cards) and choose the smallest capacity-valid stage count. For Ornith-like large models assume TP1 is host-offloaded and TP2/3/4 fit only after exact context/draft budget validation. Examples below are planner seeds, not hard-coded count recipes:

| Available N | Single-stream latency default | Aggregate throughput default once later phases ship |
|---:|---|---|
| 1 | TP1 resident if it fits, otherwise host-backed weights | One TP1 engine, batched only if beneficial. |
| 2 | Best of TP1 and TP2; large Ornith starts TP2 | Two TP1 replicas only if each fits; otherwise one TP2. |
| 3 | **Explicit owner target TP3**; auto compares TP2 and TP3 and reports why | TP3, TP2+independent TP1, or three TP1 replicas as fit permits; a leftover rank can be a drafter. |
| 4 | Compare TP3 vs TP4 vs TP2; PP only if no single group fits | Two TP2 vs one TP4 vs four TP1; choose measured aggregate service rate. |
| 5 | Fastest 1..4-card group, spare idle/draft; if capacity demands, PP 3+2 | TP3+TP2 replicas for a fitting large model, or one capacity-spanning PP replica. |
| 6 | Best small TP group; PP 3×2 if capacity demands | Two TP3 vs three TP2; topology and measured rate decide. |
| 7–8 | Best small TP; PP only for capacity | Integer packing of best fitting groups (e.g. 3+4, 3+3+2, or 2+2+2+2). |
| 9–12 | Same rule; more devices do not force higher TP | 3-card or 2-card replicas; PP stage groups only for oversized models. |
| 13–19 | Same rule; minimum-cost capacity-valid pipeline if needed | Topology-aware integer replica packing with remainders idle/draft/smaller-model. |
| 20+ | Do not make a 20-rank decode collective by default; e.g. TP4×PP5 only if the model needs that capacity | E.g. TP2×DP10, TP4×DP5, or six TP3 plus a TP2 at N=20, subject to topology and fit. |

Uniform groups are expressible as TP×PP×DP; heterogeneous layouts use explicit `{stages:[{ranks:...}], replicas:[...]}`. Arbitrary N is handled by graph partitioning and integer packing, including unused devices, not by insisting every card participate.

## 9. Phase 1 shipping slices, CPU gates and the owner's three-card test list

Each slice is independently reviewable/shippable behind an opt-in flag; none by itself means the whole TP3 objective is complete.

| Slice | Proposed flag/capability | CPU-verifiable gate | Required GPU gate |
|---|---|---|---|
| 1A | `--plan-format v2 --tp 3` (plan only) | Property tests for coverage, maps/replication, quant blocks, exact memory; model fixtures including dense GQA/MQA, GDN, MoE, vocab tails and D>4096 | Read-only topology/capacity cross-check on actual BDFs; all six directed P2P probes |
| 1B | `--bench-ar --tp 3 --strict-p2p` | Existing TP3 wire/protocol tests plus split-flag ABI, delayed-rank and epoch exhaustion tests | Literal-header and optimized collective oracles, millions of epochs, wire variants, skew/failure injection, per-AR distribution |
| 1C | `--tp 3 --decode-path generic` | CPU reference sharded forwards and state/packing comparison, fake queue/fault integration | Unfused raw decode and service correctness on supported architectures; every rank demonstrably executes |
| 1D | `--tp 3 --decode-path fused` | Descriptor/layout/occupancy admission and golden intermediate checks | Generalized superkernel parity, no spills, stable barriers and raw latency improvement over matched TP2 |
| 1E | `--tp 3 --draft-placement auto|rank:BDF|tp` | Transactional state commit/rollback, placement budget and sampling tests | MTP/external-draft/combined hybrid acceptance, catch-up, forced rejection; committed tok/s vs TP3 raw and TP2 hybrid |
| 1F | `--tp 3 --capacity host` | Pager/eviction, incomplete IO, tile coverage and peak memory accounting | Model beyond resident capacity, controlled misses, H2D overlap correctness, fault recovery; performance reported as streaming, separately |

Phase 1 is accepted only when resident raw decode **and** hybrid speculative decoding work through bench and serve with model-derived dimensions and no hidden two-rank path. Capacity fallback must be explicit and functional for models whose bytes exceed the group; unsupported semantic operators must be named rather than misplanned. Prefill may use a correct slower path; optimize its throughput in Phase 3 rather than sacrificing decode work.

### 9.1 Commands actually available today

On the owner's GPU machine, from its checkout, choose `VJ` as the GPU-enabled binary and `MODEL` as a real GGUF path. Current CLI supports ordinal selection; use full BDF selection only after the parallel v1.2 work lands. Re-enumerate before using ordinals. The following are inspection/test commands, not a request to change system configuration or power caps:

```sh
VJ=./build-gpu/vegajet
MODEL=/absolute/path/to/model.gguf
"$VJ" serve --gpu list
"$VJ" doctor --gpu all --gsf12
"$VJ" doctor --gpu all --os
"$VJ" doctor --gpu all --live-link --seconds 10
bash hardware/check_p2p_ecc.sh
python3 -B tools/vj_tp3_preflight.py profile "$MODEL"
python3 -B tools/vj_tp3_preflight.py plan --profile "$MODEL" --ctx 32768 --json
"$VJ" plan "$MODEL" --gpus 3

# Select each physical pair; each p2p soak iterates both directions.
"$VJ" soak --gpu 0,1 --p2p --seconds 60
"$VJ" soak --gpu 0,2 --p2p --seconds 60
"$VJ" soak --gpu 1,2 --p2p --seconds 60
"$VJ" soak --gpu all --p2p --seconds 60

# Existing benchmark remains TP2 only; baseline every useful pair.
"$VJ" bench --bench-ar --gpu 0,1
"$VJ" bench --bench-ar --gpu 0,2
"$VJ" bench --bench-ar --gpu 1,2
VEGAJET_RAW_ONLY=1 "$VJ" bench "$MODEL" --tp 2 --gpu 0,1 --ctx 2048 --n-predict 128
```

Important caveats: current `soak --p2p` (`cli/soak.cpp:89-139`) executes directed copies serially and rejects host transport; it does **not** measure simultaneous six-edge congestion. `--live-link` is a load-dependent link check, not an all-to-all benchmark. The generic rig planner has approximate budgets. The preflight command's default overhead is calibrated for one qwen35moe case; for other architectures supply a justified `--overhead-gib`, and treat the estimate as incomplete until the new allocator manifest exists. For a tiny/non-Ornith model the final TP2 Ornith benchmark may be unsupported; use the appropriate implemented runner, not a claim that this CLI is already universal.

**There is no runnable end-to-end TP3 decode command in this snapshot.** `cli/bench.cpp:246-259` rejects it, the session factory rejects three cards, and the existing AR CLI requires exactly two. The owner can qualify links and obtain pairwise baselines today. The new Phase 1B KFD harness must be added before claiming the TP3 header ran on hardware; CPU tests and an HSACO compilation are not substitutes.

### 9.2 GPU acceptance matrix once the Phase 1 harness lands

1. **Topology:** all six directed pushes verify data; concurrent two-destination pushes and full three-source traffic verify under load; capture negotiated links, ACS/IOMMU/root/chipset ancestry, bandwidth and p99. Repeat with swapped rank↔BDF assignments. Missing P2P must fail closed, not copy via host.
2. **Collective:** FP16 first; INT8 only as an A/B mode. Constant, random, cancellation-heavy, extreme/invalid values, rank-specific tags and canaries; check all ranks against the same CPU wire oracle. Test D=3072 and generic D/tails, M=1 and actual verify widths; million-epoch parity reuse, delayed producer/consumer, stuck rank, timeout and near-wrap. Measure completion, not just queue read-pointer progress. A fault must return no token and permit safe teardown.
3. **Shard correctness:** 1024 split into 320/352/352, coarse 256/512/256 reference, unaligned vocab; 32 and 64 global V-head fixtures, non-contiguous GDN IDs; GQA 4 KV, MQA 1 KV, and head counts smaller than ranks. Compare repacked matrices to source-dequantized weights and per-layer hidden/state dumps.
4. **Raw latency:** matched quant/model/context/prompt/token count, warmed weights, no speculative flags, same eligible power/clock configuration. Compare TP2 fast path, TP3 generic, TP3 fused, symmetric versus calibrated shares and FP16 versus INT8. Record tg128 and tg512, wall ms/token, collective median/p95/p99, launch count, per-rank critical segment timing, HBM traffic, thermal/clocks, and full memory high-water mark. Require statistically repeatable improvement for the target model/topology; if TP3 loses, keep the explicit mode but do not recommend it in auto-latency policy.
5. **Hybrid decode:** MTP K sweep, single-rank vs spread MTP, external draft placement, suffix/MTP combination if supported, raw TP3 reference. Record draft cost, verify cost, accepted/committed tokens, catch-up and rollback separately. Force zero/all/partial acceptance and rejections at every position; check committed output against same-engine greedy reference, GDN state/conv history and KV after each commit. Test context boundary, request cancellation, consecutive requests and drafter failure.
6. **Model breadth/capacity:** at least a small dense model, GQA/MQA variants, a hybrid recurrence model and a MoE; same contract with different dimensions/quants, no model-name rank patch. Test memory-near-limit context and draft coexistence; host-paged model larger than available aggregate VRAM; one tensor larger than a streaming slot; unreadable/short source IO and staging allocation failure.
7. **Soak:** ten minutes minimum with telemetry, then longer overnight continuous decode/rollback; zero invalid outputs, stale epochs, out-of-bounds canaries, uncorrectable ECC or unexplained device faults. Stage faults must be attributed to BDF/rank/epoch and fail safely.

### 9.3 CPU evidence actually obtained for this design

Executed existing binaries from `build-cpu` and Python without building or editing source:

* `/data/src/vegajet/build-cpu/test_tp3_plan` printed `test_tp3_plan: TP=3 sharding, balance, VRAM, power caps, direct/ring all-reduce exact, cost model OK` and `PASS`. Its deliberately imbalanced 32-row fixture prints a `FAIL balance` diagnostic that the test expects; the overall test passes. This confirms a planner test, not the validity of independent Q/KV splits for all GPU kernels.
* `/data/src/vegajet/build-cpu/test_ar_tp3_alltoall` printed `tp3 compact wire, partitions, four asynchronous epochs, parity isolation, fail-stop: PASS`.
* `PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s /data/src/vegajet/tests -p 'test_tp3_preflight*.py'` ran **17 tests**, `OK`.
* Split and time-budget arithmetic above was evaluated with Python. Existing CPU binaries may reflect an earlier build; no claim is made that they were freshly rebuilt from this exact source commit.

## 10. Risks, non-goals and unresolved hardware questions

* **Memory ordering is a release gate:** glc/slc, system fences, BAR memory types and queue completion need real gfx906 verification. CPU protocol models cannot prove PCIe visibility. Do not reuse a flag design without reconciling the header and runtime's actual mappings.
* **Topology may dominate:** a third card behind chipset/shared lanes can erase HBM savings. No peer fallback is allowed in the fast claim; auto may recommend two cards with an explanation.
* **Loss of fused kernels can dominate:** functional TP3 on standalone dispatches is an intermediate milestone, not the performance result. Fine-grained shards can worsen occupancy or make format expansion consume the added VRAM.
* **Numerical agreement:** changed reduction order, FP16 rounding and quant repacks can alter borderline token choices; INT8 adds more error. Keep source-layout, wire, per-layer and target-speculative correctness tests distinct. No unapproved requantization under a sharding flag.
* **Capacity accounting:** source GGUF bytes are not resident packed bytes; the nominal 3×32 GB is not fully allocatable model capacity. KV replication, GDN snapshots, draft duplication, pinned staging and allocator peak usage can dominate.
* **Architecture breadth:** new semantics require an adapter/reference, not just setting `n_layer`; unsupported operators must be explicit. Whole-expert EP and exotic attention are not secretly supported by the current Ornith engine.
* **Speculation is workload-dependent:** acceptance and same-rank contention decide whether MTP helps; do not promise an uplift for arbitrary models/prompts. Draft-only FR shortlist never truncates target verification.
* **Universal N is later:** fixed runtime arrays/masks, group-local mappings, memory-pressure and kernel-resource limits must be addressed before 20+ devices; a 20-rank AR is not the default scaling path.
* **Hardware-only unknowns:** achievable three-card AR latency, effective per-format BW at three-card power/thermal limits, concurrent P2P routing, large-BAR mapping reliability, fused occupancy, long-run queue/epoch behavior, true raw/speculative tok/s and acceptance rates. None is validated by this VM.

## Appendix A — repeated rank-bound sites in the Ornith executor

All line numbers here refer to `models/ornith/ornith_gpu.cpp` at the source revision above. They form a migration checklist in addition to the semantic table, not instructions to blindly replace every numeral 2.

**Rank loops bounded by two:** 902, 1111, 1204, 1275, 1364, 1403, 1438, 1485, 1564, 1572, 1808, 1839, 2000, 2004, 2030, 2054, 2090, 2292, 2334, 2348, 2477, 2548, 2566, 2581, 2601, 2657, 2820, 2947, 3290, 3354, 3365, 3470, 3514, 3620, 3769, 3789, 3832, 3871, 3973, 4016, 4035, 4065, 4507, 4659, 4676, 4686, 4695, 4709, 4721, 4736, 4882, 5096, 5125, 5148, 5162, 5401, 5765, 6081, 6104, 6122, 6149, 6169, 6179, 6188, 6195, 6213.

**Two-rank containers (not parity/numeric arrays):** K 899; voc_rows/off 1077/1079; ff_size/begin 1089/1090; GDN maps 1109; embedding/output/FR 1130-1132; output norm 1138; layers 1227; MTP 1559 and GU rows 1563; IO fault slots 1806; AR kernel table 1832; state 1838; LRU layouts/addresses 1973-1974; stage timing 2255; timeline addresses 4641 and totals 4735; totals 5764; AR benchmark buffers 6078,6097,6120,6158,6162-6163,6189,6201,6211-6212.

**Single-peer addressing:** 1629, 2092, 2478, 2602, 2794, 2889, 2964, 3048, 3229, 3236, 3471, 3770, 4017, 4066, 4710, 6105, 6180.

**Paired group submit/wait sites before the local-only GEMV benchmark section:** 2083, 2259, 2624-2625, 3310-3311, 3362-3363, 3387-3388, 3489-3490, 3518-3519, 3632, 3884-3885, 4085-4086, 4155-4156, 4254-4255, 4331-4332, 4356-4357, 4729-4730, 4913, 5103, 5414-5415, 6116.

**Do not misclassify:** `spec_gdn_s_slot_va[2]` at 255/2078-2079 is double buffering, not two GPUs; `C/2` at 411 and `local_ff/2` at 1899 are packed nibbles; `[2]` at 1345/1347 indexes quant words; `sum[2][6]` at 5166 indexes the two collective classes per layer; `512` in workgroup launch arithmetic is often block size. Preserve these unless a separate layout/shape contract requires a change.

# S3 tree verification — host-side checkpoint

Status: **partial, not an enabled GPU tree verifier**. `VEGAJET_V12_TREE_VERIFY=1`
fails closed in `run_ornith_gpu` before opening a model or allocating GPU memory.
Unset or any value other than exactly `1` uses the unchanged runner. No kernel,
HSACO ABI, launch geometry, default flags, or existing chain loop was changed.
The new host helpers are not invoked on that default path.

## Finished host pieces

`core/spec_tree.{hpp,cpp}` provides:

- Seven-row budget: one already-emitted/unconsumed anchor, two ranked distinct
  first-slot DFlash2 candidates, and two second-slot candidates per child.
  The caller supplies merged/ranked token IDs; per-block Top-16 is **not** already
  globally sorted. Empty slots and invalid candidate sentinels are handled.
- Alternatively converts the first six `spec::SuffixTree::draft` nodes, retaining
  parent links and allowing deeper trees. Sibling tokens must be unique.
- Ancestor-plus-self bit masks, logical depths for RoPE, parent snapshot indices,
  and ancestor-only three-token raw convolution histories. Negative indices
  `-3,-2,-1` denote the immutable pre-verify convolution history, oldest first.
- Longest greedy matching path: start at the anchor, take only a child equal to
  that parent's target argmax, stop at the first missing match or emission cap,
  and return the final row's target prediction as an unconsumed bonus token.
- Validated accepted-path plans: gather/replay node IDs for spec-GDN records,
  final per-node state selection, convolution tail, ring and position advancement,
  and byte-addressed KV row copies with capacity and uint64 overflow checks.
  Zero accepted drafts still commits the anchor, not the bonus.

KV source rows are `pos0 + node`; final rows are `pos0 + path_offset`.
Logical rotary positions are instead `pos0 + depth[node]`. Parent-before-child
ordering guarantees `source >= destination` and strictly increasing sources.
Execute the plan's row copies **serially in ascending path order**, per K/V
buffer, layer and GPU. A later copy may reuse an earlier source row as a
_destination_; dispatching all row copies concurrently would be a data race.
Each individual row copy has disjoint ranges. All local KV heads belong in
`kv_row_bytes` (FP16 bytes are preserved, never requantized).

For spec-GDN replay, gather selected FP32 k/v/decay/beta records into **disjoint**
contiguous scratch and run the existing replay recurrence from the immutable
pre-verify state. Restore/commit the convolution tail separately. Do not use the
chain's `acc == dk` active-slot swap: a tree's flattened final row need not be
on the accepted path. `state_node` is an alternative commit source only if real
per-node FP32 snapshots were computed from the proper parent snapshots.

## Why greedy acceptance is lossless (conditional on the missing GPU work)

Induct on accepted depth. The anchor is already a raw-greedy emitted token. A
node may see only the committed prefix and its ancestors/self, has the same
logical position as serial decode, and starts GDN and convolution from the
identical parent state. **If GPU arithmetic and argmax tie breaking also match
raw decode**, its full-vocabulary target argmax is the next raw-greedy token.
Acceptance uses only that argmax; draft scores or a shortlist never override it.
Unique sibling tokens make the matching continuation unique. On a missing
child, the target argmax is returned as the bonus and is consumed on the next
cycle. Only that path's exact KV bytes, FP32 recurrent state and convolution
history are retained. Therefore no rejected sibling can affect later tokens.

Masking alone does not establish bit identity: physical tree width must not
change Split-KV chunk boundaries or reductions versus the logical serial
context, and MoE/router/quantized GEMM paths must be checked too. This commit
makes **no GPU identity or performance claim**.

## Exactly what remains for GPU integration

1. Preserve merged Top-k per DFlash2 slot (currently reduced to a chain), or
   select the suffix draft tree, and feed the host constructors into each chosen
   speculative decode loop. Keep a raw-greedy fallback when fewer rows fit in
   the remaining KV arena. Cap accepted/emitted tokens at output/EOS boundaries.
2. Extend the GQA prep/RoPE ABI to take logical depth per row while writing K/V
   at separate physical rows. Current prep uses `pos0 + m` for both.
3. Add an ancestor mask operand to a dedicated tree verify flash kernel and its
   host launch ABI. Prefix `[0,pos0)` is visible; tree positions are allowed only
   when the query's bitmask contains that node. Preserve serial arithmetic,
   chunk partition and reduction order, including context-length thresholds.
   Register any new kernel/gates in CMake and `hsaco_gates.txt`. Keep all default
   kernel arguments and behavior untouched.
4. Make the k=4 convolution gather each node's three ancestor raw inputs instead
   of rolling through flattened siblings (`vj_ornith_attn.hip`, conv token loop).
5. Compute each node's FP32 GDN output from its **parent** state (root from the
   immutable pre-verify snapshot). The existing `vj_spec_gdn.hip` recurrence
   loops through rows as one chain. Allocate/manage bounded per-node states or
   an equivalent ancestry-aware record replay schedule for all 36 GDN layers.
6. Gather accepted record rows into separate scratch and reuse spec-GDN replay,
   or select the accepted terminal snapshot; install the matching convolution
   tail. Replace the chain-only `commit_spec_gdn` prefix/swap logic in the opt-in
   branch. Commit only anchor plus accepted drafts, never the bonus prediction.
7. Enqueue ordered KV compaction copies on both GPUs and all GQA layers; order
   them before reuse/attention. Advance ring/conv heads, position, MTP/DFlash2
   hidden-state/catchup metadata using the accepted **path indices**, not `acc`
   as a flattened row index. Rejected scratch rows must never enter live cache.
8. Remove the fail-closed guard only after integration. Run the owner MI50
   greedy-losslessness suite (including reject-at-root, sibling branch, deepest
   suffix path, full output cap, EOS, ring wrap, and context-threshold cases).

These are required correctness work, not optional optimizations. The inspection
found incompatible chain ABIs in attention, RoPE, convolution, recurrent replay
and hidden-state catchup. This bounded change deliberately commits the finished
host portion allowed by the task instead of exposing an unsafe partial GPU path.

## Verification and owner commands

Builds use `-j1` on the shared VM:

```sh
cmake -S . -B build-cpu -G Ninja -DVEGAJET_CPU_ONLY=ON
cmake --build build-cpu -j1
ctest --test-dir build-cpu -j1 --output-on-failure
cmake -S . -B build-gpu -G Ninja -DVEGAJET_CPU_ONLY=OFF \
  -DVEGAJET_ROCM_PATH=/data/tools/therock712/lib/python3.14/site-packages/_rocm_sdk_devel
cmake --build build-gpu -j1 -- -k 0
```

`test_spec_tree` checks construction from both draft sources; exact masks/depths;
zero/partial/full/alternate-branch acceptance; stale metadata and malformed
paths; KV capacity/overflow and byte offsets above 32 bits; ring wrap and
convolution history; fail-closed env handling. It enumerates all 720 ordered
six-draft parent layouts and every possible terminal node, actually copies KV
rows in place, and compares scalar FP32 parent-snapshot and gathered-path replay
models. Scalar recurrence tests establish the **host plan**, not device GDN
numerics. No model file or GPU is required.

On the owner GPU machine, from this worktree (set MODEL to the exact local
Ornith GGUF first), this partial checkpoint must reject before model execution:

```sh
VEGAJET_V12_TREE_VERIFY=1 ./build-gpu/vegajet bench "$MODEL" --gpu 0,1 --tp 2 \
  --prompt 'Explain speculative decoding.' --n-predict 32
```

Expected: `VEGAJET_V12_TREE_VERIFY=1 unavailable: S3 host planning only` and no
GPU work. Compare unset and `VEGAJET_V12_TREE_VERIFY=0` baseline runs using the
same invocation/model/other options; generated token IDs should be identical
(the default path is unchanged). No speedup is expected from this host-only
commit. After integration, run `bench` with `--suffix-decode --spec-gdn
--spec-flash` (and `--draft-model "$DRAFT" --mask-embedding "$MASK"` for DFlash2).
Its existing raw-versus-speculative comparisons print `lossless=...` and
`first_diff=...` for code/prose. Enable the tree flag and require all token IDs
to match raw greedy before measuring accepted depth or tokens/s. Do not treat
the current guard's rejection, CPU models, or the compiler gate as a GPU
losslessness result.

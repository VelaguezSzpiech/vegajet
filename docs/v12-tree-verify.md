# S3 GPU tree verification (v13 experiment)

Implemented on the standalone `v13/tree-gpu` branch; **not part of release/v1.2**.
`VEGAJET_V12_TREE_VERIFY=1` is the sole opt-in (exact string `1`). Unset, `0`,
`true`, and `10` keep the existing chain/raw executor and allocations unchanged.
There is no GPU in the development VM: CPU references and ROCm compilation are
not GPU execution, end-to-end parity, or throughput evidence.

## Supported configuration and boundaries

- Native raw Ornith target on two gfx906 GPUs, `--spec-gdn`, with either
  `--suffix-decode` or a DFlash2 drafter (`bench ... --draft ...`).
- Target files containing `blk.48.nextn.eh_proj.weight` (MTP), a non-DFlash2
  drafter, `VEGAJET_RAW_ONLY=1` (one-slot recurrent allocation), serving,
  interactive/sweep/profile/dump modes, zero-tax catch-up, INT4 prefill
  activations and INT8 decode wire are rejected with a named error. CPU-only
  builds reject the toggle before opening the model.
- DFlash2 uses the first two distinct valid IDs from each of its first two
  Top-16 draft slots: anchor + two children + four grandchildren = seven rows.
  The second-depth proposal is shared across first-depth branches; proposals
  need not be accurate to preserve greedy acceptance.
- Suffix mode requests at most six draft nodes, with branching limited to two.
  Sparse tries and the context tail can produce fewer rows, including just
  the anchor. No synthetic candidates are added to force seven rows.
- Every batch retains the already-emitted, not-yet-consumed anchor as row zero.
  It emits the accepted children and one bonus token, never more than the
  remaining output-token budget. All target argmaxes are full-vocabulary.

## GPU implementation

### Logical positions and masked attention

The opt-in selects `spec_flash` for the runner (including its raw/chain benchmark
controls), so raw M=1 and tree rows share the split-attention reduction order.
It does not change this option when the toggle is off. Equality against a separate
all-off fused-attention binary remains a required hardware check: floating-point
near ties across different kernel families are not resolved by tree acceptance.

`ornith_tree_attn_qkv_prep_kernel` uses `pos0 + depth[row]` for Q/K RoPE while
writing quantized KV at physical `pos0 + row`. The original prep ABI is unchanged.

`vj_tree_flash_partial_kernel` accepts ancestor-mask and depth operands. It
always exposes committed prefix positions `< pos0` and then only the row's
root-to-self ancestors. It enumerates them in depth order, maps them to physical
KV rows, and computes chunks over the **logical** sequence rather than skipping
siblings in a flattened physical sequence. Per-row chunk lengths use the same
32/64/128/256 thresholds as serial split attention; extra common-grid chunks
write identity partials. The existing merge/quant kernel is reused. The host
checks all rows, chunks, heads and actual scratch capacities before launch.

### Branch-local GDN

`vj_tree_gdn.hsaco` is a separate, gated object, loaded only with the toggle on.
`vj_tree_gdn_conv_kernel` gathers three raw convolution ancestors per row; negative
indices address the immutable pre-verify history. It writes separate post-node
convolution snapshots, without reading a sibling's mutable history.

`vj_tree_gdn_verify_kernel` starts each row from its parent's recurrent state.
Its partition owner writes/reads its own snapshot elements; no inter-WG barrier
or new persistent-barrier protocol is introduced. Seven full node states for one
layer are reused across all layers (14 MiB per GPU), not retained for every layer.
The pre-verify spec-GDN slot is immutable. All nodes record the same FP32
K/V/decay/beta representation used by the existing spec-GDN verifier.

### Acceptance, commit, compaction, catch-up

`accept_tree` follows the unique child equal to each target argmax until there is
no matching child or the output budget is reached. This is the unique longest
greedy path, independent of draft probabilities. This proves the acceptance
rule, **not** equality of GPU floating-point logits with another kernel path.

Only accepted records are gathered into disjoint replay scratch. The original
spec-GDN replay kernel evaluates that path from the immutable base into the
inactive slot. The accepted leaf's convolution history is installed at the
logical next ring slot. Rejected branch state is discarded; no sibling is
committed. Accepted attention KV rows are compacted serially, source-order,
to `pos0 + path_index`, using `plan_tree_commit` byte offsets. Topological row
ordering gives `dst <= src`, so no later source is overwritten.

Suffix catch-up extends its trie with accepted tokens and the bonus only.
DFlash2 target taps are compacted in the same source order before exactly one
`dflash_inject_kv(pos0, accepted_count)` call. The bonus remains pending; rejected
taps and draft-cache suffixes are not exposed as committed prefix state.
Fault checks and the sticky run state precede commit/catch-up. Queued copies and
replay complete and are checked before the active recurrent slot or host output
is advanced. A failed dispatch unwinds without callbacks or later catch-up.

## CPU and compiler checks

Run in this worktree, never the integration checkout:

```sh
cmake -S . -B build-cpu -G Ninja -DVEGAJET_CPU_ONLY=ON
cmake --build build-cpu -j2
CI=1 OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 \
  ctest --test-dir build-cpu -j2 --output-on-failure
cmake -S . -B build-gpu -G Ninja -DVEGAJET_CPU_ONLY=OFF \
  -DVEGAJET_ROCM_PATH=/data/tools/therock712/lib/python3.14/site-packages/_rocm_sdk_devel
cmake --build build-gpu -j2 -- -k 0
```

The VM's active-spin OpenMP sampler latency test can fail under shared CPU load;
its existing `CI` bounds and passive wait policy avoid that unrelated contention.
No latency bound is removed. `vegajet_layer_test` skips without a GPU.

CPU tests cover tree topology, masks, greedy paths, overflow/capacity validation,
branch-local convolution/recurrence versus serial paths, accepted record replay,
physical KV byte copies, and accepted tap/catch-up layout. Tree attention references
include sibling poisoning and logical chunk-threshold boundaries. Existing chain
kernels and their ABIs remain intact; new entry points have resource-gate entries.

## GPU validation (required before enabling in production)

Use a raw target file **without** blk.48 MTP, a supported DFlash2 draft (optional),
and two MI50s. Build with ROCm 7.12 as above. Do not use ROCm 7.14 gate results.
Set absolute paths for these local variables; they are not literal model paths:

```sh
export TARGET=/absolute/path/to/raw-target.gg-uf
export DRAFT=/absolute/path/to/dflash2.gg-uf
export HSACO="$PWD/build-gpu/hsaco"
# Remove inherited experimental flags before each independent run.
for key in $(env | cut -d= -f1 | sed -n '/^VEGAJET_V12_/p'); do unset "$key"; done
unset VEGAJET_RAW_ONLY
VEGAJET_V12_TREE_VERIFY=1 build-gpu/vegajet bench "$TARGET" \
  --hsaco-dir "$HSACO" --spec-gdn --spec-flash --suffix-decode --ctx 4096
VEGAJET_V12_TREE_VERIFY=1 build-gpu/vegajet bench "$TARGET" \
  --hsaco-dir "$HSACO" --spec-gdn --spec-flash --draft "$DRAFT" --dflash-k 2 --ctx 4096
```

The opt-in bench runs five deterministic prompts, **512 tokens each**, comparing
complete token-ID vectors for raw greedy, existing chain verification and tree
verification. Require five `[TREE PARITY ...] ... PASS` lines and a successful
process exit for **each** drafter. Any shorter output or token mismatch is a hard
failure. Each line also reports raw/chain/tree tok/s and tree/chain ratio; these
are observed measurements, not an assumed win. Repeat warmed runs at least three
times, alternate order externally if comparing separate binaries, and compare
medians. Expected direction for a useful optimization is `tree/chain > 1`; a
ratio below one is a regression even if token parity passes. The seven-node
prototype may lose to a chain because of snapshot, replay and host dispatch cost.

Also test: complete rejection, partial acceptance ending on either sibling,
accepted rightmost leaf, all branches containing the same token at different
depths, context tails with one through six free rows, ring wrap at every head,
and poisoned rejected KV/taps/states. Compare committed KV, convolution history,
and recurrent snapshots against serial execution. Inject asymmetric peer faults
before verification and before commit and ensure the requesting process fails
without output/catch-up after the fault. Check threshold contexts around
1920/3840/7680 logical positions, scratch red zones, and startup rejection cases.
Finally compare all-off complete token IDs and scheduling against the base binary.

**Unverified in the VM:** GPU numeric parity, memory/fault behavior on hardware,
all-off runtime byte parity, target/drafter integration, accepted depth, throughput,
power/thermal behavior, and whether this experiment is a performance improvement.

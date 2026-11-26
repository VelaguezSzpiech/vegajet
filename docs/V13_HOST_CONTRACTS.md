# v1.3 host contracts (M1/M2)

`core/tp_sharder.hpp` is opt-in planning only; it does not change the TP2 executor.
Ranks are ordered by the caller. Persist `DeviceProfile::key()` (`uid:` from sysfs
`unique_id`, falling back to `bdf:` with a full canonical PCI identity), never a
runtime ordinal. `device_key()` only reads sysfs; it opens no device nodes.

`profiles_from_json()` reuses schema-1 `hw_profile.json` GPU records and their
legacy field aliases, without the old eight-card limit. New per-GPU fields are
`unique_id`, `usable_vram_bytes`, `pcie_link_gbs`, `numa_node`, and `mclk` (MHz).
Existing `bw_hbm_gbs`/`bw_gbs`, `n_cu`/`cu`, and memory-clock aliases are accepted.
For legacy records, the caller's explicit reserve is deducted from `vram_bytes`.
When `usable_vram_bytes` is present it already excludes the reserve, so no second
reserve is deducted; it must not exceed `vram_bytes` when both are supplied.
Absent link/NUMA information is represented by zero/-1, not guessed. MCLK, CU
and PCIe bandwidth are metadata; row shares use the supplied HBM bandwidth only.

The model adapter supplies **resident packed** byte costs, including all experts,
requested-context KV/state, quant padding and temporary peak allocations. It may
reuse `rig_planner`'s tensor classification, but must not substitute its generic
GGUF-size estimate for these costs. Replicated allocations are charged on every
rank. Attention groups are allocated first, then row tensors in caller order.
Every row allocation charges complete physical tiles (16 or 32), including the
last vocabulary tail. Shards expose logical ranges and physical padded sizes.
Capacity clamping is deterministic; allocation failure names the capacity or
attention error. This is a safe sequential allocator, not a global bin-packing
solver: fragmentation can require changing tensor order or owner allocations.

Q heads have explicit global IDs and a local Q-to-KV map. By default a complete
GQA group is indivisible, including MQA. Ranks can have no heads or rows. KV is
replicated only with `replicate_kv:true`, and is then charged once on each rank
that needs it. Split-K is a recommendation on nonempty row shards only, with
configurable default threshold 240; the owner/executor decides whether to use it.

Owner JSON, passed to `shard_options_from_json`:

```json
{"schema":1,"row_tile":16,"split_k_threshold":240,"replicate_kv":false,
 "tensors":{"ff":{"rows":[320,352,352]},
            "vocab":{"fractions":[0.3,0.35,0.35]}}}
```

Exactly one allocation form per named tensor is permitted. Fractions sum to one;
Hamilton rounding applies in tile/group units. Explicit row counts are exact
logical rows (Q heads for attention), preserve boundaries and cover the tensor.
Owner overrides that exceed capacity fail rather than silently changing shares.
Duplicate/unknown option or tensor names fail. JSON integers must be exactly
representable (at most 2^53-1); internal byte arithmetic is overflow checked.

## Collective and topology contracts (M2)

`core/tp_collective.hpp` emits ABI 1: FP16 full-partial push-only all-to-all,
including a local contribution frame; each rank owns a separate RX allocation.
For N>1 each allocation contains two banks of N source frames. Each payload
stride is `align64(2 * elements)` and epoch flags are in a separate allocation
with one 64-byte line per bank/source. No source polls peer memory or uses remote
atomics. N=1 emits bypass, zero RX/flag bytes and no routes, and the CPU reference
returns the original float values without wire rounding. The reference reuses
the existing TP3 header's FP16 conversion and canonical source-rank sum order.

The reference is a **single-threaded interleaving simulation**, not hardware
coherency evidence. `begin` encodes one source partial; independent `push` calls
can deliver destinations in arbitrary order, including a host-bounce edge.
Publication means the entire payload has arrived before its destination-local
flag. `try_consume` polls only the rank's local inbox and requires that rank to
finish every outgoing push first. A rank consumes e before beginning e+1. Two
parity banks prevent e+1 from overwriting a slow rank's e; reaching e+2 implies
all sources published e+1 and therefore consumed e. The async test randomizes
individual deliveries across 20 epochs without barriers for 2..20 ranks.

Epochs are sequential 32-bit nonzero tags; exhaustion **fails closed**, never
wraps. Any timeout/protocol error poisons the entire group and discards stored
outputs. `result` exposes tentative data only after all ranks retire that epoch;
call it before the rank starts its next epoch. `reset(true)` models external
acknowledgement of complete queue quiescence and clears all banks/tags, restarting
at 1. A real executor must establish quiescence, not trust a Boolean. Constructor
`first_epoch` is for testing a continuation near exhaustion, not a GPU reset API.

Topology uses both directions of the pairwise capability/bandwidth matrix.
Disjoint P2P islands are a deterministic greedy **clique cover**, not connected
components or an optimal partition. Starting from the lowest unassigned rank,
append the strongest compatible peer (ties by rank). Default TP groups each span
one island; each group becomes a pipeline stage. `max_tp` can split an island into
smaller stage groups. Directed absent P2P edges emit `host-bounce`, never
optimistic peer reads. Stage transport describes the proposed leader-to-leader
activation transfer, not an already allocated pipeline schedule. `min_p2p_gbs`
controls grouping only: a valid directed P2P edge below that threshold still uses
peer push for a stage transfer or an explicitly requested flat collective. Force
`host_bounce` explicitly to change that transport.

The printed benchmark collective is deliberately **whole-matrix**, spanning all
`--tp N` ranks even when the separate topology recommendation has multiple stages.
The two APIs are modular: M3 must instantiate one collective on each stage's
induced peer submatrix and map its local ranks to the owner's global rank IDs.
`max_tp`/`stages` affect the topology recommendation, not the flat AR benchmark.

Owner topology JSON:

```json
{"schema":1,"max_tp":4,"min_p2p_gbs":4,"allow_host_tp":false,
 "stages":[[0,1,2],[3,4]],"host_bounce":[[2,3],[3,2]]}
```

`stages` must cover every rank exactly once; a TP group crossing a missing/weak
P2P edge needs explicit `allow_host_tp:true`. `host_bounce` can force directed
edges through the host but cannot force unsupported P2P. Unknown/duplicate keys,
invalid bandwidths, omitted ranks and duplicate ranks fail. Both collective and
topology planners accept arbitrary matrix sizes independently of the old
`topology_planner` scoring policy and its eight-GPU cap; no performance scores
or strategy decisions are imported from it.

Examples (CPU only):

```sh
./build-cpu/vegajet bench --bench-ar --tp 20 --dry-run
./build-cpu/vegajet bench --bench-ar --tp 3 --dry-run --tp-profile profile.json --tp-plan owner.json
```

Profiles use the existing hardware schema's `gpus[].p2p` matrix in owner rank
order. An absent profile means unknown links and host-bounce planning. The CLI
has a 256-rank matrix resource bound (the library has no small fixed-rank bound).
`--bench-ar` defaults to TP2 for legacy compatibility; explicit TP is parsed
strictly. Execution of TP != 2, or of any supplied profile/override plan, is
rejected until M3. Normal Ornith GPU bench/run likewise refuses TP != 2 rather
than silently launching its old two-rank runner. Planning alone is not device
execution. No new automatic policy changes the existing TP2 path.

### M3 executor handoff

* Model adapters must populate packed row/attention byte costs for every layer,
  including quant layouts, all experts, context-dependent state, replicated
  scratch and peak reservations. Bind identities through the BDF selector and
  sysfs unique IDs before allocating; do not persist runtime ordinals.
* Allocate N independent owner-local RX payload/flag mappings using this ABI,
  with local-only polling, explicit payload-before-flag system-scope publication,
  NC/UC mapping policy verified on the actual directed P2P links, and a pinned
  host relay with completed copies before notification on bounce edges.
* Generalize runtime queues, kernel descriptors/arguments, row/head maps and
  reduction sites; N=1 allocates no collective storage and launches no AR kernel.
  Apply owner stage order to explicit model layer boundaries and activation IO.
* Keep consume-before-next-publish and two-bank reuse, fail-stop the whole group
  on a partial failure, invalidate tentative outputs, quiesce all queues and reset
  every bank/epoch before restart. Epoch zero or wrap must never appear on wire.
* Add GPU FP16 oracles for tails and arbitrary supported widths/rank counts,
  delayed producer/receiver tests, exhaustion/reset and timeout injection, real
  directed-link/mapping checks and the ROCm code-object gates. Only then enable
  N-rank execution admission. This VM supplies no GPU execution evidence.

## Verification notes

The strict build needed one pre-existing warning fixed: explicitly initialize
`GgufReader::why` in `core/tokenizer.cpp` (no behavioral change). On this shared
VM use `CI=1 OMP_THREAD_LIMIT=1` for ctest: the existing sampler latency gate
otherwise oversubscribes OpenMP and fails. No timing threshold was changed.
The GPU layer test is an intentional CPU-build skip, not GPU evidence. Two
pre-existing sanitizer fixture problems were corrected without touching GPU
runtime behavior: the fake-ioctl runtime test now supplies a HighMem GPUVM
aperture rather than colliding with ASan's shadow mapping, and the signed DOT8
oracle uses multiplication rather than undefined left-shift of a negative sum.

The baseline `tools/privacy_audit.sh` fails on existing tracked example PCI
identities, model filename examples and full Git history. No suppression or
history rewrite is part of these milestones; the release owner must resolve
that pre-existing gate failure. New source avoids adding private identifiers.

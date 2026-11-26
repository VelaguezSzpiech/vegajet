# v1.3 host contracts (M1)

`core/tp_sharder.hpp` is opt-in planning only; it does not change the TP2 executor.
Ranks are ordered by the caller. Persist `DeviceProfile::key()` (`uid:` from sysfs
`unique_id`, falling back to `bdf:` with a full canonical PCI identity), never a
runtime ordinal. `device_key()` only reads sysfs; it opens no device nodes.

`profiles_from_json()` reuses schema-1 `hw_profile.json` GPU records and their
legacy field aliases, without the old eight-card limit. New per-GPU fields are
`unique_id`, `usable_vram_bytes`, `pcie_link_gbs`, `numa_node`, and `mclk` (MHz).
Existing `bw_hbm_gbs`/`bw_gbs`, `n_cu`/`cu`, and memory-clock aliases are accepted.
For legacy records, the caller's explicit reserve is deducted from `vram_bytes`.
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

## Verification notes

The strict build needed one pre-existing warning fixed: explicitly initialize
`GgufReader::why` in `core/tokenizer.cpp` (no behavioral change). On this shared
VM use `CI=1 OMP_THREAD_LIMIT=1` for ctest: the existing sampler latency gate
otherwise oversubscribes OpenMP and fails. No timing threshold was changed.
The GPU layer test is an intentional CPU-build skip, not GPU evidence.

The baseline `tools/privacy_audit.sh` fails on existing tracked example PCI
identities, model filename examples and full Git history. No suppression or
history rewrite is part of these milestones; the release owner must resolve
that pre-existing gate failure. New source avoids adding private identifiers.

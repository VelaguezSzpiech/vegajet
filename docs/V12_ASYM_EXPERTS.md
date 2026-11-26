# D1: bandwidth-proportional expert intermediate TP

`VEGAJET_V12_ASYM_EXPERTS=1` enables a two-rank split of the 1024 intermediate
rows. Widths are positive multiples of 32 and sum to 1024. The planner uses the
same calibrated, PCI-BDF-matched HBM ceilings as vocabulary/GDN/GQA sharding.
The default is OFF; other values (including `0`) retain 512/512, the original
Down repack, all legacy kernel choices and the existing MTP argument layout.
The new code object is loaded only with the toggle set to exactly `1`.

The split uses Hamilton apportionment with a one-block minimum per rank. Small
bandwidth differences can round to 512/512: 1080/1140 does, whereas 840/972 gives
480/544. MCLK alone does not override the calibrated profile. Check the startup
`[ornith::asym] expert FF=...` line before attributing an effect to imbalance.

## Implementation boundaries

- Gate/Up source offsets use `expert * n_ff_exp + rank_begin`, not
  `expert * (2 * local_ff) + rank * local_ff`. Shared Gate/Up rows and shared
  Down columns use the same per-rank range; MTP experts follow it too.
- Toggle-only Down stores one `{u32 sub-scale, f32 superblock d}` pair per 32
  weights. Slicing through a 256-weight IQ3_XXS block is exact, without decoding
  to float and requantizing. Original layouts and kernels are not modified.
- The fused s224 back kernel statically distributes `8 * local_ff / 32` routed
  tiles plus 96 shared-Down tiles across 224 resident workgroups, then reduces
  arbitrary-width routed Down. Wider LDS buffers cover all widths through 992.
- Nonfused Down supports every batch size, with optional FP16 prepush. MTP's
  Q4A8 Down also handles the actual rank width. The all-reduce remains `[M,D]`:
  both producers publish D outputs per token, never `local_ff` outputs.
- LRU slot/expert identities remain shared across ranks; cache layouts, physical
  row offsets, promotion sizes, repack launches and GU launches are per-rank.
  Capacity estimates conservatively use the larger shard.
- The 512-column-only monolithic superkernel and grouped/tiled Down variants
  are bypassed only when enabled. This can reduce prefill/verify performance;
  the generic path preserves functionality. `--dot8-ab` and golden-layer dumps
  are rejected with this toggle because their fixture ABI assumes 512. Use
  `--dump-logits` / `--dump-hidden` for numerical checks instead.
- A non-224 `--back-wgs` selection falls back to the nonfused generic path.

## CPU and ROCm verification

Build with `-j1`. `test_asym_experts` checks default and proportional plans,
invalid inputs, every 32-row split's Gate/Up source offsets, s224 work coverage,
LRU plane bounds and all 62 rank-specific Down slices against the legacy full
1024-column exact repack, including scales, chunked row ranges and sentinels.
ROCm 7.12 builds and gates the added `vj_moe_asym.hsaco`; no GPU runs are possible
on the implementation VM.

## Owner's MI50 checks

Set `MODEL` to the installed Ornith GGUF, build this branch using ROCm 7.12,
and run from its repository root. Preserve the same calibrated hardware profile
and all other settings for both passes. Start with raw expert weights so the
new fused s224 path is actually exercised:

```sh
VEGAJET_V12_ASYM_EXPERTS=0 VEGAJET_RAW_ONLY=1 VEGAJET_NO_AUTO_DOT8=1   ./build-gpu/vegajet bench "$MODEL" --tp 2 --hsaco-dir ./build-gpu/hsaco   --back-wgs 224 --profile-stages --ar-timeline --pp-stages
VEGAJET_V12_ASYM_EXPERTS=1 VEGAJET_RAW_ONLY=1 VEGAJET_NO_AUTO_DOT8=1   ./build-gpu/vegajet bench "$MODEL" --tp 2 --hsaco-dir ./build-gpu/hsaco   --back-wgs 224 --profile-stages --ar-timeline --pp-stages
```

Compare raw tg128 throughput, per-stage MoE time, and GPU1-vs-GPU0 all-reduce
waiting time. With a genuinely unequal selected split, the intended direction
is less idle waiting and higher raw throughput, not a guaranteed improvement.
Timeline stamping disables prepush/fusion for the stamped pass; use the earlier
unstamped tg128 result for fused throughput. At 512/512 there is no load-balance
benefit; this run instead checks the new layout/kernels against the baseline.

Also repeat with `--no-fuse-back`, and without `VEGAJET_NO_AUTO_DOT8=1` to cover
promotion. Run a normal non-raw bench with `VEGAJET_MOE_LRU=32` and
`VEGAJET_NO_AUTO_DOT8=1` to exercise MTP/verify LRU; use a long prompt to cover
prefill. Toggle-on floating-point reduction grouping changes, and FP16
all-reduce rounding can change: compare logits/hidden vectors with the owner's
numerical tolerance and check greedy output; do not assert bit-identical
on-vs-off tokens. Off-vs-original must remain bit-identical. No numerical or
performance result has been measured on this CPU-only VM.

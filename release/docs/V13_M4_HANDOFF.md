# v1.3 M3 verification and M4 raw N-rank handoff

## Evidence boundary

This change implements and host-tests the generic raw Ornith executor. It has not run on a GPU here. CPU tests and ROCm code-object checks do not establish GPU memory ordering, layer/token parity, serving correctness on hardware, or throughput. No device workload was launched.

## M3 baseline

An isolated `git archive 23bd22f` source snapshot was built under the ignored `build-m3-snapshot/` directory, independently of the M4 working tree. All builds used `-j2`.

- CPU-only `-Wall -Wextra -Wpedantic -Werror`: `100% tests passed, 0 tests failed out of 119`.
- Debug ASan/UBSan, non-recovering sanitizers: `100% tests passed, 0 tests failed out of 119`.
- ROCm 7.12 full bundle: 26 code objects PASS, zero FAIL. Existing register waivers remain visible.
- `vegajet_layer_test` was skipped because the hardware/model fixture is absent.
- First strict run failed `test_async_table_prefetch` at its duplicate/completion assertion under simultaneous builds. The complete strict suite passed on rerun without edits. This is recorded, not classified as an M3 defect or fixed by weakening a test.
- The first combined sanitizer build/test command hit its external time limit near the last test. A standalone complete sanitizer run passed.
- `test_tokenizer_samplers` passed; the known flake did not require a pinned rerun. If it recurs, retain the failure log and run `taskset -c 0,1 ctest --test-dir BUILD -R '^test_tokenizer_samplers$' --output-on-failure`.

No M3 source fix was necessary to pass these gates; no empty M3-fix commit is warranted.

## M4 implementation

- `core/tp_executor.*`: host-only arbitrary-rank layouts, bounded buffer/epoch planning and explicit feature admission.
- `models/ornith/ornith_tp_exec.*`: concrete Ornith raw layouts, owner sharding overrides, global/local GQA maps, GDN value-head striping, per-rank scratch and packed resident capacity checks.
- `models/ornith/ornith_nrank.inc`: separate per-op engine for TP != 2, selected-device profiles, sharded uploads, recurrent/attention execution, routed/shared MoE, FP16 N-rank all-reduce and local vocabulary argmax. Every rank participates even when a shard is empty. N=1 bypasses collectives.
- IQ3 Down handles both embedded-scale and separate-scale layouts. Resident accounting uses the actual packed layout after owner row counts are known, not a linearized padding guess.
- `ops/kernels/vj_ornith_nrank.hip`: explicit-head-map attention and local argmax; both registered in the resource manifest.
- Bench/run and serving carry `--tp-shards FILE` / `--tp-plan FILE`. Session preflight, VramBudget call sites and rig planning use the requested rank count. Protected planner/repacker/stream/oracle files are unchanged.
- The existing TP=2 kernel sources and its main fast-path body remain unchanged; default Ornith execution still selects TP=2. Generic raw code does not change TP=2 kernel ABIs or launch ordering.
- Host tests cover N=1,2,3,4,8: contiguous ranges, explicit replicated head maps, GDN maps, zero-work ranks, collective participation, scratch/state/cache sizes, packed storage, capacity boundaries and named unsupported features.

The admitted mechanism bound is 1..256 ranks, covering the design's 1..200 goal. Available model work and per-device capacity still determine useful/admissible placement. The owner, not this implementation, chooses placement policy.

## Scope and explicit restrictions

The raw executor accepts the validated Ornith geometry (3072 embedding, 32 Q heads, two KV heads, 16 GDN groups/64 value heads, 256 experts/top-8, 1024 expert/shared intermediates), trunk layer metadata and context 1..8192. It uses sequential raw prefill, greedy sampling, Q4_A8 dense weights, native IQ2_S Gate/Up, asymmetric IQ3 Down and Q8 vocabulary/cache storage. Resident weights/state must fit each selected arena. Zero-sized shards skip their kernels, not the collective.

Unsupported speculation/MTP, INT8 wire, W4A4 prefill, super-kernel variants, diagnostics/dumps, interactive/sweep modes and explicit incompatible fusion/experiment flags fail closed with a feature name. TP=2 fast execution rejects owner TP JSON rather than silently ignoring it. Multi-slot generic serving and the optional `VEGAJET_V13_SUPER_NRANK=1` experiment are not implemented.

## Owner commands: three selected MI50s

Run from the checkout on the GPU host. Replace each BDF placeholder with a distinct selected device, in owner rank order; replace `MODEL_FILE` with the model path. Never use these placeholders literally. The clean environment prevents inherited experimental switches from changing the requested path.

```sh
cmake -S . -B build-m4-rocm -DVEGAJET_CPU_ONLY=OFF \
  -DVEGAJET_ROCM_PATH=/data/tools/therock712/lib/python3.14/site-packages/_rocm_sdk_devel
cmake --build build-m4-rocm -j2

# Changing-input FP16 all-reduce diagnostic on the three selected devices.
env -i PATH="$PATH" HOME="$HOME" \
  ./build-m4-rocm/vegajet bench --bench-ar --tp 3 \
  --gpu 'dddd:bb:dd.f,dddd:bb:dd.f,dddd:bb:dd.f' \
  --hsaco-dir ./build-m4-rocm/hsaco

# Generic per-op decode; prints generated token IDs and decoded text.
env -i PATH="$PATH" HOME="$HOME" VEGAJET_RAW_ONLY=1 \
  ./build-m4-rocm/vegajet run "$MODEL_FILE" --tp 3 \
  --gpu 'dddd:bb:dd.f,dddd:bb:dd.f,dddd:bb:dd.f' \
  --hsaco-dir ./build-m4-rocm/hsaco --ctx 8192 \
  --prompt 'Hello' --n-predict 128 --ar-wire fp16
```

Optional owner inputs for raw bench/run/serve:

```json
{"schema":1,"row_tile":32,"replicate_kv":true,"tensors":{"expert_ff":{"rows":[352,352,320]},"shared_ff":{"rows":[352,352,320]},"attn":{"rows":[12,12,8]}}}
```

Save as `shards.json` and append `--tp-shards shards.json`. Unspecified tensors retain planner defaults. `--tp-plan` accepts the existing strict collective/topology schema. Live execution uses selected-device topology, not a saved profile as authorization for peer access. Impossible owner allocations fail; they are never silently rewritten.

## Reproduce M4 gates

```sh
cmake -S . -B build-v13-werror -DVEGAJET_CPU_ONLY=ON \
  -DCMAKE_CXX_FLAGS='-Wall -Wextra -Wpedantic -Werror'
cmake --build build-v13-werror -j2
ctest --test-dir build-v13-werror --output-on-failure -j2

cmake -S . -B build-v13-asan -DVEGAJET_CPU_ONLY=ON -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer -fno-pie' \
  -DCMAKE_EXE_LINKER_FLAGS='-fsanitize=address,undefined -no-pie'
cmake --build build-v13-asan -j2
ctest --test-dir build-v13-asan --output-on-failure -j2

./build-m4-rocm/vj_hsaco_gate ./build-m4-rocm/hsaco/*.hsaco
```

Remaining owner checks: TP=3 AR on actual routes; per-layer and greedy-token parity against the trusted path/reference; repeated serving reset/failure behavior; capacity at the selected context; then any owner-selected placement/tuning experiments. The optional N-rank super-kernel barriers/in-kernel AR are deferred, not claimed complete.

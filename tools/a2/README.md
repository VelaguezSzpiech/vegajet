# A2 audit + first matched A/B harness

Status: **offline audit complete; no GPU/KFD execution performed here.** A2 is already
present in the composed `tree/` working tree (staged integration on top of `75c4a64`).
The pair below produces the missing runtime evidence and fails closed.

## Exact invocation

Prepare outside the GPU window (build, fixture, model path). Then, inside an exclusive
`gpuq` window:

```bash
export VJ_BIN=/path/to/build/vegajet
export VJ_MODEL=/path/to/model-file
export VJ_HSACO_DIR=/path/to/build/hsaco
export VJ_TREE=$(git rev-parse --show-toplevel)
export AB_OUT=$VJ_TREE/../results/a2-first
unset AB_MODE   # default: sweep (one engine load)

gpuq run --purpose "VegaJet A2 matched sweep" --max-sec 900 --loads 1 \
    -- bash -eu "$VJ_TREE/tools/a2/pair.sh"
```

Optional direct layer-test bitwise A/B (second engine load; needs the golden fixture):

```bash
export AB_MODE=both
export VJ_LAYER_TEST_BIN=/path/to/build/vegajet_layer_test
export VJ_GOLDEN_DIR=/path/to/golden-layers
gpuq run --purpose "VegaJet A2 sweep + layer A/B" --max-sec 1200 --loads 2 \
    -- bash -eu "$VJ_TREE/tools/a2/pair.sh"
```

The script refuses to run without `GPUQ_TOKEN` / `gpuq verify`, a readable
`~/local-ai/run`, and with `run/TRIPPED` or
`run/NO-GPU-STOP` present. It also atomically creates
`results/a2-work/.a2-pair.lock`; a stale lock must be removed by hand after
verifying no A2 run is active. It never builds, downloads, hashes the model,
changes caps, releases the window, or runs outside `gpuq`.

## What the sweep actually runs

One engine load; two warm variants on the same weights:

```
--no-gdn-splitv8 --bench-mode raw ; --gdn-splitv8 --bench-mode raw
```

with process environment:

```
VEGAJET_ENABLE_SUPER=0
VEGAJET_RAW_TG512=1
VEGAJET_DISPATCH_LOG=<out>/sweep.dispatch.log
```

`VEGAJET_ENABLE_SUPER=0` is **required on both arms**: the default baseline would use
the M=1 super kernel, while `--gdn-splitv8` disables it, so the default comparison is
not matched. `--bench-mode raw` is only understood as a `--sweep` variant argument
(it is not a `vegajet bench` CLI flag). The raw variant runs the same
`run_greedy_raw()` / `step_batch(M=1)` loop as `tg128`/`tg512`, and the second
variant's `vs_base` line is a real in-process token comparison against the first.

Expected enqueue counts per variant (`AB_TG512=1`): 36 GDN layers x 2 ranks x
671 M=1 steps = **48,312** GDN ConvRec enqueues, i.e. 24,156 per rank.
671 = 128 tg128 decode + 512 tg512 decode + 31 golden-code decode (prompt 4,
`n_gen=32`). With `AB_TG512=0` the total is 11,448 (159 steps). Override with
`AB_STEPS` / `AB_GDN_LAYERS` only for a changed bench/model contract.

## Gates (all fail-closed)

`check_a2_sweep.py` requires, from the dispatch journal:

- baseline phase: only the fused `ornith_gdn_conv_recur_splitv_quant_m1_kernel`
  (grid 32768x1x1, block 256, barrier=1), exactly 48,312 times;
- candidate phase: **zero** fused-kernel enqueues after the first stage1 (no silent
  fallback / partial selection);
- candidate stage1 `ornith_gdn_conv_recur_splitv8_recur_m1_kernel` and stage2
  `..._splitv8_quant_m1_kernel`: exactly 48,312 each, grid 65536x1x1, block 256,
  barrier=1, every stage2 immediately after its same-rank stage1;
- per-rank symmetry (24,156 each);
- canonical enqueue sequence: one baseline fused record == one candidate
  stage1+stage2 pair; every other kernel/grid/block/barrier and S/C order identical
  (seq/kernarg dropped because the extra packet changes them);
- stdout: both variants `ok`, golden parity 32/32, candidate `vs_base=129/129
  (first_diff=-1)`, equal `disp/tok`, ECC ue/de = 0, TMem < 65 C, final telemetry
  PASS or SKIPPED (samples absent);
- engine exit 0; KFD drained after each arm.

`AB_MODE=layer|both` additionally requires `vegajet_layer_test --stage gdn --m 1
--iters 5 --gdn-splitv8` to exit 0 with a `gdn-ab M=1 ... PASS (splitv4==splitv8
bit-exact ...)` row and `{"status":"PASS"}`. That is the direct q8/d+bsum/conv-ring/
S/partial bitwise oracle; the sweep is the real-path reachability/parity/timing oracle.

Outputs: `meta.txt`, `sweep.stdout/stderr`, `sweep.dispatch.log`,
`sweep.command.txt`, `verdict.json`, `summary.txt`, `verdict.txt` (and `layer.*` in
layer modes).

## Audit findings (this snapshot)

Tree `75c4a64` + staged integration; audited hashes:
`models/ornith/ornith_gpu.cpp 3800c9b4…`, `cli/bench.cpp af0123e8…`,
`ops/kernels/vj_ornith_gdn_sv8.hip 59eb02e7…` (identical to `inputs/codex/series/A2.patch`
post-image `776d3555…`), `tests/vegajet_layer_test.cpp 4d9f5177…`.

- Selector wiring is present: `cli/bench.cpp` parses `--gdn-splitv8` and sets
  `gopt.gdn_splitv8`; `use_m1_super` excludes it; the per-layer M=1 GDN block selects
  `gdn_sv8` when both optional-module kernels are resolved and enqueues stage1 then
  stage2 with barrier=1; `conv_head` advance covers the pair.
- **No engine wiring patch is needed.** `VEGAJET_DISPATCH_LOG` already journals every
  enqueue with kernel/grid/block/barrier; the checker uses it as positive runtime
  evidence. Adding counters would touch `ornith_gpu.cpp`, which M1/M2 owners are
  editing.
- Static path is reachable in `tg128`/`tg512` (`run_greedy_raw` -> M=1 `step_batch` ->
  the A2 branch), but it is **not self-proving**: the engine silently falls back to
  Split-V=4 if the optional module is missing, and `disp/tok` is unchanged by the
  second dispatch. Hence the log check.
- A2's `ONE_LAYER_AB.md` engine spot-check uses `--bench-mode raw` on `vegajet bench`;
  the CLI rejects that as an unknown option. Use the `--sweep` variant form above.
- In a single full-bench process `[BENCH tg128] vs_base` compares against its own
  first run (static `trusted_raw128`); only the second sweep variant's line is a real
  A/B. Do not use the full-bench `vs_base` as parity evidence.
- Host kernargs are stage1 104 B / stage2 48 B; the code objects declare 100 B / 48 B.
  The 4-byte pad matches the existing fused 144-vs-140 pattern and is harmless here
  (neither kernel reads implicit args); sizes are recorded, not gated.
- Candidate throughput is expected to be lower (two 256-WG dispatches per GDN layer
  instead of one 128-WG kernel). The harness reports the ratio; it gates correctness,
  not speed.

Caveats: no GPU/device parity is claimed by this audit. AQL barrier visibility and
device bit-equality remain to be demonstrated by the pair. `AB_STEPS` is tied to the
current bench's internal prompts (4/32/129/513); a bench change must update it (the
checker will fail loudly otherwise). Do not clear `TRIPPED`, do not run outside
`gpuq`, and do not combine `--json-out` with `--sweep`.

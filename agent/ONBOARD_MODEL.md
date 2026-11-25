# Onboarding a new model onto VegaJet

This playbook takes a new GGUF architecture to a measured HBM2 utilization figure. A plugin never touches a device.
It turns file metadata into an `ops::ModelDesc`, and `ops::build_step_plan()` emits every op graph from that. All
sharding, split-K and split-KV decisions come from the shared planners, so the plugin carries no model dimensions.

Model files are passed only as CLI arguments or environment variables. Never commit a model path, file name or host
detail.

**Current state.** Steps 1–4 and the spill gate in step 5 run on the CPU build today. The parity check in step 5 and
the benchmark in step 6 need a GPU backend that executes the emitted `OpGraph`. Until the descriptor executor lands,
`Runtime::execute` validates kernel symbols only, and the one GPU-executed model is the hand-scheduled Ornith backend.

## Step 1: inspect

```sh
build-cpu/vegajet inspect "$MODEL"
```

Record the following:

- `general.architecture` and `block_count`;
- attention head counts and `key_length`;
- recurrent keys (`ssm.*`, `full_attention_interval` or `attention.recurrent_layers`);
- expert keys (`expert_count`, `expert_used_count`, `expert_feed_forward_length`, `expert_shared_feed_forward_length`);
- the per-type tensor breakdown.

**Pass:** no `types without a CPU decoder` warning.

## Step 2: map formats

Every disk type needs a CPU decoder in `quant::DecoderRegistry`. Add one with `register_format()` plus a
golden-hash case in `tests/test_quant_repacker.cpp`. Pick native VRAM formats with
`VramBudgetPlanner::plan_roofline_knapsack` (`VJ_Q8_DP4A`, `VJ_Q4_DOT8`, `VJ_IQ_PACKED`). Every plane of every
packed tensor uses `quant::select_tcc_layout`, scale planes included (`core/tcc_stagger.hpp` has the per-plane
helpers).

**Pass:**
- `ctest -R test_quant_repacker` passes.
- `test_tcc_anti_camping` shows 16/16 channels for every plane width you introduce.

## Step 3: map sharding, KV and recurrent state

Fill an `ops::ModelDesc` from metadata only: `d_model`, `vocab`, the per-layer `mixer` (GQA / GDN) and `ffn`
(dense / MoE) lists, attention / GDN / MoE dims, and native formats.

Choose `ExpertShard::SplitFf` (every expert split by FF width) or `WholeExperts` (expert parallel). Prefer
`WholeExperts` when `n_experts` is large relative to the GPU count and `top_k` is small.

Size the KV cache with `kv::plan_kv` and recurrent state with `ssm::describe`.

**Pass:** `build_step_plan()` succeeds for N = 1..4 GPUs, with your measured per-GPU bandwidths in `TpDesc`.

## Step 4: emit the OpDesc graph

Copy `agent/MODEL_TEMPLATE/` to `models/<name>/` and rename the namespace and `kArchitecture`. Then:

- adapt `describe()` to your key and tensor names;
- register the plugin in `models/builtin_models.cpp`;
- add the sources to `vegajet_models`.

`prefill()`, `decode_step()` and `draft_step()` all delegate to `build_step_plan()`.

**Pass:** a unit test on a synthetic GGUF header, like `tests/test_declarative_template.cpp`, checks all of:

- per-layer op histograms;
- AllReduce count: `2 × layers` for N > 1, and 0 for N = 1;
- prefill kinds versus decode kinds;
- shard sums: q heads, FF width, experts and vocab add up across GPUs.

It must also pass the validator in `tests/test_opgraph_multi_arch.cpp`: no norm reads a partial sum, and the only
barrier-free edge is router ‖ shared-expert gate/up.

## Step 5: verify 0 B spill and parity

Every code object your graph binds to must pass `vj_hsaco_gate`:

- `private_segment_fixed_size == 0`;
- VGPRs ≤ 48 for streaming GEMVs and ≤ 64 for persistent and tiled kernels;
- `.text` < 24,576 B per kernel.

Any waiver goes in `ops/kernels/hsaco_gates.txt` and may only ratchet down.

Parity:
- Dump raw float32 logits for a fixed prompt from VegaJet and from a reference implementation.
- Run `python3 agent/verify_model.py --build build-gpu --model "$MODEL" --candidate-logits <file> --reference-logits <file> --vocab <n>`.

**Pass:**
- 0 B spill on every kernel.
- Greedy tokens identical to the reference over the checked window.
- Cosine and argmax agreement inside the gate.
- Lossless speculative modes report `first_diff = -1`.

## Step 6: benchmark HBM2 utilization

Run this only on a host where you own the GPUs, one job at a time, after `vegajet doctor` passes.

```text
HBM2 % = (bytes actually read per token per GPU) × (tokens/s) / (measured streaming ceiling of that GPU)
```

- **Bytes:** count the bytes the kernels read. Do not count row padding (odd-256 B strides), and count each format
  at its real size, for example DOT8 at 4.25 bpw rather than the disk size.
- **Ceiling:** use the per-GPU number from `vegajet soak`, not the datasheet peak. Pass it to
  `verify_model.py --hbm-peak-gbs`, with `--decode-tok-s` and `--gpus`.
- **Asymmetric GPUs:** report each GPU separately; the slowest one sets the pace.

**Pass:** record the result with the commit hash, flags and ceiling in the scorecard. Unrecorded numbers do not go
into docs.

## Checklist

| Step | Command / test | Gate |
| :--- | :--- | :--- |
| 1 | `vegajet inspect` | every tensor type decodable |
| 2 | `test_quant_repacker`, `test_tcc_anti_camping` | bit-exact decode, 16/16 TCC channels |
| 3 | `build_step_plan` N=1..4 | valid descriptor and shards |
| 4 | plugin unit test + `test_opgraph_multi_arch` validator | histograms, AllReduce, shard sums, barriers |
| 5 | `vj_hsaco_gate`, `verify_model.py` | 0 B spill, logits parity |
| 6 | `vegajet bench` + `vegajet soak` | HBM2 % per GPU, recorded |

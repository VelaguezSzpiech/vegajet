# VegaJet CLI reference (v0.1.0-alpha.1)

Every subcommand, flag, environment variable and exit code of the `vegajet` binary. `tests/test_cli_reference.py`
checks this file against the argument parsers in `cli/` and every `getenv` in `core/`, `models/` and `cli/`, in
both directions: a flag or variable added to the code without a row here fails the test, and so does a row here for
something the code no longer reads. It also checks that each subcommand's `--help` text lists its parsed flags.

```
vegajet <subcommand> [options]
vegajet --help | -h
vegajet --version | -v
```

| Subcommand | What it does | Needs a GPU build |
| :--- | :--- | :--- |
| `inspect` | Header-only GGUF report: metadata, tensor patterns, per-type sizes, projected footprints for TP=1..4 | no |
| `plan` | Load through the model plugin, per-GPU footprint, graph summary, code-object validation (`bench --dry-run`) | no |
| `bench` | `plan`, then the GPU bench (tg128 raw, pp512 / pp2048, speculative modes, parity, telemetry) | yes |
| `run` | Tokenize the prompt, plan, execute | yes |
| `doctor` | Read-only host pre-flight (kernel, firmware, PCIe, power, RAS); optional live link check | no (`--live-link`, `--bench`: yes) |
| `soak` | HBM2 arena pattern soak and bandwidth audit; `--p2p` peer-to-peer variant | yes |
| `serve` | OpenAI-compatible HTTP server (chat completions, SSE, tools, reasoning, continuous batching) | yes (except `--mock`) |
| `top` | Live ANSI dashboard of a server's `GET /metrics` and the GPUs' sysfs telemetry (read-only) | no |
| `chat` | Interactive chat REPL against any OpenAI-compatible server (`/stats`, `/clear`, `/save`, `/load`) | no |

On a `-DVEGAJET_CPU_ONLY=ON` build no subcommand opens a device node: `bench` and `run` plan, validate, report that
execution is unavailable and exit with the plan's status; `soak`, `--bench-ar`, `--bench-gemv` and
`serve <model-file>` exit 3; `plan`, `inspect`, `doctor` (read-only checks), `serve --mock`, `top` and `chat` are
complete.

## Global options

| Flag | Meaning |
| :--- | :--- |
| `--help`, `-h` | Top-level usage (exit 0). Each subcommand also takes `--help` / `-h`. |
| `--version`, `-v` | Prints `vegajet 0.1.0-alpha.1` (exit 0). |
| `--gpu SPEC` | GPU selection for `plan`, `bench`, `run`, `doctor`, `soak`, `serve`; see below. Also `--gpu=SPEC`. |

### GPU selection (`--gpu`, `VJ_GPU_DEVICES`)

`SPEC` is `all` (the default), a list of indices and ranges such as `0`, `1`, `0,1`, `0-1`, or `list`. Indices are
positions in the gfx906 enumeration: KFD topology nodes with `gfx_target_version 90006`, ordered by PCI location,
the same order the runtime uses. Duplicates, reversed ranges, indices above 63 and anything else are usage errors
(exit 2), as is an index beyond the GPUs present (the message names the source and the valid range).

* Precedence: `--gpu` beats `VJ_GPU_DEVICES`, which beats `all`. The CLI exports the resolved selection as
  `VJ_GPU_DEVICES` (canonical form) so the KFD runtime and child processes see the same set.
* `--gpu list` prints the enumeration (index, node, render minor, CUs, VRAM, selected) and the active selection with
  its source, then exits 0. It reads sysfs only.
* One independent `N=1` worker per MI50 (for `tools/swarm_router.py`): `vegajet serve <model-file> --gpu 0 --port 8081`
  and `vegajet serve <model-file> --gpu 1 --port 8082`, or the same with `VJ_GPU_DEVICES=0` / `VJ_GPU_DEVICES=1`.

## `vegajet inspect`

`vegajet inspect <model-file> [--all-keys]`. Reads the GGUF header only.

| Flag | Meaning |
| :--- | :--- |
| `--all-keys` | Print every metadata key, not only the summary. |

Exit codes: 0 report printed, 1 file unreadable or not GGUF, 2 missing model argument.

## `vegajet plan`

`vegajet plan <model-file> [bench options] [--json-out FILE] [--gpu SPEC]`. Identical to `vegajet bench --dry-run`:
takes every `bench` option, plans and validates, and never constructs the KFD runtime (on any build). With
`--json-out` it writes a `vegajet.bench.v1` scorecard with `meta.command = "plan"`, `meta.executed = false` and the
plan status; throughput fields stay `null`.

**Rig estimator.** With `--gpus`, `--vram` or `--matrix`, `plan` instead reads the GGUF header only (no tensor data,
no device, any build) and estimates a rig: `vegajet plan <model-file> --gpus N --vram 16|32 [--ctx T] [--kv-bits 16|8]
[--cu 60|64] [--matrix]`. It prints the repacking decision per disk type (the core/quant_registry target, applied
only when the crossover law `bw_new / bw_old > bpw_new / bpw_old` holds and the budget allows; lossless re-layouts
always), the per-GPU VRAM breakdown for TP=N (dense, routed experts, embedding / head, norms, KV, recurrent state,
2 GiB reserve), the largest context that fits, and whether routed MoE experts must be offloaded to host RAM.
`--matrix` prints one line per standard rig (1x16, 2x16, 2x32, 3x32, 4x32, 8x32 GiB). Rig options: `--gpus` 1..8
(default 1), `--vram` 16 or 32 GiB (default 32), `--ctx` tokens (default 8192), `--kv-bits` 16 or 8 (default 16),
`--cu` 60 or 64 (default 60). Exit codes as below; a plan that fits with MoE offload exits 0.

Exit codes: 0 plan fits and code objects validate, 1 model unreadable, plan does not fit, tokenizer or code-object
validation failure, 2 usage error.

## `vegajet bench` and `vegajet run`

`vegajet bench <model-file> [options]` and `vegajet run <model-file> --prompt TEXT [options]` share one parser.

| Flag | Meaning |
| :--- | :--- |
| `--tp N` | Tensor-parallel degree (GPUs per model instance). |
| `--vram-gib G` | VRAM budget per GPU for the planner (default 32). |
| `--bw-gbs B[,B...]` | Per-GPU HBM2 bandwidth for the roofline estimate. |
| `--ctx T` | Context length to plan for (default 32768; the GPU bench caps at 8192). |
| `--reuse R` | Expected expert-reuse factor for the planner. |
| `--draft FILE`, `--draft-model FILE` | Draft model (MTP head or DFlash2 drafter). |
| `--mask-embedding FILE` | Mask-token embedding for the DFlash2 drafter. |
| `--hsaco FILE` | Extra code object to validate / load (repeatable). |
| `--hsaco-dir DIR` | Code-object directory (overrides `VEGAJET_HSACO_DIR`). |
| `--prompt TEXT` | Prompt to tokenize (required for `run`). |
| `--n-predict N`, `-n N` | Tokens to decode (default 32). |
| `--dry-run` | Plan and validate only (what `plan` does). |
| `--json-out FILE` | Write the `vegajet.bench.v1` scorecard; merges into an existing file of the same model and commit. Not with `--interactive`, `--interactive-fifo` or `--sweep` (a scorecard records one configuration; exit 2). |
| `--ar-wire fp16\|int8` | All-reduce wire format. |
| `--fuse-router`, `--no-fuse-router` | Fused MoE router kernel (default on). |
| `--gdn-splitv`, `--no-gdn-splitv` | Split-V GatedDeltaNet decode (default on). |
| `--gdn-splitv8`, `--no-gdn-splitv8` | Opt in or out of the ordered Split-V=8 GDN pair (off by default). |
| `--no-dot8-fwht` | Disable the FWHT weight and activation rotation. |
| `--pf-moe-iq-lds N`, `--pf-moe-iq-lds32 N` | Routed IQ prefill: 0 off, 1 Gate+Up, 2 also Down. First selects 64 rows, second the 32-row diagnostic. |
| `--pf-moe-iq-lds-layer N` | Restrict the IQ prefill candidate to layer N. |
| `--fuse-back`, `--no-fuse-back` | Fused MoE down/combine back end (default on). |
| `--back-wgs N` | Workgroups for the fused back end, 1..240. |
| `--gu-dot8-layers N` | Gate+Up layers promoted to DOT8, 0..48 (0 = automatic). |
| `--spec-flash` | Flash-attention path inside speculative verify. |
| `--prefill-chunk N` | Prefill chunk, 128, 256, 512 or 1024 (default 1024 since `a6ab3bb`; 512 before). |
| `--down-grp-prefill N` | Down-projection grouping in prefill: 0 per-token, 1 grouped exact, 2 grouped fast, 3 LDS-tiled. |
| `--down-grp-verify N` | Down-projection grouping in verify: 0, 1 or 2. |
| `--pf-fa2`, `--pf-fa2-fp16` | FlashAttention-2 prefill (fp32 / fp16 accumulate). |
| `--pf-gemm N` | Prefill GEMM tile, 0, 64 or 128. |
| `--pf-gdn N` | Prefill GatedDeltaNet variant, 0, 1 or 2. |
| `--pf-gu-tiled` | Tiled Gate+Up prefill. |
| `--prefill-ar-int8` | int8 all-reduce in prefill. |
| `--no-fast-prefill` | Disable every fast-prefill option (chunk 128, no FA2 / GEMM / GDN / tiling). |
| `--prefill-dot8` | DOT8 prefill path (scorecard `pp*_dot8_tok_s` and `dot8_prefill_kld`). |
| `--dot8-fwht` | With `--prefill-dot8`: block-64 orthonormal FWHT tile on the dense attention/GDN projections (rotated VJ_Q4_A8 weight copies + rotated INT4 activations; still lossy W4A4, extra ~1.3 GiB VRAM/GPU). Default off. |
| `--dot8-fwht-layers N` | With `--dot8-fwht`: rotate only layers `0..N-1` (0 = all 48; `N=1` isolates one layer for an A/B). |
| `--prefill-dot8 --dot8-fwht` (measured) | Two 150 W MI50s, 48 layers: pp512/pp2048 1,413.86/1,493.45 tok/s; first tokens 314/9338. Mean KL(W4A8 ∥ W4A4) 0.000939 on four 256-token prompts. Lossy, opt-in. |
| `--dot8-clip A` | DOT8 activation clip. |
| `--dot8-scope N` | DOT8 scope: 0 all, 1 dense only, 2 MoE Gate+Up only. |
| `--suffix-decode` | Suffix-tree speculative decoding. |
| `--suffix-max-draft N` | Suffix draft length, 1..15. |
| `--spec-gdn` | Speculative GatedDeltaNet state handling. |
| `--zero-tax-catchup` / `--no-zero-tax-catchup` | Deferred draft catch-up, **default on since v0.1.1** (TP=3 Hybrid tg128 code: 338.90 -> 367.24-368.89 tok/s, tg512 323.09 -> 361.36-361.89, 129/129 lossless). Earlier opt-in measurement: On the two-card 150 W Hybrid tg128 code benchmark: 290.21 / **311.87** / 290.00 tok/s (off/on/off), all 129/129 tokens matched. Not a prose parity or general serving claim. |
| `--fr-spec N` | FR-Spec LM-head shortlist size: 0 or a multiple of 512 up to 131072. |
| `--mtp-k N` | MTP draft length, 1..6. |
| `--gu-grp-verify` | Grouped Gate+Up in verify. |
| `--spec-force-reject` | Reject every draft (speculation overhead A/B). |
| `--dflash-k N` | DFlash2 block, 1..15. |
| `--dflash-p-min P` | DFlash2 acceptance threshold. |
| `--profile-stages` | Per-stage decode timing. |
| `--pp-stages` | Per-stage prefill timing. |
| `--ar-timeline` | All-reduce timeline. |
| `--dot8-ab` | DOT8 vs baseline A/B. |
| `--ab-no-prepush` | A/B: disable the epilogue BAR0 prepush. |
| `--dump-hidden FILE` | Dump hidden states. |
| `--dump-logits FILE` | Dump logits. |
| `--dump-tokens FILE` | Write the generated greedy token ids (one per line after a `#` header) for `tools/token_compare.py`; a write failure fails the run. Not with `--interactive`, `--interactive-fifo` or `--sweep` (exit 2). |
| `--gdn-state-dump FILE` | Dump GatedDeltaNet state. |
| `--prefill-seq` | Sequential (non-chunked) prefill reference. |
| `--interactive` | Keep the model resident and read commands (run, parity, reload of a code object) from stdin (GPU builds). |
| `--interactive-fifo PATH` | Same, reading commands from a named pipe (`tools/vj_bench_warm.sh` drives it). |
| `--sweep "A ; B"` | Run several variants in one warm process; each variant is a flag list and/or `VAR=value` settings, separated by `;`. |
| `--dump-golden-layers DIR` | Capture the golden layer fixture (layer 0 GDN + layer 3 GQA) for `vegajet_layer_test` / `tools/vj_dev.sh` into DIR. |
| `--bench-ar` | All-reduce micro-bench only (GPU builds). |
| `--bench-gemv` | GEMV A/B micro-bench only (GPU builds). |

Before the Ornith GPU runner reserves its per-GPU arenas (2 GPUs, about 30 GiB each), `bench` and `run` check the
selected GPUs (`--gpu` / `VJ_GPU_DEVICES`) against the runner's GPU count, its arena and the planned weights + KV +
state per GPU, and stop with exit 1 and a message stating what is needed and what was found (alpha portability
patch).

Exit codes: 0 success, 1 model unreadable, plan does not fit, the selected GPUs cannot hold the model, tokenizer or
code-object validation failure, 2 usage error, 3 no GPU backend (`--bench-ar`, `--bench-gemv`), runtime open, code-object load or execution failure.

## `vegajet doctor`

`vegajet doctor [--json] [--bench] [--dry-run] [--gsf12] [--os [--root DIR]] [--live-link [--seconds N]] [--gpu SPEC]`. Read-only.

Rig-specific properties are reported as WARN, not FAIL, so other hosts get an informative report: a BAR0 smaller than
32 GiB or above the 44-bit ceiling, fewer KFD P2P links than N-1, a single GPU, and a power cap above the reference
PSU budget (200 W per GPU for N <= 2, 150 W for N >= 3). Device access is checked with `stat()` only (no device node
is opened): a missing `/dev/kfd` or render node, or one the user cannot read and write, FAILs with the owning group and
the `usermod` line that fixes it. ECC errors, HBM temperature and unsafe kernel flags still FAIL. `--gsf12` keeps the
reference rig's strict rule set.

| Flag | Meaning |
| :--- | :--- |
| `--json` | Machine-readable report. |
| `--bench` | Calibrate and save the hardware roofline profile (`$HOME/.config/vegajet/hw_profile.json`; GPU). |
| `--dry-run` | Accepted for symmetry; doctor never writes hardware state. |
| `--gsf12` | G-SF-12 preflight: zero GPU jobs, guard / fan services stopped. |
| `--os` | Rules-only OS report: Linux >= 7.0, `amdgpu` `vm_block_size` / `vm_size` / `noretry` on `/proc/cmdline`, and stat-only `/dev/kfd` access; never opens a device node. |
| `--root DIR` | With `--os`: evaluate the rules against a fixture filesystem root instead of `/`. |
| `--live-link` | Sample the PCIe host link while a GPU kernel runs (idle downclock vs a stuck 2.5 GT/s). |
| `--seconds N` | Live-link sampling window (default 10, at most 3600). |

Exit codes: 0 no failing check, 1 a check failed or the profile could not be written, 2 unknown option.

## `vegajet soak`

`vegajet soak [--seconds 60] [--gib 28.5] [--chunk-mib 512] [--hsaco-dir DIR]` or `vegajet soak --p2p [--gib 4]`.

| Flag | Meaning |
| :--- | :--- |
| `--seconds N` | Soak duration (default 60). |
| `--gib G` | Arena size per GPU (default 28.5; for `--p2p`: payload size, default 4). |
| `--chunk-mib N` | Fill / verify chunk. |
| `--p2p` | Peer-to-peer copy soak between the GPUs. |
| `--hsaco-dir DIR` | Code-object directory. |

Exit codes: 0 soak passed, 1 data-pattern errors found, 2 usage error, 3 no GPU backend, runtime, allocation,
dispatch or verify failure.

## `vegajet serve`

`vegajet serve <model-file> [--gpu SPEC] [--tp N] [--ctx T] [--slots N] [--hsaco-dir DIR] [common]` or
`vegajet serve --mock [--gguf TOKENIZER-FILE] [--slots N] [common]`. On GPU builds, `serve <model-file>` serves the
Ornith-122B (`qwen35moe` / `qwen3_5_moe` / `ornith`) backend live on 2× 32 GB gfx906 (single-slot greedy GPU argmax);
`--slots > 1` continuous batching is active on `--mock`. Endpoints: `GET /health`, `GET /v1/models`,
`POST /v1/chat/completions` (JSON or SSE, `tools` / `tool_choice`, `reasoning_content`), `POST /v1/messages`
(Anthropic Messages API on the same queue and generator: JSON or SSE events, `tool_use` / `tool_result` / `thinking`
blocks, `stop_sequences`, Anthropic error bodies), and `GET /metrics` (Prometheus text exposition).
The model path is a positional argument or `--model`. The tokenizer and chat template come from the GGUF; `--mock`
answers with a canned reply and needs no model (a GGUF given with `--gguf` supplies the tokenizer).

A model file starts the engine registered for its architecture on the selected GPUs, with `--tp` equal to the number
of GPUs selected. This release registers no streaming GPU engine: `serve <model-file>` validates the file, the GPU
selection and `--tp`, then exits 3 naming the architecture (a CPU-only build says it was built with
`VEGAJET_CPU_ONLY=ON`). `--mock` is complete on every build.

| Flag | Meaning |
| :--- | :--- |
| `--model FILE` | Model file (instead of the positional argument). |
| `--mock` | CPU mock engine (no model, no GPU). |
| `--gguf FILE` | Tokenizer / chat-template source for `--mock`. |
| `--host ADDR` | Bind address (default 127.0.0.1). |
| `--port N` | Port, 0..65535 (default 8080; 0 picks a free port, printed in the listening line). |
| `--model-id NAME` | Model name reported by `/v1/models` and in responses. |
| `--slots N` | Concurrent sequences, 1..64 (continuous batching; needs an engine with a batch path, e.g. `--mock`). |
| `--max-queue N` | Queued requests before 503, 1..4096 (default 16). |
| `--tp N` | Tensor-parallel degree, 1..8; must equal the number of selected GPUs. |
| `--ctx T` | Context length, 256..1048576. |
| `--hsaco-dir DIR` | Code-object directory for the engine. |
| `--chat-format F` | `auto` (from the GGUF template), `chatml`, `qwen25`, `qwen3` or `ornith`. |
| `--enable-thinking` | Leave reasoning on in the chat template. |
| `--reasoning-format F` | `auto` (split `<think>` into `reasoning_content`) or `none`. |
| `--request-log FILE` | Append one JSON line per request (`ts`, `id`, `status`, `finish`, token counts, `ttft_ms`, `tpot_ms`, `e2e_ms`, `cancelled`); never prompt or reply text. File created 0600. |
| `--metrics-hw` | Add read-only GPU sysfs telemetry to `GET /metrics`: edge / junction / memory temperature, power, RAS UE / CE counts of umc, sdma, gfx, mmhub, pcie_bif, hdp. |

`GET /metrics` (always on) serves the Prometheus text format: request / failure / cancellation and token counters,
TTFT, TPOT and end-to-end histograms, active batch and queue gauges, prefix-cache and MoE LRU hit rates,
speculative `mean_accept`, and the HTTP front end's counters (`vegajet_http_*`).

Exit codes: 0 clean shutdown (SIGINT / SIGTERM), 1 model, tokenizer or request-log file unreadable, 2 usage error (bad flag,
GPU selection, `--tp` mismatch, `--mock` with a model), 3 no engine for the architecture, engine start failure or
bind failure.

## `vegajet top`

`vegajet top [--url http://127.0.0.1:8080] [--sysfs] [--sysfs-root DIR] [--interval S] [--frames N] [--once]`.
A zero-dependency ANSI dashboard: requests and rates, generated tok/s, TTFT / TPOT p50 and p95 (from the histogram
buckets), batch and queue, prefix-cache and MoE LRU hit rates, speculative `mean_accept`, and a GPU table (edge /
junction / memory temperature, power, RAS UE / CE per block; `UE!` marks uncorrectable errors). It only sends HTTP GETs
and reads sysfs files: it never writes hardware state.

| Flag | Meaning |
| :--- | :--- |
| `--url URL` | Server whose `GET /metrics` is shown (default `http://127.0.0.1:8080`; `http://` only). |
| `--sysfs` | Read the local GPUs' telemetry from sysfs directly (read-only); works without a server. |
| `--sysfs-root DIR` | sysfs mount point for `--sysfs` (default `/sys`; implies `--sysfs`). |
| `--interval S` | Seconds between frames, 0.1..60 (default 1). |
| `--frames N` | Stop after N frames (default: until Ctrl-C). |
| `--once` | One plain frame (no screen clearing, no colors), then exit. |

Exit codes: 0 ok, 1 nothing could be read (`--once` with the server unreachable and no sysfs data), 2 usage error.

## `vegajet chat`

`vegajet chat [--url http://127.0.0.1:8080] [--model NAME] [--system TEXT] [--max-tokens N] [--temperature T]
[--no-stream]`. Reads lines from stdin; replies stream as they are generated (reasoning dimmed on a terminal).
Commands: `/stats` (turns, prompt / completion tokens, last time-to-first-token and tok/s), `/clear` (forget the
conversation, keep the system prompt), `/save FILE` / `/load FILE` (the conversation as JSON, file created 0600),
`/system TEXT`, `/help`, `/quit` (also `/exit`, Ctrl-D).

| Flag | Meaning |
| :--- | :--- |
| `--url URL` | OpenAI-compatible server: `vegajet serve`, the swarm router, ... (default `http://127.0.0.1:8080`). |
| `--model NAME` | Model name sent with each request (default: the server's). |
| `--system TEXT` | System prompt. |
| `--max-tokens N` | Tokens per reply, 1..1000000 (default 512). |
| `--temperature T` | Sampling temperature, 0..2 (default: the server's). |
| `--no-stream` | Wait for whole replies instead of streaming them. |

Exit codes: 0 ok (end of input or `/quit`), 1 server unreachable (`GET /health` fails at start), 2 usage error.

## Environment variables

| Variable | Read by | Meaning |
| :--- | :--- | :--- |
| `VJ_GPU_DEVICES` | CLI, KFD runtime | GPU selection, same syntax as `--gpu` (`--gpu` wins). |
| `VJ_SYSFS_ROOT` | `--gpu list`, GPU selection | Alternative root for the KFD topology scan (tests use a synthetic tree). |
| `VJ_IO_URING` | table prefetch | `0` forces the thread-pool `preadv` backend instead of io_uring. |
| `VEGAJET_MILO` | SUB4 load / offline API | `0` (default) off; `1` fits rank-16 INT3 residual factors at load, only when SUB4 is on. Adds V*x and y+=U*tmp GPU launches for M=1..8 using original Q8 activations. Offline fit_milo also supports rank32 and FP16 storage. Lossy; potentially expensive load-time CPU fit. |
| `VEGAJET_SUB4_DOT8` | Ornith load / standalone dense dispatch | Unset (default) off; `q3` / `q2` decode source Q4_K to float via DecoderRegistry and pack VJ_Q3_DOT8 / VJ_Q2_DOT8, group64. M=1..8 uses non-FWHT Q8-to-Q4 activations. Forces the existing VEGAJET_ENABLE_SUPER=0 behavior and disables peer epilogue prepush. Loses persistent fusion; Q4 fallback remains for larger batches and fused consumers, so sidecars increase arena VRAM (reported per matrix). Load-time option: reload engine to change. |
| `VEGAJET_TCC_XOR` | Ornith load / standalone GEMV | `0` (default) off; `1` stores/reads physical block = logical block XOR (row & 15), on both weight and scale planes. Requires K multiple of512; Q4 K=512 keeps its original W16 kernel; other unsupported shapes use original layout. Standalone Q4 M=1 or SUB4 M=1..8; forces VEGAJET_ENABLE_SUPER=0 behavior. SUB4 metadata expands to one word/block32. Logical FP32 accumulation order unchanged; Q4 fallback copy retained for larger batches/fused consumers. Load-time option: reload engine to change. |
| `VEGAJET_BENCH_CONTEXT` | Warm Ornith `--bench-mode raw` | Decimal prompt token count; unset/`0` preserves the short prompt. Repeats the code prompt to exactly this length, prefills normally, then times decode only. Requires context + timed decode steps <= `max_ctx`; invalid counts fail before the raw measurement. Use `16384` or `32640` for 16K / near-32K with default tg128 and `max_ctx=32768`. Short golden parity remains unchanged; raw `vs_base` is scoped to context and generation length. Default off. |
| `VEGAJET_KV_QUANT` | Ornith GPU executor, Lever 4 | `q8_0` (default, existing layout/launches), `q4_0`, or `turbo3`. Compressed tiers append prepared FP32 KV directly, then grouped attention + LSE merge for all main-model rows. Disables persistent M=1/ring super paths; extra dispatches and slower prefill are expected. Unknown values (including `fp16`) fail. Process/cache lifetime selection; MTP draft stays Q8. |
| `VEGAJET_SPLIT_KV` | Ornith GPU executor, Lever 4 | `1` enables sealed Q8 Split-KV with one padded 16-slot tile per local KV group, including TP straddling. All main-model row counts are handled as M=1 launches. Lossless KV wire copy, lossy Q8 query quantization. Default off. Compressed KV takes priority if both toggles are set; incompatible with `VEGAJET_WIRE_ATTN`. |
| `VEGAJET_KV_CHRONOLOGICAL` | Ornith Lever 4 attention | `1` forces S=1 chronological processing in either opt-in path. Otherwise S=1 below 1920 tokens; above that grouped KVQ uses ceil(60/local KV heads), and sealed per-group launches use S=60, giving at least 240 launched waves per rank/partition. Default off. Does not remove quantization error. |
| `VEGAJET_SPEC_HOBBIT` | `1` enables `MoeOffload::fetch_spec` selection of an already staged lower-precision expert table for cold draft rows (rank > 0). Row 0, hot experts and top-1 use Full; absent low data falls back to Full, never Skip. Default off. Caller supplies low-format table and interprets returned precision using the selected table's `expert_bytes()`. In-place Ornith IQ2_S→DOT8 promotion retains one copy; resident Ornith currently prints a limitation and keeps its exact path. IQ2_S and DOT8 recodings in the TP=2 LRU are mathematically equivalent, not distinct precision levels. |
| `VEGAJET_OFFLOAD_L3_CPU` | `1` enables AVX2 (scalar fallback) expert miss GEMV in `ExpertExecutor` and experimental Ornith routed Down offload; default off. |
| `VEGAJET_OFFLOAD_L2_CUDA` | `1` enables the existing dlopen CUDA worker on NVIDIA device 0 for expert misses, with a fixed bounded matrix-session pool and CPU fallback when enabled; default off. |
| `VEGAJET_ENGRAM_IOURING` | `1` selects io_uring registered O_DIRECT table reads in `TableExecutor`; default off; preadv/buffered fallback when unavailable. Ornith has no Engram table and does not issue table reads. |
| `VEGAJET_BORROWED_KV` | `1` enables fixed-capacity expert hot-set ranking with decay and refresh every eight steps in `ExpertExecutor`; default off. The GPU hot callback and actual KV-space leases remain caller-owned; Ornith does not borrow its KV allocation. |
| `VEGAJET_ENABLE_SUPER` | Ornith GPU bench | Ornith M=1 super-kernels. On by default since `950c137`; `0` turns them off. At `b74f820` and earlier they were opt-in with `1`. |
| `VEGAJET_DECODE_FABRIC` | Ornith GPU engine | `1`: at engine init print read-only host fabric warnings (PCIe link below max or 2.5 GT/s, ACS ReqRedir on upstream bridges, Relaxed Ordering off, MaxPayload < 256, DPM clock latched <= 859 MHz under `high`, single NUMA node on a multi-die CPU). Checks needing root-only config space report `unknown`. |
| `VEGAJET_DISPATCH_CPUS` | Ornith GPU engine | Linux cpulist (for example `0-7,16-23`): pin the host dispatch thread to these CPUs at engine init (unprivileged `sched_setaffinity`). With `VEGAJET_DECODE_FABRIC=1` and no list, the thread is pinned to the NUMA node shared by all GPUs; on a single-node (UMA) host affinity is left unchanged. |
| `VEGAJET_GEMV_STAGGER` | Ornith GPU bench | `1` (opt-in, not bit-identical): standalone M=1 W32 GEMVs rotate each workgroup's starting 32-block tile (k_start = blockIdx*16 mod tiles) for TCC anti-camping. Super-kernel paths are unaffected. |
| `VEGAJET_HSACO_DIR` | KFD runtime | Code-object directory when `--hsaco-dir` is not given. |
| `VEGAJET_DISPATCH_LOG` | KFD runtime | Appends one line per kernel dispatch to this file. |
| `VEGAJET_VRAM_QUARANTINE` | KFD runtime | `<gpu>:<offset_mib>+<len_mib>[,...]` VRAM ranges the allocator never hands out. |
| `VEGAJET_TELEM_MS` | Ornith GPU bench | Telemetry sampling period in ms; `0` disables sampling. |
| `VEGAJET_MOE_LRU` | Ornith GPU bench | Requested maximum hot-expert slots per layer, capped at 21 and at both measured card budgets. Draft/MTP bytes already in arena use consume the 1.1 GiB reserve once; only its remainder stays free. The 2.70 GiB envelope remains a ceiling. Startup reports the actual grant, which may be lower or zero. |
| `VEGAJET_NO_AUTO_DOT8` | Ornith GPU bench | `1` disables automatic Gate+Up DOT8 promotion. |
| `VEGAJET_EVICT_ADAPT` | Ornith GPU bench | `1` enables the adaptive draft-length / eviction controller. |
| `VEGAJET_ROUTER_S240` | Ornith GPU bench | Select the 240-WG router front on the nonsuper path. |
| `VEGAJET_ROUTER_S240_SPLIT` | Ornith GPU bench | Deterministic Split-K factor, 1–8. |
| `VEGAJET_ROUTER_S240_XOR` | One-layer fixture | Repack router and/or Gate+Up planes with matching XOR masks (0–3); production shared planes stay natural. |
| `VEGAJET_EVICT_TRACE` | Ornith GPU bench | `1` enables verified-prefix and route provenance tracing. |
| `VEGAJET_EVICT_ROUTE_IDS` | Ornith GPU bench | `0` disables route-ID capture for overhead comparison. |
| `VEGAJET_V12_SPEC_GOV` | Ornith GPU bench | `1`: drive the draft depth from the measured per-M verify time (`t_ver[M]`) and the live accepted length. Default off (off = unchanged behaviour). |
| `VEGAJET_V12_BATCH_CATCHUP` | Ornith GPU bench | `1`: ingest consecutive MTP drafter prompt and catch-up rows in one M-row forward (`M <= 16`) across all TP ranks instead of per-token M=1 forwards. Default off. |
| `VEGAJET_SPEC_CASCADE` | speculative cascade router | `1`: route a draft cycle to the suffix-trie draft when it has enough confident tokens, else to the neural drafter (verification unchanged, lossless). `0` or unset: off. |
| `VEGAJET_SPEC_CASCADE_MIN` | speculative cascade router | Confident suffix tokens needed for a suffix-only draft, 1..64 (default 4; other values ignored). |
| `VEGAJET_MTP_SUFFIX_MIN4` | Ornith GPU bench | `1`: require up to four continuation tokens (bounded by MTP depth) before a suffix replaces the neural draft. Unset or any other value preserves the alpha threshold: one continuation token normally, up to four in `--zero-tax-catchup`. Default off; target verification is unchanged. |
| `VEGAJET_PREFILL_CODEX_FA2` | Ornith GPU bench | `1`: use the 53-VGPR Codex GQA FlashAttention-2 prefill kernel (`vj_prefill_gqa_fa2`) on `M >= 16` prefill batches (default off). |
| `VEGAJET_PREFILL_CODEX_GDN` | Ornith GPU bench | `1`: use the 128-WG Codex WY/Neumann GDN prefill pair (`vj_prefill_gdn_128wg` + `vj_prefill_gdn_128wg_normq`) on `M >= 16` prefill batches (default off). |
| `VEGAJET_PREFILL_C2` | Ornith GPU bench | C2 macro-batch prefill: `256` or `512` (dense + MoE macro batch M), optionally `:16` or `:32` (planned GDN sub-chunk C, default 32). Runtime GDN still consumes the whole macro batch; sub-chunk device dispatch and MTP/DFlash2 prefill scheduling are not integrated or qualified. Shapes whose scratch does not fit the rows allocated for `--prefill-chunk` keep the current path. Unset, `0` or `off`: current path (default). |
| `VEGAJET_AB_W64` | Ornith GPU bench | `1`: A/B wave64 GEMV variant. |
| `VEGAJET_AB_NO_SPLITKV` | Ornith GPU bench | `1`: A/B without split-KV attention. |
| `VEGAJET_AB_NO_FUSED_ATTN` | Ornith GPU bench | A/B control: `1` disables fused M=1 prep+flash on the nonsuper path. Default off. |
| `VEGAJET_WIRE_ATTN` | Ornith GPU engine | Opt-in normal decode/verify sealed attention. `1` selects Q8; `q4` selects lossy Q4. Unset or `0`: unchanged production path (including default-on super). Selecting wire attention disables super/fused/automatic split/spec-flash attention for that arm; M >= 16 prefill stays on FlashAttention-2. Q8 also quantizes Q: model parity is not established. |
| `VEGAJET_WIRE_ATTN_Q4` | Ornith GPU engine | `1` allows lossy Q4 wire format when wire attention is selected; default off. |
| `VEGAJET_WIRE_ATTN_STRICT` | Ornith GPU engine | `1` aborts on a requested decode/verify fallback, including unsupported batch widths. Default off. Only fully enqueued producer/partition/merge/epilogue sequences count as engaged, not fallbacks or device parity. |
| `VEGAJET_WIRE_ATTN_EXERCISE` | Ornith GPU engine | `1` relaxes the production short-context split floor for explicit diagnostic exercise. Default off. |
| `VEGAJET_WIRE_ATTN_MAX_SPLITS` | Ornith GPU engine | Split ceiling for selected wire attention, 2..64 (default 64). |
| `VEGAJET_TREEWY` | Ornith GPU engine | `1` opts into the sealed GDN tree-WY verifier with `--spec-gdn` for ring verification widths 5, 8 or 12 (context token included). Default off; unset, empty, `0`, `off`, `false` or `no` disables it. Requires both shards' verify/commit, existing norm/quant and independent GDN input-pack modules. Missing modules or unsupported cycles retain sequential verification; a selected-cycle enqueue/commit failure is fatal. CPU/ISA qualification is not device parity or speed acceptance. |
| `VEGAJET_TREEWY_FORCE_WIDTH` | Ornith GPU engine | Diagnostic only: 5, 8 or 12 pads/truncates the draft to width minus one with `--spec-gdn`. Default 0 (no override). Applies to both sequential and TreeWY arms, even with `VEGAJET_TREEWY=0`; omit for natural-use measurements. |
| `VEGAJET_TREEWY_FORCE_ACCEPT` | Ornith GPU engine | Diagnostic only with `--spec-gdn`: `natural` (default), `reject` (zero accepted drafts, context still committed), or `prefix:N` (at most N naturally accepted drafts). `full` is a compatibility alias for natural, never forced acceptance. Applies to both arms and never changes target predictions; omit for parity/timing. |
| `VEGAJET_TREEWY_VERBOSE` | Ornith GPU engine | `1` prints TreeWY configuration, cycle/fallback and acceptance-clamp diagnostics; default off. Final counters describe host dispatch, not verified device execution. |
| `VEGAJET_AB_F2_FRONT` | Ornith GPU bench | `1`: A/B fused front-end variant. |
| `VEGAJET_AB_B2_BACK` | Ornith GPU bench | `1`: A/B fused back-end variant. |
| `VEGAJET_DFLASH2_GPU0` | Ornith GPU bench | Run the DFlash2 drafter on GPU 0 instead of GPU 1. |
| `VEGAJET_DFLASH2_TAP_SHIFT` | Ornith GPU bench | Integer shift of the drafter's target-layer taps (diagnostic). |
| `VEGAJET_DFLASH2_NO_SEL` | Ornith GPU bench | Skip the DFlash2 selector walk (diagnostic). |
| `VEGAJET_DFLASH2_DEBUG` | Ornith GPU bench | Verbose DFlash2 draft / accept logging. |
| `VEGAJET_DFLASH2_BLOCK_MASK` | Ornith GPU bench | DFlash2 drafter block mask. `bidir` or unset: the shipped mask, a bidirectional block on every drafter layer. `zlab`: the reference's per-layer mask (causal block on the `sliding_attention` layers, bidirectional on the full-attention layer) as an opt-in short-context arm; `causal`: causal block on every layer (control). Both arms apply only while `s_pos + B <= attention.sliding_window` (the global window, never a per-layer ring) and refuse to start (`zlab`/`causal`) when the draft GGUF has no `dflash.attention.sliding_window_pattern`; any other value is refused. |
| `VEGAJET_DF_FULL_VOCAB` | Ornith GPU bench | DFlash2 LM head over the full vocabulary instead of the FR-Spec shortlist. |
| `VEGAJET_FR_SPEC_ROWS` | FR-Spec draft head | Draft-vocabulary rows: `32k`, `64k`, `<n>k` or a row count, rounded up to 256 rows per rank and clamped to the vocabulary. `0`, `off` or unset: full vocabulary. Not wired into a runner yet. |
| `VEGAJET_DF_SWEEP` | Ornith GPU bench | Sweep DFlash2 (k, p_min) settings after the bench. |
| `VEGAJET_VA_CLUSTER` | KFD runtime | `1`: map the always-read (HOT) tensors into the first arena chunks (default 5 x 512 MiB) for TLB locality. Off by default. |
| `VEGAJET_VA_HOT_MIB` | KFD runtime | HOT-tier size in MiB with `VEGAJET_VA_CLUSTER` (overrides the default 2560); `0` turns clustering off. |
| `VEGAJET_VOCAB_MUBUF` | Ornith GPU bench | `1`: M=1 greedy vocab argmax through the MUBUF kernel variant. Off by default. |
| `VEGAJET_HEAD_Q4` | Ornith GPU bench, serve | `1`: greedy argmax (all M, so speculative verify agrees with plain steps) through a Q4 copy of the LM head (lossy, full-vocab KL 0.011 vs Q8; raw decode ~+2.5 %, speculative throughput unchanged; ~+1 GiB/GPU). Off by default. |
| `VEGAJET_MEGA` | Ornith M=1 GPU decode | `1` enables the Wave 2 persistent Q3 megakernels at TP=3 (default off). Requires the normal super path, 240 WGs, 60+ CUs/rank, MoE local FF 384/320/320, Q3 GU and sub32 Down on every layer. Incompatible super experiments (`VEGAJET_DN_K4=1`, `VEGAJET_GU_PAIR=1`, forced AR tile tags), route capture, and other unsupported configurations retain the existing path. Three GDN+MoE layers and one GQA+MoE layer use two dispatches per four layers; final AR is folded into layer 47. Embedding, initial RMS and per-rank LM head stay separate: 100 → 27 dispatches/token with logits. Intended lossless; GPU token parity must be checked against the default path. Reserves 4 KiB barrier scratch and a disjoint diagnostic region (20 KiB total/rank). Local Top8 removes 48 full-grid barriers/token; lane-zero generation updates remove one WG barrier per grid handoff. The ordinary super stage replay is suppressed; use `VEGAJET_MEGA_PROFILE=1` for fused stage endpoints and unprofiled A/B for throughput. Warm-switchable. Measured in v0.1.1: raw 151.6 vs 174.4 tok/s (-13%), parity 129/129. |
| `VEGAJET_MEGA_PROFILE` | Ornith raw benchmark diagnostic | `1` with `VEGAJET_MEGA=1` enables critical-path `s_memrealtime` endpoints for one additional fused decode token after the throughput measurement (default `0`, off). Replays the same prompt and the first half of the generated tokens, then profiles the middle context position. Uses 97 disjoint 17-u64 records at scratch offset 4096, the `VEGAJET_SUPER_PROFILE=2` slot ordering, max over all 240 WGs per endpoint, and a separate final-AR record. Prints per-rank GDN/GQA/MoE buckets, record coverage and stage sum; timestamp/barrier/atomic overhead perturbs the diagnostic, which is excluded from tok/s. Ordinary super profiling now uses the same middle context position for comparisons. Unsupported mega configurations print a fallback notice. Warm-switchable; requires the rebuilt host binary and mega module. |
| `VEGAJET_SUPER_WGS` | Ornith GPU bench | Workgroups per M=1 super-kernel launch, 192..240 (default 240; other values ignored). |
| `VEGAJET_SPEC_GATE` | Ornith GPU bench | `0` disables the DFlash2/Hybrid acceptance gate (core/spec_gate.hpp), which runs exact raw single steps while speculation emits fewer tokens per ms than raw. On by default; lossless. |
| `VEGAJET_MOE_Q3_FAST` | Ornith GPU bench | `0` turns off the dedicated Q3 MoE super-kernel. On by default; it is used only at 240 workgroups on layers whose Gate+Up is in the VJ_Q3_IQ2S layout (`VEGAJET_GU_Q3`). |
| `VEGAJET_VERIFY_PROFILE` | Ornith GPU speculative verify | `1`: timestamp the first eligible ring verify for each M (2..16) with queue-boundary `kfd_stamp_clock_u64` on every rank; print stage ms and rank-0 per-layer expert union/pair counts. Default off. Disable between warm runs to re-arm; excluded from `--pp-stages`. Instrumentation adds dispatches and serializes overlapping stage boundaries; use unprofiled A/B for throughput. |
| `VEGAJET_VERIFY_FAST` | Ornith GPU TP=3 speculative verify | `1`: Q3 Gate+Up pairs consecutive routes to each expert, sharing weight loads between two rows with unchanged per-row arithmetic order. **Default on** since 2026-11-27 (W11: Hybrid tg128 378.4 -> 389.6, tg512 371.5 -> 383.0, 129/129); `0` disables. M=2..16 ring only; missing optional `vj_verify_moe.hsaco` or non-Q3 layers retain existing kernels. CPU oracle/gates only; GPU parity and performance pending. |
| `VEGAJET_VERIFY_N3` | Ornith standalone MoE front | `1`: allow N-agnostic F2 and s240 router/shared-GU kernels at TP=3 after checking all ranks. Default off. Existing F2/s240 selectors still required; these are M=1 standalone paths, so this does not change persistent M=1 or M>1 verify. Single-peer AR and fixed-512 Down guards remain. GPU parity pending. Measured in v0.1.1: MTP K=4 drops from 153 to 92 tok/s; Hybrid unchanged. |
| `VEGAJET_SUPER_PROFILE` | Ornith GPU bench | `1`: WG0 stage profile after the raw bench. `2`: select all four `_x` super-kernels even with feature bits clear; additionally print `[SUPER-PROFILE critical-path ...]`, using per-dispatch maximum stage-end `s_memrealtime` over WGs and WG0 start (diagnostic overhead included). Off by default. |
| `VEGAJET_TOP8_WG0` | Ornith M=1 GPU bench | `1`: experimental MoE `_x` kernels compute Top-8 only in WG0 and publish the existing 64-byte ids/weights slot through the existing router rendezvous (opaque entry bit 24). No extra grid barrier or LDS allocation. Default off; same arithmetic. Measured in v0.1.1: raw 172.4 vs 174.3 tok/s (-1.1%), parity 129/129. Compare critical-path `Rtr+ShGU` plus `Top8` and TOTAL: the router stamp moves before the fused rendezvous. Applies to the Q3 MoE super kernel (the decode default) in decode and SUPER_PROFILE; compiled out of the standard MoE `_x` kernel for the code-size cap; other paths unchanged. |
| `VEGAJET_GDN_VSHARD` | Ornith load-time | Unset/empty/`0`: existing BW-proportional GDN split. `a,b,c` (one count per TP rank) overrides V-head counts: positive multiples of model V/K, sum exactly V, no empty K shards. `auto` minimizes max(K-groups/rank bandwidth), retaining the existing split on ties. K groups, physical V-head maps, weight rows and recurrent states follow the same plan. Rejects splits exceeding the persistent Split-V grid. Needs an engine restart; FIFO `run` cannot repartition loaded weights. Boundaries change local projection/AR rounding; require 129/129 code and 32/32 parity. Default off. |
| `VEGAJET_RAW_TOKEN_REFERENCE` | Ornith warm raw bench | `save:PATH` writes the raw code token IDs (129 for tg128); `check:PATH` checks against that file across engine restarts. Missing/malformed references fail the variant, and token mismatches fail parity. Off when unset; otherwise replaces the process-local `vs_base` oracle, retaining the 32-token golden check. Save only from the baseline, then use check for all variants. |
| `VEGAJET_RAW_TG512` | Ornith GPU bench | `1`: also measure raw M=1 over 512 tokens after tg128. |
| `VEGAJET_TP3_EP` | Ornith GPU bench | `0` turns off the TP=3 speculative-verify (M = 2..16) epilogue prepush: the GDN / GQA out-projection GEMMs and the sub32 MoE Down then leave the FP16 peer push to the N-way AllReduce. On by default. |
| `VEGAJET_GU_STREAM` | Ornith GPU bench | `1`: Q3 MoE super-kernel Gate+Up uses MUBUF magnitude/sign/scale loads with uniform descriptors and immediate batch offsets (opaque bit 26 in `vj_ornith_super_x`). Keeps single-tile workgroups and the original FMA/reduction order; intended bit-identical output. Selects only the Q3 experimental kernel in decode and SUPER_PROFILE. Off by default. |
| `VEGAJET_GU_PAIR` | Ornith GPU bench | `1` (TP >= 3): the Q3 MoE super-kernel Gate+Up stage runs in pair mode (one workgroup computes both 16-row tiles of a 32-row quant block and quantizes from LDS: no `mid_f32` round trip, no `pair_cnt` atomic) using the `vj_ornith_super_x` kernels; bit-identical `mid_q8` / `mid_d`. Off by default. |
| `VEGAJET_DN_K4` | Ornith GPU bench | `1` (TP >= 3): the `_x` Q3 MoE Down stage (sub-32 layers) balances 3072 rows over 240 WGs (192 x 13 + 48 x 12), replacing the 8-slot prefetch experiment. A grid handoff preserves aligned 8-row FP16 prepush packets and tags; same slot order 0..7 and FMA chain, bit-identical output expected. Off by default. |
| `VEGAJET_AR_FUSED` | Ornith GPU bench | `1` (default off): TP >= 3, M=1 `_x` super-kernels use per-dword FP16 + 16-bit epoch wire readiness (opaque bit 25), final AR mode 4; doubles wire bytes, skips grid-counted flag/fences. Takes precedence over AR_TILE_PIPE and disables MEGA. Entire steps fall back to flags when epoch + 96 >= 65536 to prevent tag alias. Measured in v0.1.1: raw M=1 107.7 vs 174.3 tok/s (-38%), parity 129/129; stays off. |
| `VEGAJET_AR_TILE_PIPE` | Ornith GPU bench | `force` (TP >= 3, M=1 super-kernel path, A/B only): super-kernel epilogues post one ready tag per 16-byte FP16 packet and consumers wait per chunk instead of on the grid-counted peer flag; uses the `vj_ornith_super_x` kernels. Measured slower at M=1 (-5.9 %, 6 KiB payload, no producer skew), and the multi-row paths do not use super-kernels, so `1` is ignored (one-time note) and the flag path stays. |
| `VEGAJET_TRUST_HW_PROFILE` | Ornith GPU bench | `1`: on TP >= 3, skip the ~3 ms live HBM bandwidth probe and shard from the cached hardware profile (deterministic split). |
| `VEGAJET_BYTES_BREAKDOWN` | Ornith GPU bench | `1`: with the raw benchmark's HBM accounting, print the modelled bytes per token per GPU split by stage (GDN weights, GDN state, GQA weights, router, shared expert, routed Gate+Up, routed Down, LM head). Diagnostic only; no effect on execution. |
| `VEGAJET_MOE_PREFETCH` | Ornith M=1 GPU decode | Bitmask, default off. **Measured slower (W16): `1` 154 tok/s, `2` 138, `3` 132 vs 174 off; diagnostic only.** `1`: the `_x` Q3 MoE super kernel warms L2 with the Stage-1 router and shared Gate+Up weights while workgroup 0 waits for the incoming AllReduce. `2`: it also warms the first four experts' routed-Down weights before Stage 4. `3`: both. Same arithmetic (bit-exact); selects the `_x` Q3 kernel. With `VEGAJET_SUPER_PROFILE=2` the `_x` Q3 kernel also reports MoE sub-stage offsets (`MoE x-sub`). |
| `VEGAJET_PROFILE_DUMP` | Ornith M=1 GPU decode | Path. With `VEGAJET_SUPER_PROFILE=2`, also stamps every workgroup's stage end and workgroup 0's AllReduce wait window (`_x` kernels, critical mode only) and writes them as a binary file (`VJPD` header, per GPU: stage stamps, critical ends, per-workgroup ends, AR wait windows; then 48 layer-type bytes). Enlarges the super scratch by 3.3 MB per GPU. Parse with `scripts/w16_telemetry.py`. |
| `VEGAJET_VERIFY_GEMM_X` | Ornith M>1 verify | Default `1`: the dense M=9..12 Q4A8 verify GEMM uses `ornith_gemm_q4a8_m12_x_kernel`. One 1024-thread workgroup per CU stages the whole activation in LDS once and walks 32-row chunks; bit-identical to the 8-row kernels. Applies when `cols % 32 == 0` and `cols <= 4096`. **Default on** since W18 (Hybrid 390 -> 417 tok/s tg128, 384 -> 411 tg512, 129/129); `0` restores the 8-row kernel. `2` also routes M<=8 (investigation only: W19 showed it is NOT bit-identical for MTP prose, 21/129 -> 19/129). |
| `VEGAJET_SUPER_X` | Ornith M=1 GPU decode | Default `1` (since W21: raw 174.0 -> 175.6 tok/s mean, 129/129): the four M=1 super-kernels come from the lean module `vj_ornith_super_l` (same source as `_x`, every experimental bit compiled out). That module adds the TP=3 AllReduce fast path: both peers' fp16 packets are loaded together, and the RMS weights are L2-touched during the flag spin. It also replaces the post-AllReduce grid barrier with a WG0 broadcast flag. Bit-exact. `0` restores the pristine `vj_ornith_super` kernels. Ignored when an `_x` feature (AR_FUSED, AR_TILE_PIPE=force, DN_K4, GU_PAIR, GU_STREAM, TOP8_WG0, MOE_PREFETCH, SUPER_PROFILE=2) is on. |
| `VEGAJET_GDN_VERIFY` | Ornith GPU engine | Load-time GDN speculative-verify mode: `lumotree` = single-write Split-V verify with recompute-on-reject (same as `--spec-gdn`; **default** since 2026-11-27 unless `VEGAJET_GPU_BATCH>=2` is set at load: Hybrid +3.3 %, lossless, ~783 MiB/GPU less state), `treewy` = that plus the sealed tree-WY verifier (`VEGAJET_TREEWY=1`, TP=2 only), `ring` = per-token ring states (opt-out; required for independent batch rows). |
| `VEGAJET_SPEC_CASCADE` | Ornith GPU engine | `1`/`0` forces the zero-tax suffix cascade on/off (same as `--zero-tax-catchup` / `--no-zero-tax-catchup`): the neural drafter is skipped while the CPU suffix trie returns at least 4 confident tokens. |
| `VEGAJET_SPEC_EVICT` | Ornith GPU engine | `1` turns on lossless EVICT expert-union adaptation for DFlash2 drafting (sets `VEGAJET_EVICT_ADAPT=1` unless already set). |
| `VEGAJET_SPEC_ACCEPTMOE` | Ornith GPU engine | `1` (opt-in, lossy): speculative verify uses historical union-biased routing. Default off. Eligible ring-super batches (M=2..8) retain exact routing in both modes, as before agent F; only standalone verify batches use AcceptMoE. Prefill and independent batch rows are unchanged. |
| `VEGAJET_ACCEPTMOE_MODE` | Ornith GPU engine | `plain` (default) or `commit` (explicit experiment). Plain uses the pre-F kernel and ignores acceptance/recency controls. Commit enables commitment weighting and cold-expert pruning on standalone verify batches. `ACCEPTANCE=1` plus `COLD_PENALTY=1` or `LRU_WINDOW=0` dispatches the identical plain kernel. Re-read every warm step; unknown values fall back to plain. |
| `VEGAJET_ACCEPTMOE_LEGACY_PREFILL` | Ornith GPU engine | `1` (default off): plain or plain-equivalent commit also biases prompt prefill, reproducing the pre-F scope. Diagnostic only: the benchmark raw reference then shares lossy prompt state, so its parity counter is not parity against AcceptMoE-off decoding. Independent batch rows remain exact. |
| `VEGAJET_ACCEPTMOE_KEEP` | Ornith GPU engine | Own top experts kept per draft token, 1..8 (default 2). |
| `VEGAJET_ACCEPTMOE_PENALTY` | Ornith GPU engine | Nonunion multiplier, 0..1 (default 0.25). In commit mode position m uses `1 - acceptance^m * (1 - penalty)`. Ranking bias never becomes an output weight. |
| `VEGAJET_ACCEPTMOE_ACCEPTANCE` | Ornith GPU engine | Commit mode only: conditional acceptance estimate, 0..1 (default 0.75); position m has commitment `acceptance^m`. One restores plain union preference. Zero disables union preference but does not disable cold bias. This is a model parameter, not measured acceptance. |
| `VEGAJET_ACCEPTMOE_COLD_PENALTY` | Ornith GPU engine | Commit mode only: cold-expert multiplier on unprotected ranks, 0..1 (default 0.5); one disables pruning. Own top-KEEP and row 0 are protected. |
| `VEGAJET_ACCEPTMOE_LRU_WINDOW` | Ornith GPU engine | Commit mode only: recency horizon in verify cycles (default 8); zero disables pruning. Frozen per-layer/per-rank metadata resets per request and records the entire proposed union, including rejected drafts. Never-seen experts start cold. No HBM eviction or weight deletion. |
| `VEGAJET_GPU_BATCH` | Ornith bench and serve | `1..8`, unset/`1` keeps single-slot serving. `2..8` enables the request scheduler and fixed physical slots, one batched decode call over active rows (including singleton tails), greedy streaming, per-sequence context `--ctx/8`. Prompt chunks use scratch recurrent state and copy back into the physical slot; KV pointers rebase to that sequence segment. Serve selects the full state arena (`VEGAJET_RAW_ONLY=0`). Bench reports aggregate throughput and raw greedy parity; incompatible with `--spec-gdn` and raw-only allocation. |
| `VEGAJET_OFFLOAD_EXPERT_FRAC` | Ornith load | Fraction `0..1`, default `0` preserves all-resident allocation. Experimental TP=3 sub32 routed **Down slices only**: lowest-frequency experts (startup frequencies tie, id order) remain on host, compact resident Down weights plus one zero expert, misses execute via L2 CUDA or L3 CPU and merge before TP reduction. Gate/Up remains resident. Requires one execution tier enabled. Load-time setting: restart to change; not a warm-run mutation. Floating-point reduction differs; tolerance parity, not bit-exact. |
| `VEGAJET_RAW_ONLY` | Ornith GPU bench | `1`: raw M=1 only; skips the MTP head and uses the smaller raw-only VRAM budget for DOT8 promotion. |
| `VEGAJET_GU_DOT8_LAYERS` | Ornith GPU bench | Number of Gate+Up layers to promote to DOT8 (overrides the automatic choice; `0` = automatic). |
| `VEGAJET_GU_Q3` | Ornith GPU bench | `1` (any value > 0): routed Gate+Up as 3-bit VJ_Q3_IQ2S in place on all 48 layers (~+1.5 % raw). **Lossy** in practice: decode-trace max KL 3.5 vs the IQ2_S path, code parity 15/129. Off by default, including `serve`. |
| `HOME` | doctor, roofline profile | Location of `~/.config/vegajet/hw_profile.json` (unset: a file under `/tmp`). |

## Exit codes

| Code | Meaning |
| :--- | :--- |
| 0 | Success (`--help`, `--version`, `--gpu list`, a plan that fits, a passing check, a clean server shutdown). |
| 1 | No subcommand given; model / tokenizer unreadable; plan does not fit; a doctor check, validation or soak pattern check failed. |
| 2 | Usage error: unknown subcommand or option, bad value, invalid GPU selection, `--tp` mismatch. |
| 3 | GPU backend or engine unavailable (CPU-only build, no engine for the architecture), runtime / execution failure, server bind failure. |

Ornith warm `run ... --bench-mode raw --n-predict N` extends the raw timing run
when N > 128 (including N=4096 for long-context KV tests). The ordinary raw run
remains 128 timed tokens. The spec benchmark retains its fixed verify fixtures.
Lever 4 buffers and kernels are reserved at interactive/sweep initialization;
KV format changes apply only between variants, whose caches are reset and prefilled.

### Lever 5 speculative cycle measurements

`--bench-mode spec` emits `[spec_cycle_ms verifier=... source=suffix|mtp|dflash2]`
with mean draft / verify / catch-up / recompute-on-reject / total milliseconds per
speculative cycle. Timings surround completed device submissions; rejection state
reconstruction is charged only to recompute, not catch-up. MTP deferred ingestion
is catch-up. Raw governor fallback cycles are excluded. The label is the effective
verifier: a TreeWY fallback is reported as lumotree. Verifier state layout and
TreeWY backend are selected at engine load time; restart with
`VEGAJET_GDN_VERIFY=ring|lumotree|treewy` to compare layouts. Warm attempts to change
an incompatible layout fail explicitly. TreeWY currently supports TP=2, so TP=3
cannot provide a TreeWY measurement. No default verifier change is made here.

AcceptMoE is lossy opt-in even though the committed context row is exact. After
zero drafts are accepted, the correction is from exact row 0. After a nonzero
accepted prefix, correction logits and cached state may depend on lossy draft
rows; row-0 exactness alone does not guarantee lossless greedy generation. Use
unbiased settings (`KEEP=8`, or `PENALTY=1 COLD_PENALTY=1`) for exact
standalone routing. These differ from **plain-equivalent** commitment settings
(`MODE=commit ACCEPTANCE=1 COLD_PENALTY=1`), which reproduce the historical lossy
plain policy and dispatch. Plain restores the historical verify path. Historical full-run reproduction
also requires `LEGACY_PREFILL=1`: pre-F routing unintentionally biased prompt
prefill, including the benchmark's own raw reference. Its 129/129 counter therefore
does not prove parity against AcceptMoE-off decoding. Neither mode guarantees
parity for arbitrary prompts or standalone verify widths. The pre-F K=4 result included exact ring batches: it is not evidence
that weighted routing improves MTP acceptance.
HOBBIT fetch selection similarly requires exact target re-verification if its low
format is used in a target verifier. Its policy does not itself provide that replay.

L4 M=1 decode (`VEGAJET_KV_QUANT=q4_0|turbo3` or `VEGAJET_SPLIT_KV=1`) selects the optional `vj_ornith_gqa_kvq.hsaco` module on every rank, retaining baseline GDN/MoE super-kernels. Absent module or forced AR tile tags keeps the standalone fallback. Compressed prefill/verify uses native KVQ; the M=1 copy uses FP32 queries and native Q8/Q4/Turbo3 cache, without padded wire queries. All L4 toggles remain default off. `VEGAJET_KV_CHRONOLOGICAL=1` uses one chronological flash chunk. Compressed attention is lossy; changed query quantization and floating association can change tokens.

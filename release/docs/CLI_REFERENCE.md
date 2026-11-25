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
| `--dot8-clip A` | DOT8 activation clip. |
| `--dot8-scope N` | DOT8 scope: 0 all, 1 dense only, 2 MoE Gate+Up only. |
| `--suffix-decode` | Suffix-tree speculative decoding. |
| `--suffix-max-draft N` | Suffix draft length, 1..15. |
| `--spec-gdn` | Speculative GatedDeltaNet state handling. |
| `--zero-tax-catchup` | Zero-tax catch-up after rejected drafts. |
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
| `VEGAJET_ENABLE_SUPER` | Ornith GPU bench | Ornith M=1 super-kernels. On by default since `950c137`; `0` turns them off. At `b74f820` and earlier they were opt-in with `1`. |
| `VEGAJET_HSACO_DIR` | KFD runtime | Code-object directory when `--hsaco-dir` is not given. |
| `VEGAJET_DISPATCH_LOG` | KFD runtime | Appends one line per kernel dispatch to this file. |
| `VEGAJET_VRAM_QUARANTINE` | KFD runtime | `<gpu>:<offset_mib>+<len_mib>[,...]` VRAM ranges the allocator never hands out. |
| `VEGAJET_TELEM_MS` | Ornith GPU bench | Telemetry sampling period in ms; `0` disables sampling. |
| `VEGAJET_MOE_LRU` | Ornith GPU bench | MoE expert LRU cache slots (default 44 with `--spec-gdn`, else 0). |
| `VEGAJET_NO_AUTO_DOT8` | Ornith GPU bench | `1` disables automatic Gate+Up DOT8 promotion. |
| `VEGAJET_EVICT_ADAPT` | Ornith GPU bench | `1` enables the adaptive draft-length / eviction controller. |
| `VEGAJET_SPEC_CASCADE` | speculative cascade router | `1`: route a draft cycle to the suffix-trie draft when it has enough confident tokens, else to the neural drafter (verification unchanged, lossless). `0` or unset: off. |
| `VEGAJET_SPEC_CASCADE_MIN` | speculative cascade router | Confident suffix tokens needed for a suffix-only draft, 1..64 (default 4; other values ignored). |
| `VEGAJET_PREFILL_CODEX_FA2` | Ornith GPU bench | `1`: use the 53-VGPR Codex GQA FlashAttention-2 prefill kernel (`vj_prefill_gqa_fa2`) on `M >= 16` prefill batches (default off). |
| `VEGAJET_PREFILL_CODEX_GDN` | Ornith GPU bench | `1`: use the 128-WG Codex WY/Neumann GDN prefill pair (`vj_prefill_gdn_128wg` + `vj_prefill_gdn_128wg_normq`) on `M >= 16` prefill batches (default off). |
| `VEGAJET_AB_W64` | Ornith GPU bench | `1`: A/B wave64 GEMV variant. |
| `VEGAJET_AB_NO_SPLITKV` | Ornith GPU bench | `1`: A/B without split-KV attention. |
| `VEGAJET_AB_F2_FRONT` | Ornith GPU bench | `1`: A/B fused front-end variant. |
| `VEGAJET_AB_B2_BACK` | Ornith GPU bench | `1`: A/B fused back-end variant. |
| `VEGAJET_DFLASH2_GPU0` | Ornith GPU bench | Run the DFlash2 drafter on GPU 0 instead of GPU 1. |
| `VEGAJET_DFLASH2_TAP_SHIFT` | Ornith GPU bench | Integer shift of the drafter's target-layer taps (diagnostic). |
| `VEGAJET_DFLASH2_NO_SEL` | Ornith GPU bench | Skip the DFlash2 selector walk (diagnostic). |
| `VEGAJET_DFLASH2_DEBUG` | Ornith GPU bench | Verbose DFlash2 draft / accept logging. |
| `VEGAJET_DF_FULL_VOCAB` | Ornith GPU bench | DFlash2 LM head over the full vocabulary instead of the FR-Spec shortlist. |
| `VEGAJET_FR_SPEC_ROWS` | FR-Spec draft head | Draft-vocabulary rows: `32k`, `64k`, `<n>k` or a row count, rounded up to 256 rows per rank and clamped to the vocabulary. `0`, `off` or unset: full vocabulary. Not wired into a runner yet. |
| `VEGAJET_DF_SWEEP` | Ornith GPU bench | Sweep DFlash2 (k, p_min) settings after the bench. |
| `VEGAJET_VA_CLUSTER` | KFD runtime | `1`: map the always-read (HOT) tensors into the first arena chunks (default 5 x 512 MiB) for TLB locality. Off by default. |
| `VEGAJET_VA_HOT_MIB` | KFD runtime | HOT-tier size in MiB with `VEGAJET_VA_CLUSTER` (overrides the default 2560); `0` turns clustering off. |
| `VEGAJET_VOCAB_MUBUF` | Ornith GPU bench | `1`: M=1 greedy vocab argmax through the MUBUF kernel variant. Off by default. |
| `VEGAJET_SUPER_WGS` | Ornith GPU bench | Workgroups per M=1 super-kernel launch, 192..240 (default 240; other values ignored). |
| `VEGAJET_MOE_Q3_FAST` | Ornith GPU bench | `0` turns off the dedicated Q3 MoE super-kernel. On by default; it is used only at 240 workgroups on layers whose Gate+Up is in the VJ_Q3_IQ2S layout (`VEGAJET_GU_Q3`). |
| `VEGAJET_SUPER_PROFILE` | Ornith GPU bench | `1`: after the raw bench, a per-stage `s_memrealtime` profile of the super-kernels (diagnostic). |
| `VEGAJET_RAW_TG512` | Ornith GPU bench | `1`: also measure raw M=1 over 512 tokens after tg128. |
| `VEGAJET_RAW_ONLY` | Ornith GPU bench | `1`: raw M=1 only; skips the MTP head and uses the smaller raw-only VRAM budget for DOT8 promotion. |
| `VEGAJET_GU_DOT8_LAYERS` | Ornith GPU bench | Number of Gate+Up layers to promote to DOT8 (overrides the automatic choice; `0` = automatic). |
| `VEGAJET_GU_Q3` | Ornith GPU bench | `1` (any value > 0): routed Gate+Up as lossless 3-bit VJ_Q3_IQ2S in place on all 48 layers. Off by default. |
| `HOME` | doctor, roofline profile | Location of `~/.config/vegajet/hw_profile.json` (unset: a file under `/tmp`). |

## Exit codes

| Code | Meaning |
| :--- | :--- |
| 0 | Success (`--help`, `--version`, `--gpu list`, a plan that fits, a passing check, a clean server shutdown). |
| 1 | No subcommand given; model / tokenizer unreadable; plan does not fit; a doctor check, validation or soak pattern check failed. |
| 2 | Usage error: unknown subcommand or option, bad value, invalid GPU selection, `--tp` mismatch. |
| 3 | GPU backend or engine unavailable (CPU-only build, no engine for the architecture), runtime / execution failure, server bind failure. |

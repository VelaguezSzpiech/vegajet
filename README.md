# VegaJet

I had two 32 GB AMD Instinct MI50s and wanted to know how fast they could run a 122B model in my own machine. That question turned into VegaJet. It talks directly to the Linux GPU driver through `/dev/kfd`, without the ROCm runtime in the inference path. It repacks GGUF weights into native `gfx906` layouts (`VJ_Q8_DP4A`, `VJ_Q4_DOT8`, `VJ_IQ_PACKED`) and serves Ornith-122B to OpenAI- and Anthropic-compatible clients. My tuned two-card runs hit **165.22 tok/s raw** at 100 dispatches per token and **357.07 tok/s with Hybrid** on repeated code.

| Ornith-122B on 2× MI50 32 GB | Recorded result | Build / workload |
| :--- | ---: | :--- |
| Raw decode, zero overrides, 128 output tokens | **159.33 tok/s** | `496e61b`; baseline |
| Raw fast path, 128 / 512 output tokens | **165.22 / 156.41 tok/s** | `69ca5eb`; `GU_Q3`, `VA_CLUSTER`, `VOCAB_MUBUF` |
| Hybrid speculative decode, 128 / 512 output tokens | **357.07 / 346.27 tok/s** | `d7bc240`; repeated code, FR-Spec 64K, `--spec-gdn`, suffix drafting |
| DFlash2 alone / with Hybrid | **217.60 / 350.72 tok/s** | `d7bc240`; K=7 drafter on GPU 1, target on **both** cards; solo mean acceptance 5.10 |
| Batched prefill, 512 / 2,048 input tokens | **1,264.57 / 1,344.55 tok/s** | `71cd00e`; first-token IDs 314 / 9338 respectively. ~1,700 tok/s is a W4A8 tiled target, not either measurement. |

These tuned results are from development builds, not a benchmark of the downloadable binary. The speculative run used an external drafter, `--fr-spec 65536 --spec-gdn --zero-tax-catchup --suffix-max-draft 15`, and `VEGAJET_NO_AUTO_DOT8=1`. The tuned code comparisons reported **129/129 greedy tokens matching raw decode** (`first_diff=-1`). That is a token comparison, not a guarantee for every prompt or a 512-token parity test. In the separate zero-override run (`496e61b`), Hybrid accepted 100% of the draft on the repeated-code prompt and measured **311.04 / 295.43 tok/s**. The qualification recorded **0 UE / 0 DE** across the six monitored RAS blocks. The [full scorecard](#full-benchmarks) keeps the other parity oracles and ECC counts separate.

### New in v0.1.1: three cards

v0.1.1 is a source release that adds a third MI50 32 GB card (`--tp 3`). On three cards with the new defaults, Ornith-122B decodes at **174.25 tok/s raw** and **390.50 tok/s with Hybrid** on repeated code (**383.96 tok/s** over 512 tokens). The Hybrid output matched raw decode on all 129 compared tokens, and the six RAS blocks recorded **0 UE / 0 DE**. Three lossless speculative-decode changes are now on by default: the zero-tax suffix cascade, the LumoTree GDN verifier and `VERIFY_FAST` expert-weight reuse in the verify step. On the same three cards they took Hybrid from 338.9 to 390.5 tok/s. The [three-card scorecard](#recorded-3-mi50-32-gb-scorecard-v011) lists the commands, ablations and the levers that were measured and left off. The downloadable binary below is still `v0.1.0-alpha.1` (two cards); build v0.1.1 [from source](#build-from-source).

**[Download the Linux bundle](https://github.com/VelaguezSzpiech/vegajet/releases/download/v0.1.0-alpha.1/vegajet-0.1.0-alpha.1-linux-x86_64.tar.gz) · [Two-command quickstart](#quickstart-install-and-serve-a-local-bundle) · [Full benchmarks and parity](#full-benchmarks)**

## Why I built it

I came to this through PC hardware, then 3D and software, then local inference. I like getting everything I can out of hardware I already own. With 64 GB of HBM2 across the pair, I could fit a local 122B MoE. Finding out how fast it could *generate* took more than fitting the weights. I wrote persistent `gfx906` kernels, repacked weights at load time, and measured how much time the two cards spent exchanging results over their 32 GiB BAR0 peer link. I built the server so you can install it, point your own harness at a streaming endpoint, and try the same model without changing clients.

> **License: PolyForm Noncommercial 1.0.0.** VegaJet is *source-available, not open source*. You may use, modify
> and share it for noncommercial purposes under [`LICENSE`](LICENSE). Commercial use needs a separate license from
> the copyright holders. Third-party material is listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Supported configuration

| Component | Alpha support |
| :--- | :--- |
| GPU execution & serving | 2 × Instinct MI50 32 GB (gfx906), tensor parallel (`TP=2`); v0.1.1 adds 3 × MI50 32 GB (`TP=3`, benchmarked with `vegajet bench`; `serve` at `TP=3` is not re-qualified in this release) |
| GPU memory | 32 GB HBM2 per card; the runner reserves up to 30.5 GiB per GPU |
| Host link | 32 GiB BAR0 per card and PCIe P2P between them |
| Model | Ornith hybrid-MoE backend (`qwen35moe` / `qwen3_5_moe` / `ornith` architecture string) only |
| CPU-only tools | Any Linux host that meets the build prerequisites; no GPU or ROCm needed |

Instinct MI60 32 GB has the same ISA and VRAM size and the planners handle 64 CUs, but the GPU executor has not been
validated on MI60. Instinct MI50 16 GB, Radeon VII and Radeon Pro VII are below the runner's per-card arena and are
CPU-tooling targets in this alpha.

## Quickstart: install and serve a local bundle

Download the [Linux bundle](https://github.com/VelaguezSzpiech/vegajet/releases/download/v0.1.0-alpha.1/vegajet-0.1.0-alpha.1-linux-x86_64.tar.gz) and its [SHA-256 checksum](https://github.com/VelaguezSzpiech/vegajet/releases/download/v0.1.0-alpha.1/vegajet-0.1.0-alpha.1-linux-x86_64.tar.gz.sha256). With the tarball `vegajet-0.1.0-alpha.1-linux-x86_64.tar.gz` in your current directory (outside any source
checkout), installing and serving a local model is two commands (no `sudo` required):

```sh
tar -xzf vegajet-0.1.0-alpha.1-linux-x86_64.tar.gz && ./vegajet-0.1.0-alpha.1-linux-x86_64/install.sh --prefix "$HOME/.local"
"$HOME/.local/bin/vegajet" serve "/path/to/your/Ornith-model" --tp 2 --port 8080
```

Replace the quoted model path with your compatible local Ornith GGUF; the bundle does not include weights. For an OpenAI-compatible client, use `http://127.0.0.1:8080/v1` and the local API key `vegajet-local`. For an Anthropic-compatible client, use `http://127.0.0.1:8080`. Set the model in either client to the ID printed at startup (`Ornith-122B-AD-D4` in the qualified run).

For CPU-only evaluation without gfx906 hardware, replace the second command with
`"$HOME/.local/bin/vegajet" serve --mock --port 8080` (or run `"$HOME/.local/bin/stream-test"`).

The installer verifies the bundle's SHA-256 manifest before and after copying, installs under `<prefix>/opt`, and
creates relative symlinks under `<prefix>/bin`. On GPU bundles, the launcher re-verifies all packaged gfx906
`.hsaco` code objects against `share/vegajet/hsaco/SHA256SUMS` before any GPU subcommand and sets
`VEGAJET_HSACO_DIR` automatically. Run `"$HOME/.local/bin/vegajet" --bundle-check --require-gpu` to inspect bundle
integrity and code-object provenance.

## HTTP server and client APIs

When `vegajet serve` starts, it prints the exact base URLs, model ID, API key placeholder and endpoints for client
harnesses (OpenAI SDK, Anthropic SDK, `curl`, `vegajet chat`, `vegajet top`):

```text
[vegajet] OpenAI server listening on http://127.0.0.1:8080 (model=<model-id>, engine=..., slots=1, gpus=all)
[vegajet]   OpenAI API:    BASE_URL=http://127.0.0.1:8080/v1  MODEL=<model-id>  API_KEY=vegajet-local
[vegajet]   Anthropic API: ANTHROPIC_BASE_URL=http://127.0.0.1:8080  MODEL=<model-id>
[vegajet]   Endpoints:     GET /health  GET /v1/models  POST /v1/chat/completions  POST /v1/messages  GET /metrics
```

What users can actually invoke in `v0.1.0-alpha.1`:

| Endpoint / Tool | Live GPU (`serve <model-file>`) | Mock CPU (`serve --mock`) |
| :--- | :--- | :--- |
| `GET /health` | 200 JSON server status | 200 JSON server status |
| `GET /v1/models` | Lists active model ID | Lists active model ID |
| `POST /v1/chat/completions` | JSON & SSE streaming, `<think>` reasoning split, Qwen `<tool_call>` parsing, stop sequences | Same |
| `POST /v1/messages` | Anthropic Messages API (JSON & SSE events, `thinking` / `tool_use` / `tool_result`, `stop_sequences`) | Same |
| `GET /metrics` | Prometheus text exposition (add `--metrics-hw` for read-only GPU sysfs telemetry) | Same |
| `--request-log FILE` | Mode-0600 JSONL timing/token log (never logs prompt or reply text) | Same |
| `vegajet chat` / `vegajet top` | Interactive OpenAI REPL and live `/metrics` + sysfs TUI | Same |
| Decoding / batching mode | Single-slot (`--slots 1`), greedy GPU argmax (`512/512` MoE, `120192/128128` LM head) | `--slots 1..64` continuous batching |

Library-only components built and unit-tested in `core/` (not yet wired into the live GPU `serve` loop):
multi-slot GPU continuous batching (`core/cont_batch`, `core/prefix_cache`), GBNF/JSON-Schema grammar constraints
(`core/grammar`), multi-format DeepSeek/Llama tool parsers (`core/tool_parser`), and host temperature/top-p sampling
chains (`core/sampler_stack`).

## Hardware and host prerequisites

- Linux with `amdgpu` and KFD; the user must be able to read and write `/dev/kfd` and the
  `/dev/dri/renderD*` nodes (commonly the `render` and `video` groups; `vegajet doctor` names the owning group).
- Two 32 GB gfx906 cards with 32 GiB BAR0 and a KFD P2P link for the shipped `TP=2` path. The host-pinned
  fallback for a host without P2P is not validated in this alpha.
- For the GPU build: CMake 3.18 or newer, a C++17 compiler, Linux kernel headers providing `linux/kfd_ioctl.h`,
  and a ROCm or TheRock installation containing `hipcc`, `clang-offload-bundler` and `llvm-objdump` (qualified on
  TheRock / ROCm 7.12.0; set `-DVEGAJET_ROCM_PATH=<prefix>`).
- For the CPU build: CMake 3.18 or newer and a C++17 compiler. Python 3 adds the Python-based tests.

`vegajet doctor` is a read-only host report on every build: it checks kernel flags, device-node permissions,
PCIe topology, BAR0, VBIOS parity, power caps and 6-block RAS counters. Reference-rig properties (BAR0 size, P2P
link count, VBIOS parity, power cap) are WARN in normal and `--os` reports and FAIL only under `--gsf12`.

## Models

| Family | Example layout | Alpha status |
| :--- | :--- | :--- |
| Hybrid gated-delta-net + GQA MoE (`qwen35moe` class) | 48 layers, 256 experts, shared expert | **GPU-executed & live-served.** Ornith-122B-AD-D4 reference layout |
| Pure MoE (GQA attention, routed experts) | split-FF or expert-parallel | Descriptor plans + CPU tests; no GPU executor |
| Dense GQA | SwiGLU FFN | Descriptor plans + CPU tests; no GPU executor |

The GPU plugin loads an Ornith-122B-AD-D4 hybrid MoE: a Q4_K dense trunk repacked to `VJ_Q4_A8`, Q8_0 token
embeddings, an FP16 router, and IQ2_S / IQ3_XXS routed-expert data. No model file or download URL is distributed
with this repository; you supply a compatible local model file.

## Build from source

CPU-only build and tests (portable, no ROCm or device access):

```sh
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DVEGAJET_CPU_ONLY=ON
cmake --build build-cpu -j
ctest --test-dir build-cpu --output-on-failure
tools/privacy_audit.sh
build-cpu/vegajet --version
```

GPU build (`.hsaco` code objects land in `build-gpu/hsaco`, every object gated on `0 B` private spill, VGPR
ceilings, `< 24,576 B` `.text`, and `vj_isa_scan`):

```sh
cmake -S . -B build-gpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DVEGAJET_CPU_ONLY=OFF -DVEGAJET_ROCM_PATH=/opt/rocm
cmake --build build-gpu -j
build-gpu/vegajet doctor
build-gpu/vegajet bench <model-file> --tp 2 --json-out score.json
build-gpu/vegajet serve <model-file> --tp 2 --port 8080
```

Package a relocatable Linux bundle from a tested build (see [`release/linux/README.md`](release/linux/README.md)):

```sh
release/linux/make-bundle.sh --no-build --build-dir build-gpu --gpu-build --rocm-prefix /opt/rocm --out /tmp/vegajet-dist
```

<a id="full-benchmarks"></a>

## Recorded 3× MI50 32 GB scorecard (`v0.1.1`)

Measured on three AMD Instinct MI50 32 GB cards (`gfx906`, TheRock / ROCm 7.12.0, `TP=3` with bandwidth-proportional
shards, 150 W cap, BAR0 P2P). The scorecard runs used development build `226d3eb`, which has the same defaults as
v0.1.1. The engine stayed resident (`vegajet bench <model-file> --tp 3 --gpu 0,1,2 --interactive-fifo <fifo>`, with
`--draft-model <DFlash2 drafter> --mask-embedding <mask>` loaded) and each line is one warm `run` variant with no
environment overrides. Every run in the session recorded **0 UE / 0 DE**, and the memory temperature peaked at
**57 °C**.

| Metric | 3× MI50 32 GB (Measured) | Notes |
| :--- | :--- | :--- |
| Decode `tg128` raw `M=1` | **174.25 tok/s** (`5.74 ms/tok`, `100 disp/tok`) | `vs_base=129/129`, short prompt 32/32; `real_read = 6.283 GB/tok`, `1,093.9 GB/s` (46.3% of calibrated) |
| Speculative `Hybrid` `tg128` / `tg512` | **390.50 / 383.96 tok/s** (`2.24×` vs raw) | MTP K=4 plus suffix drafting on repeated code; 129/129 vs raw, 117/117 drafts accepted; repeat run 388.78 / 381.49 |
| Same, `VEGAJET_VERIFY_FAST=0` | 376.76 / 370.71 tok/s | Same engine, between the two default runs |
| DFlash2 K=7 alone / with Hybrid (code) | **262.85 / 382.43 tok/s** | 129/129; K=11 alone reached 283.47 tok/s on build `0282b20` |
| Batched decode, 4 independent sequences | **213.66 tok/s aggregate** (53.4 per sequence, `1.23×` raw) | `VEGAJET_GPU_BATCH=4` at launch (keeps the ring verifier); rows agree 4/4, row 0 vs raw 129/129; B=2 is slower than raw (164.49 aggregate) |

How the Hybrid default moved on the same three cards (repeated code, `tg128`, 129/129 each):

| Step | Hybrid `tg128` | Default since |
| :--- | ---: | :--- |
| Start of the v0.1.1 cycle | 338.9 tok/s | |
| + zero-tax suffix cascade (skip the neural drafter while the suffix trie has >= 4 confident tokens) | ~368 tok/s | v0.1.1 (`--no-zero-tax-catchup` opts out) |
| + LumoTree GDN verifier (single-write Split-V verify with recompute-on-reject, ~783 MiB/GPU less state) | ~382 tok/s | v0.1.1 (`VEGAJET_GDN_VERIFY=ring` opts out; batch rows keep ring) |
| + `VERIFY_FAST` (Q3 Gate+Up reuses each expert's weights across verify rows routed to it) | **390.5 tok/s** | v0.1.1 (`VEGAJET_VERIFY_FAST=0` opts out) |

These levers were measured on the same rig and stay **off** by default. Each is documented in
[`release/docs/CLI_REFERENCE.md`](release/docs/CLI_REFERENCE.md):

| Opt-in lever | Measured result |
| :--- | :--- |
| `VEGAJET_AR_FUSED=1` (in-band epoch FP16 AllReduce wire) | raw 107.7 tok/s (-38%): per-dword polling over BAR0 costs more than one flag |
| `VEGAJET_AR_TILE_PIPE=force` | raw -5.85% |
| `VEGAJET_TOP8_WG0=1` | raw 172.4 tok/s (-1.1%): the serial top-8 delays Gate+Up |
| `VEGAJET_MEGA=1` (fused persistent megakernel, 27 disp/tok) | raw 151.6 tok/s (-13%): every fused MoE stage is slower |
| `VEGAJET_SUB4_DOT8=q3` | raw 92.0 tok/s and lossy (32-token parity 6/32): forces standalone launches |
| `VEGAJET_SPEC_ACCEPTMOE=1` | Hybrid 275 tok/s, lossy (122/129) |
| `VEGAJET_VERIFY_N3=1` | MTP K=4 drops from 153 to 92 tok/s |
| `VEGAJET_KV_QUANT=q8_0` / `q4_0` / `turbo3` (short context) | 173.7 / 162.4 / 47.7 tok/s; turbo3 is lossy (122/129). Long contexts were not benchmarked in this release |

## Recorded 2× MI50 32 GB scorecard (`v0.1.0-alpha.1`)

The zero-override scorecard below is from benchmark build `496e61b`, not a new benchmark of the release binary.

Measured on dual AMD Instinct MI50 32 GB (`gfx906`, Linux 7.0, TheRock / ROCm 7.12.0, `512/512` MoE split,
`120192/128128` bandwidth-proportional LM-head split, `<= 200 W` cap, PCIe Gen3 x8 + Gen3 x16 host links, dual
32 GiB BAR0 P2P, `0 B` private spill across all 21 `.hsaco` modules):

| Metric | 2× MI50 32 GB (Measured) | Notes / Flags |
| :--- | :--- | :--- |
| Decode `tg128` raw `M=1` (packaged default) | **159.33 tok/s** (`6.28 ms/tok`, `100 disp/tok`) | `vegajet bench <model-file> --tp 2` (zero env overrides) |
| Decode `tg128` / `tg512` raw `M=1` (raw fast-path) | **165.07 / 160.76 tok/s** (`6.06 / 6.22 ms/tok`) | `VEGAJET_RAW_ONLY=1 VEGAJET_GU_Q3=1 VEGAJET_VA_CLUSTER=1 VEGAJET_VOCAB_MUBUF=1 VEGAJET_SUPER_WGS=240` |
| Speculative decode `Hybrid` (`tg128` / `tg512`) | **311.04 / 295.43 tok/s** (`1.95×` vs raw) | Packaged default (`p_rep_code`, `100%` acceptance) |
| Prefill `pp512` / `pp2048` (`M=1024`, W4A8) | **1,258.66 / 1,342.88 tok/s** (`WY_GDN=ON`) | Packaged default (`pp2048 / pp512 = 1.07`) |
| HBM2 utilization, raw decode (vs calibrated soak) | **54.3%** (`883.1 / 1,626.2 GB/s` calibrated) | `real_read = 5.543 GB/tok`; 60 s full-arena soak: `1,709.0 GB/s` (`834.4 + 874.6 GB/s`) |
| AllReduce microbench (`--bench-ar`, FP16 / INT8) | **7.75 / 17.30 µs/AR** (`verify_errs = 0`) | Push-only dual-BAR0 P2P, 64 rounds × 2 ranks × 3072 elems |
| Thermals & 6-block RAS ECC | **48.0 °C peak TMem**, **0 UE / 0 DE** | Isolated model delta: GPU 0 `umc CE +25` (known weak row), GPU 1 `CE 0` |

### Exact parity oracles (named explicitly)

| Comparison | Prompt & Length | Result |
| :--- | :--- | :--- |
| Packaged default (`ENABLE_SUPER=1`) vs Tier-1 `llama.cpp` (`golden::kCodeRef32`) | `"def fibonacci(n):"` (32 greedy tok) | **32/32 PASS** |
| Forced `ENABLE_SUPER=0` vs Tier-1 `llama.cpp` (`golden::kCodeRef32`) | `"def fibonacci(n):"` (32 greedy tok) | **32/32 PASS** |
| Packaged default (`ENABLE_SUPER=1`) & `ENABLE_SUPER=0` vs `golden::kProseRef32` | `"The capital of France is"` (32 tok, informational) | **16/32** (identical 32-tok output on both) |
| Packaged default `Hybrid` vs packaged default raw (`ENABLE_SUPER=1`) | `p_rep_code` (129 greedy tok) | **129/129 (`first_diff=-1`) PASS** |
| Forced `ENABLE_SUPER=0` vs packaged default `ENABLE_SUPER=1` | `p_rep_code` (129 greedy tok) | **15/129 (`first_diff=12`)** — FP32 accumulation-order drift after token 11 |
| Packaged default `MTP K=4` (`M=5` unfused verify) vs `ENABLE_SUPER=1` raw (`M=1`) | `"The capital of France is"` (129 tok) | **5/129 (`first_diff=2`)** — same unfused `M>1` vs super `M=1` FP32 drift |

## Explore the engine with your coding agent

Start with the [CPU-only build](#build-from-source). Ask your agent to trace one streamed request from [`cli/serve_cmd.cpp`](cli/serve_cmd.cpp) into [`core/server.cpp`](core/server.cpp), then run [`test_server_sse`](tests/test_server_sse.cpp) before changing it. For model work, start with the [onboarding playbook](agent/ONBOARD_MODEL.md).

## Layout

```text
core/             GGUF reader, decoders/packers, roofline knapsack, TCC layout, sharder, launch planner,
                  KV / SSM planners, speculative decoding, KFD runtime, sampler, OpenAI & Anthropic server, metrics
ops/              OpDesc / OpGraph descriptors, step planner, gfx906 kernels & sealed alpha ISA headers
models/           model plugins (Ornith GPU backend + live streaming server + plan-only plugins)
cli/              vegajet: inspect | plan | bench | run | serve | doctor | soak | top | chat
agent/            onboarding playbook, model template, verification harness
hardware/         read-only gfx906 / P2P / ECC host checks
contrib/          optional host helpers (udev/systemd templates; power-cap script writes only with --apply)
tests/            CPU & GPU unit/integration tests (ctest)
universal_draft/  experimental 3-kernel zero-spill universal-IR / VM prototype with CPU + live GPU tests
release/          CLI reference, operations guide, alpha notes and Linux bundle tooling (release/linux/)
```

## Safety

VegaJet drives the GPUs without the HIP runtime's guard rails. Before you run anything:

- Run `vegajet doctor`; fix every FAIL and read the WARN items.
- Run one GPU job at a time, on the two-card configuration described above.
- MI50 and MI60 are passively cooled server cards: provide forced airflow. VegaJet does not control fans. A
  completed run ends with a fatal telemetry gate on power (`<= 205 W` 10 s rolling avg for `N <= 2`, `< 270 W` peak)
  and HBM temperature (`< 65 °C`), and live serving checks temperature and ECC deltas between requests, but these do
  not replace an independent hardware thermal guard.
- Stop on any uncorrectable ECC error or kernel fault.

The engine never writes sysfs and contains no clock, voltage, fan, VBIOS or memory-timing tooling.

## Projected scaling: other models and rigs

**Estimates, not benchmarks or alpha support.** These roofline scenarios assume the model-specific loaders and kernels are implemented. The two-card budget is 64 GiB HBM2; the three-card scenario assumes 96 GiB HBM2, 64 GiB host DDR4 and a 150 W cap per card. Speculative rates depend on draft acceptance; offloaded rates also depend on how often experts must be fetched. No three-card rate below has been measured.

| Model and quantization | MI50s | Raw decode (tok/s) | Speculative decode (tok/s) | Prefill, 2,048 inputs (tok/s) | Assumed placement |
| :--- | ---: | ---: | ---: | ---: | :--- |
| Ornith-1.5-122B, Q4_0 / UD-Q4_K_XL | 3 | 225–333 | 450–612 | 1,620–3,420 | 70–74 GiB in HBM2 |
| Qwen3.8-Flash-Next, IQ4_XS | 2 | 155–185 | 264–351 | 1,610–1,975 | Trunk in HBM2; 4-bit PLE in host RAM |
| Qwen3.8-Flash-Next, UD-IQ4_XS | 3 | 318–373 | 550–836 | 3,000–6,833 | 62–70 GiB in HBM2; ~27 GiB pinned host RAM |
| Tencent HY4-Flash, IQ2-AH / Q3_K | 3 | 276–334 | 514–732 | 1,593–3,628 | 67–86 GiB in HBM2; ~33 GiB for 4-bit Engram in RAM |
| Ornith-1.5-35B-A3B, Q8_0 | 3 | 407–623 | 779–1,102 | 6,000–13,667 | 35.4 GiB in HBM2 |
| Qwen3.8-27B Dense, W4A4 dot8 | 1 | ~45 | 85–110 | — | 17.5 GiB; fits on one card |
| Qwen3.8-27B Dense, W4A4 dot8 | 3 | 109–158 | 333–396 | 889–1,519 | Three-card model parallelism, not three independent workers |
| GLM-5.3-Flash (320B) / DeepSeek-V4.1-Flash (552B) | 3 | 35–62 | 55–95 | 450–600 | Cold experts streamed from host RAM or NVMe |

The optimistic raw-decode scenarios for three cards reach 530+ tok/s for Qwen3.8-Flash-Next and 411+ tok/s for HY4-Flash; neither is a measured result. The experimental three-kernel Universal Engine is a starting point for this work, not a working loader for every model in the table.

## What I’m working on next

v0.1.1 got the three-card engine to **174.25 tok/s raw** and **390.50 tok/s Hybrid**. That is short of the ~200 tok/s raw target I set for this release. This release didn't touch prefill, so the 2,000+ tok/s prefill target is still open. Next:

- Raw `M=1` decode on three cards. The critical path is about 5.7 ms/token: MoE Gate+Up ~0.8 ms, MoE Down ~0.7 ms
  and about 0.9 ms of AllReduce across the 48 layers.
- The speculative verify step. A 12-row verify still costs about 30 ms/cycle, and routed-expert weight traffic
  dominates it.
- Single-GPU multi-worker serving, and re-qualifying `vegajet serve` at `TP=3`.
- Wiring multi-slot GPU continuous batching, host temperature/top-p sampling, GBNF grammar constraints, and
  multi-format tool parsers from `core/` into live GPU `vegajet serve`.

## Credits and acknowledgments

I’m **Velaguez Szpiech (Vela)**, VegaJet’s creator and maintainer. **Gemini (Neon)** was my primary coding partner on the runtime, the kernels, and the live MI50 runs. **Claude** helped me build the serving APIs, platform tools, and CI. **Codex** worked with me on kernel and systems engineering, qualification, and packaging the installer.

Thank you, **Gemini Deep Research**, for helping me understand the Vega 20 architecture and choose which experiments to run on my cards.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

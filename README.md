# VegaJet

A bare-metal LLM inference engine for AMD gfx906 GPUs (Vega 20). VegaJet talks to `/dev/kfd` directly instead of
going through the HIP runtime. It repacks GGUF weights at load time into three native formats, `VJ_Q8_DP4A`,
`VJ_Q4_DOT8` and `VJ_IQ_PACKED`, and runs tensor-parallel decode with peer-to-peer AllReduce over PCIe BAR.

> **Alpha scope (`v0.1.0-alpha.1`).** GPU execution and live GPU serving in this release cover the hand-scheduled
> Ornith hybrid-MoE backend on exactly **two 32 GB gfx906 cards**. The CPU toolchain (build, tests, `inspect`,
> `plan`, `doctor`, `serve --mock`, `top`, `chat`) is portable. One-GPU, 16 GB, Radeon VII, 3–4 GPU execution and
> multi-slot GPU continuous batching are on the [Roadmap](#roadmap). Full caveats are in
> [`release/ALPHA_NOTES.md`](release/ALPHA_NOTES.md).

> **License: PolyForm Noncommercial 1.0.0.** VegaJet is *source-available, not open source*. You may use, modify
> and share it for noncommercial purposes under [`LICENSE`](LICENSE). Commercial use needs a separate license from
> the copyright holders. Third-party material is listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Supported configuration

| Component | Alpha support |
| :--- | :--- |
| GPU execution & serving | Exactly 2 × Instinct MI50 32 GB (gfx906), tensor parallel (`TP=2`) |
| GPU memory | 32 GB HBM2 per card; the runner reserves up to 30.5 GiB per GPU |
| Host link | 32 GiB BAR0 per card and PCIe P2P between them |
| Model | Ornith hybrid-MoE backend (`qwen35moe` / `qwen3_5_moe` / `ornith` architecture string) only |
| CPU-only tools | Any Linux host that meets the build prerequisites; no GPU or ROCm needed |

Instinct MI60 32 GB has the same ISA and VRAM size and the planners handle 64 CUs, but the GPU executor has not been
validated on MI60. Instinct MI50 16 GB, Radeon VII and Radeon Pro VII are below the runner's per-card arena and are
CPU-tooling targets in this alpha.

## Quickstart: install and serve a local bundle

With the release tarball `vegajet-0.1.0-alpha.1-linux-x86_64.tar.gz` in your current directory (outside any source
checkout), installing and serving a local model is two commands (no `sudo` required):

```sh
tar -xzf vegajet-0.1.0-alpha.1-linux-x86_64.tar.gz && ./vegajet-0.1.0-alpha.1-linux-x86_64/install.sh --prefix "$HOME/.local"
"$HOME/.local/bin/vegajet" serve <model-file> --tp 2 --port 8080
```

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

## Recorded 2× MI50 32 GB scorecard (`v0.1.0-alpha.1`)

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

## Roadmap

Items planned after `v0.1.0-alpha.1`:

- Intra-superkernel barrier/L2-cache refinement (Step 3B, deferred to `v0.1.1`).
- GPU executor support for 1, 3 and 4 cards (including TP=3 using the sealed `vj_ar_tp3_alltoall` ISA header and
  single-GPU multi-worker serving).
- Wiring multi-slot GPU continuous batching, host temperature/top-p sampling, GBNF grammar constraints, and
  multi-format tool parsers from `core/` into live GPU `vegajet serve`.

## Credits and acknowledgments

VegaJet `v0.1.0-alpha.1` was engineered collaboratively across three AI development lanes under human direction:

- **Gemini (Neon)** — bare-metal `/dev/kfd` runtime, gfx906 super-kernels, live 2× MI50 execution/serving,
  hardware qualification, and release integration.
- **Claude** — platform architecture, tokenization/sampling, OpenAI & Anthropic serving APIs, observability (`top`,
  `chat`, `/metrics`), CI regression suite, and legal/export tooling.
- **Codex** — gfx906 ISA forge deliverables, systems qualification (`M4`–`M6`, `M9`), universal VM 3-kernel zero-spill
  redesign, guard-latch fixtures, and the offline Linux release bundler/installer.

**Research acknowledgment:** Architectural design, gfx906/Vega 20 ISA analysis, HBM2/TCC anti-camping models, and
speculative/hybrid-SSM strategies benefited from foundational synthesis produced with **Gemini Deep Research**.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md).

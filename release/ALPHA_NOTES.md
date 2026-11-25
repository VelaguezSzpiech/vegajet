# VegaJet v0.1.0-alpha.1

I ran Ornith-122B on the two **32 GB AMD Instinct MI50s** in my machine. My raw fast-path runs reached **165.22 tok/s for 128 output tokens** and **156.41 for 512** (`69ca5eb`). With FR-Spec 64K, `--spec-gdn` and suffix drafting, Hybrid reached **357.07 / 346.27 tok/s** on repeated code (`d7bc240`). Those are development-build measurements, not a benchmark of the shipped binary. That’s the machine I built VegaJet for. This alpha lets you point your own streaming client at it.

On `d7bc240`, the DFlash2 drafter ran on GPU 1 while the target used both cards: **217.60 tok/s** solo and **350.72 tok/s** with Hybrid. On `71cd00e`, batched prefill reached **1,264.57 / 1,344.55 tok/s** for 512 / 2,048 input tokens; ~1,700 tok/s is a separate tiled W4A8 target. All **22 shipped gfx906 code objects** passed the zero-private-spill and direct-to-LDS vector-load checks.

[Download the Linux bundle](https://github.com/VelaguezSzpiech/vegajet/releases/download/v0.1.0-alpha.1/vegajet-0.1.0-alpha.1-linux-x86_64.tar.gz), bring a compatible local Ornith GGUF, and follow the [two-command quickstart](https://github.com/VelaguezSzpiech/vegajet#quickstart-install-and-serve-a-local-bundle). The server streams to OpenAI- and Anthropic-compatible clients. Here’s what the external harness got back from the installed binary:

```text
POST /v1/chat/completions SSE -> deltas=7 finish=stop done=True elapsed=0.13s
  reply: 'The capital of France is Paris.'
```

The tuned 129-token code comparison matched raw decode exactly (`first_diff=-1`); it is not a claim of lossless parity on arbitrary chat. The separate zero-override benchmark on build `496e61b` recorded **159.33 tok/s** for 128 output tokens. Hybrid reached **311.04 / 295.43 tok/s** for 128 / 512 output tokens on the repeated-code prompt: 100% draft acceptance, with 129/129 tokens matching raw decode in the recorded comparison. See [performance and parity](#performance-and-parity) for the flags and named comparisons.

This is an evaluation alpha for exactly two MI50 32 GB cards, not a production serving release.

- **License:** PolyForm Noncommercial 1.0.0. VegaJet is source-available, not open source. See [`LICENSE`](../LICENSE).
- **Binary version string:** `vegajet 0.1.0-alpha.1`; the CMake project version is
  `0.1.0`.
- **Third-party material:** see [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

## What this alpha runs

GPU execution is deliberately narrow:

| Item | Alpha behaviour |
| :--- | :--- |
| GPUs | Exactly two Instinct MI50 32 GB cards (gfx906), tensor parallel |
| GPU memory | 32 GB HBM2 per card; the runner reserves about 30 GiB per GPU |
| Host link | 32 GiB BAR0 per card and a KFD P2P link between the pair |
| Model | The hand-scheduled Ornith hybrid-MoE backend only |
| Model architecture | GGUF `general.architecture` of `qwen35moe`, `qwen3_5_moe` or `ornith` |
| Reference layout | Ornith-122B-AD-D4: 48 layers, 256 routed experts, shared expert |

The loader repacks a Q4_K dense trunk to `VJ_Q4_A8`, keeps Q8_0 token embeddings and an FP16 router, and uses
IQ2_S / IQ3_XXS routed-expert data. `vegajet bench` additionally requires 32 greedy tokens to match the built-in
Tier-1 reference, so a different quantization mix can fail the bench even when the architecture string matches. No
model file is distributed with the repository and no download URL is shipped; you supply a compatible file.

Instinct MI60 32 GB has the same ISA and VRAM size, and the planners handle 64 CUs, but the GPU executor has not
been validated on MI60. Instinct MI50 16 GB, Radeon VII and Radeon Pro VII have less VRAM than the runner's
per-card arena and are CPU-tooling targets in this alpha.

## What works on CPU

The CPU-only build is the portable path and needs no ROCm, GPU or device access:

- `vegajet inspect` — header-only GGUF metadata and footprint report.
- `vegajet plan` / `vegajet bench --dry-run` — load through a plugin, print per-GPU footprints and graph summaries,
  validate code objects passed with `--hsaco`, and never open `/dev/kfd`.
- `vegajet doctor` — read-only host report. Plain `doctor` never opens a device node; `--bench` and `--live-link`
  do. On a host with no gfx906 it exits 1 with an explicit check.
- `vegajet serve --mock` — OpenAI-compatible and Anthropic-compatible HTTP mock engine on `GET /health`,
  `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/messages` and `GET /metrics`, with JSON or SSE responses
  and `--slots 1..64` continuous batching.
- `vegajet top` and `vegajet chat` — read-only live `/metrics` dashboard and interactive OpenAI chat REPL.
- The CPU test suite, including the Python tool tests when Python 3 is present, and `tools/privacy_audit.sh` from
  a git checkout.

`vegajet soak` needs a GPU build. It allocates the arena named by `--gib` per selected GPU and is useful as a
pattern and bandwidth diagnostic; `soak --p2p` needs both cards.

## What works on the supported GPU configuration

- The KFD backend and gfx906 code objects build with `-DVEGAJET_CPU_ONLY=OFF`.
- `vegajet run <model-file> --tp 2 --prompt ...` executes the Ornith backend.
- `vegajet bench <model-file> --tp 2` runs the Tier-1 parity check and the live two-card benchmark suite.
- `vegajet serve <model-file> --tp 2 --port 8080` serves the Ornith-122B backend live on 2× 32 GB MI50 over
  `GET /health`, `GET /v1/models`, `POST /v1/chat/completions` (JSON/SSE, reasoning, Qwen tool calls),
  `POST /v1/messages` (Anthropic Messages API, JSON/SSE), and `GET /metrics` (single-slot greedy GPU argmax).
- GPU selection with `--gpu` or `VJ_GPU_DEVICES`; `--gpu list` prints the gfx906 enumeration.
- Build-time and load-time code-object gates: zero scratch, VGPR ceilings and a `.isa` scan.
- On the Ornith path, before reserving VRAM, `run` and `bench` check the selected GPU count and VRAM against the
  runner's needs and stop with a message naming what was found and what is required.

## Not in this alpha

- GPU execution on one GPU, 16 GB cards or Radeon VII-class cards.
- Multi-slot GPU continuous batching, temperature/top-p GPU sampling, GBNF grammar constraints and multi-format
  DeepSeek/Llama tool parsers inside live `vegajet serve <model-file>` (these exist as tested CPU libraries in
  `core/`, while live GPU serving runs single-slot greedy decode; `--slots > 1` is active on `serve --mock`).
- Descriptor/OpGraph GPU execution for the generic plugins. A non-Ornith plugin in a GPU build is plan/validate
  only; the execution entry point reports the missing executor and returns a nonzero status.
- GPU execution beyond two cards. The CPU planners cover 1–4 GPUs, but the GPU runner is fixed at two ranks and
  two ~30 GiB targets. The CLI's `--tp`, `--vram-gib` and `--bw-gbs` affect planning, not the GPU runner's fixed
  layout.
- A validated no-P2P transport. A host-pinned fallback exists in the runtime, but the alpha qualification and the
  documented configuration use the large-BAR P2P path.

## Build notes and known limits

- GPU build prerequisites are CMake 3.18+, a C++17 compiler, Linux kernel headers with `linux/kfd_ioctl.h`, and a
  ROCm or TheRock installation providing `hipcc`, `clang-offload-bundler` and `llvm-objdump`. CMake defaults
  `VEGAJET_ROCM_PATH` to `/opt/rocm`.
- The gate manifest pins exact per-kernel VGPR counts. The only compiler label recorded in the tree is TheRock
  7.12 (`ops/kernels/vj_prefill_gemm.hip`); stock ROCm 6.x/7.x and distro packaging were not validated for this
  alpha. A different LLVM release can fail the build in either direction. Record the toolchain with any new waiver
  or result.
- The engine does not read or enforce `amdgpu` module parameters. `doctor` reports a missing
  `amdgpu.vm_fragment_size=9` as a WARN; correctness with the default fragment size has not been established.
  `release/docs/OPERATIONS.md` describes the reference rig's optional host procedures, not requirements enforced
  by the binary.
- Reference-rig invariants (32 GiB BAR0, N-1 P2P links, the PSU power budget, matching VBIOS) are WARN in normal and
  `doctor --os` reports. They are hard failures only under `doctor --gsf12`, which is the reference-rig pre-flight.
- The post-run telemetry gate uses the reference rig's power and HBM-temperature limits. The sampler reads only
  gfx906 cards; other gfx906 cards on the host can still affect the report. Exceeding the gate fails the run.
- MI50 and MI60 are passively cooled server cards and need forced airflow. VegaJet does not control fans and does
  not abort a run on temperature.
- The reference layout is model-specific: the Ornith backend carries 48-layer / head-layout / tensor-type
  assumptions and a built-in reference golden. Other `qwen35moe` files are not validated up front.

## Performance and parity

Recorded on 2× Instinct MI50 32 GB (Linux 7.0, TheRock / ROCm 7.12.0, `512/512` MoE split, `120192/128128` LM-head
split, `<= 200 W` cap, calibrated soak ceiling `1,626.2 GB/s`, `0 B` private spill across all 21 `.hsaco` modules):

- Packaged default (`vegajet bench <model-file> --tp 2`): raw `M=1` `tg128 = 159.33 tok/s` (`6.28 ms/tok`, `54.3%`
  of calibrated HBM2 ceiling), `Hybrid tg128 / tg512 = 311.04 / 295.43 tok/s`, W4A8 prefill `pp512 / pp2048 =
  1,258.66 / 1,342.88 tok/s`, peak `TMem = 48.0 °C`, `0 UE / 0 DE`.
- Raw fast-path (`VEGAJET_RAW_ONLY=1 VEGAJET_GU_Q3=1 VEGAJET_VA_CLUSTER=1 VEGAJET_VOCAB_MUBUF=1
  VEGAJET_SUPER_WGS=240`): `tg128 = 165.07 tok/s`, `tg512 = 160.76 tok/s`, peak `TMem = 50.0 °C`, `0 UE / 0 DE`.
- Parity by oracle:
  - Tier-1 `llama.cpp` code reference (`golden::kCodeRef32`, `"def fibonacci(n):"`): **32/32 PASS** on both
    packaged default (`ENABLE_SUPER=1`) and `ENABLE_SUPER=0`. Informational prose (`golden::kProseRef32`): `16/32`
    matched prefix (identical 32-token output on both).
  - Packaged default `Hybrid` vs packaged default raw (`ENABLE_SUPER=1`, `p_rep_code`): **129/129 (`first_diff=-1`) PASS**.
  - Forced `ENABLE_SUPER=0` vs packaged default `ENABLE_SUPER=1` (`p_rep_code`, 129 tokens): **15/129 (`first_diff=12`)**
    due to FP32 reduction-order drift after token 11 (open if super-on vs super-off 129-token identity is required).
    Default `MTP K=4` (`M=5` unfused verify vs `M=1` super-kernel raw on prose) shows the same reduction-order drift
    (`5/129`, `first_diff=2`).

## Roadmap

For v0.1.1 I’m aiming for ~200 tok/s raw decode and 2,000+ tok/s prefill. These are goals. Three-card scaling is unmeasured. Also outside this alpha:

- Intra-superkernel barrier/L2-cache trim (Step 3B, deferred to `v0.1.1`).
- GPU executor support for one, three and four cards, including a three-card TP=3 layout and a three-worker
  single-GPU layout. The CPU planners already cover 1–4 GPUs.
- Wiring multi-slot GPU continuous batching, host sampling chains, GBNF grammar masks and multi-format tool parsers
  into live GPU `vegajet serve <model-file>`.

## Acknowledgments

I’m **Velaguez Szpiech (Vela)**, VegaJet’s creator and maintainer. **Gemini (Neon)** was my primary coding partner on the runtime, the kernels, and the live MI50 runs. **Claude** helped me build the serving APIs, platform tools, and CI. **Codex** worked with me on kernel and systems engineering, qualification, and packaging the installer.

Thank you, **Gemini Deep Research**, for helping me understand the Vega 20 architecture and choose which experiments to run on my cards.

## Reporting problems

Include the exact command, the full error line, the `vegajet doctor` summary, the commit hash, the GPU model and
VRAM, and the ROCm/TheRock toolchain label. Scrub user names, hostnames, mount paths, PCI addresses and model file
names before posting. Stop on an uncorrectable ECC error or kernel fault and preserve the `dmesg` excerpt.

## Build provenance

The tag points to `75c4a64b3e6753047434e03df3004c3709c93ee6`. The GPU binary was built from `ac177850d73fe2b187ee35631d98e85358d729db`; the only change between them is how a CPU test fixture is compiled. The zero-override performance measurements are from `496e61b`.

# VegaJet v0.1.0-alpha.1

VegaJet is a bare-metal LLM inference engine for AMD gfx906 (Vega 20) GPUs. It talks to `/dev/kfd` directly,
repacks GGUF weights into native GPU formats, and runs tensor-parallel decode with peer-to-peer AllReduce over
PCIe BAR. This is the first public alpha: a source release for evaluation, not a production serving release.

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

These items are not part of v0.1.0-alpha.1:

- Intra-superkernel barrier/L2-cache trim (Step 3B, deferred to `v0.1.1`).
- GPU executor support for one, three and four cards, including a three-card TP=3 layout and a three-worker
  single-GPU layout. The CPU planners already cover 1–4 GPUs.
- Wiring multi-slot GPU continuous batching, host sampling chains, GBNF grammar masks and multi-format tool parsers
  into live GPU `vegajet serve <model-file>`.

## Research acknowledgment

Architectural design, gfx906/Vega 20 ISA analysis, HBM2/TCC anti-camping models, and speculative/hybrid-SSM
strategies benefited from foundational synthesis produced with **Gemini Deep Research**.

## Reporting problems

Include the exact command, the full error line, the `vegajet doctor` summary, the commit hash, the GPU model and
VRAM, and the ROCm/TheRock toolchain label. Scrub user names, hostnames, mount paths, PCI addresses and model file
names before posting. Stop on an uncorrectable ECC error or kernel fault and preserve the `dmesg` excerpt.

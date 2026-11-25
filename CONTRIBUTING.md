# Contributing to VegaJet

VegaJet is licensed under **PolyForm Noncommercial 1.0.0**. Contributions are accepted under the same license
(inbound = outbound). Do not submit code you cannot license that way, and keep third-party material listed in
`THIRD_PARTY_NOTICES.md`.

Read [`release/ALPHA_NOTES.md`](release/ALPHA_NOTES.md) for the supported configuration before proposing a GPU
change. GPU execution in this alpha is limited to the Ornith backend on two 32 GB gfx906 cards; other GPU
configurations and live GPU serving are roadmap material.

## Ground rules

- **CPU first.** Every change must build and pass with `-DVEGAJET_CPU_ONLY=ON`. Planners, packers, op graphs, the
  sampler and the server are all testable without a GPU. Put the logic there and keep the GPU side thin.
- **No personal data in the tree.** That covers user names, home or mount paths, hostnames, PCI addresses, firmware
  file names, model file names and serial numbers. Model files reach code only through CLI arguments or environment
  variables. `tools/privacy_audit.sh` checks the working tree and git history; run it from a git checkout.
- **No hardcoded model dimensions** outside a model plugin's metadata parsing. Shared code
  (`core/`, `ops/layer_plan.*`) takes every size from `ModelDesc`. `test_opgraph_multi_arch` scans the step planner
  for literals.
- **No throughput claims without a recorded run.** README performance cells are filled only from the final
  scorecard, with the commit, flags and measured HBM2 ceiling. Do not put a number in a document before it has a
  recorded run behind it.
- **Do not invent compatibility.** State what was tested, on which toolchain and configuration, and label the rest
  as untested. The host prerequisites and the supported two-card configuration are in the README.

## Before you open a pull request

```sh
cmake -S . -B build-cpu -DVEGAJET_CPU_ONLY=ON
cmake --build build-cpu -j
ctest --test-dir build-cpu --output-on-failure
tools/privacy_audit.sh
```

All of these must pass. A change to a kernel must also pass the code-object gates. They run at build time through
`vj_hsaco_gate` and again at load time:

| Gate | Limit |
| :--- | :--- |
| Scratch | `private_segment_fixed_size == 0` (0 B spill) |
| VGPRs | ≤ 48 for streaming GEMVs, ≤ 64 for persistent super-kernels and tiled GEMMs |
| Code size | `.text` < 24,576 B per kernel |
| ISA | no `buffer_load_dwordx4 ... lds` (gfx950-only); non-temporal loads only on one-shot decode streams |

An over-limit kernel needs a waiver line in `ops/kernels/hsaco_gates.txt` with a reason. Waivers may only go down.
The gate compares exact per-kernel register counts, so a compiler change can fail the build even when the count
drops; do not edit a waiver to absorb a toolchain difference without recording which toolchain produced it.

## Correctness bar

- Decoders are bit-exact against the reference implementation (golden hashes in `tests/golden_decode_hashes.inc`).
- Lossless kernel changes keep greedy output identical to the reference parity golden.
- Speculative decoding modes stay lossless (`first_diff = -1`).
- Lossy changes (new quantization, wire formats) are opt-in flags and report KL divergence against the lossless path.
- Every new planner gets a CPU test that covers 60-CU and 64-CU devices and N = 1..4 GPUs.

## GPU runs

GPU validation in this alpha means the supported two-card configuration. If you run GPU jobs yourself:

- Run `vegajet doctor` first and fix every FAIL. WARNs are host-specific information; read them before deciding
  they do not apply.
- Run one GPU job at a time, and make sure no other process holds `/dev/kfd` or a compute render node.
- Provide forced airflow for the passively cooled cards. VegaJet does not manage fans or abort a run on
  temperature; a completed run ends with a power/thermal telemetry gate.
- Stop on any kernel fault or uncorrectable ECC error, and report it with the `dmesg` excerpt (scrub host details).
- Do not change VBIOS, clocks, power tables or kernel boot parameters as part of a code change. Optional host
  helpers under `contrib/hardware/` are operator tools, not part of the engine.

## Style

- C++17 (the CMake standard), 4-space indent, 120 columns, `#pragma once`.
- For new project-owned C++ sources use `// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0`.
- Headers say what a component does and what it guarantees. Comments explain why, not what.
- Validate bad input with errors, not asserts. Never put work with side effects inside `assert()`.
- Standard library only on the host side: no new third-party dependencies.

## Adding a model

Follow [`agent/ONBOARD_MODEL.md`](agent/ONBOARD_MODEL.md). A model pull request includes:

- the plugin;
- a synthetic-GGUF unit test that runs in the CPU suite;
- an optional real-file check gated on an environment variable;
- for GPU support, an engine implementation, its parity result and the recorded scorecard entry. A descriptor
  plugin alone is plan/validate-only and must not be described as GPU-executed.

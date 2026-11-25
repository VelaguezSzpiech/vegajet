# Linux bundle: build, verify, install, smoke test

This directory turns an exact CMake build of VegaJet into a relocatable Linux bundle for the current user. It is
offline: nothing is downloaded, no URL is required, and no step needs sudo.

| File | Role |
| :--- | :--- |
| `make-bundle.sh` | Configures/builds the documented CMake configuration and stages the bundle + tarball |
| `vegajet-launcher.sh` | Installed as `bin/vegajet`; relocatable prefix lookup, named dependency checks, GPU-asset gate |
| `verify-bundle.sh` | SHA-256 integrity verification of every file in a staged or installed bundle |
| `install.sh` | No-sudo per-user installer (verifies, copies under `<prefix>/opt`, creates relative symlinks) |
| `stream-test` | Packaged client harness: installed CLI `serve --mock` over loopback, requires SSE content, finish reason and `DONE` |
| `selftest.sh` | Acceptance checks for a staged bundle, including failure-name checks |

## Build a bundle

Prerequisites: Linux, CMake 3.18+, a C++17 compiler, and the standard `sha256sum`, `tar`, `gzip`, `find`, `sort`,
`xargs` and `install` tools. `--smoke` and `stream-test` additionally need `python3`. A GPU bundle needs a ROCm or
TheRock installation that provides `hipcc`; the CPU bundle needs neither ROCm nor a GPU. Nothing is downloaded.

CPU tools and the mock server (default; no ROCm, no GPU):

```sh
release/linux/make-bundle.sh --out "${TMPDIR:-/tmp}/vegajet-dist"
```

That is exactly the documented CPU build, with a dedicated build directory:

```sh
cmake -S . -B build-bundle -DCMAKE_BUILD_TYPE=Release -DVEGAJET_CPU_ONLY=ON
cmake --build build-bundle --target vegajet vj_hsaco_gate vj_isa_scan --parallel <jobs>
```

The maker then stages `bin/vegajet` (the launcher), `libexec/vegajet/vegajet` (the built executable), the client
harness, the verifier/installer, `share/doc/vegajet/` (README, alpha notes, CLI reference, operations guide,
third-party notices, license, this file) and an integrity manifest, and writes
`vegajet-<version>-linux-<arch>.tar.gz` plus a `.sha256` file.

`--out` defaults to `$TMPDIR/vegajet-dist` so nothing is written into the checkout unless you ask for it; keep it
that way for release builds so binary artifacts never land in the source tree or a privacy audit. Useful
options: `--build-dir DIR`, `--no-build` (repack an existing build), `--smoke` (run `stream-test` against the
staged bundle), `--no-tarball`, `--force`, `--jobs N`.

The build adds `-ffile-prefix-map` for the checkout and build directory, so `VJ_CHECK` diagnostics in the packaged
binary do not embed the builder's paths; `--no-prefix-map` builds without it. `--no-build` cannot retrofit the
mapping, so the maker prints a note if the reused binary still contains either path.

Staged layout:

```text
vegajet-<version>-linux-<arch>/
  bin/vegajet                  relocatable launcher (dependency checks, GPU-asset gate)
  bin/stream-test              packaged mock SSE client harness
  libexec/vegajet/vegajet      the CMake-built executable
  share/vegajet/hsaco/         real gate-verified code objects (GPU bundles only)
  share/doc/vegajet/           README, alpha notes, CLI reference, operations guide, notices, license
  install.sh                   no-sudo installer
  verify-bundle.sh             SHA-256 manifest verifier
  selftest.sh                  acceptance checks
  SHA256SUMS                   integrity manifest for every file above
  BUNDLE_INFO.txt              version, arch, build mode, GPU-asset status
```

## GPU bundles and the code-object policy

GPU execution is packaged only from real code objects produced by the same CMake build:

```sh
release/linux/make-bundle.sh --out DIR --gpu-build --rocm-prefix /opt/rocm
```

With `-DVEGAJET_CPU_ONLY=OFF` the maker builds the `*_hsaco` targets, then requires **every** `.hsaco` to pass
`vj_hsaco_gate` (zero scratch, VGPR ceilings, code-size ceiling) and `vj_isa_scan`. The accepted files are copied
to `share/vegajet/hsaco/`, hashed into `share/vegajet/hsaco/SHA256SUMS`, and described in `PROVENANCE.txt`
(architecture, count, gate results, recorded toolchain). No placeholder, synthetic or downloaded code object is
ever packaged.

A CPU-only bundle records the opposite explicitly: `share/vegajet/GPU_UNAVAILABLE.txt` and
`BUNDLE_INFO.txt` say `gpu_assets: unavailable` with a reason. The launcher then refuses `run`, `bench`, `soak`
and `serve <model-file>` with `E_GPU_ASSETS_MISSING` and exit 3, instead of failing later inside the CLI. CPU
paths (`inspect`, `plan`, `bench --dry-run`, `doctor`, `serve --mock`, `--version`, `--help`) are unaffected.

Before any GPU command, the launcher re-verifies the packaged code objects against their manifest and exports
`VEGAJET_HSACO_DIR` only for that verified directory. An operator-supplied `VEGAJET_HSACO_DIR` is honored only if
it carries its own `SHA256SUMS` covering every `*.hsaco`; otherwise the launcher stops with
`E_GPU_ASSETS_INTEGRITY`. `VEGAJET_ROCM_LIB=/path/to/rocm/lib` may be set when the ROCm shared libraries are not
on the loader path.

## Verify integrity

```sh
./vegajet-<version>-linux-<arch>/verify-bundle.sh --dir ./vegajet-<version>-linux-<arch>
```

Every manifest entry must match and every regular file must be listed. Failures name `E_MANIFEST_MISSING`,
`E_INTEGRITY` or `E_EXTRA_FILE`. The installer runs the same check before and after copying, and
`vegajet --bundle-check` reports it after installation.

## Install (no sudo)

```sh
tar -xzf vegajet-<version>-linux-<arch>.tar.gz
./vegajet-<version>-linux-<arch>/install.sh --prefix "$HOME/.local"
```

The default prefix is `$HOME/.local`. The bundle is copied verbatim to `<prefix>/opt/vegajet-<version>`, and
`<prefix>/bin/vegajet`, `<prefix>/bin/stream-test` and `<prefix>/bin/vegajet-verify` are created as **relative**
symlinks. Because the launcher resolves its own real path, the whole prefix can be moved and still works.
`--dry-run` shows the actions, `--force` replaces an existing version, `--verify-only` only checks the extracted
bundle. Nothing outside the prefix is written and `sudo` is never invoked.

```sh
$HOME/.local/bin/vegajet --bundle-check          # binary, dependencies, integrity, GPU-asset status
$HOME/.local/bin/vegajet --version
$HOME/.local/bin/stream-test                     # mock SSE acceptance; does not run GPU inference
```

## stream-test

`stream-test` starts the installed `vegajet serve --mock` on `127.0.0.1:0`, waits for its listening line, requires
`engine=mock` or `engine=mock+scheduler` (any other engine is `E_NOT_MOCK`), then requires:

- `GET /health` returns 200 (`E_HEALTH`),
- the completion streams as `text/event-stream` with at least two content deltas before the terminal chunk
  (`E_STREAM`),
- a terminal chunk with a non-null `finish_reason`, which must be `stop` for the canned mock reply (`E_FINISH`),
- the literal SSE terminator `data: [DONE]` (`E_DONE`),
- a clean server exit after SIGTERM (`E_SERVER_STOP`).

It prints `PASS engine=mock …` and a note that the CPU mock engine was exercised and no GPU inference was run.
Options: `--vegajet PATH`, `--timeout SECONDS`. It needs `python3` (reported as `E_DEPENDENCY: python3`).

## Diagnostics

All tools use stable, greppable failure names.

| Tool | Names |
| :--- | :--- |
| launcher | `E_PREFIX`, `E_BUNDLE_EXEC`, `E_DEPENDENCY`, `E_GPU_ASSETS_MISSING`, `E_GPU_ASSETS_INTEGRITY`, `E_USAGE` |
| verifier | `E_MANIFEST_MISSING`, `E_INTEGRITY`, `E_EXTRA_FILE`, `E_DEPENDENCY`, `E_USAGE` |
| installer | `E_PREFIX`, `E_MANIFEST_MISSING`, `E_INTEGRITY`, `E_WRITE`, `E_CHECK`, `E_DEPENDENCY`, `E_USAGE` |
| maker | `E_REPO`, `E_DEPENDENCY`, `E_BUILD`, `E_BINARY`, `E_VERSION`, `E_STAGE_EXISTS`, `E_HSACO_MISSING`, `E_HSACO_GATE`, `E_HSACO_ARCH`, `E_WRITE`, `E_INTEGRITY`, `E_TARBALL`, `E_SMOKE`, `E_USAGE` |
| stream-test | `E_VEGAJET`, `E_DEPENDENCY`, `E_SPAWN`, `E_LISTEN`, `E_NOT_MOCK`, `E_HEALTH`, `E_STREAM`, `E_FINISH`, `E_DONE`, `E_SERVER_STOP`, `E_USAGE` |

`E_GPU_ASSETS_MISSING` exits 3, matching the CLI's "GPU backend or engine unavailable" status.

## Acceptance

From the checkout, all offline:

```sh
# 1. Build a CPU bundle and run the packaged mock client harness against the staged tree.
release/linux/make-bundle.sh --out /tmp/vegajet-dist --smoke

# 2. Run the bundle acceptance checks (tamper, extra file, missing executable, GPU refusal,
#    no-sudo install, prefix relocation, stream-test).
release/linux/selftest.sh --bundle "/tmp/vegajet-dist/vegajet-0.1.0-alpha.1-linux-$(uname -m)"

# 3. Install into a throwaway prefix and inspect the installed tree.
tar -xzf /tmp/vegajet-dist/vegajet-0.1.0-alpha.1-linux-$(uname -m).tar.gz
./vegajet-0.1.0-alpha.1-linux-$(uname -m)/install.sh --prefix /tmp/vegajet-prefix
/tmp/vegajet-prefix/bin/vegajet --bundle-check
/tmp/vegajet-prefix/bin/stream-test
```

For a GPU bundle, add `--gpu-build --rocm-prefix <prefix>` to step 1; the maker fails with `E_HSACO_MISSING`,
`E_HSACO_GATE` or `E_HSACO_ARCH` unless it was able to build and gate real gfx906 code objects.
`vegajet --bundle-check --require-gpu` exits 3 unless verified code objects are present.

## Notes

- The bundle records no machine-specific paths. `BUNDLE_INFO.txt` names the build mode and flags generically;
  `source_epoch` is `unset` unless `SOURCE_DATE_EPOCH` was provided to the maker.
- The CPU executable reads no asset files; its only optional runtime asset is `share/vegajet/hsaco` for GPU
  execution. Licenses and docs are included for redistribution compliance.
- `LICENSE` (PolyForm Noncommercial 1.0.0) and `THIRD_PARTY_NOTICES.md` are part of every bundle.

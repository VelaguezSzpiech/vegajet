#!/usr/bin/env bash
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Offline Linux bundle creator for VegaJet.
#
# Builds the exact documented CMake configuration, installs the executable into a relocatable bundle with the
# launcher, licenses, docs, integrity manifest and the stream-test client harness, then optionally packs a
# tarball. It never downloads anything and never needs sudo.
#
#   release/linux/make-bundle.sh --out DIR                 # CPU tools + mock server bundle (default)
#   release/linux/make-bundle.sh --out DIR --gpu-build     # GPU binary + verified real .hsaco code objects
#   release/linux/make-bundle.sh --out DIR --no-build      # repack an existing build directory
#
# GPU policy: a GPU bundle is created only from .hsaco files produced by the same CMake build. Every code object
# must pass vj_hsaco_gate and vj_isa_scan, is recorded in share/vegajet/hsaco/SHA256SUMS, and is re-verified by
# the launcher before any GPU command. Without verified code objects the bundle records GPU mode as explicitly
# unavailable; no placeholder or invented code object is ever packaged.
#
# Stable diagnostic names: E_USAGE, E_REPO, E_DEPENDENCY, E_BUILD, E_BINARY, E_VERSION, E_STAGE_EXISTS,
# E_HSACO_MISSING, E_HSACO_GATE, E_HSACO_ARCH, E_WRITE, E_INTEGRITY, E_TARBALL, E_SMOKE.

set -euo pipefail

PROG=vegajet-bundle-maker
repo=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)

die() {
    local name=$1 rc=$2
    shift 2
    printf '%s: %s: %s\n' "$PROG" "$name" "$*" >&2
    exit "$rc"
}

usage() {
    cat <<'EOF'
usage: make-bundle.sh [options]
  --out DIR          output directory for the bundle and tarball (default: $TMPDIR/vegajet-dist)
  --build-dir DIR    CMake build directory (default: <repo>/build-bundle)
  --arch ARCH        bundle architecture label (default: uname -m)
  --gpu-build        configure with -DVEGAJET_CPU_ONLY=OFF and -DVEGAJET_ROCM_PATH
  --rocm-prefix DIR  ROCm/TheRock prefix for --gpu-build (default: $VEGAJET_ROCM_PATH or /opt/rocm)
  --hsaco DIR        code-object directory for a GPU build (default: <build-dir>/hsaco)
  --no-build         reuse an existing build directory (CMakeCache.txt required)
  --jobs N           parallel build jobs (default: nproc)
  --version V        override the version read from the built binary
  --no-tarball       keep the staged directory but do not pack a .tar.gz
  --smoke            run the packaged stream-test against the staged bundle
  --no-prefix-map    do not add -ffile-prefix-map; the binary may embed build paths
  --force            replace an existing staged bundle directory
EOF
}

out=${TMPDIR:-/tmp}/vegajet-dist
build=
arch=$(uname -m)
gpu_build=0
rocm_prefix=${VEGAJET_ROCM_PATH:-/opt/rocm}
hsaco_dir=
hsaco_explicit=0
no_build=0
jobs=
version_override=
tarball=1
smoke=0
force=0
prefix_map=1

while [ $# -gt 0 ]; do
    case $1 in
        --out) [ $# -ge 2 ] || die E_USAGE 2 "--out needs a directory"; out=$2; shift 2 ;;
        --out=*) out=${1#--out=}; shift ;;
        --build-dir) [ $# -ge 2 ] || die E_USAGE 2 "--build-dir needs a directory"; build=$2; shift 2 ;;
        --build-dir=*) build=${1#--build-dir=}; shift ;;
        --arch) [ $# -ge 2 ] || die E_USAGE 2 "--arch needs a value"; arch=$2; shift 2 ;;
        --arch=*) arch=${1#--arch=}; shift ;;
        --gpu-build) gpu_build=1; shift ;;
        --rocm-prefix) [ $# -ge 2 ] || die E_USAGE 2 "--rocm-prefix needs a directory"; rocm_prefix=$2; shift 2 ;;
        --rocm-prefix=*) rocm_prefix=${1#--rocm-prefix=}; shift ;;
        --hsaco) [ $# -ge 2 ] || die E_USAGE 2 "--hsaco needs a directory"; hsaco_dir=$2; hsaco_explicit=1; shift 2 ;;
        --hsaco=*) hsaco_dir=${1#--hsaco=}; hsaco_explicit=1; shift ;;
        --no-build) no_build=1; shift ;;
        --jobs) [ $# -ge 2 ] || die E_USAGE 2 "--jobs needs a number"; jobs=$2; shift 2 ;;
        --jobs=*) jobs=${1#--jobs=}; shift ;;
        --version) [ $# -ge 2 ] || die E_USAGE 2 "--version needs a value"; version_override=$2; shift 2 ;;
        --version=*) version_override=${1#--version=}; shift ;;
        --no-tarball) tarball=0; shift ;;
        --smoke) smoke=1; shift ;;
        --no-prefix-map) prefix_map=0; shift ;;
        --force) force=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) die E_USAGE 2 "unknown option: $1" ;;
    esac
done

[ -n "$out" ] || die E_USAGE 2 "--out must not be empty"
[ -f "$repo/CMakeLists.txt" ] || die E_REPO 1 "not a VegaJet checkout: $repo"
for required in \
    release/linux/vegajet-launcher.sh release/linux/stream-test release/linux/verify-bundle.sh \
    release/linux/install.sh release/linux/selftest.sh release/linux/README.md \
    README.md release/ALPHA_NOTES.md release/docs/CLI_REFERENCE.md release/docs/OPERATIONS.md \
    THIRD_PARTY_NOTICES.md LICENSE; do
    [ -f "$repo/$required" ] || die E_REPO 1 "checkout is missing required bundle input: $required"
done
[ -n "$arch" ] || die E_USAGE 2 "empty architecture label"

. /dev/null 2>/dev/null || true
for tool in cmake sha256sum tar gzip find sort xargs install; do
    command -v "$tool" >/dev/null 2>&1 || die E_DEPENDENCY 1 "required tool not found: $tool"
done
case ${jobs:-} in
    "") jobs=$(command -v nproc >/dev/null 2>&1 && nproc || echo 2) ;;
    *[!0-9]*) die E_USAGE 2 "--jobs needs a positive integer" ;;
esac

build=${build:-$repo/build-bundle}
hsaco_dir=${hsaco_dir:-$build/hsaco}

if [ "$no_build" = 1 ]; then
    [ -f "$build/CMakeCache.txt" ] || die E_BUILD 1 "--no-build needs a configured build directory: $build"
    [ -x "$build/vegajet" ] || die E_BINARY 1 "missing built executable: $build/vegajet"
    [ -x "$build/vj_hsaco_gate" ] || die E_BINARY 1 "missing gate tool: $build/vj_hsaco_gate"
    [ -x "$build/vj_isa_scan" ] || die E_BINARY 1 "missing ISA scan tool: $build/vj_isa_scan"
    cpu_only=$(sed -n 's/^VEGAJET_CPU_ONLY:BOOL=//p' "$build/CMakeCache.txt" | head -n 1)
    [ -n "$cpu_only" ] || die E_BUILD 1 "CMakeCache.txt has no VEGAJET_CPU_ONLY entry"
    if [ "$gpu_build" = 1 ] && [ "$cpu_only" = "ON" ]; then
        die E_USAGE 2 "--gpu-build does not match a VEGAJET_CPU_ONLY=ON build directory"
    fi
else
    mkdir -p "$build" 2>/dev/null || die E_BUILD 1 "cannot create build directory: $build"
    cmake_cfg=(-S "$repo" -B "$build" -DCMAKE_BUILD_TYPE=Release)
    if [ "$prefix_map" = 1 ]; then
        case "$repo$build" in
            *[[:space:]]*)
                printf '%s: note: build paths contain whitespace; compiler path mapping disabled\n' "$PROG"
                prefix_map=0
                ;;
        esac
    fi
    if [ "$prefix_map" = 1 ]; then
        probe_ok=0
        for probe_cc in "${CXX:-}" c++ g++ clang++; do
            [ -n "$probe_cc" ] || continue
            command -v "$probe_cc" >/dev/null 2>&1 || continue
            if printf 'int main(){return 0;}\n' | "$probe_cc" -ffile-prefix-map=/tmp=. -x c++ -fsyntax-only - >/dev/null 2>&1; then
                probe_ok=1
                break
            fi
        done
        if [ "$probe_ok" != 1 ]; then
            printf '%s: note: compiler lacks -ffile-prefix-map; build paths may be embedded in the binary\n' "$PROG"
            prefix_map=0
        fi
    fi
    if [ "$prefix_map" = 1 ]; then
        cmake_cfg+=("-DCMAKE_CXX_FLAGS=${CXXFLAGS:-} -ffile-prefix-map=$repo=. -ffile-prefix-map=$build=.")
    fi
    if [ "$gpu_build" = 1 ]; then
        cmake_cfg+=(-DVEGAJET_CPU_ONLY=OFF -DVEGAJET_ROCM_PATH="$rocm_prefix")
    else
        cmake_cfg+=(-DVEGAJET_CPU_ONLY=ON)
    fi
    cmake "${cmake_cfg[@]}" || die E_BUILD 1 "CMake configure failed"
    cmake --build "$build" --target vegajet vj_hsaco_gate vj_isa_scan --parallel "$jobs" \
        || die E_BUILD 1 "CMake build of vegajet, vj_hsaco_gate and vj_isa_scan failed"
    if [ "$gpu_build" = 1 ]; then
        hsaco_targets=()
        while IFS= read -r target; do
            [ -n "$target" ] && hsaco_targets+=("$target")
        done < <(cmake --build "$build" --target help 2>/dev/null | sed -n 's/^\.\.\. \([A-Za-z0-9_]*_hsaco\)$/\1/p')
        [ "${#hsaco_targets[@]}" -gt 0 ] || die E_HSACO_MISSING 1 "no *_hsaco targets found in $build"
        cmake --build "$build" --target "${hsaco_targets[@]}" --parallel "$jobs" \
            || die E_BUILD 1 "CMake build of the gfx906 code objects failed"
    fi
    cpu_only=$(sed -n 's/^VEGAJET_CPU_ONLY:BOOL=//p' "$build/CMakeCache.txt" | head -n 1)
fi

[ -x "$build/vegajet" ] || die E_BINARY 1 "missing built executable: $build/vegajet"
if [ "$hsaco_explicit" = 0 ]; then
    cached_hsaco=$(sed -n 's/^VEGAJET_HSACO_DIR:PATH=//p' "$build/CMakeCache.txt" | head -n 1)
    if [ -n "$cached_hsaco" ]; then hsaco_dir=$cached_hsaco; fi
fi
if [ -n "$version_override" ]; then
    version=$version_override
else
    version=$("$build/vegajet" --version 2>/dev/null | awk 'NR==1 {print $2}') \
        || die E_BINARY 1 "built executable cannot report --version"
fi
case $version in
    ""|*[!A-Za-z0-9._+-]*) die E_VERSION 1 "unusable version string: '$version'" ;;
esac

bundle_name=vegajet-$version-linux-$arch
stage=${out%/}/$bundle_name
if [ -e "$stage" ]; then
    [ "$force" = 1 ] || die E_STAGE_EXISTS 1 "staged bundle already exists: $stage (use --force)"
    case $stage in
        */"$bundle_name") ;;
        *) die E_USAGE 2 "refusing to remove unexpected stage path: $stage" ;;
    esac
    rm -rf "$stage"
fi
mkdir -p "$stage/bin" "$stage/libexec/vegajet" "$stage/share/doc/vegajet" "$stage/share/vegajet"

# ---- installed executable, launcher, tools, licenses -----------------------------------------------------
install -m 755 "$repo/release/linux/vegajet-launcher.sh" "$stage/bin/vegajet"
install -m 755 "$build/vegajet" "$stage/libexec/vegajet/vegajet"
install -m 755 "$repo/release/linux/stream-test" "$stage/bin/stream-test"
install -m 755 "$repo/release/linux/verify-bundle.sh" "$stage/verify-bundle.sh"
install -m 755 "$repo/release/linux/install.sh" "$stage/install.sh"
install -m 755 "$repo/release/linux/selftest.sh" "$stage/selftest.sh"

install -m 644 "$repo/README.md" "$stage/share/doc/vegajet/README.md"
install -m 644 "$repo/release/ALPHA_NOTES.md" "$stage/share/doc/vegajet/ALPHA_NOTES.md"
install -m 644 "$repo/release/docs/CLI_REFERENCE.md" "$stage/share/doc/vegajet/CLI_REFERENCE.md"
install -m 644 "$repo/release/docs/OPERATIONS.md" "$stage/share/doc/vegajet/OPERATIONS.md"
install -m 644 "$repo/release/linux/README.md" "$stage/share/doc/vegajet/BUNDLE.md"
install -m 644 "$repo/THIRD_PARTY_NOTICES.md" "$stage/share/doc/vegajet/THIRD_PARTY_NOTICES.md"
install -m 644 "$repo/LICENSE" "$stage/share/doc/vegajet/LICENSE"

if grep -aqF -- "$repo" "$stage/libexec/vegajet/vegajet" 2>/dev/null \
    || grep -aqF -- "$build" "$stage/libexec/vegajet/vegajet" 2>/dev/null; then
    printf '%s: note: packaged binary embeds a build path; bundling maps source paths by default\n' "$PROG"
    printf '%s:       (a --no-build tree may predate that mapping; use a neutral build directory for release)\n' "$PROG"
fi

# ---- gfx906 code objects --------------------------------------------------------------------------------
gpu_assets=unavailable
gpu_reason=
if [ "$cpu_only" = "ON" ]; then
    if [ "$hsaco_explicit" = 1 ]; then
        die E_USAGE 2 "cannot package code objects with a CPU-only binary (use --gpu-build)"
    fi
    gpu_reason="this bundle was built with VEGAJET_CPU_ONLY=ON; no gfx906 code objects are packaged"
else
    [ -d "$hsaco_dir" ] || die E_HSACO_MISSING 1 "GPU build has no code-object directory: $hsaco_dir"
    mkdir -p "$stage/share/vegajet/hsaco" || die E_WRITE 1 "cannot create the code-object directory in the staged bundle"
    hsacos=()
    mapfile -t hsacos < <(find "$hsaco_dir" -maxdepth 1 -type f -name '*.hsaco' -print | LC_ALL=C sort)
    [ "${#hsacos[@]}" -gt 0 ] || die E_HSACO_MISSING 1 "GPU build produced no .hsaco files in $hsaco_dir"
    for f in "${hsacos[@]}"; do
        if ! "$build/vj_hsaco_gate" "$f" >/dev/null 2>&1; then
            die E_HSACO_GATE 1 "hsaco gate rejected $(basename "$f")"
        fi
        if ! "$build/vj_isa_scan" "$f" >/dev/null 2>&1; then
            die E_HSACO_ARCH 1 "ISA scan rejected $(basename "$f")"
        fi
    done
    for f in "${hsacos[@]}"; do
        install -m 644 "$f" "$stage/share/vegajet/hsaco/$(basename "$f")"
    done
    ( cd "$stage/share/vegajet/hsaco" && sha256sum ./*.hsaco > SHA256SUMS ) \
        || die E_INTEGRITY 1 "cannot write the code-object manifest"
    {
        gfx_arch=$(sed -n 's/^VEGAJET_GFX_ARCH:STRING=//p' "$build/CMakeCache.txt" | head -n 1)
        [ -n "$gfx_arch" ] || gfx_arch=gfx906
        printf 'gfx_arch: %s\n' "$gfx_arch"
        printf 'code_objects: %s\n' "${#hsacos[@]}"
        printf 'gate: vj_hsaco_gate PASS (0 B spill, VGPR ceilings, code-size ceiling)\n'
        printf 'isa_scan: vj_isa_scan PASS\n'
        if [ -x "$rocm_prefix/bin/hipcc" ]; then
            printf 'toolchain: %s\n' "$("$rocm_prefix/bin/hipcc" --version 2>/dev/null | head -n 1)"
        else
            printf 'toolchain: not recorded (hipcc not present at packaging time)\n'
        fi
    } > "$stage/share/vegajet/hsaco/PROVENANCE.txt" || die E_WRITE 1 "cannot write code-object provenance"
    gpu_assets=verified
    gpu_reason=
fi

if [ "$gpu_assets" = unavailable ]; then
    {
        printf 'gpu_assets: unavailable\n'
        printf 'reason: %s\n' "$gpu_reason"
    } > "$stage/share/vegajet/GPU_UNAVAILABLE.txt"
fi

# ---- bundle metadata ------------------------------------------------------------------------------------
mode_label=CPU_ONLY
[ "$cpu_only" = "ON" ] || mode_label=FULL
{
    printf 'name: vegajet\n'
    printf 'version: %s\n' "$version"
    printf 'platform: linux\n'
    printf 'arch: %s\n' "$arch"
    printf 'build_type: Release\n'
    printf 'cpu_only: %s\n' "$([ "$cpu_only" = "ON" ] && echo 1 || echo 0)"
    printf 'gpu_assets: %s\n' "$gpu_assets"
    if [ -n "$gpu_reason" ]; then printf 'gpu_reason: %s\n' "$gpu_reason"; fi
    printf 'build_command: cmake -S <source> -B <build-dir> -DCMAKE_BUILD_TYPE=Release -DVEGAJET_CPU_ONLY=%s\n' "$cpu_only"
    printf 'build_mode: %s\n' "$mode_label"
    printf 'source_epoch: %s\n' "${SOURCE_DATE_EPOCH:-unset}"
} > "$stage/BUNDLE_INFO.txt"

# ---- integrity manifest ---------------------------------------------------------------------------------
(
    cd "$stage"
    find . -type f ! -path './SHA256SUMS' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
) > "$stage/SHA256SUMS" || die E_INTEGRITY 1 "cannot write the bundle manifest"

"$stage/verify-bundle.sh" --dir "$stage" --quiet || die E_INTEGRITY 1 "staged bundle failed its own verification"

# ---- optional packaged-client smoke test ----------------------------------------------------------------
if [ "$smoke" = 1 ]; then
    command -v python3 >/dev/null 2>&1 || die E_DEPENDENCY 1 "--smoke needs python3 for stream-test"
    "$stage/bin/stream-test" --vegajet "$stage/bin/vegajet" || die E_SMOKE 1 "packaged stream-test failed"
fi

# ---- tarball --------------------------------------------------------------------------------------------
tarball_path=
if [ "$tarball" = 1 ]; then
    mkdir -p "$out" || die E_TARBALL 1 "cannot create output directory: $out"
    tarball_path=${out%/}/$bundle_name.tar.gz
    tmp_tar=$tarball_path.tmp.$$
    mtime_args=()
    if [ -n "${SOURCE_DATE_EPOCH:-}" ]; then mtime_args=(--mtime="@$SOURCE_DATE_EPOCH"); fi
    if ! ( cd "$out" && tar --sort=name --owner=0 --group=0 --numeric-owner "${mtime_args[@]}" -cf - "$bundle_name" ) | gzip -n > "$tmp_tar"; then
        rm -f "$tmp_tar"
        die E_TARBALL 1 "cannot create tarball: $tarball_path"
    fi
    mv "$tmp_tar" "$tarball_path" || die E_TARBALL 1 "cannot move tarball into place"
    ( cd "$out" && sha256sum "$bundle_name.tar.gz" > "$bundle_name.tar.gz.sha256" ) \
        || die E_TARBALL 1 "cannot write tarball checksum"
fi

printf '%s: bundle ok version=%s arch=%s cpu_only=%s gpu_assets=%s\n' \
    "$PROG" "$version" "$arch" "$cpu_only" "$gpu_assets"
if [ -n "$gpu_reason" ]; then printf '%s: gpu reason: %s\n' "$PROG" "$gpu_reason"; fi
printf '%s: staged: %s\n' "$PROG" "$stage"
if [ -n "$tarball_path" ]; then
    printf '%s: tarball: %s\n' "$PROG" "$tarball_path"
    printf '%s: sha256: %s\n' "$PROG" "$(cut -d' ' -f1 "$tarball_path.sha256")"
fi
printf '%s: install from outside the checkout: tar -xzf %s && ./%s/install.sh --prefix "$HOME/.local"\n' \
    "$PROG" "$bundle_name.tar.gz" "$bundle_name"

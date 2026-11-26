#!/usr/bin/env bash
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Fast offline check: incremental build of one target (hsaco targets run the 0 B spill / VGPR / .text / ISA gate as
# part of the build), then the matching CPU tests with /dev/kfd and render nodes masked (no GPU is ever opened).
#   scripts/fast_check.sh <cmake-target> [ctest-regex]      e.g. scripts/fast_check.sh hsaco_vj_ornith_super_x super
# Env: VJ_BUILD_DIR (default: build)
set -euo pipefail
target=${1:?usage: fast_check.sh <cmake-target> [ctest-regex]}; rx=${2:-.}
B=${VJ_BUILD_DIR:-build}
cmake --build "$B" --target "$target" -j"$(nproc)" 2>&1 | grep -E 'warning|error|FAIL|gate' || true
[ "${PIPESTATUS[0]}" -eq 0 ] || { echo "fast_check: build/gate FAILED"; exit 1; }
mask=(--dev-bind / / --ro-bind /dev/null /dev/kfd)
for d in /dev/dri/renderD*; do [ -e "$d" ] && mask+=(--ro-bind /dev/null "$d"); done
if command -v bwrap >/dev/null; then bwrap "${mask[@]}" ctest --test-dir "$B" -R "$rx" -j"$(nproc)" 2>&1 | tail -3
else echo "fast_check: bwrap missing, skipping ctest (never run GPU tests unmasked here)"; fi

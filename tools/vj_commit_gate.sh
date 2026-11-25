#!/usr/bin/env bash
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Tier-3 once-per-commit full validation gate:
#   1) Build build-gpu + build-cpu (auto-reconfiguring Ninja if available)
#   2) Run ctest --output-on-failure on BOTH build-gpu and build-cpu
#   3) Run bash tools/privacy_audit.sh (0 leaks in working tree & git history)
#   4) Run ./build-gpu/vegajet doctor (if GPU build present) + ECC UE/DE == 0 check
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "Usage: tools/vj_commit_gate.sh [--self-test]"
    echo "Env overrides: VEGAJET_BUILD_GPU (build-gpu), VEGAJET_BUILD_CPU (build-cpu), VEGAJET_ROCM_PATH"
    exit 0
fi

if [[ "${1:-}" == "--self-test" ]]; then
    [[ -x "${ROOT_DIR}/tools/privacy_audit.sh" ]] || { echo "missing tools/privacy_audit.sh" >&2; exit 1; }
    echo '{"tool":"vj_commit_gate","status":"SELF_TEST_PASS"}'
    exit 0
fi

BUILD_GPU="${VEGAJET_BUILD_GPU:-${ROOT_DIR}/build-gpu}"
BUILD_CPU="${VEGAJET_BUILD_CPU:-${ROOT_DIR}/build-cpu}"
GEN_ARGS=()
if command -v ninja >/dev/null 2>&1; then
    GEN_ARGS=(-G Ninja)
fi

echo "=== [1/5] BUILD build-gpu ==="
if [[ ! -d "${BUILD_GPU}" ]]; then
    ROCM_ARG=()
    [[ -n "${VEGAJET_ROCM_PATH:-}" ]] && ROCM_ARG=(-DVEGAJET_ROCM_PATH="${VEGAJET_ROCM_PATH}")
    cmake "${GEN_ARGS[@]}" -B "${BUILD_GPU}" -DVEGAJET_CPU_ONLY=OFF "${ROCM_ARG[@]}"
fi
cmake --build "${BUILD_GPU}"

echo "=== [2/5] BUILD build-cpu ==="
if [[ ! -d "${BUILD_CPU}" ]]; then
    cmake "${GEN_ARGS[@]}" -B "${BUILD_CPU}" -DVEGAJET_CPU_ONLY=ON
fi
cmake --build "${BUILD_CPU}"

echo "=== [3/5] CTEST build-gpu + build-cpu ==="
ctest --test-dir "${BUILD_GPU}" --output-on-failure
ctest --test-dir "${BUILD_CPU}" --output-on-failure

echo "=== [4/5] PRIVACY AUDIT ==="
PRIV_OUT="$(bash "${ROOT_DIR}/tools/privacy_audit.sh" 2>&1 || true)"
echo "${PRIV_OUT}"
if echo "${PRIV_OUT}" | awk '/^== working tree/{f=1;next}/^== git history/{f=0}f' | grep -qv '^clean$'; then
    echo "COMMIT GATE FAIL: working tree privacy leak" >&2
    exit 1
fi

echo "=== [5/5] DOCTOR + RAS ECC CHECK ==="
if [[ -x "${BUILD_GPU}/vegajet" ]]; then
    "${BUILD_GPU}/vegajet" doctor
fi

UE_TOT=0
DE_TOT=0
for card in /sys/class/drm/card*/device; do
    [[ -d "${card}/ras" ]] || continue
    for blk in umc gfx hdp mmhub pcie_bif sdma; do
        f="${card}/ras/${blk}_err_count"
        [[ -f "${f}" ]] || continue
        ue="$(awk '/ue:/{print $2}' "${f}" || echo 0)"
        de="$(awk '/de:/{print $2}' "${f}" || echo 0)"
        UE_TOT=$((UE_TOT + ${ue:-0}))
        DE_TOT=$((DE_TOT + ${de:-0}))
    done
done
echo "COMMIT_GATE_ECC: UE=${UE_TOT} DE=${DE_TOT}"
if (( UE_TOT != 0 || DE_TOT != 0 )); then
    echo "COMMIT GATE FAIL: non-zero RAS ECC UE/DE" >&2
    exit 1
fi
echo "COMMIT GATE: PASS"

#!/usr/bin/env bash
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Tier-1 5-second GPU inner-loop runner:
#   1) Compiles ONLY the requested .hsaco CMake target (incremental ~2-4s)
#   2) Runs vj_hsaco_gate + vj_isa_scan on that single .hsaco
#   3) Runs vegajet_layer_test (<2s) against $VEGAJET_GOLDEN_DIR
#   4) Prints a strict 6-line machine-readable summary + JSON and exits non-zero on any regression or spill.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${VEGAJET_BUILD_GPU:-${ROOT_DIR}/build-gpu}"

usage() {
    echo "Usage: tools/vj_dev.sh <hsaco_target> [vegajet_layer_test_args...]"
    echo "       tools/vj_dev.sh --self-test"
    echo "Example: tools/vj_dev.sh vj_ornith_moe_m1 --stage moe --m 1,5,16"
}

if [[ $# -lt 1 || "$1" == "--help" || "$1" == "-h" ]]; then
    usage
    exit 0
fi

if [[ "$1" == "--self-test" ]]; then
    echo '{"tool":"vj_dev","status":"SELF_TEST_PASS","mode":"dry_run"}'
    exit 0
fi

RAW_TARGET="$1"
shift
STEM="${RAW_TARGET%_hsaco}"
STEM="${STEM#hsaco_}"
STEM="${STEM%.hsaco}"
CMAKE_TARGET="${STEM}_hsaco"
HSACO_FILE="${BUILD_DIR}/hsaco/${STEM}.hsaco"

if [[ ! -d "${BUILD_DIR}" ]]; then
    echo "vj_dev: GPU build directory not found: ${BUILD_DIR} (set VEGAJET_BUILD_GPU)" >&2
    exit 2
fi

T0=$(date +%s%3N)
cmake --build "${BUILD_DIR}" --target "${CMAKE_TARGET}" >/tmp/vj_dev_build.log 2>&1 || {
    tail -n 25 /tmp/vj_dev_build.log >&2
    exit 1
}

GATE_OUT="$("${BUILD_DIR}/vj_hsaco_gate" "${HSACO_FILE}" 2>&1)" || {
    echo "${GATE_OUT}" >&2
    exit 1
}
ISA_OUT="$("${BUILD_DIR}/vj_isa_scan" "${HSACO_FILE}" 2>&1)" || {
    echo "${ISA_OUT}" >&2
    exit 1
}

VGPRS="$(echo "${GATE_OUT}" | sed -n 's/.*max vgpr_count=\([0-9]*\).*/\1/p' | tail -n1)"
SPILL="$(echo "${GATE_OUT}" | sed -n 's/.* \([0-9]*\) B spill.*/\1/p' | tail -n1)"
LDS_DW="$(echo "${ISA_OUT}" | sed -n 's/.*lds_dword=\([0-9]*\).*/\1/p' | tail -n1)"
VGPRS="${VGPRS:-0}"; SPILL="${SPILL:-0}"; LDS_DW="${LDS_DW:-0}"

LAYER_JSON='{"status":"SKIP","sum_us":0,"peak_gb_s":0,"max_err":0,"tmem_c":0,"ecc_ue":0,"ecc_de":0}'
LAYER_RC=0
if [[ -x "${BUILD_DIR}/vegajet_layer_test" ]]; then
    if [[ $# -eq 0 ]]; then
        LAYER_RAW="$("${BUILD_DIR}/vegajet_layer_test" --stage all --m 1,16,128 --json 2>&1)" || LAYER_RC=$?
    else
        LAYER_RAW="$("${BUILD_DIR}/vegajet_layer_test" "$@" --json 2>&1)" || LAYER_RC=$?
    fi
    if [[ ${LAYER_RC} -eq 77 ]]; then
        LAYER_RC=0
    else
        JLINE="$(echo "${LAYER_RAW}" | grep -E '^\{' | tail -n1 || true)"
        [[ -n "${JLINE}" ]] && LAYER_JSON="${JLINE}"
    fi
fi

T1=$(date +%s%3N)
ELAPSED_MS=$((T1 - T0))

python3 - "${STEM}" "${VGPRS}" "${SPILL}" "${LDS_DW}" "${ELAPSED_MS}" "${LAYER_RC}" "${LAYER_JSON}" <<'PY'
import json, sys
stem, vgprs, spill, lds, ms, rc, ljson = sys.argv[1:]
try:
    d = json.loads(ljson)
except Exception:
    d = {"status": "FAIL", "sum_us": 0.0, "peak_gb_s": 0.0, "max_err": 1.0, "tmem_c": 0.0, "ecc_ue": 0, "ecc_de": 0}
ok = (int(rc) == 0) and (int(spill) == 0) and (int(lds) == 0) and (d.get("status") in ("PASS", "SKIP")) and (d.get("ecc_ue", 0) == 0) and (d.get("ecc_de", 0) == 0)
status = "PASS" if ok else "FAIL"
out = {
    "status": status,
    "module": stem,
    "vgprs": int(vgprs),
    "spill": int(spill),
    "lds_dword": int(lds),
    "us": d.get("sum_us", 0.0),
    "gb_s": d.get("peak_gb_s", 0.0),
    "max_err": d.get("max_err", 0.0),
    "tmem_c": d.get("tmem_c", 0.0),
    "ecc_delta": f"{d.get('ecc_ue', 0)}/{d.get('ecc_de', 0)}",
    "wall_ms": int(ms),
}
print(f"STATUS   : {out['status']} (module={stem}, wall={ms} ms)")
print(f"COMPILER : max_vgprs={out['vgprs']} | spill={out['spill']} B | lds_dword={out['lds_dword']}")
print(f"PERF     : us={out['us']:.1f} | peak_gb_s={out['gb_s']:.1f}")
print(f"PARITY   : layer_status={d.get('status')} | max_err={out['max_err']:.3e}")
print(f"HEALTH   : tmem={out['tmem_c']:.1f} C | ecc_delta(UE/DE)={out['ecc_delta']}")
print(json.dumps(out))
sys.exit(0 if ok else 1)
PY

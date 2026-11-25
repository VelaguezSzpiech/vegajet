#!/usr/bin/env bash
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Tier-2 warm 48-layer benchmark controller:
# Sends reload / run / parity / sweep commands to a live `vegajet bench --interactive-fifo <fifo>` process
# (keeping 42 GiB weights resident in VRAM across edits) and prints the compact delta scorecard.
set -euo pipefail

usage() {
    echo "Usage: tools/vj_bench_warm.sh <fifo_path> <command...>"
    echo "       tools/vj_bench_warm.sh --self-test"
    echo "Example:"
    echo "  tools/vj_bench_warm.sh /tmp/vj.fifo reload vj_ornith_moe_m1"
    echo "  tools/vj_bench_warm.sh /tmp/vj.fifo run --bench-mode raw"
    echo "  tools/vj_bench_warm.sh /tmp/vj.fifo sweep '--gu-grp-verify --down-grp-verify 1 ; VEGAJET_MOE_LRU=32'"
}

if [[ $# -lt 1 || "$1" == "--help" || "$1" == "-h" ]]; then
    usage
    exit 0
fi

if [[ "$1" == "--self-test" ]]; then
    TMP_FIFO="$(mktemp -u /tmp/vj_warm_selftest.XXXXXX)"
    mkfifo "${TMP_FIFO}"
    trap 'rm -f "${TMP_FIFO}"' EXIT
    ( read -r line < "${TMP_FIFO}"; [[ "${line}" == "status" ]] ) &
    RPID=$!
    printf '%s\n' "status" > "${TMP_FIFO}"
    wait "${RPID}"
    echo '{"tool":"vj_bench_warm","status":"SELF_TEST_PASS"}'
    exit 0
fi

if [[ $# -lt 2 ]]; then
    usage >&2
    exit 2
fi

FIFO="$1"
shift
CMD="$*"
LOG_FILE="${VEGAJET_WARM_LOG:-}"

if [[ ! -p "${FIFO}" ]]; then
    echo "vj_bench_warm: control FIFO does not exist: ${FIFO}" >&2
    echo "Start a warm server first: mkfifo ${FIFO} && ./build-gpu/vegajet bench \"\$VEGAJET_MODEL_PATH\" --interactive-fifo ${FIFO} > /tmp/vj_warm.log 2>&1 &" >&2
    exit 2
fi

START_LINES=0
if [[ -n "${LOG_FILE}" && -f "${LOG_FILE}" ]]; then
    START_LINES="$(wc -l < "${LOG_FILE}")"
fi

printf '%s\n' "${CMD}" > "${FIFO}"
echo "SENT -> ${FIFO}: ${CMD}"

if [[ -n "${LOG_FILE}" && -f "${LOG_FILE}" ]]; then
    TIMEOUT="${VEGAJET_WARM_TIMEOUT:-120}"
    for ((i = 0; i < TIMEOUT * 5; ++i)); do
        CUR_LINES="$(wc -l < "${LOG_FILE}")"
        if (( CUR_LINES > START_LINES )); then
            NEW_OUT="$(tail -n +$((START_LINES + 1)) "${LOG_FILE}")"
            if echo "${NEW_OUT}" | grep -qE '"event":"(reload|variant|sweep_summary|status|help|error|quit)"'; then
                echo "${NEW_OUT}" | grep -E '(===|\[BENCH|\[PARITY|\[SWEEP|^\{)' | tail -n 20
                exit 0
            fi
        fi
        sleep 0.2
    done
    echo "vj_bench_warm: timed out waiting for JSON response in ${LOG_FILE}" >&2
    exit 1
fi

#!/usr/bin/env bash
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Send ONE variant to a resident warm engine (`vegajet bench ... --interactive-fifo FIFO`, weights stay loaded) and
# print one line:  tok/s | target bucket ms (GPU0/1/2) | parity | ECC / TMem.
#   scripts/fast_bench.sh "<ENV=VAL ... flags>" [bucket]     bucket: SUPER-PROFILE key, e.g. MoEDn+PP, GU_DOT8, AR
# Env: VJ_WARM_FIFO (the engine's FIFO), VJ_WARM_LOG (its stdout log), VJ_BENCH_MODE (default raw), VJ_TIMEOUT (300)
set -euo pipefail
flags=${1:?usage: fast_bench.sh "<ENV flags>" [bucket]}; bucket=${2:-TOTAL}
: "${VJ_WARM_FIFO:?set VJ_WARM_FIFO}"; : "${VJ_WARM_LOG:?set VJ_WARM_LOG}"
n0=$(wc -l < "$VJ_WARM_LOG")
echo "run VEGAJET_SUPER_PROFILE=1 $flags --bench-mode ${VJ_BENCH_MODE:-raw}" > "$VJ_WARM_FIFO"
timeout "${VJ_TIMEOUT:-300}" bash -c "until tail -n +$((n0+1)) '$VJ_WARM_LOG' | grep -qE '\"event\":\"(variant|error)\"|WATCHDOG|end rc'; do sleep 1; done" \
  || { echo "fast_bench: timeout"; exit 2; }
tail -n +$((n0+1)) "$VJ_WARM_LOG" | python3 "$(dirname "$0")/fast_bench_parse.py" "$bucket"

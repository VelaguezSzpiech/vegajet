#!/usr/bin/env bash
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# A2 Split-V=8 GDN ConvRec: first matched A/B pair (exclusive gpuq window only).
#
# One engine load, two warm-sweep variants on the same weights:
#   baseline : --no-gdn-splitv8 --bench-mode raw   (VEGAJET_ENABLE_SUPER=0)
#   candidate: --gdn-splitv8    --bench-mode raw   (VEGAJET_ENABLE_SUPER=0)
#
# The raw sweep path runs the same run_greedy_raw()/step_batch(M=1) decode loop
# as `[BENCH tg128]` / `[BENCH tg512]`, prints in-process token parity against
# variant 1, and journals every enqueue to VEGAJET_DISPATCH_LOG.  The verifier
# requires both A2 dispatches for every candidate M=1 GDN ConvRec, no fused
# Split-V=4 fallback, and an otherwise identical enqueue sequence.
#
# This script NEVER builds, downloads, hashes the model, or changes hardware
# settings.  Run it through gpuq:
#
#   gpuq run --purpose "VegaJet A2 matched sweep" --max-sec 900 --loads 1 \
#       -- bash -eu tools/a2/pair.sh
#
# Required env:
#   VJ_BIN            built vegajet executable
#   VJ_MODEL          Ornith GGUF used for the build/golden fixture
#   VJ_HSACO_DIR      directory containing vj_ornith_gdn_sv8.hsaco and friends
# Optional:
#   AB_MODE=sweep|layer|both   (default sweep)
#   VJ_LAYER_TEST_BIN, VJ_GOLDEN_DIR   required for layer/both
#   AB_OUT            output directory (default results/a2-work/out/<utc>)
#   AB_TG512=1        run tg512 as well (default 1; use --loads 1)
#   AB_STEPS          671 with tg512, 159 without (default computed)
#   AB_GDN_LAYERS=36
#   AB_ARM_TIMEOUT=600
#   AB_LOCK_DIR       (default <script dir>/.a2-pair.lock)
#
set -Eeuo pipefail
umask 077

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODE=${AB_MODE:-sweep}
TG512=${AB_TG512:-1}
GDN_LAYERS=${AB_GDN_LAYERS:-36}
ARM_TIMEOUT=${AB_ARM_TIMEOUT:-600}
LATCH_ROOT=$(getent passwd "$(id -u)" | cut -d: -f6)/local-ai
OUT=${AB_OUT:-$SCRIPT_DIR/out/$(date -u +%Y%m%dT%H%M%SZ)}
LOCK_DIR=${AB_LOCK_DIR:-$SCRIPT_DIR/.a2-pair.lock}
VERDICT=FAIL
WHY=()

die() { printf 'A2 PAIR FAIL: %s\n' "$*" >&2; exit 1; }
note() { printf 'A2 PAIR: %s\n' "$*" >&2; }

LOCK_HELD=0
cleanup() {
    if (( LOCK_HELD )); then
        rmdir -- "$LOCK_DIR" 2>/dev/null || true
    fi
}
trap cleanup EXIT

case "$MODE" in sweep|layer|both) ;; *) die "AB_MODE must be sweep|layer|both (got '$MODE')" ;; esac
case "$TG512" in 0|1) ;; *) die "AB_TG512 must be 0 or 1 (got '$TG512')" ;; esac
[[ "$GDN_LAYERS" =~ ^[0-9]+$ ]] || die "AB_GDN_LAYERS must be an integer"
STEPS=${AB_STEPS:-}
if [[ -z "$STEPS" ]]; then
    if [[ "$TG512" == 1 ]]; then STEPS=671; else STEPS=159; fi
fi
[[ "$STEPS" =~ ^[0-9]+$ ]] || die "AB_STEPS must be an integer"

# ---------------------------------------------------------------- exclusive guard
command -v gpuq >/dev/null || die "gpuq not on PATH"
command -v timeout >/dev/null || die "timeout not on PATH"
[[ -n "${GPUQ_TOKEN:-}" ]] || die "no GPUQ_TOKEN: run this script through 'gpuq run'"
gpuq verify --token "${GPUQ_TOKEN}" >/dev/null 2>&1 \
    || die "GPUQ_TOKEN does not own the live window (run through 'gpuq run')"
[[ -d "$LATCH_ROOT/run" && -r "$LATCH_ROOT/run" && -x "$LATCH_ROOT/run" ]] \
    || die "latch directory '$LATCH_ROOT/run' is missing or unreadable"
[[ ! -e "$LATCH_ROOT/run/TRIPPED" && ! -L "$LATCH_ROOT/run/TRIPPED" ]] \
    || die "TRIPPED latch present ($LATCH_ROOT/run/TRIPPED); owner must clear it, do not bypass"
[[ ! -e "$LATCH_ROOT/run/NO-GPU-STOP" ]] \
    || die "NO-GPU-STOP latch present; refusing to start"
[[ -d /sys/class/kfd/kfd/proc && -r /sys/class/kfd/kfd/proc ]] \
    || die "/sys/class/kfd/kfd/proc is missing or unreadable; cannot prove exclusivity"
shopt -s nullglob
kfd_procs=(/sys/class/kfd/kfd/proc/*)
shopt -u nullglob
(( ${#kfd_procs[@]} == 0 )) \
    || die "${#kfd_procs[@]} process(es) already attached to KFD; refusing to start"
mkdir -- "$LOCK_DIR" 2>/dev/null \
    || die "A2 pair latch '$LOCK_DIR' already exists (another run or stale lock)"
LOCK_HELD=1
mkdir -p -- "$OUT" || die "cannot create output directory $OUT"

# ------------------------------------------------------------------- prerequisites
[[ -n "${VJ_BIN:-}" && -x "${VJ_BIN:-}" ]] || die "VJ_BIN is unset or not executable"
[[ -n "${VJ_MODEL:-}" && -r "${VJ_MODEL:-}" ]] || die "VJ_MODEL is unset or unreadable"
[[ -n "${VJ_HSACO_DIR:-}" && -d "${VJ_HSACO_DIR:-}" ]] || die "VJ_HSACO_DIR is unset or not a directory"
SV8_MOD="$VJ_HSACO_DIR/vj_ornith_gdn_sv8.hsaco"
[[ -r "$SV8_MOD" ]] || die "missing $SV8_MOD (build target hsaco_vj_ornith_gdn_sv8)"
grep -aq 'ornith_gdn_conv_recur_splitv8_recur_m1_kernel' "$SV8_MOD" \
    || die "stage-1 symbol absent from $SV8_MOD"
grep -aq 'ornith_gdn_conv_recur_splitv8_quant_m1_kernel' "$SV8_MOD" \
    || die "stage-2 symbol absent from $SV8_MOD"

LAYER_BIN=${VJ_LAYER_TEST_BIN:-}
GOLDEN_DIR=${VJ_GOLDEN_DIR:-}
if [[ "$MODE" == layer || "$MODE" == both ]]; then
    [[ -n "$LAYER_BIN" && -x "$LAYER_BIN" ]] \
        || die "AB_MODE=$MODE needs VJ_LAYER_TEST_BIN (built vegajet_layer_test)"
    [[ -n "$GOLDEN_DIR" && -d "$GOLDEN_DIR" ]] \
        || die "AB_MODE=$MODE needs VJ_GOLDEN_DIR (golden layer fixture)"
    [[ -r "$GOLDEN_DIR/m1_gdn_partial_out.bin" ]] \
        || die "$GOLDEN_DIR does not look like the GDN golden fixture"
fi

# ------------------------------------------------------------------------ metadata
BIN_VER=$("$VJ_BIN" --version 2>/dev/null || echo unknown)
{
    printf 'utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'host=%s\n' "$(hostname)"
    printf 'mode=%s tg512=%s gdn_layers=%s steps=%s\n' "$MODE" "$TG512" "$GDN_LAYERS" "$STEPS"
    printf 'gpuq_session=%s\n' "${GPUQ_SESSION:-}"
    printf 'gpuq_purpose=%s\n' "${GPUQ_PURPOSE:-}"
    printf 'gpuq_max_sec=%s\n' "${GPUQ_MAX_SEC:-}"
    printf 'vegajet=%s sha256=%s\n' "$BIN_VER" "$(sha256sum -- "$VJ_BIN" | cut -d' ' -f1)"
    printf 'model=%s size=%s mtime=%s\n' "$VJ_MODEL" "$(stat -Lc %s -- "$VJ_MODEL")" "$(stat -Lc %y -- "$VJ_MODEL")"
    printf 'hsaco_dir=%s\n' "$VJ_HSACO_DIR"
    for f in "$VJ_HSACO_DIR"/*.hsaco; do
        [[ -e "$f" ]] || continue
        printf 'hsaco %s %s\n' "$(sha256sum -- "$f" | cut -d' ' -f1)" "$(basename -- "$f")"
    done
    if [[ -n "${VJ_TREE:-}" ]]; then
        printf 'tree=%s head=%s\n' "$VJ_TREE" "$(git -C "$VJ_TREE" rev-parse HEAD 2>/dev/null || echo n/a)"
    fi
} >"$OUT/meta.txt"

note "output $OUT"
note "guard ok: gpuq window ${GPUQ_SESSION:-?} purpose '${GPUQ_PURPOSE:-?}'; latch clean"
note "sweep expected: $GDN_LAYERS GDN layers x 2 ranks x $STEPS M=1 steps = $((GDN_LAYERS * 2 * STEPS)) GDN ConvRec enqueues per variant"

# ------------------------------------------------------------------- env sanitation
sanitize_env() {
    local var
    while IFS= read -r var; do
        [[ -n "$var" ]] || continue
        unset "$var"
    done < <(
        compgen -e | grep -E '^(VEGAJET_|VJ_GPU_DEVICES$|HIP_VISIBLE_DEVICES$|ROCR_VISIBLE_DEVICES$)' || true
    )
}

wait_kfd_drain() {
    local i
    for i in $(seq 1 30); do
        shopt -s nullglob
        local p=(/sys/class/kfd/kfd/proc/*)
        shopt -u nullglob
        if (( ${#p[@]} == 0 )); then return 0; fi
        sleep 1
    done
    return 1
}

gpuq_load() {
    gpuq load --token "${GPUQ_TOKEN}" --cards 2 >/dev/null \
        || die "gpuq load refused (declare the real number of engine starts in --loads)"
}

# ------------------------------------------------------------------------- sweep
run_sweep() {
    sanitize_env
    export VEGAJET_ENABLE_SUPER=0
    export VEGAJET_RAW_TG512="$TG512"
    export VEGAJET_DISPATCH_LOG="$OUT/sweep.dispatch.log"
    [[ ! -e "$VEGAJET_DISPATCH_LOG" ]] || die "dispatch log already exists: $VEGAJET_DISPATCH_LOG"

    local spec='--no-gdn-splitv8 --bench-mode raw ; --gdn-splitv8 --bench-mode raw'
    local -a cmd=("$VJ_BIN" bench "$VJ_MODEL" --tp 2 --hsaco-dir "$VJ_HSACO_DIR" --sweep "$spec")
    printf '%q ' "${cmd[@]}" >"$OUT/sweep.command.txt"
    printf '\n' >>"$OUT/sweep.command.txt"

    gpuq_load
    note "starting sweep (one engine load)"
    local rc=0
    set +e
    timeout -k 15 "$ARM_TIMEOUT" "${cmd[@]}" >"$OUT/sweep.stdout" 2>"$OUT/sweep.stderr"
    rc=$?
    set -e
    printf 'engine_rc=%s\n' "$rc" >>"$OUT/sweep.command.txt"
    if (( rc == 124 || rc == 137 )); then
        WHY+=("sweep engine timed out after ${ARM_TIMEOUT}s (rc=$rc)")
    fi

    wait_kfd_drain || WHY+=("KFD still has attached process(es) after the sweep")

    local -a args=(
        "$SCRIPT_DIR/check_a2_sweep.py"
        --stdout "$OUT/sweep.stdout"
        --dispatch-log "$OUT/sweep.dispatch.log"
        --exit-code "$rc"
        --gdn-layers "$GDN_LAYERS"
        --steps "$STEPS"
        --json-out "$OUT/verdict.json"
    )
    if [[ "$TG512" == 1 ]]; then args+=(--require-tg512); fi
    local arc=0
    set +e
    python3 "${args[@]}" >"$OUT/summary.txt" 2>&1
    arc=$?
    set -e
    cat "$OUT/summary.txt" >&2
    (( arc == 0 )) || WHY+=("dispatch/stdout verifier failed (see summary.txt)")
}

# -------------------------------------------------------------------- layer A/B
run_layer() {
    sanitize_env
    export VEGAJET_ENABLE_SUPER=0
    export VEGAJET_DISPATCH_LOG="$OUT/layer.dispatch.log"
    local -a cmd=("$LAYER_BIN" --golden-dir "$GOLDEN_DIR" --hsaco-dir "$VJ_HSACO_DIR"
                  --stage gdn --m 1 --iters 5 --gdn-splitv8)
    printf '%q ' "${cmd[@]}" >"$OUT/layer.command.txt"
    printf '\n' >>"$OUT/layer.command.txt"

    gpuq_load
    note "starting layer-test A/B (one engine load)"
    local rc=0
    set +e
    timeout -k 15 "$ARM_TIMEOUT" "${cmd[@]}" >"$OUT/layer.stdout" 2>"$OUT/layer.stderr"
    rc=$?
    set -e
    printf 'engine_rc=%s\n' "$rc" >>"$OUT/layer.command.txt"
    (( rc == 0 )) || WHY+=("layer test exited $rc")
    grep -aq '"status":"PASS"' "$OUT/layer.stdout" \
        || WHY+=("layer test did not report status PASS")
    grep -aqE 'gdn-ab[[:space:]]+M=1.*PASS' "$OUT/layer.stdout" \
        || WHY+=("layer test gdn-ab row did not report bit-exact PASS")
    grep -aqE 'gdn[[:space:]]+M=1.*PASS' "$OUT/layer.stdout" \
        || WHY+=("layer test gdn M=1 row did not report PASS")
    wait_kfd_drain || WHY+=("KFD still has attached process(es) after the layer test")
}

# --------------------------------------------------------------------------- main
if [[ "$MODE" == sweep || "$MODE" == both ]]; then run_sweep; fi
if [[ "$MODE" == layer || "$MODE" == both ]]; then run_layer; fi

if (( ${#WHY[@]} == 0 )); then
    VERDICT=PASS
fi
{
    printf 'status=%s\n' "$VERDICT"
    printf 'mode=%s tg512=%s gdn_layers=%s steps=%s\n' "$MODE" "$TG512" "$GDN_LAYERS" "$STEPS"
    printf 'expected_per_variant=%s\n' "$((GDN_LAYERS * 2 * STEPS))"
    printf 'output=%s\n' "$OUT"
    if (( ${#WHY[@]} )); then
        printf 'reasons:\n'
        printf '  - %s\n' "${WHY[@]}"
    fi
} >"$OUT/verdict.txt"
cat "$OUT/verdict.txt" >&2

if [[ "$VERDICT" == PASS ]]; then
    note "PASS; verdict $OUT/verdict.txt"
    exit 0
fi
note "FAIL; verdict $OUT/verdict.txt"
exit 1

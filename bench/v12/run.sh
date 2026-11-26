#!/usr/bin/env bash
# VegaJet v1.2 A/B runner for the inference machine (2x MI50 32 GB).
# Runs each v1.2 toggle OFF vs ON, 1 warm-up + N timed runs, and stores the raw logs.
# Usage:
#   MODEL=<model-file> [DRAFT=<dflash2-file>] [MTP=<mtp-file>] \
#   [RUNS=5] [N=128] [ONLY="VOCAB_ILP DENSE_PERSIST"] bench/v12/run.sh
# Pin the same two MI50s for every run (third card must not be used for v1.2 numbers):
#   export the device-selection your rig uses (e.g. ROCR_VISIBLE_DEVICES / the hw_profile) BEFORE running.
set -euo pipefail

BIN=${BIN:-./build-gpu/vegajet}
MODEL=${MODEL:?set MODEL=<model-file>}
RUNS=${RUNS:-5}
N=${N:-128}
OUT=${OUT:-bench/v12/out/$(date +%Y%m%d-%H%M%S)-$(git rev-parse --short HEAD)}
mkdir -p "$OUT"

# Add a fixed prompt with EXTRA="--prompt '...'" to test normal text as well as the default prompt.
EXTRA=${EXTRA:-}

run() { # name, env, args...
  local name=$1 envs=$2; shift 2
  echo "== $name [$envs]" | tee -a "$OUT/summary.txt"
  env $envs "$BIN" bench "$MODEL" "$@" $EXTRA > "$OUT/${name}.warmup.log" 2>&1 || true
  for i in $(seq 1 "$RUNS"); do
    if ! env $envs "$BIN" bench "$MODEL" "$@" $EXTRA --json-out "$OUT/${name}.$i.json" > "$OUT/${name}.$i.log" 2>&1; then
      echo "   run $i: FAILED (exit $?) -> see $OUT/${name}.$i.log" | tee -a "$OUT/summary.txt"
    fi
  done
}

# Each entry: NAME|env when ON|bench args
RAW="-n $N"
SPEC="-n $N --draft ${DRAFT:-} --fr-spec 65536 --spec-gdn --zero-tax-catchup --suffix-max-draft 15"
MTPA="-n $N --draft ${MTP:-} --mtp-k 4 --spec-gdn"
LEVERS=(
  "VOCAB_ILP|VEGAJET_V12_VOCAB_ILP=1|$RAW"
  "DENSE_PERSIST|VEGAJET_V12_DENSE_PERSIST=1|$RAW"
  "IQ2S_REG|VEGAJET_V12_IQ2S_REG=1|$RAW"
  "BARRIER_VOL|VEGAJET_V12_BARRIER_VOL=1|$RAW"
  "SPEC_GOV|VEGAJET_V12_SPEC_GOV=1|$MTPA"
  "BATCH_CATCHUP|VEGAJET_V12_BATCH_CATCHUP=1|$MTPA"
)
# Group-2 levers are appended here by later commits (GDN_SPLITV, VERIFY_FUSED_NQ, NARROW_VERIFY, TREE_VERIFY, ASYM_EXPERTS).
[ -f bench/v12/levers_extra.sh ] && source bench/v12/levers_extra.sh

run baseline_raw "VEGAJET_V12_NONE=1" $RAW
[ -n "${DRAFT:-}" ] && run baseline_spec "VEGAJET_V12_NONE=1" $SPEC
[ -n "${MTP:-}" ]   && run baseline_mtp  "VEGAJET_V12_NONE=1" $MTPA

for e in "${LEVERS[@]}"; do
  IFS='|' read -r name envs args <<< "$e"
  [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $name "* ]] && continue
  if [[ "$args" == *"--draft  "* || "$args" == *"--draft --"* ]]; then echo "skip $name (no draft model set)"; continue; fi
  run "$name" "$envs" $args
done

echo "Logs in $OUT. Compare tok/s medians and the [PARITY]/[BENCH-VALIDATE] lines per lever."
echo "Any lever with a token mismatch vs baseline_raw is a FAIL, whatever its speed."

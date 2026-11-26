#!/usr/bin/env bash
# VegaJet v1.2 A/B runner for the inference machine (2x MI50 32 GB).
# Runs each v1.2 toggle OFF vs ON, 1 warm-up + N timed runs, and stores the raw logs.
# Usage:
#   MODEL=<model-file> [DRAFT=<dflash2-file>] [MTP=<mtp-file>] \
#   [RUNS=5] [N=128] [ONLY="VOCAB_ILP DENSE_PERSIST"] bench/v12/run.sh
# Pin the same two MI50s for every run by PCI address (see `vegajet serve --gpu list`):
#   GPUS=dddd:bb:dd.f,dddd:bb:dd.f MODEL=... bench/v12/run.sh
set -euo pipefail

BIN=${BIN:-./build-gpu/vegajet}
MODEL=${MODEL:?set MODEL=<model-file>}
RUNS=${RUNS:-5}
N=${N:-128}
OUT=${OUT:-bench/v12/out/$(date +%Y%m%d-%H%M%S)-$(git rev-parse --short HEAD)}
mkdir -p "$OUT"

# Add a fixed prompt with EXTRA="--prompt '...'" to test normal text as well as the default prompt.
EXTRA=${EXTRA:-}
GPUS=${GPUS:-}
if [[ -n $GPUS ]]; then
  EXTRA="$EXTRA --gpu $GPUS"
else
  echo "WARNING: GPUS is not set; the engine selects the cards. Set GPUS=<bdf0>,<bdf1> (vegajet serve --gpu list)." >&2
fi

failed=0
run() { # name, env, args...
  local name=$1 envs=$2 key entry status; shift 2
  local -a clean_env=(env)
  # Remove even unknown/future inherited v1.2 flags; VEGAJET_V12_NONE is not an engine switch.
  while IFS= read -r key; do
    [[ $key != VEGAJET_V12_* ]] || clean_env+=(-u "$key")
  done < <(compgen -e)
  echo "== $name [$envs]" | tee -a "$OUT/summary.txt"
  "${clean_env[@]}" $envs env | while IFS= read -r entry; do
    if [[ $entry == VEGAJET_V12_*=* || $entry == VEGAJET_ENABLE_SUPER=* ]]; then printf '%s\n' "$entry"; fi
  done > "$OUT/${name}.env"
  if "${clean_env[@]}" $envs "$BIN" bench "$MODEL" "$@" $EXTRA > "$OUT/${name}.warmup.log" 2>&1; then :; else
    status=$?
    echo "   warmup: FAILED (exit $status)" | tee -a "$OUT/summary.txt"
    failed=1
    return
  fi
  for i in $(seq 1 "$RUNS"); do
    if "${clean_env[@]}" $envs "$BIN" bench "$MODEL" "$@" $EXTRA --json-out "$OUT/${name}.$i.json" > "$OUT/${name}.$i.log" 2>&1; then :; else
      status=$?
      echo "   run $i: FAILED (exit $status) -> see $OUT/${name}.$i.log" | tee -a "$OUT/summary.txt"
      failed=1
    fi
  done
}

# Each entry: NAME|env when ON|bench args
RAW="-n $N"
SPEC="-n $N --draft ${DRAFT:-} --fr-spec 65536 --spec-gdn --zero-tax-catchup --suffix-max-draft 15"
MTPA="-n $N --draft ${MTP:-} --mtp-k 4 --spec-gdn"
LEVERS=(
  "VOCAB_ILP|VEGAJET_V12_VOCAB_ILP=1|$RAW"
  "DENSE_PERSIST|VEGAJET_ENABLE_SUPER=0 VEGAJET_V12_DENSE_PERSIST=1|$RAW"
  "IQ2S_REG|VEGAJET_V12_IQ2S_REG=1|$RAW"
  "BARRIER_VOL|VEGAJET_V12_BARRIER_VOL=1|$RAW"
  "SPEC_GOV|VEGAJET_V12_SPEC_GOV=1|$MTPA"
  "BATCH_CATCHUP|VEGAJET_V12_BATCH_CATCHUP=1|$MTPA"
  "NARROW_VERIFY|VEGAJET_V12_NARROW_VERIFY=1|$MTPA"
  "VERIFY_FUSED_NQ|VEGAJET_V12_VERIFY_FUSED_NQ=1|$MTPA"
  "GDN_SPLITV|VEGAJET_ENABLE_SUPER=0 VEGAJET_V12_GDN_SPLITV=1|$RAW"
  "ASYM_EXPERTS|VEGAJET_V12_ASYM_EXPERTS=1|$RAW"
)
# TREE_VERIFY is host-side only in v1.2 and fails closed when enabled; it is not benchmarked.

run baseline_raw "" $RAW
[ -n "${DRAFT:-}" ] && run baseline_spec "" $SPEC
[ -n "${MTP:-}" ]   && run baseline_mtp  "" $MTPA

for e in "${LEVERS[@]}"; do
  IFS='|' read -r name envs args <<< "$e"
  [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $name "* ]] && continue
  if [[ "$args" == *"--draft  "* || "$args" == *"--draft --"* ]]; then echo "skip $name (no draft model set)"; continue; fi
  if [[ $name == DENSE_PERSIST || $name == GDN_SPLITV ]] && [[ ! -e "$OUT/baseline_nosuper.env" ]]; then run baseline_nosuper "VEGAJET_ENABLE_SUPER=0" $RAW; fi
  run "$name" "$envs" $args
done

echo "Logs in $OUT. Compare tok/s medians and the [PARITY]/[BENCH-VALIDATE] lines per lever."
echo "Any lever with a token mismatch vs baseline_raw is a FAIL, whatever its speed."
echo "This runner is NOT evidence of main/all-off token parity; full token-ID comparison still needs a GPU harness."
exit "$failed"

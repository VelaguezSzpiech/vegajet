#!/usr/bin/env bash
# VegaJet v1.2 A/B runner for the inference machine (2x MI50 32 GB).
# Runs each v1.2 toggle OFF vs ON, 1 warm-up + N timed runs, and stores the raw logs.
# Usage:
#   MODEL=<model-file> [DRAFT=<dflash2-file>] [MTP=<mtp-file>] \
#   [RUNS=5] [N=128] [ONLY="VOCAB_ILP DENSE_PERSIST"] bench/v12/run.sh
# Pin the same two MI50s for every run by PCI address (see `vegajet serve --gpu list`):
#   GPUS=dddd:bb:dd.f,dddd:bb:dd.f MODEL=... bench/v12/run.sh
# Token parity (REVIEW M6): every timed run writes `--dump-tokens "$OUT/<name>.<i>.tokens"`. Set
#   REF_TOKENS=<dir>  to compare each run's ids against a reference dump from an all-off/main run
#                     (uses "$REF_TOKENS/<name>.tokens", else "$REF_TOKENS/baseline_raw.tokens").
#                     A mismatch (or a missing dump) is a runner failure, never a speed result.
#   DUMP_TOKENS=0     turns the per-run token dump off (for a binary that predates --dump-tokens).
set -euo pipefail

BIN=${BIN:-./build-gpu/vegajet}
MODEL=${MODEL:?set MODEL=<model-file>}
RUNS=${RUNS:-5}
N=${N:-128}
OUT=${OUT:-bench/v12/out/$(date +%Y%m%d-%H%M%S)-$(git rev-parse --short HEAD)}
mkdir -p "$OUT"

# Optional reference token dumps and the comparison gate (tools/token_compare.py, stdlib only).
REF_TOKENS=${REF_TOKENS:-}
DUMP_TOKENS=${DUMP_TOKENS:-1}
PYTHON=${PYTHON:-python3}
TOKEN_COMPARE=${TOKEN_COMPARE:-tools/token_compare.py}
if [[ -n $REF_TOKENS && ! -d $REF_TOKENS ]]; then
  echo "REF_TOKENS=$REF_TOKENS is not a directory of token dumps" >&2
  exit 2
fi
if [[ -n $REF_TOKENS ]]; then
  command -v "$PYTHON" >/dev/null 2>&1 || { echo "REF_TOKENS set but $PYTHON is not on PATH" >&2; exit 2; }
  [[ -f $TOKEN_COMPARE ]] || { echo "REF_TOKENS set but $TOKEN_COMPARE is missing" >&2; exit 2; }
fi
ref_warned=0

# Add a fixed prompt with EXTRA="--prompt '...'" to test normal text as well as the default prompt.
EXTRA=${EXTRA:-}
GPUS=${GPUS:-}
if [[ -n $GPUS ]]; then
  EXTRA="$EXTRA --gpu $GPUS"
else
  echo "WARNING: GPUS is not set; the engine selects the cards. Set GPUS=<bdf0>,<bdf1> (vegajet serve --gpu list)." >&2
fi

failed=0
# Compare one run's token dump to the reference sequence (REVIEW M6). A mismatch, an unusable dump or a missing
# dump is a runner failure: throughput from a diverged run must never be reported as a result.
compare_tokens() { # name, run index, dump path
  local name=$1 idx=$2 dump=$3 ref out rc
  ref="$REF_TOKENS/$name.tokens"
  [[ -f $ref ]] || ref="$REF_TOKENS/baseline_raw.tokens"
  if [[ ! -f $ref ]]; then
    if (( ref_warned == 0 )); then
      echo "WARNING: no $REF_TOKENS/$name.tokens and no $REF_TOKENS/baseline_raw.tokens; token compare skipped" \
        | tee -a "$OUT/summary.txt"
      ref_warned=1
    fi
    return 0
  fi
  if [[ ! -f $dump ]]; then
    echo "   run $idx: NO TOKEN DUMP ($dump missing, DUMP_TOKENS=$DUMP_TOKENS) -> FAIL" | tee -a "$OUT/summary.txt"
    failed=1
    return 0
  fi
  if out=$("$PYTHON" "$TOKEN_COMPARE" "$ref" "$dump" 2>&1); then
    echo "   run $idx: $out" | tee -a "$OUT/summary.txt"
  else
    rc=$?
    echo "   run $idx: TOKEN MISMATCH vs $ref (token_compare exit $rc): $out" | tee -a "$OUT/summary.txt"
    failed=1
  fi
}

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
    local -a dump_args=()
    [[ $DUMP_TOKENS == 0 ]] || dump_args=(--dump-tokens "$OUT/${name}.$i.tokens")
    if "${clean_env[@]}" $envs "$BIN" bench "$MODEL" "$@" $EXTRA ${dump_args[@]+"${dump_args[@]}"} \
        --json-out "$OUT/${name}.$i.json" > "$OUT/${name}.$i.log" 2>&1; then
      [[ -z $REF_TOKENS ]] || compare_tokens "$name" "$i" "$OUT/${name}.$i.tokens"
    else
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
if [[ -z $REF_TOKENS ]]; then
  echo "Token dumps are in $OUT/*.tokens; re-run with REF_TOKENS=<dir of reference dumps> to gate id parity per run"
  echo "(tools/token_compare.py REF CAND; a mismatch is counted as a runner failure)."
else
  echo "Token ids compared against $REF_TOKENS for every run; mismatches are counted as failures."
fi
exit "$failed"

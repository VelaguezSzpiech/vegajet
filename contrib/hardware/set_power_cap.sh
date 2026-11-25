#!/usr/bin/env bash
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Set hwmon power1_cap on every gfx906 GPU, within the PSU-safe ceiling for the number of GPUs.
#   ROOT=<fixture root> ./set_power_cap.sh <watts> [--apply]     (default ROOT=/, default mode: dry run)
# Ceilings: <= 200 W per GPU with 1-2 GPUs, <= 150 W with 3 or more (shared PSU budget). Requests above the ceiling,
# below the driver's power1_cap_min, above power1_cap_max, or under 50 W are refused. Without --apply nothing is
# written; with --apply the value is written in microwatts and read back. Writing the real /sys needs root.
set -euo pipefail
ROOT="${ROOT:-/}"; ROOT="${ROOT%/}"
usage() { echo "usage: [ROOT=dir] $0 <watts> [--apply]" >&2; exit 2; }
[[ $# -ge 1 && $1 =~ ^[0-9]+$ ]] || usage
W=$1; APPLY=0
[[ "${2:-}" == "--apply" ]] && APPLY=1
[[ $# -le 2 ]] || usage
[[ $# -eq 1 || $APPLY == 1 ]] || usage

caps=()
for dev in "$ROOT"/sys/bus/pci/devices/*; do
  [[ -r "$dev/vendor" && -r "$dev/device" ]] || continue
  [[ "$(cat "$dev/vendor")" == "0x1002" ]] || continue
  case "$(cat "$dev/device")" in 0x66a0|0x66a1|0x66a3) ;; *) continue ;; esac
  for h in "$dev"/hwmon/hwmon*; do [[ -e "$h/power1_cap" ]] && { caps+=("$h"); break; }; done
done
N=${#caps[@]}
(( N > 0 )) || { echo "no gfx906 hwmon power1_cap found" >&2; exit 1; }
CEIL=200; (( N >= 3 )) && CEIL=150
if (( W > CEIL )); then echo "refused: $W W exceeds the $CEIL W per-GPU ceiling for $N GPU(s)" >&2; exit 1; fi
if (( W < 50 )); then echo "refused: $W W is below the 50 W floor" >&2; exit 1; fi
UW=$(( W * 1000000 ))
for h in "${caps[@]}"; do
  lo=$(cat "$h/power1_cap_min" 2>/dev/null || echo 0); hi=$(cat "$h/power1_cap_max" 2>/dev/null || echo "$UW")
  if (( UW < lo || UW > hi )); then echo "refused: $W W outside the driver range [$((lo / 1000000)), $((hi / 1000000))] W for ${h%/hwmon/*}" >&2; exit 1; fi
done
if (( APPLY )) && [[ -z ${ROOT} ]] && (( EUID != 0 )); then echo "refused: writing /sys needs root" >&2; exit 1; fi
for h in "${caps[@]}"; do
  name=${h%/hwmon/*}; name=${name##*/}
  cur=$(( $(cat "$h/power1_cap") / 1000000 ))
  if (( APPLY )); then
    printf '%s\n' "$UW" > "$h/power1_cap"
    got=$(( $(cat "$h/power1_cap") / 1000000 ))
    [[ $got == "$W" ]] || { echo "$name: read back $got W, expected $W W" >&2; exit 1; }
    echo "$name: power1_cap $cur W -> $got W"
  else
    echo "$name: power1_cap $cur W -> $W W (dry run; add --apply to write)"
  fi
done

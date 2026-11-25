#!/usr/bin/env bash
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# G-SF-12 preflight, read-only: is it safe to write the power-play table (pp_table / upp) right now?
#   ROOT=<fixture root> ./gsf12_preflight.sh [--post]    (default ROOT=/)
# Checks (never stops, starts or queries services; never opens a device node):
#   * no process holds /dev/kfd (readlink of /proc/<pid>/fd/*; unreadable fd tables are reported, run as root)
#   * the guard, fan and sensor-reader daemons are stopped (a unit is running when its cgroup directory exists)
#   * pp_dpm_mclk level 0 is 350 MHz on every gfx906 (FreqTableUclk/0 invariant)
# --post inverts the service expectation for the after-write check: guard and fan daemon must be running again.
# Unit names can be overridden: GUARD_UNITS, FAN_UNITS, READER_UNITS (space separated).
# Exit 0 = safe, 1 = not safe.
set -euo pipefail
ROOT="${ROOT:-/}"; ROOT="${ROOT%/}"
POST=0; [[ "${1:-}" == "--post" ]] && POST=1
GUARD_UNITS="${GUARD_UNITS:-mi50-guard.service}"
FAN_UNITS="${FAN_UNITS:-vifted.service}"
READER_UNITS="${READER_UNITS:-coolercontrold.service}"
fail=0; warn=0
say() { printf '[%s] %s\n' "$1" "$2"; [[ $1 == FAIL ]] && fail=$((fail + 1)); [[ $1 == WARN ]] && warn=$((warn + 1)); return 0; }
active() { [[ -d "$ROOT/sys/fs/cgroup/system.slice/$1" ]]; }

holders=""; unreadable=0
for p in "$ROOT"/proc/[0-9]*; do
  [[ -d $p ]] || continue
  if [[ ! -r "$p/fd" || ! -x "$p/fd" ]]; then unreadable=$((unreadable + 1)); continue; fi
  for fd in "$p"/fd/*; do
    [[ -L $fd ]] || continue
    if [[ "$(readlink "$fd" 2>/dev/null || true)" == /dev/kfd ]]; then holders+=" ${p##*/}"; break; fi
  done
done
if [[ -n $holders ]]; then say FAIL "/dev/kfd held by pid$holders: finish or stop every GPU job first"
elif (( unreadable > 0 )); then say WARN "$unreadable process fd table(s) unreadable: run as root to prove zero GPU jobs"
else say PASS "no process holds /dev/kfd"; fi

for u in $GUARD_UNITS $FAN_UNITS; do
  if (( POST )); then
    if active "$u"; then say PASS "$u running again"; else say FAIL "$u not running: restart it now"; fi
  else
    if active "$u"; then say FAIL "$u running: stop it before the write (it reads amdgpu sensors)"; else say PASS "$u stopped"; fi
  fi
done
for u in $READER_UNITS; do
  if active "$u"; then say FAIL "$u running: competing amdgpu sensor reader"; else say PASS "$u inactive"; fi
done

n=0
for dev in "$ROOT"/sys/bus/pci/devices/*; do
  [[ -r "$dev/vendor" && -r "$dev/device" ]] || continue
  [[ "$(cat "$dev/vendor")" == "0x1002" ]] || continue
  case "$(cat "$dev/device")" in 0x66a0|0x66a1|0x66a3) ;; *) continue ;; esac
  n=$((n + 1))
  name=${dev##*/}
  l0=$(awk -F'[: ]+' '$1 == "0" {sub(/[Mm][Hh]z.*/, "", $2); print $2; exit}' "$dev/pp_dpm_mclk" 2>/dev/null || true)
  if [[ -z $l0 ]]; then say WARN "$name pp_dpm_mclk unreadable"
  elif [[ $l0 == 350 ]]; then say PASS "$name mclk level 0 = 350 MHz"
  else say FAIL "$name mclk level 0 = $l0 MHz (G-SF-12 keeps 350)"; fi
done
(( n > 0 )) || say WARN "no gfx906 device found"
echo "summary: $fail fail, $warn warn -> $([[ $fail == 0 ]] && echo SAFE || echo NOT SAFE)"
if (( ! POST && fail == 0 )); then
  echo "after the write: restart $GUARD_UNITS $FAN_UNITS, then run this script with --post and soak-verify"
fi
(( fail == 0 ))

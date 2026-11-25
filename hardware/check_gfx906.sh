#!/usr/bin/env bash
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Read-only listing of Vega 20 (gfx906) devices from sysfs. Never writes, never opens device nodes.
set -euo pipefail
found=0
for dev in /sys/bus/pci/devices/*; do
  [[ -r "$dev/vendor" && -r "$dev/device" ]] || continue
  [[ "$(cat "$dev/vendor")" == "0x1002" ]] || continue
  case "$(cat "$dev/device")" in 0x66a0|0x66a1|0x66a3) ;; *) continue ;; esac
  found=$((found + 1))
  bdf=$(basename "$dev")
  bar0=$(awk 'NR==1 {printf "%d", (strtonum($2) - strtonum($1) + 1) / 1073741824}' "$dev/resource" 2>/dev/null || echo "?")
  vram=$(awk '{printf "%.1f", $1 / 1073741824}' "$dev/mem_info_vram_total" 2>/dev/null || echo "?")
  cap="?"
  for h in "$dev"/hwmon/hwmon*; do [[ -r "$h/power1_cap" ]] && cap=$(( $(cat "$h/power1_cap") / 1000000 )); done
  sclk=$(grep '\*' "$dev/pp_dpm_sclk" 2>/dev/null | awk '{print $2}' || true)
  mclk=$(grep '\*' "$dev/pp_dpm_mclk" 2>/dev/null | awk '{print $2}' || true)
  echo "gpu[$found] $bdf device=$(cat "$dev/device") bar0=${bar0}GiB vram=${vram}GiB power_cap=${cap}W sclk=${sclk:-?} mclk=${mclk:-?}"
done
[[ $found -gt 0 ]] || { echo "no gfx906 devices found"; exit 1; }

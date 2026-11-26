#!/usr/bin/env bash
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Read-only X399 / MI50 decode-fabric audit. It never writes sysfs, PCI config, GRUB or daemons, and runs no
# tuning tool. Without root, lspci hides extended capabilities; those checks then print UNKNOWN.
set -u
say() { printf '%-8s %-14s %s\n' "$1" "$2" "$3"; }
for bdf in $(lspci -D -d 1002: 2>/dev/null | awk '/Vega 20|MI50|MI60|Display|Processing/ {print $1}'); do
  echo "=== GPU $bdf ==="
  base="/sys/bus/pci/devices/$bdf"
  cur=$(cut -d' ' -f1 "$base/current_link_speed" 2>/dev/null); max=$(cut -d' ' -f1 "$base/max_link_speed" 2>/dev/null)
  w=$(cat "$base/current_link_width" 2>/dev/null); mw=$(cat "$base/max_link_width" 2>/dev/null)
  if [[ -z "${cur:-}" ]]; then say UNKNOWN pcie_link "no sysfs link files"
  elif [[ "$cur" == "2.5" || "$cur" != "$max" || "$w" != "$mw" ]]; then say WARN pcie_link "${cur} GT/s x${w} (max ${max} GT/s x${mw})"
  else say PASS pcie_link "${cur} GT/s x${w}"; fi
  v=$(lspci -s "$bdf" -vvv 2>/dev/null)
  devctl=$(grep -A2 -E '^\s+DevCtl:' <<<"$v" | tr '\n' ' ')
  if [[ -z "$devctl" ]]; then say UNKNOWN relaxed_order "DevCtl not visible (run as root for full lspci)"; say UNKNOWN max_payload "DevCtl not visible"
  else
    if grep -q 'RlxdOrd+' <<<"$devctl"; then say PASS relaxed_order RlxdOrd+; else say WARN relaxed_order RlxdOrd-; fi
    mps=$(grep -oE 'MaxPayload [0-9]+' <<<"$devctl" | head -1 | awk '{print $2}')
    if [[ -n "$mps" && "$mps" -lt 256 ]]; then say WARN max_payload "${mps} B"; else say PASS max_payload "${mps:-?} B"; fi
  fi
  # ACS on every upstream bridge
  dev=$(readlink -f "$base"); acs_seen=0
  while dev=$(dirname "$dev"); [[ "$dev" == /sys/devices/pci* ]]; do
    ub=$(basename "$dev"); [[ "$ub" =~ ^[0-9a-f]{4}: ]] || continue
    a=$(lspci -s "$ub" -vvv 2>/dev/null | grep -E 'ACSCtl:')
    [[ -z "$a" ]] && continue; acs_seen=1
    if grep -q 'ReqRedir+' <<<"$a"; then say WARN acs "$ub ReqRedir+"; else say PASS acs "$ub ReqRedir-"; fi
  done
  [[ $acs_seen -eq 0 ]] && say UNKNOWN acs "ACSCtl not visible (root needed) or no ACS-capable bridge"
  lvl=$(cat "$base/power_dpm_force_performance_level" 2>/dev/null)
  act=$(grep -h '\*' "$base/pp_dpm_sclk" "$base/pp_dpm_mclk" 2>/dev/null | grep -oE '[0-9]+Mhz' | tr -d 'Mhz' | tr '\n' ' ')
  low=0; for c in $act; do [[ "$c" -le 859 ]] && low=1; done
  if [[ "$lvl" == high && $low -eq 1 ]]; then say WARN dpm_clock "level=high but active clocks: $act MHz"; else say PASS dpm_clock "level=${lvl:-?} active: ${act:-?} MHz"; fi
done
echo "=== HOST ==="
nodes=$(cat /sys/devices/system/node/possible 2>/dev/null)
l3=$(cat /sys/devices/system/cpu/cpu*/cache/index3/shared_cpu_list 2>/dev/null | sort -u | wc -l)
if [[ "$nodes" == "0" && "$l3" -gt 1 ]]; then say WARN numa "1 NUMA node exposed, $l3 L3 domains (memory interleave hides die locality)"
else say PASS numa "nodes=${nodes:-?} l3_domains=$l3"; fi

#!/usr/bin/env bash
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Read-only KFD P2P topology + RAS ECC report for gfx906 nodes. Exit 2 if any uncorrectable error is reported.
set -euo pipefail
nodes=/sys/class/kfd/kfd/topology/nodes
ue_total=0
if [[ -d $nodes ]]; then
  for n in "$nodes"/*; do
    gfx=$(awk '$1=="gfx_target_version"{print $2}' "$n/properties" 2>/dev/null || echo 0)
    [[ "$gfx" == "90006" ]] || continue
    links=$(ls "$n/io_links" 2>/dev/null | wc -l)
    p2p=$(ls "$n/p2p_links" 2>/dev/null | wc -l || echo 0)
    echo "kfd node $(basename "$n"): gfx906 io_links=$links p2p_links=$p2p"
  done
else
  echo "KFD topology not present (amdgpu not loaded?)"
fi
for dev in /sys/bus/pci/devices/*; do
  [[ -d "$dev/ras" ]] || continue
  [[ "$(cat "$dev/vendor" 2>/dev/null)" == "0x1002" ]] || continue
  for f in "$dev"/ras/*_err_count; do
    [[ -r $f ]] || continue
    ue=$(awk '/^ue:/{print $2}' "$f"); ce=$(awk '/^ce:/{print $2}' "$f")
    echo "$(basename "$dev") $(basename "$f" _err_count): ue=${ue:-0} ce=${ce:-0}"
    ue_total=$((ue_total + ${ue:-0}))
  done
done
[[ $ue_total -eq 0 ]] || { echo "UNCORRECTABLE ECC ERRORS: $ue_total"; exit 2; }

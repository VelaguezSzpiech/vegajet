#!/usr/bin/env bash
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Read-only boot sanity for gfx906 hosts: kernel release and amdgpu kernel command line.
#   ROOT=<fixture root> ./check_boot_flags.sh     (default ROOT=/)
# Exit 0 = no FAIL (WARNs allowed), 1 = at least one FAIL. Reads two text files; writes nothing.
set -euo pipefail
ROOT="${ROOT:-/}"; ROOT="${ROOT%/}"
fail=0; warn=0
say() { printf '[%s] %s\n' "$1" "$2"; [[ $1 == FAIL ]] && fail=$((fail + 1)); [[ $1 == WARN ]] && warn=$((warn + 1)); return 0; }

rel=$(head -n1 "$ROOT/proc/sys/kernel/osrelease" 2>/dev/null || true)
if [[ $rel =~ ^([0-9]+)\.([0-9]+) ]]; then
  if (( BASH_REMATCH[1] >= 7 )); then say PASS "kernel $rel (>= 7.0 KFD baseline)"; else say FAIL "kernel $rel is older than 7.0"; fi
else
  say WARN "kernel release unreadable"
fi

tok=()
read -r -a tok < <(head -n1 "$ROOT/proc/cmdline" 2>/dev/null || true) || true
has() { local t; for t in "${tok[@]}"; do [[ $t == "$1" ]] && return 0; done; return 1; }
bad=""
for t in "${tok[@]}"; do
  [[ $t == amdgpu.vm_block_size=* || $t == amdgpu.vm_size=* ]] && bad+=" $t"
done
if [[ -n $bad ]]; then say FAIL "unsafe VM flag(s):$bad (keep only amdgpu.vm_fragment_size=9)"; else say PASS "no unsafe VM flags"; fi
for t in "${tok[@]}"; do
  case $t in
    amdgpu.noretry|amdgpu.noretry=0) ;;                       # bare/0: default or explicitly disabled
    amdgpu.noretry=*) say WARN "$t set; recommend dropping it" ;;
  esac
done
if has amdgpu.vm_fragment_size=9; then say PASS "amdgpu.vm_fragment_size=9"; else say WARN "amdgpu.vm_fragment_size=9 missing"; fi
if has iommu=pt; then say PASS "iommu=pt"; else say WARN "iommu=pt missing"; fi
mask=""
for t in "${tok[@]}"; do [[ $t == amdgpu.ppfeaturemask=* ]] && mask=${t#*=}; done
if [[ -n $mask ]]; then
  if (( (mask & 0x4) == 0 )); then say PASS "ppfeaturemask=$mask (PCIe DPM bit 2 cleared)"; else say WARN "ppfeaturemask=$mask keeps PCIe DPM (bit 2) enabled"; fi
fi
echo "summary: $fail fail, $warn warn"
(( fail == 0 ))

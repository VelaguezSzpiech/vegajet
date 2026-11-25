#!/usr/bin/env bash
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# release/contrib/hardware scripts against a synthetic fixture tree ONLY. ROOT always points into a fresh temp
# directory; the real /sys, /proc and /dev are never read or written by this test.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
C="$HERE/release/contrib/hardware"; [[ -d $C ]] || C="$HERE/contrib/hardware"   # sidecar / repo layout
FX="$(mktemp -d "${TMPDIR:-/tmp}/vj-contrib-XXXXXX")"
trap 'rm -rf "$FX"' EXIT
[[ -n $FX && $FX != / ]] || { echo "bad fixture dir"; exit 1; }
export ROOT="$FX"
pass=0
ok() { pass=$((pass + 1)); }
die() { echo "FAIL: $*"; exit 1; }
expect() {  # expect <exit code> <grep pattern> <cmd...>
  local want=$1 pat=$2; shift 2
  set +e; out=$("$@" 2>&1); rc=$?; set -e
  [[ $rc == "$want" ]] || die "$* -> rc $rc, want $want"$'\n'"$out"
  grep -q -- "$pat" <<<"$out" || die "$* -> missing '$pat'"$'\n'"$out"
  ok
}

gpu() {  # gpu <i>
  local d="$FX/sys/devices/host0/rp$1/up$1/dn$1/gpu$1"
  mkdir -p "$d/hwmon/hwmon$1" "$FX/sys/bus/pci/devices"
  echo 0x1002 > "$d/vendor"; echo 0x66a1 > "$d/device"
  printf '0: 350Mhz\n1: 800Mhz *\n' > "$d/pp_dpm_mclk"
  echo 150000000 > "$d/hwmon/hwmon$1/power1_cap"
  echo 90000000 > "$d/hwmon/hwmon$1/power1_cap_min"
  echo 225000000 > "$d/hwmon/hwmon$1/power1_cap_max"
  ln -s "../../../devices/host0/rp$1/up$1/dn$1/gpu$1" "$FX/sys/bus/pci/devices/gpu$1"
}
mkdir -p "$FX/proc/sys/kernel" "$FX/proc/1/fd" "$FX/sys/fs/cgroup/system.slice" "$FX/sys/devices/host0/nic0"
echo 0x8086 > "$FX/sys/devices/host0/nic0/vendor"; echo 0x1533 > "$FX/sys/devices/host0/nic0/device"
gpu 0; gpu 1
ln -s ../../../devices/host0/nic0 "$FX/sys/bus/pci/devices/nic0"
ln -s /dev/null "$FX/proc/1/fd/0"

# ---- check_boot_flags.sh
echo "7.0.0-43-generic" > "$FX/proc/sys/kernel/osrelease"
echo "ro amdgpu.vm_fragment_size=9 iommu=pt amdgpu.ppfeaturemask=0xfffffffb" > "$FX/proc/cmdline"
expect 0 "0 fail, 0 warn" "$C/check_boot_flags.sh"
expect 0 "PCIe DPM bit 2 cleared" "$C/check_boot_flags.sh"
echo "ro amdgpu.vm_fragment_size=90 amdgpu.noretry=1 amdgpu.ppfeaturemask=0xffffffff" > "$FX/proc/cmdline"
expect 0 "0 fail, 4 warn" "$C/check_boot_flags.sh"
echo "ro amdgpu.vm_block_size=11 amdgpu.vm_size=2048" > "$FX/proc/cmdline"
expect 1 "vm_block_size=11 amdgpu.vm_size=2048" "$C/check_boot_flags.sh"
echo "6.8.0" > "$FX/proc/sys/kernel/osrelease"; echo "ro amdgpu.vm_fragment_size=9 iommu=pt" > "$FX/proc/cmdline"
expect 1 "older than 7.0" "$C/check_boot_flags.sh"
: > "$FX/proc/cmdline"; rm "$FX/proc/sys/kernel/osrelease"
expect 0 "kernel release unreadable" "$C/check_boot_flags.sh"

# ---- gsf12_preflight.sh
mkdir -p "$FX/sys/fs/cgroup/system.slice/mi50-guard.service" "$FX/sys/fs/cgroup/system.slice/vifted.service"
expect 1 "mi50-guard.service running" "$C/gsf12_preflight.sh"
expect 0 "-> SAFE" "$C/gsf12_preflight.sh" --post
rmdir "$FX/sys/fs/cgroup/system.slice/mi50-guard.service" "$FX/sys/fs/cgroup/system.slice/vifted.service"
expect 0 "-> SAFE" "$C/gsf12_preflight.sh"
expect 0 "restart mi50-guard.service vifted.service" "$C/gsf12_preflight.sh"
expect 1 "vifted.service not running" "$C/gsf12_preflight.sh" --post
mkdir -p "$FX/sys/fs/cgroup/system.slice/coolercontrold.service"
expect 1 "competing amdgpu sensor reader" "$C/gsf12_preflight.sh"
rmdir "$FX/sys/fs/cgroup/system.slice/coolercontrold.service"
mkdir -p "$FX/proc/777/fd"; ln -s /dev/kfd "$FX/proc/777/fd/5"
expect 1 "held by pid 777" "$C/gsf12_preflight.sh"
rm -r "$FX/proc/777"
printf '0: 400Mhz\n1: 800Mhz *\n' > "$FX/sys/devices/host0/rp1/up1/dn1/gpu1/pp_dpm_mclk"
expect 1 "gpu1 mclk level 0 = 400 MHz" "$C/gsf12_preflight.sh"
printf '0: 350Mhz\n1: 800Mhz *\n' > "$FX/sys/devices/host0/rp1/up1/dn1/gpu1/pp_dpm_mclk"
GUARD_UNITS="other-guard.service" expect 0 "other-guard.service stopped" "$C/gsf12_preflight.sh"

# ---- set_power_cap.sh (dry run never writes; --apply writes the fixture and reads back)
cap0="$FX/sys/devices/host0/rp0/up0/dn0/gpu0/hwmon/hwmon0/power1_cap"
expect 0 "150 W -> 180 W (dry run" "$C/set_power_cap.sh" 180
[[ $(cat "$cap0") == 150000000 ]] || die "dry run wrote power1_cap"; ok
expect 1 "exceeds the 200 W per-GPU ceiling for 2 GPU(s)" "$C/set_power_cap.sh" 210 --apply
expect 1 "below the 50 W floor" "$C/set_power_cap.sh" 40
expect 1 "outside the driver range" "$C/set_power_cap.sh" 60
expect 2 "usage" "$C/set_power_cap.sh" abc
expect 2 "usage" "$C/set_power_cap.sh" 180 --force
expect 0 "gpu1: power1_cap 150 W -> 180 W" "$C/set_power_cap.sh" 180 --apply
[[ $(cat "$cap0") == 180000000 ]] || die "apply did not write power1_cap"; ok
gpu 2  # three GPUs: the ceiling drops to 150 W
expect 1 "exceeds the 150 W per-GPU ceiling for 3 GPU(s)" "$C/set_power_cap.sh" 180 --apply
expect 0 "gpu2: power1_cap 150 W -> 140 W" "$C/set_power_cap.sh" 140 --apply
rm -r "$FX/sys/bus/pci/devices/"gpu*
expect 1 "no gfx906 hwmon power1_cap found" "$C/set_power_cap.sh" 140

echo "test_contrib_hardware: $pass checks on fixtures OK"
echo PASS

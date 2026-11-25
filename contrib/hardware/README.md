# contrib/hardware: host setup helpers for gfx906 rigs

Small shell scripts for preparing and checking a multi-GPU gfx906 host. Each script reads a filesystem root from
`ROOT` (default `/`), so it can run against a synthetic `/proc` + `/sys` tree. That is how
`tests/test_contrib_hardware.sh` tests them; it never points them at the real system. They contain no device addresses
and no machine-specific paths. GPUs are found by vendor `0x1002` and device ID `0x66a0` / `0x66a1` / `0x66a3`.

| Script | Writes? | What it does |
| :--- | :--- | :--- |
| `check_boot_flags.sh` | never | Kernel ≥ 7.0. An explicit `amdgpu.vm_block_size=` / `amdgpu.vm_size=` FAILs (never set VM geometry). A missing `vm_fragment_size=9`, `amdgpu.noretry=1`, a missing `iommu=pt`, or a `ppfeaturemask` that keeps PCIe DPM (bit 2) each WARN. |
| `gsf12_preflight.sh` | never | G-SF-12 gate before a power-play table write. Checks that no process holds `/dev/kfd`, that the guard, fan and sensor-reader daemons are stopped, and that `pp_dpm_mclk` level 0 = 350 MHz. `--post` checks that the daemons are running again afterwards. |
| `set_power_cap.sh <W> [--apply]` | only with `--apply` | Sets `power1_cap` on every gfx906. Ceilings: ≤ 200 W/GPU for 1–2 GPUs, ≤ 150 W/GPU for 3 or more. Refuses values outside the driver's `power1_cap_min..max` and below 50 W. Dry run by default. Reads the value back after writing. |

## Notes

- **Read-only means read-only.**
  - The checks only read small text files and `readlink` `/proc/<pid>/fd` entries.
  - They never open `/dev/kfd`, `/dev/dri/*` or `pp_table`.
  - They never start, stop or query services. A unit counts as running when
    `sys/fs/cgroup/system.slice/<unit>` exists, which holds for long-running daemons under cgroup v2.
- **G-SF-12 itself is a procedure; the script only checks it.** The procedure:
  1. Run `gsf12_preflight.sh` and stop the units it names.
  2. Write the table, keeping `FreqTableUclk/0 = 350` and `Tvr_memLimit = 115`.
  3. Restart the units right away.
  4. Run `gsf12_preflight.sh --post`, then a soak.
- **Unit names** default to `mi50-guard.service`, `vifted.service` and `coolercontrold.service`. Override them with
  `GUARD_UNITS`, `FAN_UNITS` and `READER_UNITS`.
- **Power caps are per GPU.** The N-aware ceilings come from a shared-PSU budget. The alpha GPU executor is
  two-card only; the 3+ card ceilings are for the roadmap. The engine's telemetry gate separately keeps the 10 s
  rolling `power1_input` average ≤ 205 W (N ≤ 2) or ≤ 155 W (N ≥ 3), and the instantaneous draw < 270 W.
- **The same rules exist in C++** in the engine's `core/doctor_rules` (root-relative, fixture-tested). There they
  are combined with PCIe host-link, HBM2 TMem, 6-block RAS ECC, BAR0, VBIOS-parity and KFD-topology checks.

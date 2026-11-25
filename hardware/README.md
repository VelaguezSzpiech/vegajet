# VegaJet `hardware/`

**Alpha scope.** These are reference-rig notes, not alpha installation requirements. The alpha GPU executor
supports two MI50 32 GB cards with 32 GiB BAR0 and PCIe P2P; VBIOS and boot-parameter changes below are optional
operator procedures, not first-run steps. Do not flash a VBIOS on a host you have not qualified.

Generic, read-only host checks for gfx906 (Vega 20: MI50 / MI60 / Radeon VII / Radeon Pro VII) boxes.
Nothing here writes to sysfs, changes clocks/power, or opens a device node.

| Script | Purpose |
| :--- | :--- |
| `check_gfx906.sh` | Lists Vega 20 devices discovered from `/sys/bus/pci/devices`, their BAR0 size, VRAM, power cap and current SCLK/MCLK levels. |
| `check_p2p_ecc.sh` | Reports KFD topology P2P (`io_links`) per gfx906 node and per-block RAS ECC counters (`ras/*_err_count`). Exits non-zero on uncorrectable errors. |

Both scripts only print what the kernel exposes; interpret results with `vegajet doctor`, which applies the same
checks with pass/warn/fail thresholds.

## Firmware

No firmware images, flashing procedure or board-specific recommendations are supplied. Check host compatibility using the read-only reports above.

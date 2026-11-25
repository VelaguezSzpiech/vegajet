# VegaJet operations guide

This guide is for running VegaJet on a multi-GPU gfx906 host (MI50 / MI60 / Radeon VII class, 2 or 3 cards). It
covers kernel boot flags, the G-SF-12 power-play workflow, keeping the desktop off the compute GPUs, and CPU planning fixtures. It contains no machine-specific values. Wherever you see `<cardN>`, `<model-file>`,
`<display-vendor:device>` and similar placeholders, substitute values from your own system.

**Who may touch the GPUs.** Only one operator process runs GPU work at a time. That covers GPU binaries, `/dev/kfd`
tests, `upp` / `pp_table` writes and guard-service changes, and two GPU runs must never overlap. If the guard has
tripped or a reboot is required, stop and clear the cause first. Never clear the latch or work around the guard.

**Running without a GPU.** The CPU-only test suite, the `contrib/hardware` scripts (with `ROOT=` fixtures) and
`vegajet serve --mock` run on the CPU and never open `/dev/kfd`.

---

## 1. Kernel and boot flags (Linux 7.0)

VegaJet drives the GPUs through `/dev/kfd` directly. Its memory layout assumes 2 MiB GPU page-table fragments:
- The KFD arena allocates 512 MiB buffer objects.
- Queue buffers and executable code objects are aligned to 2 MiB in the GPU virtual address space.

| Flag | Setting | Why |
| :--- | :--- | :--- |
| `amdgpu.vm_fragment_size=9` | **required on the reference rig** | 2^9 × 4 KiB = 2 MiB PTE fragments. These are the huge-PTE streaming throughput that the arena and the 2 MiB VA isolation rely on. The engine does not read or enforce this parameter; correctness without it is unverified. |
| `iommu=pt` | recommended | IOMMU passthrough for P2P and host DMA. |
| `pci=realloc=on` | recommended | Lets the kernel reassign BARs so large (32 GiB) BAR0 windows fit. |
| `amdgpu.ppfeaturemask=0xfffffffb` | recommended | Clears bit 2 (`PP_PCIE_DPM_MASK`), so PCIe link DPM cannot park a link at Gen1. |
| `amdgpu.runpm=0` | recommended | No runtime power-down of compute cards. |
| `amdgpu.vm_block_size=*`, `amdgpu.vm_size=*` | **never** | Unsafe VM geometry for this engine; `doctor` fails on any explicit override. |
| `amdgpu.noretry=1` | **never** | Changes fault handling that the runtime depends on. |

Setting the flags on a GRUB system:

```sh
sudoedit /etc/default/grub        # append the flags to GRUB_CMDLINE_LINUX_DEFAULT
sudo update-grub                  # or: grub2-mkconfig -o /boot/grub2/grub.cfg
# reboot, then verify:
cat /proc/cmdline
contrib/hardware/check_boot_flags.sh        # read-only: kernel >= 7.0, required / forbidden flags
build-cpu/vegajet doctor                    # same rules plus PCIe, BAR0, VBIOS parity, ECC, KFD topology
```

`check_boot_flags.sh` treats any explicit `vm_block_size=` and `vm_size=` as FAIL. It gives WARN for a missing
`vm_fragment_size=9`, for `noretry=1`, for a missing `iommu=pt`, and for a `ppfeaturemask` that keeps PCIe DPM.
`=90` does not count as `=9`.

**Link speed.** An idle card may downclock its link, so a single idle reading proves nothing. Check the link under
load with `vegajet doctor --live-link`:
- A host link below 8 GT/s under load is a WARN.
- A link stuck at 2.5 GT/s under load points to the VBIOS or the PCIe-DPM feature bits. See `hardware/README.md` for
  the VBIOS matrix.
- On a card installed in an x8 slot, x8 width is expected, not a failure.

---

## 2. Hardware safety

The alpha does not distribute firmware images, board-specific power tables, or instructions for disabling host guards. Keep independent thermal protection active and provide forced airflow. Do not benchmark after an uncorrectable ECC event or failed guard check.

## 3. Desktop Vulkan isolation

The compute GPUs must have no other clients:
- **Compositor:** keep the desktop compositor off them.
- **Desktop Vulkan / OpenGL applications** (browsers, video players, launchers): keep them off too.
- **Sensor and overlay tools:** keep them off too.

Any such client holds `/dev/dri/renderD*` or `/dev/kfd`, competes for VRAM and queues, and counts as a sensor reader
under G-SF-12.

1. **Use a separate display GPU.** Use an iGPU or a small dGPU, and make sure the firmware picks it as the boot VGA
   device. Some VGA/GOP builds of the compute card's VBIOS can claim `boot_vga`; a compute-class VBIOS cannot. Check
   with `cat /sys/bus/pci/devices/*/boot_vga`: exactly one device prints `1`, and it must be the display GPU.
2. **Pin desktop Vulkan to the display GPU** with Mesa's device-select layer, for the whole graphical session (for
   example in `~/.config/environment.d/90-vulkan-display.conf`):

   ```ini
   MESA_VK_DEVICE_SELECT=<display-vendor:device>
   MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE=1
   DRI_PRIME=0
   ```

   `vulkaninfo --summary` (run inside the session) lists the device order, and `MESA_VK_DEVICE_SELECT=list vkcube`
   prints the IDs Mesa sees.
3. **When the display GPU uses a different Vulkan driver** than the compute cards, you can also restrict the session
   to that driver's ICD manifest with `VK_DRIVER_FILES` (older loaders: `VK_ICD_FILENAMES`). Do not do this when both
   use RADV, because it would hide nothing.
4. **Verify before every GPU run:**

   ```sh
   build-cpu/vegajet doctor            # KFD holders, sensor readers, topology
   sudo fuser -v /dev/kfd /dev/dri/renderD<compute-minor>   # must list nothing
   ```

   If anything still holds a compute render node, close it or move it to the display GPU. Do not start a benchmark
   next to it.

---

## 4. Roadmap: TP=3 and the 3× swarm (not in this alpha)

This section is roadmap procedure. The alpha GPU runner is fixed at two ranks; do not treat any step here as an
alpha acceptance test.

There are two ways to use three cards.

| Mode | What runs | Status in this tree |
| :--- | :--- | :--- |
| **TP=3** | One engine; weights sharded by bandwidth; an all-reduce per layer. | Planner and CPU checks available: `core/tp3_plan`, `core/sharder` with `row_unit`, the ISA scanner, `doctor`. GPU execution follows the engine's TP path. |
| **3× swarm** | Three single-GPU (N=1) servers behind `tools/swarm_router.py`. | Router and stress tests run today against `vj_mock_serve` workers and `vegajet serve --mock`. Per-process GPU selection (`--gpu`, `VJ_GPU_DEVICES`) and `vegajet serve <model-file>` are in place; live GPU workers still need a **streaming GPU engine** registered for the model's architecture (until then `serve <model-file>` exits 3). |

### 4.1 Preconditions (both modes)

1. **CPU gates are green:**
   - `ctest --test-dir build-cpu --output-on-failure`;
   - the sidecar suite (`run_tests.sh`) at `-O3` and under ASan/UBSan;
   - `tools/privacy_audit.sh`.
2. **The host passes the checks in sections 1–3:** `doctor` has no FAIL, `--live-link` was checked under load, and
   the compute GPUs have no desktop clients.
3. **All three cards run the identical VBIOS** (`doctor` enforces VBIOS parity) and have no ECC CE delta trend.
4. **Power caps are 150 W per GPU** (`set_power_cap.sh 150 --apply`). The PSU budget covers 3 × 150 W plus the host
   at the rolling 155 W gate.
5. **Each card passes on its own (N=1)** before you combine them:
   - parity `32/32` against the Tier-1 reference and `129/129 (first_diff = -1)` on the lossless speculative modes;
   - the thermal and ECC gates;
   - a scorecard saved per card.

### 4.2 TP=3

1. **Plan on the CPU.** Use `core/tp3_plan` (`plan_tp3()`) with the measured per-GPU bandwidth shares. The reference
   plan is 31.7 / 34.1 / 34.1 %; the card in the narrower slot gets the smallest share.
   - **Balance:** at `expert_ff = 512` with the default 32-row unit, the experts split 5/6/5 and one GPU streams 37.5 %
     of expert bytes. Use `row_unit = 16` (`split_rows(..., 16)`) or whole-expert sharding; both are within 1 % of
     target. The planner reports the 32-row case as a failed balance check.
   - **VRAM and power:** every GPU's weights + KV + workspace fit in its VRAM, and every cap is ≤ 150 W.
   - **All-reduce:** direct (2 phases, summed in a fixed order, so the result is bit-identical on every rank) is the
     planner's choice at decode sizes; it chooses ring only when the per-link bandwidth makes ring cheaper. FP16 is
     the all-reduce wire format. The INT8 wire stays opt-in.
2. **Bring up in stages.** Go TP=2 on the two widest links, then TP=3. At each stage:
   - parity is exact (`32/32`, `129/129`);
   - TMem, UE / DE and power stay inside the section 2 gates;
   - no kernel faults appear.
3. **Soak** for at least 10 minutes at TP=3 under telemetry, then record the scorecard.
4. **Gate:** `tools/regression_gate.py <tp2-or-previous-scorecard> <tp3-scorecard>`.

### 4.3 3× swarm

1. **Today (CPU):** test the router and the HTTP stack.

   ```sh
   python3 -m unittest discover -s tests -p "test_swarm_router*.py"     # with VJ_MOCK_SERVE=<build>/vj_mock_serve
   python3 tools/api_stress_test.py --spawn <build>/vj_mock_serve --clients 24 --rounds 3 \
       --spawn-arg=--max-queue --spawn-arg=2 --expect-backpressure
   ```

2. **Once a streaming GPU engine is registered** (GPU selection already exists):
   - start three workers, each pinned to one GPU, each with `--max-queue 1` (one sequence per GPU) on its own
     loopback port: `vegajet serve <model-file> --gpu 0 --port <p1> --max-queue 1`, the same with `--gpu 1` and
     `--gpu 2` (or `VJ_GPU_DEVICES=<i>`). Indices are positions in `vegajet serve --gpu list`;
   - put the router in front:

     ```sh
     python3 tools/swarm_router.py --listen 127.0.0.1:8090 --worker http://127.0.0.1:<p1> \
         --worker http://127.0.0.1:<p2> --worker http://127.0.0.1:<p3>
     ```

   - keep the router on loopback or behind your own authenticating proxy.

   The router routes by conversation affinity, never sends a worker two requests, answers 503 with Retry-After when
   every worker is busy and the queue is full, fails over when a worker's health check fails, and propagates client
   disconnects.
3. **Per worker**, run the same N=1 gates (parity, telemetry) as in 4.1. Then run `api_stress_test.py --url
   http://127.0.0.1:<router-port>` against the router to check determinism (SHA-256 of identical requests),
   disconnects and back-pressure under real load.

---

## 5. Release gate

Every published number comes from a recorded run, with its commit, flags and measured HBM2 ceiling.

```sh
# regression: candidate vs baseline scorecard (JSON or text; VegaJet [BENCH]/[PARITY]/telemetry lines are understood)
python3 tools/regression_gate.py baseline.json candidate.json
#   fails on: > 2 % throughput drop on any tok/s metric (or a baseline metric missing from the candidate),
#             KLD > 0.045, inexact parity, spill > 0, TMem >= 65 C, UE > 0 or DE > 0

# readiness: no "fill from final scorecard" placeholder may remain in README.md or release/docs/*.md
python3 tools/regression_gate.py --release-check
```

Exit codes: 0 pass, 1 gate failed, 2 unusable input. Do not tag a release while either command is non-zero.
`tools/bench_diff.py` stays the descriptive before/after table. `regression_gate.py` is the pass/fail decision.

Before starting an A/B sweep, set `VEGAJET_GUARD_LATCH` to the absolute path of the supervisor’s real trip marker in its existing readable and searchable guard directory. A present marker blocks the run, regardless of contents. Never clear or substitute a marker to launch a sweep. An unconfigured or inaccessible guard refuses launch.

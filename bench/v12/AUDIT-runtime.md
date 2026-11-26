# VegaJet low-level runtime audit

## Scope, verdict, and evidence boundary

**Do not treat this runtime as release-safe yet.** This audit identifies two additional blockers and seven majors, distinct from the already reported v1.2 change-review findings. In particular, adding a third MI50 **can change which physical cards an ordinal TP=2 selection uses**. The current interface cannot pin a physical pair by persistent identity.

Source: `/data/src/vegajet`, branch `release/v1.2`, HEAD `3bc0015049af85d3b39743f9cf08326a59bceb80`. All source references below are relative to that checkout and refer to that revision. This was a read-only static audit; the only repository file created was this report. No source edits, commits, GPU runs, builds, or performance measurements were performed. The VM has no GPU. Concrete control-flow/interleaving failures are distinguished below from hardware-dependent manifestations.

Reviewed: `core/kfd_runtime.{cpp,hpp}`, `core/hsaco_{bundle,gate,isa_scan}.*`, `core/gpu_select.*`, `hardware/*`, runtime/allocation/dispatch/reload sections of `models/ornith/ornith_gpu.cpp`, `ops/kernels/vj_ornith_ar.hip`, the AR body in `vj_ornith_common.hiph`, relevant super-kernel epoch consumers, `vj_kfd_base.hip`, and CLI/server shutdown integration. Read `/data/work/vegajet-outreach/research/07-code-audit.md` and the sole matching `bench/v12/REVIEW*.md`, `REVIEW.md`, first. Their performance findings, v1.2 fault-buffer UAF, stale launch ABI sites, S2 rank-epoch mismatch, speculative capacity bugs, and cancellation/backpressure findings are not repeated as new findings here.

Severity convention: **BLOCKER** = supported operation has a concrete deadlock or dangling-pointer execution path; **MAJOR** = substantial lifetime, dispatch, loader, identity, or recovery defect; **MODERATE** = narrower API/error/long-run robustness defect. These are static findings, not observed GPU failures.

## BLOCKER findings

### RT-B1 — Consecutive slices overwrite a readiness epoch before the peer consumes it

**Locations:** `models/ornith/ornith_gpu.cpp:2443–2486`; `ops/kernels/vj_ornith_common.hiph:280–304`; `ops/kernels/vj_ornith_ar.hip:126–129`.

**Quotes:**

```cpp
const uint32_t slice_rows = (use_pf_i8_all || opt.ar_wire_int8) ? 512u : 128u;
for (uint32_t m0 = 0u; m0 < M; m0 += slice_rows) {
    const uint32_t ep = ar_epoch++;
    const uint32_t pp = ep & 1u;
    // payload uses pp; flags do not:
    flg_base + peer * 4096ULL + (slot & 63u) * 64ULL,
    flg_base + g    * 4096ULL + (slot & 63u) * 64ULL,
```

```cpp
__atomic_store_n(peer_flag, epoch, __ATOMIC_RELEASE);
while (__atomic_load_n(local_flag, __ATOMIC_ACQUIRE) != epoch) {
    __builtin_amdgcn_s_sleep(1);
}
```

**Why:** Multiple slices of one collective reuse the same flag address immediately, although payloads alternate. There is no receive/consumption acknowledgement. A legal schedule is:

1. Both ranks publish readiness for slice epoch E.
2. A observes B's E and completes its slice. B is descheduled before its own readiness load.
3. A starts its next slice and overwrites B's incoming flag with E+1.
4. B resumes and waits for equality with E forever; A waits for B's E+1 forever.

The per-queue dispatch barrier only orders work on the same GPU; it does not make B consume A's flag before A advances. Two payload slots do not fix a single readiness flag. FP16 batches above 128 rows take this path; `max_batch` explicitly permits larger `prefill_chunk` values (`ornith_gpu.cpp:1239`). This is independent of the S2 per-rank epoch bug in `REVIEW.md`.

**Concrete fix:** Use separately versioned readiness slots for outstanding slices, with an explicit consumed/credit acknowledgement before reusing either flag or payload slot; define the maximum producer lead. A simple correctness-first alternative is a host completion boundary on both ranks between slices. Add an asymmetric scheduling test that delays B between publication and observation, for two and three consecutive slices. All waits need a propagated timeout, not merely a spin bound followed by continued computation.

**Verification:** An abstract state-transition check reproduced `B waits for epoch 1; flag is 2; A waits for B epoch 2`. This is a protocol counterexample, not a GPU execution result.

### RT-B2 — Failed reload-all leaves cached kernel pointers dangling, then accepts another run

**Locations:** `core/kfd_runtime.cpp:932–950,954–969,1314–1317`; `models/ornith/ornith_gpu.cpp:908–915,5936–5958`.

**Quotes:**

```cpp
for (const auto& old_kn : existing->kernel_names) kernels_[g].erase(old_kn);
if (g < existing->code_buffers.size()) free_buffer(existing->code_buffers[g]);
...
for (const auto& nm : names) {
    if (!reload_hsaco(nm, err)) return false;
}
```

```cpp
const bool ok_rel = ... rt.reload_all_hsacos(&rerr) ...;
const bool ok_res = ok_rel && resolve_kernels(&rerr);
...
return;  // interactive command loop remains available
```

**Why:** `get_kernel()` returns pointers to values inside the kernel maps, and the Ornith `K[]` tables retain them. Reloading module A destroys those map objects and replaces its code BOs. If module B then fails to load (e.g. a missing/truncated replacement), reload-all returns false after A has already committed. The short-circuit skips `resolve_kernels()`. A subsequent `run` or `parity` uses stale pointers to A's erased `LoadedKernel` objects; allocator reuse does not make those pointers valid. The first module is normally `vj_kfd_base`, so the reset/copy helpers themselves can be affected. A single-module replacement that omits required symbols is also committed before the caller's resolver reports failure, leaving a non-runnable table while commands remain enabled.

**Concrete fix:** Stage and validate the complete requested module set, required symbol set, and compatible launch ABI before committing any replacement. Swap one immutable generation, retain old generation ownership until all users release it, and refresh cached handles only as part of that commit. On any failed post-commit rebind, poison the runner and refuse further execution; do not continue the interactive loop as a usable engine. Add a CPU ownership test for valid first module + invalid second module + subsequent run, preferably under ASan.

## MAJOR findings

### RT-M1 — Free unmaps every GPU, not the GPUs actually mapped; later-rank BOs leak

**Locations:** `core/kfd_runtime.cpp:627–652`; owner-only mapping sites `:585–586,691,730,1090`; chunk allocation `:400–416`.

**Quote:**

```cpp
uargs.device_ids_array_ptr = reinterpret_cast<uint64_t>(all_gpu_ids_.data());
uargs.n_devices = static_cast<uint32_t>(all_gpu_ids_.size());
(void) kfd_ioctl(kfd_fd_, AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, &uargs);
...
(void) kfd_ioctl(kfd_fd_, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &fargs);
...
buf = KfdBuffer{};
```

**Why:** Queue ring/control/kernarg buffers, EOP/CWSR, code images, and arena chunks can be mapped only to their owner. For a rank-1-only BO, the unmap list begins with rank 0, where the BO is not mapped. KFD stops on that error before unmapping rank 1. FREE then fails because a mapping remains, but both errors are discarded and the handle is erased. This is an ordinary multi-GPU free/hot-reload path, not just hypothetical partial initialization. For a rank-0-only BO the first unmap may succeed, but the later invalid target still aborts the ioctl's completion/synchronization path.

The same lost ownership affects partially successful MAP calls: `map_to_gpus()` returns only a bool and discards the successfully mapped prefix. Cleanup cannot reliably reconstruct what needs unmapping.

**Kernel ABI corroboration:** `/usr/include/linux/kfd_ioctl.h:441–471` specifies `n_success` as a resumable prefix count. Upstream Linux v6.8 and the current upstream source were read: `kfd_ioctl_unmap_memory_from_gpu()` stops on the first failed target; `amdgpu_amdkfd_gpuvm_unmap_memory_from_gpu()` returns `-EINVAL` if no mapped VM exists; `amdgpu_amdkfd_gpuvm_free_memory_of_gpu()` returns `-EBUSY` while `mapped_to_gpu_memory > 0`. Reproducible reference sources:

- <https://github.com/torvalds/linux/blob/v6.8/drivers/gpu/drm/amd/amdkfd/kfd_chardev.c>
- <https://github.com/torvalds/linux/blob/v6.8/drivers/gpu/drm/amd/amdgpu/amdgpu_amdkfd_gpuvm.c>

The owner's exact kernel was not available; this is ABI/source corroboration, not a measurement on that machine.

**Concrete fix:** Give each owning buffer an explicit successful mapping set, including partial MAP progress. Unmap exactly that set, preserve `n_success` across retries, check both unmap and free results, and retain ownership/diagnostics on failure. Separate borrowed mailbox views from owning buffers. Fault-inject every target position and repeatedly reload code on ranks 0 and 1 while checking BO accounting.

### RT-M2 — One reusable completion signal cannot represent overlapping submissions or unsubmitted work

**Locations:** `core/kfd_runtime.cpp:1495–1532,1535–1558`; forced publication `:195–203`; reload drain `:897–904`.

**Quotes:**

```cpp
__atomic_store_n(&q.completion_sig->value, 1LL, __ATOMIC_RELEASE);
...
raw[7] = q.rw_ptr_buf.va + 0x200;
```

```cpp
while (__atomic_load_n(&q.completion_sig->value, __ATOMIC_ACQUIRE) > 0) { ... }
```

**Why:** Every submission stores 1 into the same signal. Two `dispatch_async()`/`submit_batch()` calls on one queue before waiting can both reference it. If the first barrier completes after the second store, it decrements 1 to 0; `wait_idle()` returns while the second batch is still outstanding. The second completion can subsequently make the value negative, also treated as idle. No API check rejects a second outstanding submission.

Conversely `ring_doorbell()` and ring-full forced publication do not arm a completion signal at all. `wait_idle()` on such work can see the previous zero immediately. In particular, “drain all active queues” in reload is not established merely by calling this function when pending/rung work lacks a current completion barrier. Most current call sites submit/wait in pairs, reducing exposure, but the runtime API itself does not enforce that discipline.

**Concrete fix:** Either enforce exactly one submitted batch per queue, with explicit pending/submitted/failed states and a required drain before reuse, or allocate generation-specific completion signals/fence records. `wait_idle()` must fence all previously enqueued work, not merely inspect the last signal. Treat unexpected negative values as errors. Test two submissions before the first completion, doorbell-only work, empty waits, forced flushes, and reload after enqueue.

**Verification:** An abstract check of two stores of 1 followed by the first decrement produced signal 0 while batch 2 remained outstanding. No GPU completion behavior was measured.

### RT-M3 — Kernarg reuse is governed by a wrapping counter, not dispatch completion

**Locations:** `core/kfd_runtime.cpp:733–746,1436–1446,1468`; `:195–209`.

**Quote:**

```cpp
const uint32_t kslot = (q.kernarg_slot++) & 2047u;
...
std::memset(ka_dst_host, 0, total_ka);
if (kernargs && kernarg_bytes > 0) std::memcpy(ka_dst_host, kernargs, kernarg_bytes);
...
if (!aql_wait_space(q, 2u, 5000000ULL)) return false;
```

**Why:** There are 2048 argument slots. The next slot is cleared and overwritten before any capacity wait, and its lifetime is not associated with a completed dispatch. AQL packet consumption/read-index progress is not proof that a running kernel has finished reading its kernarg storage. With overlapping (`barrier=false`) dispatches, a long-lived early kernel can still use a slot when subsequent launches wrap the counter. The code permits such launches. Failure to get packet space also consumes/mutates a kernarg slot before returning false. This is a lifetime issue, distinct from the earlier audit's ABI-size/implicit-layout findings.

**Concrete fix:** Reserve packet and argument capacity before writing, retire kernarg storage only against a real completion fence, and retain a batch's arguments until that batch completes. Start with a separate per-batch arena and drain on exhaustion if necessary. Add a test where an early non-barrier kernel delays its argument load while more than one argument-ring capacity of work is submitted, plus a capacity-timeout test that confirms no live argument storage changed.

### RT-M4 — The live ELF loader trusts unchecked offsets before its validation gates

**Locations:** `core/kfd_runtime.cpp:1038–1066,1092–1103,1243–1271`; contrast `core/hsaco_bundle.cpp:805–895` and `core/hsaco_gate.cpp:29–60`.

**Quotes:**

```cpp
if (fsize < sizeof(Elf64_Ehdr) || std::memcmp(elf.data(), ELFMAG, SELFMAG) != 0) ...
const auto* phdrs = reinterpret_cast<const Elf64_Phdr*>(elf.data() + ehdr->e_phoff);
...
std::memcpy(... + (phdrs[i].p_vaddr - min_vaddr),
            elf.data() + phdrs[i].p_offset, phdrs[i].p_filesz);
```

```cpp
const char* strtab = ... + shdrs[shdrs[i].sh_link].sh_offset;
std::string sname = strtab + syms[s].st_name;
...
const auto* kd = ... + (vaddr - min_vaddr);
```

**Why:** Only ELF magic/minimum header length is checked before dereferencing the program-header table. Program/section table bounds and entry sizes, integer overflow, `p_filesz <= p_memsz`, segment file bounds, string-table links/termination, and descriptor bounds are not validated. A truncated or malformed hot-reload artifact can cause host OOB reads/writes before the spill/VGPR gate runs. The live path also does not establish ELF64 little-endian EM_AMDGPU/gfx906/code-object ABI compatibility, nor implement/reject unsupported relocation requirements. Stronger checks in the standalone bundle inspector are not called by this runtime loader; its presence is not protection here.

**Concrete fix:** Use one checked parser shared with the bundle verifier, extend it with complete PT_LOAD/descriptor/entry-point/relocation checks, and validate the whole file before allocating or copying. Use subtraction-based bounds checks and capped image/file sizes; reject unsupported ELF class, machine, target, code-object versions, and relocation types. Check the actual read length. Fuzz the parser separately on CPU and test truncated tables, overflowed segments, nonterminated symbols, and out-of-image `.kd` entries.

### RT-M5 — A third MI50 can silently change the physical TP=2 pair and expand resource ownership

**Locations:** `core/gpu_select.hpp:4–12`; `core/gpu_select.cpp:88–100,128–142`; `core/kfd_runtime.cpp:155–187,291–340,373–379,493–540`; `models/ornith/ornith_gpu.cpp:865–872`; `cli/serve_cmd.cpp:196–208`.

**Quotes:**

```cpp
return a.location_id < b.location_id; // runtime ordering
```

```cpp
if (rt.num_gpus() < 2) { res.error = "Ornith-122B GPU runner requires 2x MI50 GPUs"; return res; }
```

```cpp
for (const auto& g : devices_) total_p2p += g.p2p_links_count;
...
if (!map_to_gpus(out, all_gpu_ids_)) { ... }
```

**Why / answer for the new card:** Selection is a list of positions in the current PCI-location-sorted enumeration, not persistent identities. Inserting a lower-location card changes the meaning of `--gpu 0,1` and `VJ_GPU_DEVICES=0,1`. The same happens if firmware renumbers PCI buses. The scan ignores PCI domain, so equal `location_id` values across domains lack a complete ordering key; CLI uses `stable_sort` while runtime uses `sort`.

With default `all`, the direct Ornith runner accepts three devices because it checks only `< 2`, then uses ranks 0 and 1 for its two-rank computation. Before that it has already allocated arenas/queues/code on every selected device and maps shared/mailbox storage to all of them. The new card can therefore replace a compute rank, consume an otherwise unrelated card's VRAM, or make a previously valid pair fail P2P setup. Summing link counts does not validate connectivity between the actual intended two endpoints. A new smaller-VRAM card can also fail the all-device arena preflight.

**Important existing protection:** `serve --tp 2` with all three selected is rejected by the count check at `serve_cmd.cpp:205–208`; that specific route does not silently truncate to two. It does **not** protect an explicit ordinal `--gpu 0,1`, or the direct runner's first-two behavior. `bench-ar` separately requires exactly two (`ornith_gpu.cpp:6037`).

**Concrete fix:** Support full PCI domain:BDF and preferably a persistent device unique ID; resolve once to an immutable ordered rank-to-identity list. Print/log both identities and validate them at runtime init. For a TP=2-only executor reject anything other than exactly two selected devices before opening/allocating; allocate/map only the participating set. Verify directed peer connectivity for that pair, not aggregate link counts. Reject incomplete/duplicate identity data. Add fake-topology tests for insertion before/between/after the old cards, domain collisions, bus renumbering, and CLI/runtime agreement.

**Immediate operational mitigation:** Re-run the GPU listing and independently map render nodes to PCI devices after installing the card; explicitly select the intended pair. This is an operator workaround, **not** a stable guarantee across topology changes. `hardware/check_gfx906.sh` prints a one-based `gpu[$found]` label (`:10–18`), unlike zero-based `--gpu`; do not copy that label as a CLI index.

### RT-M6 — Failed CPU mapping still declares BAR P2P ready, making mailbox reset a silent no-op

**Locations:** `core/kfd_runtime.cpp:479–490,510–548`; resets `models/ornith/ornith_gpu.cpp:2059–2061`.

**Quotes:**

```cpp
if (args.mmap_offset != 0) {
    void* p = mmap(...);
    if (p != MAP_FAILED) out.host_ptr = p;
}
...
return true;
```

```cpp
for (auto& b : p2p_flag_slices_) if (b.host_ptr) std::memset(b.host_ptr, 0, b.size);
```

**Why:** GPU mapping success is accepted even when the CPU mapping is absent or failed. Reset then skips that flag slice, but setup still selects `PeerBarP2P`, prints success, and the model restarts its epoch at 1. Old flags can satisfy an equality test before the current peer payload is published, or wait on unusable state. The reset interface is `void`, so callers cannot fail closed. This is especially relevant to BAR/aperture/mapping failures, not just initial allocation failure.

Reset also has no internal quiescence requirement/enforcement: CPU zeroing while old queues are running would race their flag updates and payload traffic. Normal successful run boundaries may already be idle, but failed/time-out run boundaries are not proven idle by this API.

**Concrete fix:** Require a valid CPU flag mapping if reset uses host stores, or use an explicit GPU reset kernel plus a verified system-scope completion fence. Return errors, verify all rank slices, and require a successfully drained/healthy queue generation before resetting epochs. Reject P2P readiness or explicitly choose a supported fallback if any required mapping/reset fails. Inject CPU mmap failure independently of GPU MAP success; test reset after a timeout separately from reset after success.

### RT-M7 — Teardown discards queue-destruction failure and frees memory the queue may still use

**Locations:** `core/kfd_runtime.cpp:784–806,636–652`; retry loop `:264–269`.

**Quote:**

```cpp
(void) kfd_ioctl(kfd_fd_, AMDKFD_IOC_DESTROY_QUEUE, &dargs);
q.queue_id = 0;
q.queue_active = false;
...
free_buffer(q.kernarg_buf);
free_buffer(q.rw_ptr_buf);
```

**Why:** Shutdown has the right intended order—queues before executable/mailbox/arena storage—but marks the queue destroyed irrespective of the ioctl result. On a failed destroy it immediately releases control, completion-signal, kernarg and ring storage and then code/data. The runtime loses the information needed to distinguish a quiesced queue from a failed device teardown. The ignored unmap/free results in RT-M1 compound this. A GPU timeout is not a completion fence and must not authorize resource reuse.

The common ioctl wrapper also retries `EAGAIN` and `EINTR` forever with no deadline/backoff/cancellation check. The serve signal handler requests stop, but this loop cannot observe that request; a signal is not a reliable bounded way out of an unhealthy ioctl/recovery sequence.

**Concrete fix:** Implement an explicit healthy/draining/poisoned/destroyed lifecycle. Check DESTROY_QUEUE, do not free backing storage until a supported kernel teardown/revocation boundary is established, and retain failure diagnostics/ownership until final device/process cleanup. Bound retryable operations where the ABI allows, integrate cancellation, and never retry non-idempotent operations without preserving their progress state. Test destroy failure, partial unmap, repeated EAGAIN, and SIGTERM during a stalled drain with mocked syscalls; separately qualify the real driver's reset behavior.

## Additional robustness findings

### RT-R1 — MODERATE: epoch zero is both a valid wrapped counter and a “skip collective” sentinel

**Locations:** `models/ornith/ornith_gpu.cpp:2053,2061,2450,2512–2513`; `ops/kernels/vj_ornith_super.hip:760,989,1045,1273,1331,1352,1643`.

**Quotes:** `uint32_t ar_epoch = 1u`; `const uint32_t attn_ep = ar_epoch++;`; `if (prev_ar_epoch != 0u)`; `if (out_ar_epoch != 0u)`.

**Why:** There is no rollover policy. The unsigned counter eventually emits zero, which super kernels interpret as absent AR/prepush. Flag equality and doubled grid-barrier senses also require an explicit generation/reuse argument. Resets at ordinary request/run boundaries reduce practical exposure; this audit does not claim a normal bounded request reaches rollover. It remains an unguarded protocol limit for extended warm/profiling use or future persistent execution.

**Concrete fix:** Before the remaining nonzero epoch space is insufficient for a complete batch, drain both ranks successfully, reset all protocol state, and restart at 1; preferably use a wider generation paired with slot ownership. Test near `UINT32_MAX` and doubled-sense rollover, not billions of actual iterations. A failed drain must poison the run rather than reset underneath it.

### RT-R2 — MODERATE: byte-copy helpers round outward and zero staging capacity makes no progress

**Locations:** `core/kfd_runtime.cpp:453–459,1571–1600`; `ops/kernels/vj_kfd_base.hip:6–13`.

**Quote:** `(chunk + 3u) / 4u` (the `n_u32` field in the aggregate argument construction); `const size_t chunk_cap = std::min<size_t>(host_pinned_arena_.size, 4ULL * 1024ULL * 1024ULL);`; `done += chunk;`.

**Why:** Public helpers accept byte counts but copy full words. An upload of one to three trailing bytes writes bytes beyond the requested destination, sourced from stale staging data; download can read past the requested source range. Existing float/word-aligned callers usually avoid it, but the API does not state or check a word-alignment precondition. Separately `init(..., host_pinned_mb=0, ...)` is accepted, leaving `chunk_cap=0`; a nonempty copy then repeatedly makes zero progress if its zero-grid dispatch succeeds, or fails indirectly rather than reporting missing staging.

**Concrete fix:** Either reject unaligned counts/addresses explicitly or implement byte-exact tails. Reject nonempty transfers when staging is unavailable, and check source/destination range ownership. Test 0/1/2/3/4-byte and chunk-boundary transfers, preserving guard bytes, and host staging disabled.

### RT-R3 — MODERATE: some error/optional-buffer paths lose ownership before cleanup

**Locations:** `core/kfd_runtime.cpp:714–726`; `models/ornith/ornith_gpu.cpp:4831–4832,4888–4890,5987–6002`; ownership type `core/kfd_runtime.hpp:122–132`.

**Quotes:**

```cpp
if (cm == MAP_FAILED) { munmap(cwsr_res, cwsr_total); return false; }
q.cwsr_buf = KfdBuffer{}; // handle is recorded only after mmap succeeds
```

```cpp
if (!run_full_bench_body(true, nullptr)) return res;
...
rt.free_buffer(io_buf);
```

**Why:** CWSR allocation succeeds before a failed CPU mmap, but its KFD handle is not transferred to queue ownership and is never explicitly freed on that path. Local model `KfdBuffer`s have no destructor/RAII ownership and are not registered in `Runtime`: failure to open the GDN dump at line 4832, or failure to obtain the DOT8 comparison buffer at line 4890, returns false from `run_full_bench_body()` without freeing the already allocated IO buffer; line 5989 then returns past its final cleanup. Other specialized branches do free these buffers, so not every false return leaks. KFD process teardown does not substitute for freeing a `posix_memalign` host allocation during a long-lived process. This is separate from the already reported fault-buffer use-after-free after the successful IO free.

**Concrete fix:** Introduce a move-only owning BO/host-storage guard, register ownership immediately after ALLOC, and make every subsequent failure unwind it. Use borrowed views for aliases. Test mmap failure after each successful ALLOC and all early model returns with allocation counters. Do not solve this by blindly freeing IO earlier while later fault checks still reference it.

### RT-R4 — MODERATE: fixed four-rank mailbox layout has no corresponding runtime limit

**Locations:** `core/kfd_runtime.hpp:94–104`; `core/kfd_runtime.cpp:291–296,507–508,536–540`.

**Quotes:** `constexpr uint32_t kMaxGpus = 4;`; `reserve_va(kMaxGpus * p_slice, ...)`; `for (size_t g = 0; ok && g < devices_.size(); ++g)`.

**Why:** The runtime accepts all discovered gfx906 devices, reserves only four ranks of mailbox VA, then allocates a slice for every device. Above four selected devices, slices extend outside their reservation and can overlap following flag/other VA reservations; host fallback storage is also fixed at four ranks. This does not trigger merely from the owner's third card, but it contradicts unrestricted `all` enumeration.

**Concrete fix:** Reject more than the supported maximum before allocating anything, and separately enforce TP=2 at the executor boundary. Alternatively size the full host/device protocol and all rank masks dynamically with checked arithmetic. Test the maximum and maximum-plus-one topology without a GPU.

## Coverage notes and positive checks

- **KFD ioctl setup:** required open/ACQUIRE_VM/aperture/allocation/queue-create failures generally propagate and call shutdown. MAP checks `n_success` on success, and the retry wrapper preserves the same argument object. However `SET_MEMORY_POLICY` is explicitly discarded (`kfd_runtime.cpp:333–339`), and `mlock` failures are ignored (`:595,668`). KFD's own USERPTR pinning can make failed `mlock` nonfatal; it should not be advertised as verified host locking. Make required memory policy failures fatal and report optional locking degradation. No driver-version/feature negotiation or per-GPU aperture-match validation was found; these remain qualification requirements.
- **Host ordering:** dispatch fields precede a release fence and release publication of the packet header (`:1474–1491`); write pointer precedes the doorbell (`:1513–1514`). The batch barrier uses system acquire/release and wait loads use acquire. These are useful safeguards, not proof of signal ownership (RT-M2), argument lifetime (RT-M3), or all supported CPU/device memory-type behavior. No unsupported claim is made that these atomics alone prove PCIe posted-write draining or cross-agent cache visibility.
- **Code-object/kernarg layouts:** fixed packet/signal/descriptor sizes are statically asserted; loaded kernels carry spill/register/code-size gates. Runtime launch construction still derives the implicit-argument offset from host `kernarg_bytes` (`:1437,1466`) and hardcodes a V5 implicit layout, without an argument-offset/type/ABI contract; block sizes are narrowed to 16 bits and launch resource limits are not comprehensively checked. The earlier audit's C6 and `REVIEW.md` B2 already cover the missing ABI validation and concrete stale callers. The systemic fix is generated host/device launch layouts, parsed argument metadata/ABI, `offsetof`/size checks and runtime validation—not merely correcting those three callers. RT-M4 adds the separate unchecked-loader defect.
- **P2P protocol:** the source does issue `s_waitcnt vmcnt(0)`, system fences, and acquire/release readiness operations. This review does not infer actual BAR/cache behavior from comments claiming previous throughput. Real peer-write ordering, MMIO/UC/WC mappings, and reset visibility need owner-hardware qualification. RT-B1 is a logical lost notification even with perfectly ordered, coherent memory.
- **Allocation/double-free:** the slab is a non-owning summary of owning chunks; shutdown frees chunks and only frees slab/mailbox aliases when they have handles (`:821–827`). This avoids an obvious alias double-free. `Runtime` itself is noncopyable. No additional concrete normal-path double-free was established. The actionable defects are mapping ownership, failure cleanup and unregistered local buffers above.
- **Signals:** `serve_cmd.cpp:22–24,296–297` routes SIGINT/SIGTERM to `request_stop`; `server.cpp:1097–1100` sets the stop atomic and wakes the server, rather than doing KFD cleanup in the handler. Server stop joins workers (`:1102–1110`). That is not proof that GPU waits are cancellable. Non-serve commands mostly rely on default termination/process cleanup. The earlier cancellation/backpressure findings are not duplicated; RT-M7 is the additional low-level teardown/retry defect.
- **Hardware scripts:** they are read-only diagnostics, not a P2P test, stable GPU-selection contract, queue test, or reset verification. `check_p2p_ecc.sh` reports link counts and ECC counters; it does not identify and qualify the chosen directed pair.

## Verification performed and required follow-up

Observed tools established the branch/HEAD above and a clean worktree before this report. Source reads and searches supplied all cited runtime paths. Upstream kernel implementations were read to check unmap/free behavior. Small abstract checks exercised the AR lost-notification schedule, completion-signal overwrite, ordinal-selection insertion, and 32-bit wrap-to-zero. These are deliberately **not** represented as VegaJet GPU tests.

Before accepting fixes, add CPU mocked-syscall/fault-injection coverage for resource ownership and every partial failure; parser fuzz/ASan coverage; queue state-machine tests; fake-sysfs identity tests. On the owner's MI50 host, require delayed-peer multi-slice AR tests, wrap-seeded epochs, reset-after-success and reset-after-failure tests, repeated hot reload including a corrupt second module, BO/leak accounting, timeout/destroy failure behavior, and SIGTERM under active execution. Record the actual rank-to-PCI/unique-ID mapping for both two-card and three-card topologies. No GPU acceptance criterion is verified by this static audit.

## BLOCKERS + MAJORS

### BLOCKERS

- **RT-B1 — Consecutive AR slices overwrite an unconsumed readiness epoch, allowing a two-rank deadlock.** `models/ornith/ornith_gpu.cpp:2443–2486`; `ops/kernels/vj_ornith_common.hiph:289–304`.
- **RT-B2 — Partially failed reload-all destroys cached kernel objects, skips rebinding, and accepts subsequent execution.** `core/kfd_runtime.cpp:932–967`; `models/ornith/ornith_gpu.cpp:5936–5958`.

### MAJORS

- **RT-M1 — Free unmaps the wrong GPU set, discards partial progress/errors, and leaks later-rank BOs.** `core/kfd_runtime.cpp:627–652`.
- **RT-M2 — Reused completion signal can report idle with submitted or doorbell-only work still running.** `core/kfd_runtime.cpp:1495–1558`.
- **RT-M3 — Kernarg slots wrap and are overwritten without completion-based retirement.** `core/kfd_runtime.cpp:1436–1468`.
- **RT-M4 — Live HSACO loading dereferences unchecked ELF tables/segments/symbols/descriptors.** `core/kfd_runtime.cpp:1048–1096,1243–1271`.
- **RT-M5 — Third-GPU insertion can change ordinal TP=2 identities; default-all also expands allocation/mapping beyond the compute pair.** `core/kfd_runtime.cpp:185–187,291–340,493–540`; `models/ornith/ornith_gpu.cpp:865–872`.
- **RT-M6 — CPU BAR mapping failure is accepted; mailbox reset silently skips flags while epochs restart.** `core/kfd_runtime.cpp:479–490,523–548`.
- **RT-M7 — Failed queue destruction is treated as success before freeing GPU-visible backing storage; retries are unbounded.** `core/kfd_runtime.cpp:264–269,784–806`.

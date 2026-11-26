// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/kfd_runtime.hpp"
#include "core/gpu_select.hpp"
#include "core/hsaco_gate.hpp"
#include "core/hsaco_loader.hpp"
#include "core/v12_modules.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <thread>

#include "core/host_preflight.hpp"

#ifndef VEGAJET_CPU_ONLY
#include <dirent.h>
#include <elf.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/kfd_ioctl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace vegajet::kfd {

namespace {

uint64_t align_up(uint64_t v, uint64_t a) {
    return (v + a - 1ULL) & ~(a - 1ULL);
}

bool read_u64_file(const std::string& path, uint64_t& out) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) return false;
    ifs >> out;
    return !ifs.fail();
}

} // namespace

namespace {

// First aligned start >= cursor whose [start, start + bytes) avoids every quarantined range, if it ends <= limit.
bool arena_fit(const KfdArena& a, uint64_t cursor, uint64_t limit, uint64_t bytes, uint64_t align, uint64_t& start) {
    if (bytes > limit) return false;
    start = align_up(cursor, align);
    for (bool moved = true; moved;) {
        moved = false;
        for (const auto& q : a.quarantine)
            if (start < q.second && start + bytes > q.first) { start = align_up(q.second, align); moved = true; }
    }
    return start <= limit - bytes;
}

} // namespace

std::vector<uint64_t> arena_chunk_sizes(uint64_t arena_bytes, uint64_t chunk_bytes) {
    std::vector<uint64_t> out;
    if (chunk_bytes == 0) return out;
    for (uint64_t off = 0; off < arena_bytes; off += chunk_bytes) out.push_back(std::min(chunk_bytes, arena_bytes - off));
    return out;
}

uint64_t va_cluster_hot_bytes(const char* cluster_env, const char* hot_mib_env) {
    if (!cluster_env || std::strcmp(cluster_env, "1") != 0) return 0;
    uint64_t mib = kDefaultHotChunks * (kArenaChunkBytes >> 20);
    if (hot_mib_env && hot_mib_env[0]) {
        char* end = nullptr;
        const unsigned long long v = std::strtoull(hot_mib_env, &end, 10);
        if (end && *end == '\0' && v > 0 && v <= (1ull << 20)) mib = v;  // bounded: <= 1 TiB, cannot wrap
    }
    return mib << 20;
}

uint64_t KfdArena::alloc(uint64_t bytes, uint64_t align) { return alloc(bytes, align, default_tier); }

uint64_t KfdArena::alloc(uint64_t bytes, uint64_t align, Tier tier) {
    uint64_t start = 0;
    if (hot_limit == 0) {  // untiered: the original single bump allocator
        if (!arena_fit(*this, offset, slab.size, bytes, align, start)) return 0;
        offset = start + bytes;
        return slab.va + start;
    }
    const auto try_hot = [&]() {
        if (!arena_fit(*this, hot_offset, hot_limit, bytes, align, start)) return false;
        hot_offset = start + bytes;
        return true;
    };
    const auto try_cold = [&]() {
        if (!arena_fit(*this, offset, slab.size, bytes, align, start)) return false;
        offset = start + bytes;
        return true;
    };
    if (tier == Tier::HOT) {
        if (try_hot()) return slab.va + start;
        if (try_cold()) { hot_spill_bytes += bytes; return slab.va + start; }
    } else {
        if (try_cold()) return slab.va + start;
        if (try_hot()) { cold_spill_bytes += bytes; return slab.va + start; }
    }
    return 0;
}

bool KfdArena::enable_tiers(uint64_t hot_bytes, uint64_t chunk_bytes) {
    if (hot_limit != 0 || offset != 0 || hot_offset != 0 || hot_bytes == 0 || chunk_bytes == 0) return false;
    if (hot_bytes > slab.size) return false;
    const uint64_t hot = ((hot_bytes + chunk_bytes - 1) / chunk_bytes) * chunk_bytes;
    if (hot >= slab.size) return false;  // keep at least part of a chunk for the COLD tier
    hot_limit = hot;
    offset = hot;
    default_tier = Tier::HOT;
    return true;
}

std::string resolve_hsaco_dir(const std::string& cli_override) {
    if (!cli_override.empty()) return cli_override;
    if (const char* env = std::getenv("VEGAJET_HSACO_DIR"); env && env[0]) return env;
#ifdef VEGAJET_DEFAULT_HSACO_DIR
    {
        std::error_code ec;
        if (std::filesystem::is_directory(VEGAJET_DEFAULT_HSACO_DIR, ec)) {
            return VEGAJET_DEFAULT_HSACO_DIR;
        }
    }
#endif
    return "./hsaco";
}

std::vector<DeviceInfo> Runtime::scan_gfx906_topology() {
    std::vector<DeviceInfo> out;
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path base("/sys/class/kfd/kfd/topology/nodes");
    if (!fs::exists(base, ec)) return out;

    std::vector<uint32_t> node_ids;
    for (const auto& ent : fs::directory_iterator(base, ec)) {
        const std::string nm = ent.path().filename().string();
        if (nm.empty() || nm[0] < '0' || nm[0] > '9') continue;
        node_ids.push_back(static_cast<uint32_t>(std::stoul(nm)));
    }
    std::sort(node_ids.begin(), node_ids.end());

    for (uint32_t nid : node_ids) {
        const fs::path ndir = base / std::to_string(nid);
        std::ifstream pfile(ndir / "properties");
        if (!pfile.is_open()) continue;

        DeviceInfo info{};
        info.node_id = nid;
        std::string key;
        uint64_t val = 0;
        while (pfile >> key >> val) {
            if (key == "gfx_target_version") info.gfx_target_version = static_cast<uint32_t>(val);
            else if (key == "simd_count") info.simd_count = static_cast<uint32_t>(val);
            else if (key == "drm_render_minor") info.drm_render_minor = static_cast<uint32_t>(val);
            else if (key == "location_id") info.location_id = static_cast<uint32_t>(val);
            else if (key == "p2p_links_count") info.p2p_links_count = static_cast<uint32_t>(val);
            else if (key == "cwsr_size") info.cwsr_size = static_cast<uint32_t>(val);
            else if (key == "ctl_stack_size") info.ctl_stack_size = static_cast<uint32_t>(val);
        }
        if (info.gfx_target_version != 90006u) continue;
        info.n_cu = info.simd_count / 4u;

        uint64_t gid = 0;
        if (read_u64_file((ndir / "gpu_id").string(), gid)) info.gpu_id = static_cast<uint32_t>(gid);

        std::ifstream mfile(ndir / "mem_banks/0/properties");
        if (mfile.is_open()) {
            while (mfile >> key >> val) {
                if (key == "size_in_bytes") {
                    info.vram_bytes = val;
                    break;
                }
            }
        }
        out.push_back(info);
    }

    std::sort(out.begin(), out.end(), [](const DeviceInfo& a, const DeviceInfo& b) {
        return a.location_id < b.location_id;
    });
    return out;
}

Runtime::~Runtime() {
    shutdown();
}

bool aql_wait_space(KfdAqlQueue& q, uint32_t keep_free, uint64_t timeout_us) {
    if (q.ring_packets <= keep_free || !q.read_ptr || !q.write_ptr || !q.doorbell) return false;
    const uint64_t limit = static_cast<uint64_t>(q.ring_packets - keep_free);
    auto has_space = [&] { return q.next_write_idx - __atomic_load_n(q.read_ptr, __ATOMIC_ACQUIRE) < limit; };
    if (has_space()) return true;
    if (__atomic_load_n(q.write_ptr, __ATOMIC_ACQUIRE) != q.next_write_idx) {
        __atomic_store_n(q.write_ptr, q.next_write_idx, __ATOMIC_RELEASE);
        __atomic_store_n(q.doorbell, q.next_write_idx - 1u, __ATOMIC_RELEASE);
        ++q.forced_flushes;
    }
    const auto t0 = std::chrono::steady_clock::now();
    while (!has_space()) {
        if (static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - t0).count()) >= timeout_us) return false;
        std::this_thread::yield();
    }
    return true;
}

#ifdef VEGAJET_CPU_ONLY

bool Runtime::compiled_with_gpu() { return false; }
bool Runtime::open(std::string* err) {
    if (err) *err = "VegaJet was built with VEGAJET_CPU_ONLY=ON; the KFD runtime is disabled";
    return false;
}
bool Runtime::init(double, size_t, std::string* err) { return open(err); }
void Runtime::shutdown() { open_ = false; }
void Runtime::abort_dispatches() {}
bool Runtime::load_code_object(const std::string&, std::string* err) {
    if (err) *err = "code objects cannot be loaded in a CPU-only build";
    return false;
}
bool Runtime::load_hsaco(size_t, const std::string&, std::string* err) { return load_code_object("", err); }
bool Runtime::load_hsaco_dir(const std::string&, std::string* err) { return load_code_object("", err); }
bool Runtime::reload_hsaco(const std::string& module_name, std::string* err) {
    if (err) *err = "cannot reload '" + module_name + "' in a CPU-only build (runtime not open)";
    return false;
}
bool Runtime::reload_all_hsacos(std::string* err) {
    if (err) *err = "cannot reload code objects in a CPU-only build (runtime not open)";
    return false;
}
std::vector<std::string> Runtime::loaded_module_names() const { return {}; }
const LoadedKernel* Runtime::get_kernel(size_t, const std::string&) const { return nullptr; }
bool Runtime::execute(const ops::OpGraph&, std::string* err) {
    if (err) *err = "op graphs cannot execute in a CPU-only build";
    return false;
}
bool Runtime::execute_plan(const ops::StepPlan&, const OpBinderFn&, bool, ExecuteStats*, std::string* err) {
    if (err) *err = "step plans cannot execute in a CPU-only build";
    return false;
}
void Runtime::reset_allreduce_mailboxes() {}
bool Runtime::alloc_userptr_pinned(uint64_t, KfdBuffer&, bool) { return false; }
bool Runtime::free_buffer(KfdBuffer&) { return true; }
bool Runtime::enqueue_kernel(size_t, const LoadedKernel&, uint32_t, uint32_t, uint32_t,
                             uint32_t, uint32_t, uint32_t, const void*, size_t, bool, uint32_t) { return false; }
bool Runtime::submit_batch(size_t) { return false; }
void Runtime::ring_doorbell(size_t) {}
bool Runtime::wait_idle(size_t, uint64_t) { return false; }
bool Runtime::dispatch_async(size_t, const LoadedKernel&, uint32_t, uint32_t, uint32_t,
                             uint32_t, uint32_t, uint32_t, const void*, size_t, uint32_t) { return false; }
bool Runtime::dispatch_sync(size_t, const LoadedKernel&, uint32_t, uint32_t, uint32_t,
                            uint32_t, uint32_t, uint32_t, const void*, size_t, uint32_t) { return false; }
bool Runtime::upload_to_vram(size_t, uint64_t, const void*, size_t) { return false; }
bool Runtime::download_from_vram(size_t, void*, uint64_t, size_t) { return false; }

#else  // !VEGAJET_CPU_ONLY

namespace {
int kfd_ioctl(int fd, unsigned long req, void* arg) {
    int r;
    do {
        r = ioctl(fd, req, arg);
    } while (r == -1 && (errno == EINTR || errno == EAGAIN));
    return r;
}
} // namespace

bool Runtime::compiled_with_gpu() { return true; }

uint64_t Runtime::reserve_va(uint64_t size, uint64_t align) {
    const uint64_t addr = align_up(va_cursor_, align);
    if (addr + size > va_limit_) return 0;
    va_cursor_ = addr + size;
    return addr;
}

bool Runtime::open(std::string* err) {
    return init(30.5, 64, err);
}

bool Runtime::init(double vram_arena_gib, size_t host_pinned_mb, std::string* err) {
    if (const char* dl = std::getenv("VEGAJET_DISPATCH_LOG"); dl && dl[0] && dlog_fd_ < 0)
        dlog_fd_ = ::open(dl, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (open_) return true;

    devices_ = scan_gfx906_topology();
    if (!gpusel::filter_from_env(devices_, err)) return false;  // per-process selection (--gpu / VJ_GPU_DEVICES)
    if (devices_.empty()) {
        if (err) *err = "No gfx906 (90006) GPUs found in /sys/class/kfd/kfd/topology/nodes";
        return false;
    }
    {   // Refuse an arena the cards cannot hold before any device is opened (16 GB cards used to fail late).
        std::vector<uint64_t> vram;
        for (const auto& g : devices_) vram.push_back(g.vram_bytes);
        const auto budget = preflight::check_arena(vram, vram_arena_gib, 1, "the KFD runtime");
        if (!budget.ok) {
            if (err) *err = budget.message;
            devices_.clear();
            return false;
        }
    }

    kfd_fd_ = ::open("/dev/kfd", O_RDWR | O_CLOEXEC);
    if (kfd_fd_ < 0) {
        const int e = errno;
        if (err) *err = std::string("open(/dev/kfd) failed: ") + std::strerror(e) + preflight::device_open_hint(e);
        return false;
    }

    all_gpu_ids_.clear();
    for (auto& g : devices_) {
        const std::string rpath = "/dev/dri/renderD" + std::to_string(g.drm_render_minor);
        g.drm_fd = ::open(rpath.c_str(), O_RDWR | O_CLOEXEC);
        if (g.drm_fd < 0) {
            const int e = errno;
            if (err) *err = "open(" + rpath + ") failed: " + std::strerror(e) + preflight::device_open_hint(e);
            shutdown();
            return false;
        }
        kfd_ioctl_acquire_vm_args vm_args{};
        vm_args.drm_fd = static_cast<uint32_t>(g.drm_fd);
        vm_args.gpu_id = g.gpu_id;
        if (kfd_ioctl(kfd_fd_, AMDKFD_IOC_ACQUIRE_VM, &vm_args) != 0) {
            if (err) *err = std::string("AMDKFD_IOC_ACQUIRE_VM failed: ") + std::strerror(errno);
            shutdown();
            return false;
        }
        kfd_ioctl_set_memory_policy_args mp_args{};
        mp_args.gpu_id                  = g.gpu_id;
        mp_args.default_policy          = KFD_IOC_CACHE_POLICY_NONCOHERENT;
        mp_args.alternate_policy        = KFD_IOC_CACHE_POLICY_COHERENT;
        mp_args.alternate_aperture_base = 0x200000ULL;
        mp_args.alternate_aperture_size = 0x7fffffe00000ULL;
        (void) kfd_ioctl(kfd_fd_, AMDKFD_IOC_SET_MEMORY_POLICY, &mp_args);
        all_gpu_ids_.push_back(g.gpu_id);
    }

    kfd_ioctl_get_process_apertures_new_args ap_args{};
    if (kfd_ioctl(kfd_fd_, AMDKFD_IOC_GET_PROCESS_APERTURES_NEW, &ap_args) != 0) {
        if (err) *err = "AMDKFD_IOC_GET_PROCESS_APERTURES_NEW (count) failed";
        shutdown();
        return false;
    }
    std::vector<kfd_process_device_apertures> aps(ap_args.num_of_nodes);
    ap_args.kfd_process_device_apertures_ptr = reinterpret_cast<uint64_t>(aps.data());
    if (kfd_ioctl(kfd_fd_, AMDKFD_IOC_GET_PROCESS_APERTURES_NEW, &ap_args) != 0) {
        if (err) *err = "AMDKFD_IOC_GET_PROCESS_APERTURES_NEW (fill) failed";
        shutdown();
        return false;
    }

    uint64_t common_base  = 0;
    uint64_t common_limit = UINT64_MAX;
    for (const auto& g : devices_) {
        for (const auto& a : aps) {
            if (a.gpu_id == g.gpu_id) {
                if (a.gpuvm_base > common_base) common_base = a.gpuvm_base;
                if (a.gpuvm_limit < common_limit) common_limit = a.gpuvm_limit;
            }
        }
    }
    va_cursor_ = align_up(std::max<uint64_t>(common_base, 0x0000001000000000ULL), 0x200000ULL);
    va_limit_  = common_limit;

    if (vram_arena_gib > 31.0) vram_arena_gib = 31.0;
    const uint64_t arena_bytes = align_up(static_cast<uint64_t>(vram_arena_gib * 1024.0 * 1024.0 * 1024.0), 0x200000ULL);

    vram_arenas_.resize(devices_.size());
    kernels_.resize(devices_.size());
    queues_.resize(devices_.size());

    // 512 MiB 2 MiB-aligned BO chunking (Linux 7.0 vm_fragment_size=9 invariant)
    constexpr uint64_t kMaxChunkBytes = 512ULL * 1024ULL * 1024ULL;
    for (size_t i = 0; i < devices_.size(); ++i) {
        const uint64_t base_va = reserve_va(arena_bytes, 0x200000ULL);
        if (!base_va) {
            if (err) *err = "GPUVM VA space exhausted for VRAM arena";
            shutdown();
            return false;
        }
        vram_arenas_[i].slab = KfdBuffer{};
        vram_arenas_[i].slab.va           = base_va;
        vram_arenas_[i].slab.size         = arena_bytes;
        vram_arenas_[i].slab.owner_gpu_id = devices_[i].gpu_id;
        vram_arenas_[i].chunks.clear();
        uint64_t off = 0;
        for (const uint64_t csz : arena_chunk_sizes(arena_bytes, kMaxChunkBytes)) {
            kfd_ioctl_alloc_memory_of_gpu_args args{};
            args.va_addr = base_va + off;
            args.size    = csz;
            args.gpu_id  = devices_[i].gpu_id;
            args.flags   = KFD_IOC_ALLOC_MEM_FLAGS_VRAM |
                           KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
                           KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE;
            if (kfd_ioctl(kfd_fd_, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &args) != 0) {
                if (err) *err = std::string("VRAM 512 MiB BO alloc failed: ") + std::strerror(errno);
                shutdown();
                return false;
            }
            KfdBuffer cb{};
            cb.va           = args.va_addr;
            cb.size         = args.size;
            cb.handle       = args.handle;
            cb.owner_gpu_id = args.gpu_id;
            cb.flags        = args.flags;
            if (!map_to_gpus(cb, {devices_[i].gpu_id})) {
                free_buffer(cb);
                if (err) *err = "VRAM chunk MAP_MEMORY_TO_GPU failed";
                shutdown();
                return false;
            }
            vram_arenas_[i].chunks.push_back(cb);
            off += csz;
        }
        vram_arenas_[i].offset = 0;
        vram_arenas_[i].quarantine.clear();
        if (const char* q = std::getenv("VEGAJET_VRAM_QUARANTINE"); q && q[0]) {
            // "<gpu>:<offset_mib>+<len_mib>[,...]"; malformed entries are reported and ignored.
            for (const char* c = q; *c;) {
                unsigned long long g = 0, off = 0, len = 0;
                int used = 0;
                if (*c >= '0' && *c <= '9' && std::sscanf(c, "%llu:%llu+%llu%n", &g, &off, &len, &used) == 3 && len > 0 &&
                    off <= (1ull << 20) && len <= (1ull << 20)) {  // MiB values; bounded so the byte range cannot wrap
                    if (g == i) {
                        vram_arenas_[i].quarantine.emplace_back(off << 20, (off + len) << 20);
                        std::printf("[vegajet::kfd] GPU %zu: arena quarantine +%llu MiB .. +%llu MiB\n", static_cast<size_t>(i), off, off + len);
                    }
                    c += used;
                } else {
                    std::fprintf(stderr, "[vegajet::kfd] VEGAJET_VRAM_QUARANTINE: ignoring malformed entry at '%s'\n", c);
                    while (*c && *c != ',') ++c;
                }
                while (*c == ',') ++c;
            }
        }
        if (const uint64_t hot = va_cluster_hot_bytes(std::getenv("VEGAJET_VA_CLUSTER"), std::getenv("VEGAJET_VA_HOT_MIB")); hot > 0) {
            if (vram_arenas_[i].enable_tiers(hot)) {
                std::printf("[vegajet::kfd] GPU %zu: hot-tensor VA clustering, HOT tier = first %llu x 512 MiB chunks (%llu MiB)\n",
                            static_cast<size_t>(i), static_cast<unsigned long long>(vram_arenas_[i].hot_limit / kMaxChunkBytes),
                            static_cast<unsigned long long>(vram_arenas_[i].hot_limit >> 20));
            } else {
                std::fprintf(stderr, "[vegajet::kfd] GPU %zu: VEGAJET_VA_CLUSTER ignored (HOT %llu MiB does not fit the arena)\n",
                             static_cast<size_t>(i), static_cast<unsigned long long>(hot >> 20));
            }
        }
    }

    if (host_pinned_mb > 0) {
        if (!alloc_userptr_pinned(host_pinned_mb * 1024ULL * 1024ULL, host_pinned_arena_, true)) {
            if (err) *err = "Failed to allocate pinned USERPTR staging arena";
            shutdown();
            return false;
        }
    }

    if (!setup_allreduce_mailbox()) {
        if (err) *err = "Failed to initialize AllReduce mailbox";
        shutdown();
        return false;
    }

    for (size_t i = 0; i < devices_.size(); ++i) {
        if (!create_aql_queue(i)) {
            if (err) *err = "Failed to create AQL queue on GPU " + std::to_string(i);
            shutdown();
            return false;
        }
    }

    open_ = true;
    return true;
}

void Runtime::reset_allreduce_mailboxes() {
    if (!p2p_flag_slices_.empty()) {
        for (auto& b : p2p_flag_slices_) if (b.host_ptr) std::memset(b.host_ptr, 0, b.size);
    } else if (mailbox_flag_buf_.host_ptr) {
        std::memset(mailbox_flag_buf_.host_ptr, 0, sizeof(AllReduceFlags));
    }
    if (!p2p_payload_slices_.empty()) {
        for (auto& b : p2p_payload_slices_) if (b.host_ptr) std::memset(b.host_ptr, 0, b.size);
    } else if (mailbox_payload_buf_.host_ptr) {
        std::memset(mailbox_payload_buf_.host_ptr, 0, sizeof(AllReducePayload));
    }
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

bool Runtime::setup_allreduce_mailbox() {
    uint32_t total_p2p = 0;
    for (const auto& g : devices_) total_p2p += g.p2p_links_count;

    if (total_p2p > 0 && devices_.size() > 1) {
        const uint32_t uc_flags = KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC |
                                  KFD_IOC_ALLOC_MEM_FLAGS_COHERENT |
                                  KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED;
        // Vega 20 (gfx906): PUBLIC|COHERENT maps P2P BAR0 payload to write-combining (verified 6.31 GB/s,
        // 99% of PCIe Gen3 x8 peak, verify_errs=0); dropping COHERENT caches TX in local L2 (MTYPE_NC).
        const uint32_t cc_flags = KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC |
                                  KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;
        const uint64_t p_slice = kArGpuBytes;
        const uint64_t f_slice = 4096ULL;
        const uint64_t p_base  = reserve_va(kMaxGpus * p_slice, 0x200000ULL);
        const uint64_t f_base  = reserve_va(kMaxGpus * f_slice, 4096ULL);

        auto alloc_vram_at = [&](size_t gidx, uint64_t va, uint64_t sz, uint32_t eflags, KfdBuffer& out) -> bool {
            kfd_ioctl_alloc_memory_of_gpu_args args{};
            args.va_addr = va;
            args.size    = sz;
            args.gpu_id  = devices_[gidx].gpu_id;
            args.flags   = KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | eflags;
            if (kfd_ioctl(kfd_fd_, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &args) != 0) return false;
            out = KfdBuffer{};
            out.va           = args.va_addr;
            out.size         = args.size;
            out.handle       = args.handle;
            out.owner_gpu_id = args.gpu_id;
            out.flags        = args.flags;
            if (args.mmap_offset != 0) {
                void* p = mmap(nullptr, sz, PROT_READ | PROT_WRITE, MAP_SHARED, devices_[gidx].drm_fd,
                               static_cast<off_t>(args.mmap_offset));
                if (p != MAP_FAILED) out.host_ptr = p;
            }
            if (!map_to_gpus(out, all_gpu_ids_)) {
                free_buffer(out);
                return false;
            }
            return true;
        };

        bool ok = (p_base != 0 && f_base != 0);
        p2p_payload_slices_.resize(devices_.size());
        p2p_flag_slices_.resize(devices_.size());
        for (size_t g = 0; ok && g < devices_.size(); ++g) {
            ok = ok && alloc_vram_at(g, p_base + g * p_slice, p_slice, cc_flags, p2p_payload_slices_[g]);
            ok = ok && alloc_vram_at(g, f_base + g * f_slice, f_slice, uc_flags, p2p_flag_slices_[g]);
        }
        if (ok) {
            mailbox_payload_buf_.va       = p_base;
            mailbox_payload_buf_.host_ptr = p2p_payload_slices_[0].host_ptr;
            mailbox_flag_buf_.va          = f_base;
            mailbox_flag_buf_.host_ptr    = p2p_flag_slices_[0].host_ptr;
            transport_ = AllReduceTransport::PeerBarP2P;
            reset_allreduce_mailboxes();
            std::printf("[vegajet::kfd] AllReduce: PEER_BAR_P2P PUSH-ONLY DUAL-BAR (payload @ 0x%llx [COH-WC], flags @ 0x%llx [UC])\n",
                        static_cast<unsigned long long>(p_base),
                        static_cast<unsigned long long>(f_base));
            return true;
        }
        for (auto& b : p2p_payload_slices_) free_buffer(b);
        for (auto& b : p2p_flag_slices_) free_buffer(b);
        p2p_payload_slices_.clear();
        p2p_flag_slices_.clear();
    }

    const uint64_t payload_bytes = align_up(sizeof(AllReducePayload), 0x200000ULL);
    const uint64_t flag_bytes    = align_up(sizeof(AllReduceFlags), 4096ULL);
    if (!alloc_userptr_pinned(payload_bytes, mailbox_payload_buf_, true)) return false;
    if (!alloc_userptr_pinned(flag_bytes, mailbox_flag_buf_, true)) return false;
    transport_ = AllReduceTransport::HostPinnedUserptr;
    reset_allreduce_mailboxes();
    return true;
}

bool Runtime::alloc_vram(size_t gpu_idx, uint64_t size, uint32_t extra_flags, KfdBuffer& out, bool map_all_gpus) {
    size = align_up(size, 4096ULL);
    const uint64_t va = reserve_va(size, (size >= 0x200000ULL) ? 0x200000ULL : 4096ULL);
    if (!va) return false;
    kfd_ioctl_alloc_memory_of_gpu_args args{};
    args.va_addr = va;
    args.size    = size;
    args.gpu_id  = devices_[gpu_idx].gpu_id;
    args.flags   = KFD_IOC_ALLOC_MEM_FLAGS_VRAM | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | extra_flags;
    if (kfd_ioctl(kfd_fd_, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &args) != 0) return false;
    out = KfdBuffer{};
    out.va           = args.va_addr;
    out.size         = args.size;
    out.handle       = args.handle;
    out.owner_gpu_id = args.gpu_id;
    out.flags        = args.flags;
    const std::vector<uint32_t> targets = map_all_gpus ? all_gpu_ids_ : std::vector<uint32_t>{devices_[gpu_idx].gpu_id};
    if (!map_to_gpus(out, targets)) { free_buffer(out); return false; }
    return true;
}

bool Runtime::alloc_userptr_pinned(uint64_t size, KfdBuffer& out, bool map_all_gpus) {
    size = align_up(size, 4096ULL);
    void* host_mem = nullptr;
    if (posix_memalign(&host_mem, 4096, size) != 0 || !host_mem) return false;
    std::memset(host_mem, 0, size);
    (void) mlock(host_mem, size);

    const uint64_t va = reserve_va(size, 4096ULL);
    if (!va) { munlock(host_mem, size); std::free(host_mem); return false; }

    kfd_ioctl_alloc_memory_of_gpu_args args{};
    args.va_addr     = va;
    args.size        = size;
    args.mmap_offset = reinterpret_cast<uint64_t>(host_mem);
    args.gpu_id      = devices_[0].gpu_id;
    args.flags       = KFD_IOC_ALLOC_MEM_FLAGS_USERPTR |
                       KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
                       KFD_IOC_ALLOC_MEM_FLAGS_COHERENT |
                       KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED;
    if (kfd_ioctl(kfd_fd_, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &args) != 0) {
        munlock(host_mem, size);
        std::free(host_mem);
        return false;
    }
    out = KfdBuffer{};
    out.va           = args.va_addr;
    out.size         = args.size;
    out.handle       = args.handle;
    out.host_ptr     = host_mem;
    out.owner_gpu_id = args.gpu_id;
    out.flags        = args.flags;
    out.is_userptr   = true;
    const std::vector<uint32_t> targets = map_all_gpus ? all_gpu_ids_ : std::vector<uint32_t>{devices_[0].gpu_id};
    if (!map_to_gpus(out, targets)) { free_buffer(out); return false; }
    return true;
}

bool Runtime::map_to_gpus(KfdBuffer& buf, const std::vector<uint32_t>& gpu_ids) {
    std::vector<uint32_t> targets;
    for (uint32_t id : gpu_ids) {
        if (std::find(buf.mapped_gpu_ids.begin(), buf.mapped_gpu_ids.end(), id) == buf.mapped_gpu_ids.end())
            targets.push_back(id);
    }
    if (targets.empty()) return true;
    kfd_ioctl_map_memory_to_gpu_args margs{};
    margs.handle               = buf.handle;
    margs.device_ids_array_ptr = reinterpret_cast<uint64_t>(targets.data());
    margs.n_devices            = static_cast<uint32_t>(targets.size());
    const int rc = kfd_ioctl(kfd_fd_, AMDKFD_IOC_MAP_MEMORY_TO_GPU, &margs);
    const size_t done = std::min<size_t>(margs.n_success, targets.size());
    buf.mapped_gpu_ids.insert(buf.mapped_gpu_ids.end(), targets.begin(), targets.begin() + done);
    if (rc == 0 && margs.n_success == margs.n_devices) return true;
    std::fprintf(stderr, "[vegajet::kfd] MAP handle=%llu failed (%u/%u GPUs): %s\n",
                 static_cast<unsigned long long>(buf.handle), margs.n_success, margs.n_devices, std::strerror(errno));
    return false;
}

bool Runtime::free_buffer(KfdBuffer& buf) {
    auto retain = [&](const char* op) {
        retained_buffers_[buf.handle] = buf;
        std::fprintf(stderr, "[vegajet::kfd] %s handle=%llu failed; retaining BO and host storage: %s\n",
                     op, static_cast<unsigned long long>(buf.handle), std::strerror(errno));
        return false;
    };
    if (buf.handle != 0) {
        if (kfd_fd_ < 0) return retain("cleanup without KFD fd");
        if (!buf.mapped_gpu_ids.empty()) {
            kfd_ioctl_unmap_memory_from_gpu_args uargs{};
            uargs.handle               = buf.handle;
            uargs.device_ids_array_ptr = reinterpret_cast<uint64_t>(buf.mapped_gpu_ids.data());
            uargs.n_devices            = static_cast<uint32_t>(buf.mapped_gpu_ids.size());
            const int rc = kfd_ioctl(kfd_fd_, AMDKFD_IOC_UNMAP_MEMORY_FROM_GPU, &uargs);
            const size_t done = std::min<size_t>(uargs.n_success, buf.mapped_gpu_ids.size());
            buf.mapped_gpu_ids.erase(buf.mapped_gpu_ids.begin(), buf.mapped_gpu_ids.begin() + done);
            if (rc != 0 || uargs.n_success != uargs.n_devices) return retain("UNMAP");
        }
        kfd_ioctl_free_memory_of_gpu_args fargs{};
        fargs.handle = buf.handle;
        if (kfd_ioctl(kfd_fd_, AMDKFD_IOC_FREE_MEMORY_OF_GPU, &fargs) != 0) return retain("FREE");
        retained_buffers_.erase(buf.handle);
    }
    if (buf.host_ptr) {
        if (buf.is_hugetlb) munmap(buf.host_ptr, buf.size);
        else if (buf.is_userptr) { munlock(buf.host_ptr, buf.size); std::free(buf.host_ptr); }
        else munmap(buf.host_ptr, buf.size);
    }
    buf = KfdBuffer{};
    return true;
}

bool Runtime::create_aql_queue(size_t gpu_idx) {
    auto& q = queues_[gpu_idx];
    q.gpu_idx      = static_cast<uint32_t>(gpu_idx);
    q.ring_packets = 2048u;
    const uint64_t ring_bytes = static_cast<uint64_t>(q.ring_packets) * sizeof(AqlDispatchPacket);

    auto alloc_iso_userptr = [&](uint64_t bytes, uint32_t extra, KfdBuffer& out) -> bool {
        const uint64_t va = reserve_va(bytes, 0x200000ULL);
        if (!va) return false;
        void* p = mmap(reinterpret_cast<void*>(va), bytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
        if (p == MAP_FAILED) return false;
        std::memset(p, 0, bytes);
        (void) mlock(p, bytes);
        kfd_ioctl_alloc_memory_of_gpu_args args{};
        args.va_addr     = va;
        args.size        = bytes;
        args.mmap_offset = va;
        args.gpu_id      = devices_[gpu_idx].gpu_id;
        args.flags       = KFD_IOC_ALLOC_MEM_FLAGS_USERPTR |
                           KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE |
                           KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE |
                           KFD_IOC_ALLOC_MEM_FLAGS_COHERENT |
                           KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED | extra;
        if (kfd_ioctl(kfd_fd_, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &args) != 0) {
            munmap(p, bytes);
            return false;
        }
        out = KfdBuffer{};
        out.va           = args.va_addr;
        out.size         = bytes;
        out.handle       = args.handle;
        out.host_ptr     = p;
        out.owner_gpu_id = args.gpu_id;
        out.flags        = args.flags;
        out.is_userptr   = false;
        if (!map_to_gpus(out, {devices_[gpu_idx].gpu_id})) { free_buffer(out); return false; }
        return true;
    };

    // 1. 2 MiB-isolated AQL ring pre-initialized to Format 1 (HSA_PACKET_TYPE_INVALID)
    if (!alloc_iso_userptr(ring_bytes, KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE, q.ring_buf)) return false;
    auto* pkts = reinterpret_cast<AqlDispatchPacket*>(q.ring_buf.host_ptr);
    for (uint32_t i = 0; i < q.ring_packets; ++i) pkts[i].header = kPacketTypeInvalid;

    // 2. 4 KiB EOP buffer in VRAM
    if (!alloc_vram(gpu_idx, 4096ULL,
                    KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE,
                    q.eop_buf, false)) return false;

    // 3. CWSR buffer (GTT, 2 MiB-aligned VA)
    const uint32_t cwsr_req_size = devices_[gpu_idx].cwsr_size + devices_[gpu_idx].ctl_stack_size;
    if (cwsr_req_size > 0) {
        const uint64_t cwsr_total = align_up(static_cast<uint64_t>(cwsr_req_size) + 65536ULL, 4096ULL);
        const uint64_t cwsr_va = reserve_va(cwsr_total, 0x200000ULL);
        if (!cwsr_va) return false;
        void* cwsr_res = mmap(reinterpret_cast<void*>(cwsr_va), cwsr_total, PROT_NONE,
                              MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
        if (cwsr_res == MAP_FAILED) return false;
        kfd_ioctl_alloc_memory_of_gpu_args cargs{};
        cargs.va_addr = cwsr_va;
        cargs.size    = cwsr_total;
        cargs.gpu_id  = devices_[gpu_idx].gpu_id;
        cargs.flags   = KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE | KFD_IOC_ALLOC_MEM_FLAGS_COHERENT;
        if (kfd_ioctl(kfd_fd_, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &cargs) != 0) { munmap(cwsr_res, cwsr_total); return false; }
        void* cm = mmap(cwsr_res, cwsr_total, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED,
                        devices_[gpu_idx].drm_fd, static_cast<off_t>(cargs.mmap_offset));
        if (cm == MAP_FAILED) { munmap(cwsr_res, cwsr_total); return false; }
        q.cwsr_buf = KfdBuffer{};
        q.cwsr_buf.va           = cargs.va_addr;
        q.cwsr_buf.size         = cwsr_total;
        q.cwsr_buf.handle       = cargs.handle;
        q.cwsr_buf.host_ptr     = cwsr_res;
        q.cwsr_buf.owner_gpu_id = cargs.gpu_id;
        q.cwsr_buf.flags        = cargs.flags;
        if (!map_to_gpus(q.cwsr_buf, {devices_[gpu_idx].gpu_id})) { free_buffer(q.cwsr_buf); return false; }
    }

    // 4. 2 MiB-isolated 4 KiB ctrl page + 1 MiB kernarg ring (2048 x 512 B)
    if (!alloc_iso_userptr(4096ULL, 0u, q.rw_ptr_buf)) return false;
    auto* base_host = reinterpret_cast<uint8_t*>(q.rw_ptr_buf.host_ptr);
    q.write_ptr      = reinterpret_cast<volatile uint64_t*>(base_host + 0x38);
    q.read_ptr       = reinterpret_cast<volatile uint64_t*>(base_host + 0x80);
    q.completion_sig = reinterpret_cast<AmdSignal*>(base_host + 0x200);
    q.completion_sig->kind  = 1;
    q.completion_sig->value = 0;

    if (!alloc_iso_userptr(2048ULL * 512ULL, 0u, q.kernarg_buf)) return false;
    q.kernarg_host     = q.kernarg_buf.host_ptr;
    q.kernarg_va       = q.kernarg_buf.va;
    q.kernarg_capacity = q.kernarg_buf.size;
    q.kernarg_slot     = 0;

    // 5. Create AQL Compute Queue
    kfd_ioctl_create_queue_args cargs{};
    cargs.ring_base_address        = q.ring_buf.va;
    cargs.ring_size                = static_cast<uint32_t>(ring_bytes);
    cargs.gpu_id                   = devices_[gpu_idx].gpu_id;
    cargs.queue_type               = KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
    cargs.queue_percentage         = KFD_MAX_QUEUE_PERCENTAGE;
    cargs.queue_priority           = 7;
    cargs.write_pointer_address    = reinterpret_cast<uint64_t>(q.write_ptr);
    cargs.read_pointer_address     = reinterpret_cast<uint64_t>(q.read_ptr);
    cargs.eop_buffer_address       = q.eop_buf.va;
    cargs.eop_buffer_size          = 4096ULL;
    cargs.ctx_save_restore_address = q.cwsr_buf.va;
    cargs.ctx_save_restore_size    = cwsr_req_size;
    cargs.ctl_stack_size           = devices_[gpu_idx].ctl_stack_size;
    if (kfd_ioctl(kfd_fd_, AMDKFD_IOC_CREATE_QUEUE, &cargs) != 0) return false;
    q.queue_id     = cargs.queue_id;
    q.queue_active = true;

    const uint64_t page_size   = static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
    const uint64_t db_page_off = cargs.doorbell_offset & ~(page_size - 1ULL);
    const uint64_t db_in_page  = cargs.doorbell_offset & (page_size - 1ULL);
    q.doorbell_map_size = page_size * 2ULL;
    q.doorbell_map = mmap(nullptr, q.doorbell_map_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                          kfd_fd_, static_cast<off_t>(db_page_off));
    if (q.doorbell_map == MAP_FAILED) {
        q.doorbell_map_size = page_size;
        q.doorbell_map = mmap(nullptr, q.doorbell_map_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                              kfd_fd_, static_cast<off_t>(db_page_off));
    }
    if (q.doorbell_map == MAP_FAILED) { q.doorbell_map = nullptr; return false; }
    q.doorbell = reinterpret_cast<volatile uint64_t*>(reinterpret_cast<uint8_t*>(q.doorbell_map) + db_in_page);
    q.next_write_idx = 0;
    return true;
}

void Runtime::destroy_aql_queue(size_t gpu_idx) {
    auto& q = queues_[gpu_idx];
    if (q.doorbell_map) {
        munmap(q.doorbell_map, q.doorbell_map_size);
        q.doorbell_map = nullptr;
        q.doorbell     = nullptr;
    }
    if (q.queue_active && kfd_fd_ >= 0) {
        kfd_ioctl_destroy_queue_args dargs{};
        dargs.queue_id = q.queue_id;
        (void) kfd_ioctl(kfd_fd_, AMDKFD_IOC_DESTROY_QUEUE, &dargs);
        q.queue_id     = 0;
        q.queue_active = false;
    }
    free_buffer(q.kernarg_buf);
    free_buffer(q.rw_ptr_buf);
    free_buffer(q.cwsr_buf);
    free_buffer(q.eop_buf);
    free_buffer(q.ring_buf);
}

void Runtime::abort_dispatches() {
    for (size_t i = 0; i < queues_.size(); ++i) destroy_aql_queue(i);
    queues_.clear();
}

void Runtime::shutdown() {
    abort_dispatches();
    for (auto& m : loaded_modules_) {
        for (auto& cb : m.code_buffers) free_buffer(cb);
    }
    loaded_modules_.clear();
    for (auto& cb : code_buffers_) free_buffer(cb);
    code_buffers_.clear();
    kernels_.clear();
    last_hsaco_dir_.clear();
    reload_generation_ = 0;
    for (auto& b : p2p_flag_slices_) free_buffer(b);
    for (auto& b : p2p_payload_slices_) free_buffer(b);
    p2p_flag_slices_.clear();
    p2p_payload_slices_.clear();
    if (mailbox_flag_buf_.handle) free_buffer(mailbox_flag_buf_);
    if (mailbox_payload_buf_.handle) free_buffer(mailbox_payload_buf_);
    free_buffer(host_pinned_arena_);
    for (auto& a : vram_arenas_) {
        for (auto& cb : a.chunks) free_buffer(cb);
        a.chunks.clear();
        if (a.slab.handle) free_buffer(a.slab);
    }
    vram_arenas_.clear();
    const auto retry = retained_buffers_;
    for (const auto& entry : retry) {
        KfdBuffer b = entry.second;
        free_buffer(b);
    }
    if (!retained_buffers_.empty()) {
        std::fprintf(stderr, "[vegajet::kfd] shutdown incomplete: %zu BOs retained; KFD fd kept open\n",
                     retained_buffers_.size());
        open_ = false;
        return;
    }
    for (auto& g : devices_) {
        if (g.drm_fd >= 0) { ::close(g.drm_fd); g.drm_fd = -1; }
    }
    devices_.clear();
    if (kfd_fd_ >= 0) { ::close(kfd_fd_); kfd_fd_ = -1; }
    if (dlog_fd_ >= 0) { ::close(dlog_fd_); dlog_fd_ = -1; }
    open_ = false;
}

std::vector<std::string> Runtime::loaded_module_names() const {
    std::vector<std::string> names;
    names.reserve(loaded_modules_.size());
    for (const auto& m : loaded_modules_) names.push_back(m.name);
    std::sort(names.begin(), names.end());
    return names;
}

bool Runtime::load_code_object(const std::string& path, std::string* err) {
    if (!open_) {
        if (err) *err = "Runtime is not open";
        return false;
    }
    namespace fs = std::filesystem;
    const std::string stem = fs::path(path).stem().string();
    for (const auto& m : loaded_modules_) {
        if (m.name == stem) {
            return reload_hsaco(stem, err);
        }
    }
    const size_t ng = devices_.size();
    std::vector<KfdBuffer> staged_bufs(ng);
    std::vector<std::unordered_map<std::string, LoadedKernel>> staged_k(ng);
    for (size_t g = 0; g < ng; ++g) {
        if (!load_hsaco_staged(g, path, staged_bufs[g], staged_k[g], err)) {
            for (size_t j = 0; j < g; ++j) free_buffer(staged_bufs[j]);
            return false;
        }
    }
    LoadedModule mod;
    mod.name = stem;
    mod.path = path;
    mod.code_buffers = std::move(staged_bufs);
    if (!staged_k.empty()) {
        mod.kernel_names.reserve(staged_k[0].size());
        for (const auto& [kn, _] : staged_k[0]) mod.kernel_names.push_back(kn);
    }
    for (size_t g = 0; g < ng; ++g) {
        for (const auto& [kn, lk] : staged_k[g]) kernels_[g][kn] = lk;
    }
    loaded_modules_.push_back(std::move(mod));
    return true;
}

bool Runtime::reload_hsaco(const std::string& module_name, std::string* err) {
    if (!open_) {
        if (err) *err = "Runtime is not open";
        return false;
    }
    if (module_name.empty() || module_name == "all") {
        return reload_all_hsacos(err);
    }
    namespace fs = std::filesystem;
    std::string stem = fs::path(module_name).stem().string();
    if (stem.empty()) {
        if (err) *err = "Empty module name for reload_hsaco";
        return false;
    }
    // Drain all active AQL queues completely before swapping executable code BOs.
    const size_t ng = devices_.size();
    for (size_t g = 0; g < ng; ++g) {
        if (!wait_idle(g)) {
            if (err) *err = "Timed out draining AQL queue on GPU " + std::to_string(g) + " before hot-reload";
            return false;
        }
    }
    LoadedModule* existing = nullptr;
    for (auto& m : loaded_modules_) {
        if (m.name == stem) { existing = &m; break; }
    }
    std::string path;
    if (existing) {
        path = existing->path;
    } else if (!last_hsaco_dir_.empty()) {
        path = (fs::path(last_hsaco_dir_) / (stem + ".hsaco")).string();
    } else {
        if (err) *err = "Unknown module '" + stem + "' and no hsaco_dir recorded";
        return false;
    }
    // Stage and validate the new .hsaco onto all GPUs BEFORE unmapping old executable BOs.
    std::vector<KfdBuffer> staged_bufs(ng);
    std::vector<std::unordered_map<std::string, LoadedKernel>> staged_k(ng);
    for (size_t g = 0; g < ng; ++g) {
        if (!load_hsaco_staged(g, path, staged_bufs[g], staged_k[g], err)) {
            for (size_t j = 0; j < g; ++j) free_buffer(staged_bufs[j]);
            return false;
        }
    }
    std::vector<std::string> new_names;
    if (!staged_k.empty()) {
        new_names.reserve(staged_k[0].size());
        for (const auto& [kn, _] : staged_k[0]) new_names.push_back(kn);
    }
    if (existing) {
        for (size_t g = 0; g < ng; ++g) {
            for (const auto& old_kn : existing->kernel_names) kernels_[g].erase(old_kn);
            if (g < existing->code_buffers.size() && !free_buffer(existing->code_buffers[g])) {
                for (auto& cb : staged_bufs) free_buffer(cb);
                if (err) *err = "Hot-reload cleanup failed; runtime stopped";
                abort_dispatches();
                open_ = false;
                return false;
            }
        }
        existing->code_buffers = std::move(staged_bufs);
        existing->kernel_names = std::move(new_names);
    } else {
        LoadedModule mod;
        mod.name = stem;
        mod.path = path;
        mod.code_buffers = std::move(staged_bufs);
        mod.kernel_names = std::move(new_names);
        loaded_modules_.push_back(std::move(mod));
    }
    for (size_t g = 0; g < ng; ++g) {
        for (const auto& [kn, lk] : staged_k[g]) kernels_[g][kn] = lk;
    }
    ++reload_generation_;
    return true;
}

bool Runtime::reload_all_hsacos(std::string* err) {
    if (!open_) {
        if (err) *err = "Runtime is not open";
        return false;
    }
    if (loaded_modules_.empty()) {
        if (err) *err = "No loaded .hsaco modules to reload";
        return false;
    }
    std::vector<std::string> names;
    names.reserve(loaded_modules_.size());
    for (const auto& m : loaded_modules_) names.push_back(m.name);
    for (const auto& nm : names) {
        if (!reload_hsaco(nm, err)) {
            // Earlier modules may already have replaced executable BOs and cached kernel objects.
            if (err) *err = "reload-all failed at '" + nm + "'; runtime stopped: " + *err;
            abort_dispatches();
            open_ = false;
            return false;
        }
    }
    return true;
}

bool Runtime::load_hsaco_dir(const std::string& hsaco_dir, std::string* err) {
    namespace fs = std::filesystem;
    const std::string dir = resolve_hsaco_dir(hsaco_dir);
    last_hsaco_dir_ = dir;
    const char* required[] = {
        "vj_kfd_base.hsaco",
        "vj_quant_ops.hsaco",
        "vj_kv_flash_ops.hsaco",
        "vj_ssm_neumann_ops.hsaco",
        "vegajet_megakernel_vm.hsaco",
    };
    for (const char* f : required) {
        const fs::path p = fs::path(dir) / f;
        if (!load_code_object(p.string(), err)) return false;
    }
    // Optional modules: loaded (and gated) when present, so builds without them keep working.
    const char* optional[] = {
        "vj_ornith_ar.hsaco",
        "vj_ornith_gemm.hsaco",
        "vj_ornith_attn.hsaco",
        "vj_ornith_moe.hsaco",
        "vj_ornith_moe_m1.hsaco",
        "vj_ornith_vocab.hsaco",
        "vj_ornith_ops.hsaco",
        "vj_moe_dot8.hsaco",
        "vj_prefill_fa2.hsaco",
        "vj_prefill_gemm.hsaco",
        "vj_prefill_dot8.hsaco",
        "vj_prefill_gdn.hsaco",
        "vj_moe_down_grouped.hsaco",
        "vj_spec_flash.hsaco",
        "vj_spec_gdn.hsaco",
        "vj_dflash2.hsaco",
        "vj_ornith_super.hsaco",

    };
    for (const char* f : optional) {
        const fs::path p = fs::path(dir) / f;
        std::error_code ec;
        if (fs::exists(p, ec)) {
            if (!load_code_object(p.string(), err)) return false;
        }
    }
    const char* vol = std::getenv("VEGAJET_V12_BARRIER_VOL");
    const char* iq2s = std::getenv("VEGAJET_V12_IQ2S_REG");
    const auto variant = select_super_variant(vol && vol[0] == '1', iq2s && iq2s[0] == '1');
    if (!variant.valid) {
        if (err) *err = "BARRIER_VOL and IQ2S_REG are mutually exclusive";
        return false;
    }
    if (variant.object && !load_code_object((fs::path(dir) / variant.object).string(), err)) return false;
    const char* asym = std::getenv("VEGAJET_V12_ASYM_EXPERTS");
    if (asym && std::strcmp(asym, "1") == 0) {
        if (!load_code_object((fs::path(dir) / "vj_moe_asym.hsaco").string(), err)) return false;
    }
    return true;
}

bool Runtime::load_hsaco(size_t gpu_idx, const std::string& hsaco_path, std::string* err) {
    KfdBuffer cbuf{};
    std::unordered_map<std::string, LoadedKernel> staged;
    if (!load_hsaco_staged(gpu_idx, hsaco_path, cbuf, staged, err)) return false;
    for (const auto& [kn, lk] : staged) kernels_[gpu_idx][kn] = lk;
    code_buffers_.push_back(cbuf);
    return true;
}

bool Runtime::load_hsaco_staged(size_t gpu_idx, const std::string& hsaco_path,
                                KfdBuffer& out_buf,
                                std::unordered_map<std::string, LoadedKernel>& out_kernels,
                                std::string* err) {
    out_buf = KfdBuffer{};
    out_kernels.clear();
    std::ifstream ifs(hsaco_path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        if (err) *err = "Cannot open .hsaco: " + hsaco_path;
        return false;
    }
    const auto end = ifs.tellg();
    if (end <= 0 || static_cast<uint64_t>(end) > kMaxCodeBytes) {
        if (err) *err = "HSACO file size outside live-loader limit: " + hsaco_path;
        return false;
    }
    const size_t fsize = static_cast<size_t>(end);
    ifs.seekg(0);
    std::vector<uint8_t> elf(fsize);
    if (!ifs.read(reinterpret_cast<char*>(elf.data()), static_cast<std::streamsize>(fsize))) {
        if (err) *err = "Short HSACO read: " + hsaco_path;
        return false;
    }
    CodeLoadPlan plan;
    if (gpu_idx >= devices_.size() || !plan_hsaco_load(elf.data(), elf.size(), plan, err)) return false;
    const uint64_t image_size = plan.image_bytes;
    // 2 MiB VA alignment enforces amdgpu.vm_fragment_size=9 fragment isolation for executable code BOs.
    const uint64_t code_va = reserve_va(image_size, 0x200000ULL);
    if (!code_va) { if (err) *err = "GPUVM VA exhausted loading " + hsaco_path; return false; }

    kfd_ioctl_alloc_memory_of_gpu_args cargs{};
    cargs.va_addr = code_va;
    cargs.size    = image_size;
    cargs.gpu_id  = devices_[gpu_idx].gpu_id;
    cargs.flags   = KFD_IOC_ALLOC_MEM_FLAGS_GTT | KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE | KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE;
    if (kfd_ioctl(kfd_fd_, AMDKFD_IOC_ALLOC_MEMORY_OF_GPU, &cargs) != 0) {
        if (err) *err = "GTT code alloc failed for " + hsaco_path;
        return false;
    }
    KfdBuffer cbuf{};
    cbuf.va           = cargs.va_addr;
    cbuf.size         = image_size;
    cbuf.handle       = cargs.handle;
    cbuf.owner_gpu_id = cargs.gpu_id;
    cbuf.flags        = cargs.flags;
    void* mapped = mmap(nullptr, image_size, PROT_READ | PROT_WRITE, MAP_SHARED,
                        devices_[gpu_idx].drm_fd, static_cast<off_t>(cargs.mmap_offset));
    if (mapped == MAP_FAILED) { free_buffer(cbuf); if (err) *err = "mmap(code GTT) failed"; return false; }
    cbuf.host_ptr = mapped;
    if (!map_to_gpus(cbuf, {devices_[gpu_idx].gpu_id})) { free_buffer(cbuf); if (err) *err = "map_to_gpus(code) failed"; return false; }

    std::memset(cbuf.host_ptr, 0, image_size);
    for (const auto& seg : plan.segments)
        std::memcpy(static_cast<uint8_t*>(cbuf.host_ptr) + seg.image_offset,
                    elf.data() + seg.file_offset, seg.file_bytes);
    for (auto& entry : plan.kernels) entry.second.kd_gpu_va += cbuf.va;
    out_kernels = std::move(plan.kernels);
    out_buf = cbuf;
    return true;
}

const LoadedKernel* Runtime::get_kernel(size_t gpu_idx, const std::string& name) const {
    if (gpu_idx >= kernels_.size()) return nullptr;
    auto it = kernels_[gpu_idx].find(name);
    return (it == kernels_[gpu_idx].end()) ? nullptr : &it->second;
}

bool Runtime::execute(const ops::OpGraph& graph, std::string* err) {
    if (!open_) {
        if (err) *err = "Runtime is not open";
        return false;
    }
    const size_t ng = devices_.size();
    for (const auto& op : graph) {
        const char* sym = ops::kernel_symbol(op.kind);
        if (!sym || !sym[0]) {
            if (err) *err = "Missing kernel symbol for op " + std::string(ops::op_name(op.kind));
            return false;
        }
        if (op.kind == ops::OpKind::AllReduce) {
            if (ng <= 1) continue; // N == 1 cleanly bypasses AllReduce
            for (size_t g = 0; g < ng; ++g) {
                if ((op.gpu & (1u << g)) == 0u) continue;
                if (g >= kernels_.size()) {
                    if (err) *err = "AllReduce rank out of range";
                    return false;
                }
            }
        } else if (op.gpu >= ng) {
            if (err) *err = "Op " + std::string(ops::op_name(op.kind)) + " targets invalid GPU rank " + std::to_string(op.gpu);
            return false;
        }
    }
    return true;
}

bool Runtime::execute_plan(const ops::StepPlan& plan, const OpBinderFn& binder,
                           bool submit_and_wait, ExecuteStats* stats, std::string* err) {
    if (!open_) {
        if (err) *err = "Runtime is not open";
        return false;
    }
    if (!binder) {
        if (err) *err = "Runtime::execute_plan requires a non-null OpBinderFn";
        return false;
    }
    ExecuteStats st{};
    const size_t ng = devices_.size();
    auto dispatch_one = [&](size_t g, const ops::PlannedOp& pop) -> bool {
        if (g >= ng) {
            if (err) *err = "StepPlan rank " + std::to_string(g) + " >= num_gpus (" + std::to_string(ng) + ")";
            return false;
        }
        OpDispatchSpec spec{};
        if (!binder(g, pop, spec, err)) return false;
        if (spec.skip) {
            ++st.ops_skipped;
            return true;
        }
        if (!spec.kernel || spec.grid_x == 0u) {
            if (err) *err = "OpBinder returned null kernel or zero grid_x for " + std::string(ops::op_name(pop.op.kind));
            return false;
        }
        if (spec.kernel->private_segment_fixed_size != 0u) {
            if (err) *err = "FATAL: refusing to dispatch spilled kernel '" + spec.kernel->name + "'";
            return false;
        }
        if (!enqueue_kernel(g, *spec.kernel, spec.grid_x, spec.grid_y, spec.grid_z,
                            spec.block_x, spec.block_y, spec.block_z,
                            spec.kernargs, spec.kernarg_bytes, pop.barrier, spec.dynamic_lds_bytes)) {
            if (err) *err = "enqueue_kernel failed for " + spec.kernel->name;
            return false;
        }
        ++st.ops_enqueued;
        st.gpus_touched_mask |= (1u << g);
        return true;
    };

    for (const auto& pop : plan.ops) {
        if (pop.op.kind == ops::OpKind::AllReduce) {
            if (ng <= 1u) {
                ++st.ops_skipped;
                continue;
            }
            ++st.allreduce_ops;
            for (size_t g = 0; g < ng; ++g) {
                if ((pop.op.gpu & (1u << g)) == 0u) continue;
                if (!dispatch_one(g, pop)) return false;
            }
        } else {
            if (!dispatch_one(pop.op.gpu, pop)) return false;
        }
    }

    if (submit_and_wait) {
        for (size_t g = 0; g < ng; ++g) {
            if ((st.gpus_touched_mask & (1u << g)) == 0u) continue;
            if (!submit_batch(g)) {
                if (err) *err = "submit_batch failed on GPU " + std::to_string(g);
                return false;
            }
        }
        for (size_t g = 0; g < ng; ++g) {
            if ((st.gpus_touched_mask & (1u << g)) == 0u) continue;
            if (!wait_idle(g)) {
                if (err) *err = "wait_idle failed on GPU " + std::to_string(g);
                return false;
            }
        }
    }
    if (stats) *stats = st;
    return true;
}

bool Runtime::enqueue_kernel(size_t gpu_idx, const LoadedKernel& kernel,
                             uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                             uint32_t block_x, uint32_t block_y, uint32_t block_z,
                             const void* kernargs, size_t kernarg_bytes,
                             bool barrier, uint32_t dynamic_lds_bytes) {
    if (dlog_fd_ >= 0)
        dprintf(dlog_fd_, "E rank=%zu seq=%llu kernel=%s grid=%ux%ux%u block=%ux%ux%u kernarg=%zu barrier=%d\n", gpu_idx,
                static_cast<unsigned long long>(++dlog_seq_), kernel.name.c_str(), grid_x, grid_y, grid_z, block_x, block_y, block_z,
                kernarg_bytes, barrier ? 1 : 0);
    if (!open_ || gpu_idx >= queues_.size()) return false;
    auto& q = queues_[gpu_idx];
    if (q.submission_in_flight) {
        std::fprintf(stderr, "[vegajet::kfd] enqueue GPU %zu rejected: wait for outstanding batch first\n", gpu_idx);
        return false;
    }
    if (kernarg_bytes > 256 || (kernarg_bytes && !kernargs)) {
        std::fprintf(stderr, "[vegajet::kfd] invalid kernargs GPU %zu (bytes=%zu)\n", gpu_idx, kernarg_bytes);
        return false;
    }
    const size_t hidden_off = align_up(kernarg_bytes, 8ULL);
    const size_t total_ka   = std::max<size_t>(hidden_off + 256, kernel.kernarg_size);
    if (total_ka > 512 || !q.kernarg_host || q.kernarg_slot >= q.kernarg_capacity / 512) {
        std::fprintf(stderr, "[vegajet::kfd] kernarg capacity exhausted GPU %zu; submit and wait before reuse\n", gpu_idx);
        return false;
    }
    // Reserve AQL space before touching either live kernargs or the allocation cursor.
    if (!aql_wait_space(q, 2u, 5000000ULL)) {
        std::fprintf(stderr, "[vegajet::kfd] AQL space timeout GPU %zu; runtime stopped\n", gpu_idx);
        open_ = false;
        return false;
    }
    const uint32_t kslot = q.kernarg_slot++;
    uint8_t* ka_dst_host = reinterpret_cast<uint8_t*>(q.kernarg_host) + static_cast<size_t>(kslot) * 512ULL;
    const uint64_t ka_dst_va = q.kernarg_va + static_cast<uint64_t>(kslot) * 512ULL;

    std::memset(ka_dst_host, 0, total_ka);
    if (kernargs && kernarg_bytes > 0) std::memcpy(ka_dst_host, kernargs, kernarg_bytes);
    struct AmdhsaImplicitKernargsV5 {
        uint32_t block_count_x, block_count_y, block_count_z;
        uint16_t group_size_x, group_size_y, group_size_z;
        uint16_t remainder_x, remainder_y, remainder_z;
        uint8_t  reserved0[16];
        uint64_t global_offset_x, global_offset_y, global_offset_z;
        uint16_t grid_dims;
    };
    AmdhsaImplicitKernargsV5 imp{};
    imp.block_count_x = (block_x > 0) ? (grid_x / block_x) : 1u;
    imp.block_count_y = (block_y > 0) ? (grid_y / block_y) : 1u;
    imp.block_count_z = (block_z > 0) ? (grid_z / block_z) : 1u;
    imp.group_size_x  = static_cast<uint16_t>(block_x);
    imp.group_size_y  = static_cast<uint16_t>(block_y);
    imp.group_size_z  = static_cast<uint16_t>(block_z);
    imp.remainder_x   = (block_x > 0) ? static_cast<uint16_t>(grid_x % block_x) : 0u;
    imp.remainder_y   = (block_y > 0) ? static_cast<uint16_t>(grid_y % block_y) : 0u;
    imp.remainder_z   = (block_z > 0) ? static_cast<uint16_t>(grid_z % block_z) : 0u;
    imp.grid_dims     = 3u;
    std::memcpy(ka_dst_host + hidden_off, &imp, sizeof(imp));

    const uint64_t idx  = q.next_write_idx++;
    const uint32_t slot = static_cast<uint32_t>(idx % q.ring_packets);
    auto* pkt = reinterpret_cast<AqlDispatchPacket*>(q.ring_buf.host_ptr) + slot;

    pkt->setup                = 3;
    pkt->workgroup_size_x     = static_cast<uint16_t>(block_x);
    pkt->workgroup_size_y     = static_cast<uint16_t>(block_y);
    pkt->workgroup_size_z     = static_cast<uint16_t>(block_z);
    pkt->reserved0            = 0;
    pkt->grid_size_x          = grid_x;
    pkt->grid_size_y          = grid_y;
    pkt->grid_size_z          = grid_z;
    pkt->private_segment_size = kernel.private_segment_fixed_size;
    pkt->group_segment_size   = kernel.group_segment_fixed_size + dynamic_lds_bytes;
    pkt->kernel_object        = kernel.kd_gpu_va;
    pkt->kernarg_address      = ka_dst_va;
    pkt->reserved2            = 0;
    pkt->completion_signal    = 0;

    __atomic_thread_fence(__ATOMIC_RELEASE);
    const uint16_t hdr = make_aql_header(kPacketTypeKernelDispatch, barrier, kFenceScopeAgent, kFenceScopeAgent);
    __atomic_store_n(&pkt->header, hdr, __ATOMIC_RELEASE);
    return true;
}

bool Runtime::submit_batch(size_t gpu_idx) {
    if (!open_ || gpu_idx >= queues_.size()) return false;
    auto& q = queues_[gpu_idx];
    if (q.submission_in_flight) {
        std::fprintf(stderr, "[vegajet::kfd] submit GPU %zu rejected: completion signal still owned by outstanding batch\n", gpu_idx);
        return false;
    }
    if (q.next_write_idx == q.completed_write_idx) return true;
    if (dlog_fd_ >= 0) dprintf(dlog_fd_, "S rank=%zu upto_seq=%llu\n", gpu_idx, static_cast<unsigned long long>(dlog_seq_));
    if (!aql_wait_space(q, 1u, 5000000ULL)) {
        std::fprintf(stderr, "[vegajet::kfd] AQL fence space timeout GPU %zu; runtime stopped\n", gpu_idx);
        open_ = false;
        return false;
    }
    __atomic_store_n(&q.completion_sig->value, 1LL, __ATOMIC_RELEASE);

    const uint64_t idx  = q.next_write_idx++;
    const uint32_t slot = static_cast<uint32_t>(idx % q.ring_packets);
    auto* raw   = reinterpret_cast<uint64_t*>(reinterpret_cast<AqlDispatchPacket*>(q.ring_buf.host_ptr) + slot);
    auto* raw16 = reinterpret_cast<uint16_t*>(raw);
    raw16[1] = 0; raw16[2] = 0; raw16[3] = 0;
    for (int i = 1; i < 7; ++i) raw[i] = 0;
    raw[7] = q.rw_ptr_buf.va + 0x200;

    __atomic_thread_fence(__ATOMIC_RELEASE);
    const uint16_t hdr = make_aql_header(kPacketTypeBarrierAnd, true, kFenceScopeSystem, kFenceScopeSystem);
    __atomic_store_n(&raw16[0], hdr, __ATOMIC_RELEASE);

    q.submission_in_flight = true;
    __atomic_store_n(q.write_ptr, q.next_write_idx, __ATOMIC_RELEASE);
    __atomic_store_n(q.doorbell, idx, __ATOMIC_RELEASE);
    return true;
}

void Runtime::ring_doorbell(size_t gpu_idx) {
    if (!open_ || gpu_idx >= queues_.size()) return;
    auto& q = queues_[gpu_idx];
    if (q.next_write_idx == 0) return;
    __atomic_store_n(q.write_ptr, q.next_write_idx, __ATOMIC_RELEASE);
    __atomic_store_n(q.doorbell, q.next_write_idx - 1u, __ATOMIC_RELEASE);
}

bool Runtime::dispatch_async(size_t gpu_idx, const LoadedKernel& kernel,
                             uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                             uint32_t block_x, uint32_t block_y, uint32_t block_z,
                             const void* kernargs, size_t kernarg_bytes,
                             uint32_t dynamic_lds_bytes) {
    if (!enqueue_kernel(gpu_idx, kernel, grid_x, grid_y, grid_z, block_x, block_y, block_z,
                        kernargs, kernarg_bytes, true, dynamic_lds_bytes)) return false;
    return submit_batch(gpu_idx);
}

bool Runtime::wait_idle(size_t gpu_idx, uint64_t timeout_us) {
    if (!open_ || gpu_idx >= queues_.size()) return false;
    auto& q = queues_[gpu_idx];
    // Pending/rung packets without a completion barrier must be fenced before zero means idle.
    if (!q.submission_in_flight && q.next_write_idx != q.completed_write_idx && !submit_batch(gpu_idx)) return false;
    auto t0 = std::chrono::steady_clock::now();
    uint32_t spins = 0;
    int64_t value;
    while ((value = __atomic_load_n(&q.completion_sig->value, __ATOMIC_ACQUIRE)) != 0) {
        if (value < 0) {
            std::fprintf(stderr, "[vegajet::kfd] completion underflow GPU %zu (sig=%lld); runtime stopped\n",
                         gpu_idx, static_cast<long long>(value));
            open_ = false;
            return false;
        }
        if ((++spins & 1023u) == 0u) {
            const auto el = std::chrono::duration_cast<std::chrono::microseconds>(
                                std::chrono::steady_clock::now() - t0).count();
            if (static_cast<uint64_t>(el) > timeout_us) {
                std::fprintf(stderr, "[vegajet::kfd] TIMEOUT GPU %zu (sig=%lld rptr=%llu wptr=%llu)\n",
                             gpu_idx, static_cast<long long>(q.completion_sig->value),
                             static_cast<unsigned long long>(*q.read_ptr),
                             static_cast<unsigned long long>(*q.write_ptr));
                open_ = false;
                return false;
            }
        }
#if defined(__x86_64__)
        __builtin_ia32_pause();
#else
        std::this_thread::yield();
#endif
    }
    q.submission_in_flight = false;
    q.completed_write_idx = q.next_write_idx;
    q.kernarg_slot = 0;  // only a completed system fence retires kernarg ownership
    if (dlog_fd_ >= 0) dprintf(dlog_fd_, "C rank=%zu upto_seq=%llu\n", gpu_idx, static_cast<unsigned long long>(dlog_seq_));
    return true;
}

bool Runtime::dispatch_sync(size_t gpu_idx, const LoadedKernel& kernel,
                            uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                            uint32_t block_x, uint32_t block_y, uint32_t block_z,
                            const void* kernargs, size_t kernarg_bytes,
                            uint32_t dynamic_lds_bytes) {
    if (!dispatch_async(gpu_idx, kernel, grid_x, grid_y, grid_z, block_x, block_y, block_z,
                        kernargs, kernarg_bytes, dynamic_lds_bytes)) return false;
    return wait_idle(gpu_idx);
}

bool Runtime::upload_to_vram(size_t gpu_idx, uint64_t dst_vram_va, const void* src, size_t bytes) {
    const auto* k = get_kernel(gpu_idx, "kfd_memcpy_u32");
    if (!k) return false;
    const size_t chunk_cap = std::min<size_t>(host_pinned_arena_.size, 4ULL * 1024ULL * 1024ULL);
    const auto* src_b = reinterpret_cast<const uint8_t*>(src);
    size_t done = 0;
    while (done < bytes) {
        const size_t chunk = std::min(bytes - done, chunk_cap);
        std::memcpy(host_pinned_arena_.host_ptr, src_b + done, chunk);
        struct { uint64_t dst, src, n_u32; } ka{dst_vram_va + done, host_pinned_arena_.va, (chunk + 3u) / 4u};
        const uint32_t grid = static_cast<uint32_t>(((ka.n_u32 + 255u) / 256u) * 256u);
        if (!dispatch_sync(gpu_idx, *k, grid, 1, 1, 256, 1, 1, &ka, sizeof(ka))) return false;
        done += chunk;
    }
    return true;
}

bool Runtime::download_from_vram(size_t gpu_idx, void* dst, uint64_t src_vram_va, size_t bytes) {
    const auto* k = get_kernel(gpu_idx, "kfd_memcpy_u32");
    if (!k) return false;
    const size_t chunk_cap = std::min<size_t>(host_pinned_arena_.size, 4ULL * 1024ULL * 1024ULL);
    auto* dst_b = reinterpret_cast<uint8_t*>(dst);
    size_t done = 0;
    while (done < bytes) {
        const size_t chunk = std::min(bytes - done, chunk_cap);
        struct { uint64_t dst, src, n_u32; } ka{host_pinned_arena_.va, src_vram_va + done, (chunk + 3u) / 4u};
        const uint32_t grid = static_cast<uint32_t>(((ka.n_u32 + 255u) / 256u) * 256u);
        if (!dispatch_sync(gpu_idx, *k, grid, 1, 1, 256, 1, 1, &ka, sizeof(ka))) return false;
        std::memcpy(dst_b + done, host_pinned_arena_.host_ptr, chunk);
        done += chunk;
    }
    return true;
}

#endif  // VEGAJET_CPU_ONLY

} // namespace vegajet::kfd

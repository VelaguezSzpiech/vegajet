// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Bare-metal /dev/kfd runtime interface for VegaJet (gfx906 / Vega 20).
// Under VEGAJET_CPU_ONLY=1 every entry point is a clean stub that never opens a device node;
// under VEGAJET_CPU_ONLY=0 this implements the full hardened Linux 7.0 KFD runtime:
//   - 512 MiB 2 MiB-aligned BO chunking in KfdArena
//   - 4 MiB max VRAM upload/download chunks (<0.6 ms)
//   - mlock() + 2 MiB VA isolation for ring/cwsr/ctrl/kernarg
//   - AQL Format 1 pre-init + release fence + atomic 16-bit header write last
//   - Push-Only Dual-BAR0 PEER_BAR_P2P AllReduce mailboxes
//   - AMDGPU MessagePack .amdhsa.kernels ELF parser with HARD runtime private_segment_fixed_size == 0 assert
#pragma once

#include "ops/layer_plan.hpp"
#include "ops/op_desc.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vegajet::kfd {

// HSA AQL kernel dispatch packet (64 bytes).
struct alignas(64) AqlDispatchPacket {
    uint16_t header;
    uint16_t setup;
    uint16_t workgroup_size_x, workgroup_size_y, workgroup_size_z;
    uint16_t reserved0;
    uint32_t grid_size_x, grid_size_y, grid_size_z;
    uint32_t private_segment_size;
    uint32_t group_segment_size;
    uint64_t kernel_object;
    uint64_t kernarg_address;
    uint64_t reserved2;
    uint64_t completion_signal;
};
static_assert(sizeof(AqlDispatchPacket) == 64, "AQL packet must be 64 bytes");

// AMD HSA signal layout expected by Vega 20 CP firmware (64 bytes, 64-byte aligned).
struct alignas(64) AmdSignal {
    int64_t kind;  // 1 = AMD_SIGNAL_KIND_USER
    union {
        volatile int64_t value;
        volatile uint64_t* legacy_hardware_doorbell_ptr;
    };
    uint64_t event_mailbox_ptr;
    uint32_t event_id;
    uint32_t reserved1;
    uint64_t start_ts;
    uint64_t end_ts;
    uint64_t queue_ptr;
    uint32_t reserved3[2];
};
static_assert(sizeof(AmdSignal) == 64, "AmdSignal must be 64 bytes");

// AMDHSA kernel descriptor (the `.kd` symbol of a code object, 64 bytes).
struct alignas(64) KernelDescriptor {
    uint32_t group_segment_fixed_size;
    uint32_t private_segment_fixed_size;
    uint32_t kernarg_size;
    uint8_t reserved0[4];
    int64_t kernel_code_entry_byte_offset;
    uint8_t reserved1[20];
    uint32_t compute_pgm_rsrc3;
    uint32_t compute_pgm_rsrc1;
    uint32_t compute_pgm_rsrc2;
    uint16_t kernel_code_properties;
    uint16_t kernarg_preload;
    uint8_t reserved2[4];
};
static_assert(sizeof(KernelDescriptor) == 64, "kernel descriptor must be 64 bytes");

constexpr uint16_t kPacketTypeInvalid        = 1;
constexpr uint16_t kPacketTypeKernelDispatch = 2;
constexpr uint16_t kPacketTypeBarrierAnd     = 3;
constexpr uint16_t kFenceScopeNone           = 0;
constexpr uint16_t kFenceScopeAgent          = 1;
constexpr uint16_t kFenceScopeSystem         = 2;

inline uint16_t make_aql_header(uint16_t type, bool barrier, uint16_t acquire, uint16_t release) {
    return static_cast<uint16_t>((type & 0xFFu) | ((barrier ? 1u : 0u) << 8) | ((acquire & 3u) << 9) |
                                 ((release & 3u) << 11));
}

enum class AllReduceTransport {
    PeerBarP2P,
    HostPinnedUserptr,
};

constexpr uint32_t kMaxGpus = 4;

struct alignas(4096) AllReduceFlags {
    volatile uint32_t epoch[kMaxGpus][64][16];  // 4 KiB per dst_rank (64B isolated slots)
};

constexpr uint64_t kArSlotBytes = 2ULL * 1048576ULL; // 2 MiB per ping-pong slot (fits M=512 INT8 AR in 1 slice!)
constexpr uint64_t kArGpuBytes  = 2ULL * kArSlotBytes; // 4 MiB per GPU (2 ping-pong slots)

struct alignas(4096) AllReducePayload {
    uint8_t rx_rank[kMaxGpus][2][kArSlotBytes];  // 4 MiB per dst_rank (2 ping-pong slots x 2 MiB)
};

struct DeviceInfo {
    uint32_t node_id = 0;
    uint32_t gpu_id = 0;
    uint32_t drm_render_minor = 0;
    uint32_t gfx_target_version = 0;  // 90006 for gfx906
    uint32_t n_cu = 0;
    uint32_t simd_count = 0;
    uint32_t location_id = 0;
    uint32_t p2p_links_count = 0;
    uint32_t cwsr_size = 0;
    uint32_t ctl_stack_size = 0;
    uint64_t vram_bytes = 0;
    int      drm_fd = -1;
};

struct KfdBuffer {
    uint64_t va = 0;
    uint64_t size = 0;
    uint64_t handle = 0;
    void*    host_ptr = nullptr;
    uint32_t owner_gpu_id = 0;
    uint32_t flags = 0;
    bool     is_userptr = false;
    bool     is_hugetlb = false;
};

// Hot-tensor VA clustering (opt-in). A tiered arena packs every always-touched tensor (dense projections,
// shared experts, router, LM head, KV cache, GDN state, workspace) into [0, hot_limit) -- the first few
// 512 MiB BO chunks -- and pushes sparsely-read data (routed MoE experts, token embedding) above it, so
// the per-token working set spans as few UTCL2 2 MiB translations as possible.
enum class Tier : uint8_t { HOT = 0, COLD = 1 };

constexpr uint64_t kArenaChunkBytes  = 512ULL * 1024ULL * 1024ULL;  // BO chunk size (vm_fragment_size=9)
constexpr uint64_t kDefaultHotChunks = 5;                           // 2.5 GiB HOT tier by default

// Sizes of the BO chunks init() maps back-to-back for an arena of `arena_bytes`: all chunk_bytes, then one tail.
std::vector<uint64_t> arena_chunk_sizes(uint64_t arena_bytes, uint64_t chunk_bytes = kArenaChunkBytes);

// HOT tier size requested by the environment: VEGAJET_VA_CLUSTER=1 turns clustering on (default
// kDefaultHotChunks chunks), VEGAJET_VA_HOT_MIB=<n> overrides the size. 0 = off (legacy arena).
uint64_t va_cluster_hot_bytes(const char* cluster_env, const char* hot_mib_env);

struct KfdArena {
    KfdBuffer slab;
    std::vector<KfdBuffer> chunks;  // <=512 MiB 2 MiB-aligned BOs mapped back-to-back
    uint64_t offset = 0;            // COLD (and untiered) bump cursor, arena-relative
    // Tiering: off while hot_limit == 0 (every call behaves exactly like the untiered arena, `tier` is ignored).
    // On: HOT allocations bump hot_offset inside [0, hot_limit), COLD ones bump offset inside [hot_limit,
    // capacity). A request that does not fit its tier spills into the other one (counted) instead of failing.
    uint64_t hot_limit = 0;
    uint64_t hot_offset = 0;
    Tier     default_tier = Tier::COLD;  // tier of alloc(bytes, align); enable_tiers() sets HOT
    uint64_t hot_spill_bytes = 0;        // HOT requests served from the COLD tier
    uint64_t cold_spill_bytes = 0;       // COLD requests served from the HOT tier
    // Quarantined [begin, end) arena offsets that alloc() never hands out (HBM2 regions logging correctable
    // errors, found with `vegajet soak`). Set from VEGAJET_VRAM_QUARANTINE="<gpu>:<offset_mib>+<len_mib>[,...]".
    std::vector<std::pair<uint64_t, uint64_t>> quarantine;
    uint64_t alloc(uint64_t bytes, uint64_t align = 4096);
    uint64_t alloc(uint64_t bytes, uint64_t align, Tier tier);
    // Splits a still-empty arena: HOT = hot_bytes rounded up to whole chunk_bytes chunks, so the tier boundary is
    // a BO chunk boundary. Fails (arena unchanged) when the arena is not empty or HOT would not leave a COLD chunk.
    bool     enable_tiers(uint64_t hot_bytes, uint64_t chunk_bytes = kArenaChunkBytes);
    bool     tiered() const { return hot_limit != 0; }
    void     reset() { offset = hot_limit; hot_offset = 0; hot_spill_bytes = cold_spill_bytes = 0; }
    uint64_t used() const { return offset - hot_limit + hot_offset; }  // bytes handed out (both tiers)
    uint64_t capacity() const { return slab.size; }
    uint64_t hot_used() const { return hot_offset; }
    uint64_t cold_used() const { return offset - hot_limit; }
    // Bump-cursor snapshot for an all-or-nothing group of allocations.
    struct Mark { uint64_t offset, hot_offset, hot_spill, cold_spill; };
    Mark     mark() const { return {offset, hot_offset, hot_spill_bytes, cold_spill_bytes}; }
    void     rollback(const Mark& m) { offset = m.offset; hot_offset = m.hot_offset; hot_spill_bytes = m.hot_spill; cold_spill_bytes = m.cold_spill; }
};

// Scoped default tier: allocations made through alloc(bytes, align) inside the scope land in `tier`.
class ArenaTierScope {
public:
    ArenaTierScope(KfdArena& a, Tier tier) : a_(a), prev_(a.default_tier) { a_.default_tier = tier; }
    ~ArenaTierScope() { a_.default_tier = prev_; }
    ArenaTierScope(const ArenaTierScope&) = delete;
    ArenaTierScope& operator=(const ArenaTierScope&) = delete;
private:
    KfdArena& a_;
    Tier prev_;
};

struct LoadedKernel {
    std::string name;
    uint64_t    kd_gpu_va = 0;
    uint32_t    group_segment_fixed_size = 0;
    uint32_t    private_segment_fixed_size = 0;
    uint32_t    kernarg_size = 0;
    uint32_t    vgpr_count = 0;
    uint32_t    sgpr_count = 0;
    uint32_t    wavefront_size = 64;
    uint32_t    max_flat_workgroup_size = 256;
    uint32_t    resident_waves_per_simd = 4;
    bool        from_msgpack_note = false;
};

struct KfdAqlQueue {
    uint32_t  gpu_idx = 0;
    uint32_t  queue_id = 0;
    bool      queue_active = false;
    uint32_t  ring_packets = 2048;
    KfdBuffer ring_buf;
    KfdBuffer eop_buf;
    KfdBuffer cwsr_buf;
    KfdBuffer rw_ptr_buf;
    KfdBuffer kernarg_buf;
    volatile uint64_t* write_ptr = nullptr;
    volatile uint64_t* read_ptr  = nullptr;
    volatile uint64_t* doorbell  = nullptr;
    void*     doorbell_map = nullptr;
    size_t    doorbell_map_size = 0;
    uint64_t  next_write_idx = 0;
    AmdSignal* completion_sig = nullptr;
    uint64_t  kernarg_va = 0;
    void*     kernarg_host = nullptr;
    size_t    kernarg_capacity = 0;
    uint32_t  kernarg_slot = 0;
    uint64_t  forced_flushes = 0;  // ring-full publishes done by aql_wait_space (see below)
};

// Ring-full guard (host-memory logic only, no ioctl; shared by Runtime and the CPU mock-ring test).
// Waits until fewer than ring_packets - keep_free packets are outstanding (bounded by timeout_us). If the ring is
// full of packets that were written but never published (write_ptr/doorbell not yet advanced, e.g. a long token
// enqueued between the early ring_doorbell() and submit_batch()), the CP cannot drain them and the old wait spun
// until its 5 s timeout. Here the written packets are published first (write_ptr, then the doorbell), counted in
// forced_flushes. Every published packet already carries its final header, so this only starts work early.
bool aql_wait_space(KfdAqlQueue& q, uint32_t keep_free, uint64_t timeout_us);

// Resolves default .hsaco directory: cli_override -> $VEGAJET_HSACO_DIR -> build dir -> ./hsaco.
std::string resolve_hsaco_dir(const std::string& cli_override = "");

struct OpDispatchSpec {
    const LoadedKernel* kernel = nullptr;
    uint32_t grid_x = 0, grid_y = 1, grid_z = 1;
    uint32_t block_x = 256, block_y = 1, block_z = 1;
    const void* kernargs = nullptr;
    size_t kernarg_bytes = 0;
    uint32_t dynamic_lds_bytes = 0;
    bool skip = false;
};

struct ExecuteStats {
    uint32_t ops_enqueued = 0;
    uint32_t ops_skipped = 0;
    uint32_t allreduce_ops = 0;
    uint32_t gpus_touched_mask = 0;
};

using OpBinderFn = std::function<bool(size_t gpu_idx, const ops::PlannedOp& pop, OpDispatchSpec& out, std::string* err)>;

class Runtime {
public:
    Runtime() = default;
    ~Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    // True only in GPU builds with /dev/kfd support compiled in.
    static bool compiled_with_gpu();

    // Read-only topology scan of /sys/class/kfd/kfd/topology/nodes/* (safe without opening /dev/kfd).
    static std::vector<DeviceInfo> scan_gfx906_topology();

    // Opens /dev/kfd with default arena sizing (30.5 GiB ceiling, 64 MiB pinned staging).
    bool open(std::string* err);
    bool init(double vram_arena_gib, size_t host_pinned_mb, std::string* err = nullptr);
    void shutdown();
    // Irreversible fail-closed stop; keep allocations valid for caller cleanup.
    void abort_dispatches();

    // Loads a single unbundled gfx906 .hsaco onto all open GPUs (hard-asserts private_segment_fixed_size == 0!).
    bool load_code_object(const std::string& path, std::string* err);
    bool load_hsaco(size_t gpu_idx, const std::string& path, std::string* err = nullptr);
    // Loads standard VegaJet .hsaco set from hsaco_dir.
    bool load_hsaco_dir(const std::string& hsaco_dir, std::string* err = nullptr);

    // Zero-reload .hsaco hot-reload: drains all active AQL queues, validates & stages the updated .hsaco
    // (re-running the MessagePack 0 B spill + VGPR + code-size gate), then atomically unmaps/frees the old
    // executable BOs without touching the resident weight/KV/scratch VRAM arena.
    bool reload_hsaco(const std::string& module_name, std::string* err = nullptr);
    bool reload_all_hsacos(std::string* err = nullptr);
    std::vector<std::string> loaded_module_names() const;
    uint64_t reload_generation() const { return reload_generation_; }

    const LoadedKernel* get_kernel(size_t gpu_idx, const std::string& name) const;
    bool execute(const ops::OpGraph& graph, std::string* err);
    // Descriptor-driven executor consuming ops::StepPlan (from ops::build_step_plan): resolves per-op
    // kernel/grid/kernargs via `binder`, skips AllReduce cleanly when num_gpus() <= 1, enforces
    // private_segment_fixed_size == 0, respects barrier bits, and optionally submits + waits all touched GPUs.
    bool execute_plan(const ops::StepPlan& plan, const OpBinderFn& binder,
                      bool submit_and_wait = true, ExecuteStats* stats = nullptr, std::string* err = nullptr);

    const std::vector<DeviceInfo>& devices() const { return devices_; }
    size_t num_gpus() const { return devices_.size(); }
    KfdArena& vram_arena(size_t gpu_idx) { return vram_arenas_[gpu_idx]; }
    AllReduceTransport allreduce_transport() const { return transport_; }
    uint64_t mailbox_payload_gpu_va() const { return mailbox_payload_buf_.va; }
    uint64_t mailbox_flags_gpu_va() const { return mailbox_flag_buf_.va; }
    // GPU VA and size of the peer-mapped payload slice owned by `gpu_idx` (0 when the transport is not PEER_BAR_P2P).
    uint64_t p2p_payload_slice_va(size_t gpu_idx) const { return gpu_idx < p2p_payload_slices_.size() ? p2p_payload_slices_[gpu_idx].va : 0; }
    uint64_t p2p_payload_slice_bytes(size_t gpu_idx) const { return gpu_idx < p2p_payload_slices_.size() ? p2p_payload_slices_[gpu_idx].size : 0; }
    void     reset_allreduce_mailboxes();

    const KfdBuffer& host_pinned_arena() const { return host_pinned_arena_; }
    bool alloc_userptr_pinned(uint64_t size, KfdBuffer& out, bool map_all_gpus = true);
    void free_buffer(KfdBuffer& buf);

    // AQL kernel dispatch
    bool enqueue_kernel(size_t gpu_idx, const LoadedKernel& kernel,
                        uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                        uint32_t block_x, uint32_t block_y, uint32_t block_z,
                        const void* kernargs, size_t kernarg_bytes,
                        bool barrier = true, uint32_t dynamic_lds_bytes = 0);
    bool submit_batch(size_t gpu_idx);
    void ring_doorbell(size_t gpu_idx);
    bool wait_idle(size_t gpu_idx, uint64_t timeout_us = 10000000ULL);
    bool dispatch_async(size_t gpu_idx, const LoadedKernel& kernel,
                        uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                        uint32_t block_x, uint32_t block_y, uint32_t block_z,
                        const void* kernargs, size_t kernarg_bytes,
                        uint32_t dynamic_lds_bytes = 0);
    bool dispatch_sync(size_t gpu_idx, const LoadedKernel& kernel,
                       uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                       uint32_t block_x, uint32_t block_y, uint32_t block_z,
                       const void* kernargs, size_t kernarg_bytes,
                       uint32_t dynamic_lds_bytes = 0);

    // 4 MiB chunk-capped VRAM transfers (safe against Linux 7.0 CWSR preemption)
    bool upload_to_vram(size_t gpu_idx, uint64_t dst_vram_va, const void* src, size_t bytes);
    bool download_from_vram(size_t gpu_idx, void* dst, uint64_t src_vram_va, size_t bytes);

private:
    friend struct RuntimeTestAccess;
    struct LoadedModule {
        std::string name;
        std::string path;
        std::vector<KfdBuffer> code_buffers;
        std::vector<std::string> kernel_names;
    };

#ifndef VEGAJET_CPU_ONLY
    uint64_t reserve_va(uint64_t size, uint64_t align = 0x200000ULL);
    bool alloc_vram(size_t gpu_idx, uint64_t size, uint32_t extra_flags, KfdBuffer& out, bool map_all_gpus = false);
    bool map_to_gpus(const KfdBuffer& buf, const std::vector<uint32_t>& gpu_ids);
    bool create_aql_queue(size_t gpu_idx);
    void destroy_aql_queue(size_t gpu_idx);
    bool setup_allreduce_mailbox();
    bool load_hsaco_staged(size_t gpu_idx, const std::string& path,
                           KfdBuffer& out_buf,
                           std::unordered_map<std::string, LoadedKernel>& out_kernels,
                           std::string* err);
#endif

    int kfd_fd_ = -1;
    // VEGAJET_DISPATCH_LOG=<file>: unbuffered host-side journal, one record per enqueue (E), submit (S) and completed
    // wait (C), written before the action reaches the GPU so a fault can be attributed to a dispatch boundary.
    int dlog_fd_ = -1;
    uint64_t dlog_seq_ = 0;
    std::vector<DeviceInfo> devices_;
    std::vector<uint32_t> all_gpu_ids_;
    std::vector<KfdArena> vram_arenas_;
    std::vector<KfdAqlQueue> queues_;
    std::vector<KfdBuffer> code_buffers_;
    std::vector<LoadedModule> loaded_modules_;
    std::string last_hsaco_dir_;
    uint64_t reload_generation_ = 0;
    std::vector<std::unordered_map<std::string, LoadedKernel>> kernels_;

    KfdBuffer host_pinned_arena_;
    KfdBuffer mailbox_payload_buf_;
    KfdBuffer mailbox_flag_buf_;
    std::vector<KfdBuffer> p2p_payload_slices_;
    std::vector<KfdBuffer> p2p_flag_slices_;
    AllReduceTransport transport_ = AllReduceTransport::HostPinnedUserptr;

    uint64_t va_cursor_ = 0;
    uint64_t va_limit_  = 0;
    bool open_ = false;
};

} // namespace vegajet::kfd

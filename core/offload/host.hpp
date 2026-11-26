// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>
namespace vegajet::offload {
constexpr size_t huge_bytes = 2*1024*1024;
struct Switches {
    bool l3, l2, engram, borrowed_kv, hobbit;
    static Switches environment();
};
// Single producer / single consumer. Each worker has its own queue.
template<class T, size_t N> class Ring {
    static_assert(N > 1 && (N & (N-1)) == 0 && std::atomic<size_t>::is_always_lock_free);
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
    std::array<T,N> slots_{};
public:
    bool push(T v) { auto t=tail_.load(std::memory_order_relaxed);
        if(t-head_.load(std::memory_order_acquire)==N) return false;
        slots_[t%N]=v; tail_.store(t+1,std::memory_order_release); return true; }
    bool pop(T& v) { auto h=head_.load(std::memory_order_relaxed);
        if(h==tail_.load(std::memory_order_acquire)) return false;
        v=slots_[h%N]; head_.store(h+1,std::memory_order_release); return true; }
};
class HostArena {
    void* base_=nullptr; size_t bytes_=0;
    bool huge_=false, pinned_=false, numa_=false;
public:
    explicit HostArena(size_t bytes, int node=-1, bool pin=false);
    ~HostArena();
    HostArena(const HostArena&)=delete; HostArena& operator=(const HostArena&)=delete;
    void* data() const { return base_; } size_t size() const { return bytes_; }
    bool hugetlb() const { return huge_; } bool pinned() const { return pinned_; }
    bool numa_bound() const { return numa_; }
};
enum class Format { Q4_0, Dot8Iq3Sub32 };
// Q4_0: 18-byte blocks (fp16 d, then 16 low/high nibble pairs).
// Dot8Iq3Sub32: existing dot8_iq3_sub32_layout, 16 code bytes + f32 scale/block.
struct Matrix {
    const uint8_t* weights=nullptr; const uint8_t* scales=nullptr;
    uint32_t rows=0, cols=0; size_t weight_stride=0, scale_stride=0;
    Format format=Format::Q4_0;
    bool valid() const;
};
struct Activation { const int8_t* values=nullptr; const float* scales=nullptr; };
void gemv_reference(Matrix m, Activation x, float* y);
void gemv(Matrix m, Activation x, float* y); // AVX2 when available, scalar otherwise
struct Job { Matrix matrix; Activation x; float* y=nullptr; std::atomic<bool> done{false}; };
struct Node { int id; std::vector<int> cpus; };
std::vector<Node> host_nodes(); // allowed CPU affinity intersected with sysfs NUMA nodes
class CpuTier {
    struct Worker;
    std::vector<std::unique_ptr<Worker>> workers_;
public:
    explicit CpuTier(std::vector<Node> nodes=host_nodes());
    ~CpuTier();
    bool submit(size_t worker, Job& job); // caller owns inputs/output/job until done.acquire()
    size_t workers() const { return workers_.size(); }
    bool affinity_ok(size_t worker) const;
};
} // namespace vegajet::offload

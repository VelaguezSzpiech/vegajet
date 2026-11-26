// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Host-side expert streaming for models whose routed experts live in host RAM / NVMe instead of VRAM.
//
//   * ExpertStream reads a contiguous expert pack (core/expert_pack: header + index + 4 KiB-aligned records)
//     through the io_uring / O_DIRECT reader with its pread fallback;
//   * LruCache is the decode-only recency cache: a fixed number of staging slots, a hash map (expert key -> slot)
//     and an intrusive doubly linked list over the slots (MRU head, LRU tail), so lookup, hit-promotion, insert
//     and evict are all O(1) and allocation-free;
//   * during prefill the cache is bypassed by an explicit mode flag (Phase::Prefill): every expert goes straight
//     from the pack to the caller, nothing is promoted or evicted. That keeps the prefill working set from
//     flushing the decode working set and is why the cache policy is a phase, not a heuristic.
//
// Hook points for the GPU executor (not wired here):
//   1. per request, after the batch is classified: stream.set_phase(Phase::Prefill) while prompt tokens run and
//      stream.set_phase(Phase::Decode) for the decode loop;
//   2. per MoE layer, after routing: for every routed (layer, expert) call stream.expert(...) into the staging
//      buffer the expert GEMV reads, then upload;
//   3. the PLE / Engram tables use core/host_q4 (4-bit host packing) and core/ple_pregather (one contiguous
//      pre-gather per step); both are CPU-only and independent of this stream.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/expert_pack.hpp"

namespace vegajet::expertstream {

enum class Phase : uint8_t { Prefill = 0, Decode = 1 };
const char* phase_name(Phase p);

// Fixed-capacity recency cache. Keys are opaque 64-bit ids (e.g. layer << 32 | expert).
class LruCache {
public:
    explicit LruCache(uint32_t capacity);
    void set_phase(Phase p) { phase_ = p; }
    Phase phase() const { return phase_; }

    // Hit: slot holding key, promoted to MRU (-1 on miss). Bypassed (always -1, no mutation) in Prefill.
    int32_t get(uint64_t key);
    // Slot for key: existing slot promoted, a free slot, or the LRU slot (its key is dropped). Bypassed in Prefill.
    int32_t admit(uint64_t key);
    bool contains(uint64_t key) const;
    uint32_t capacity() const { return capacity_; }
    uint32_t size() const { return uint32_t(map_.size()); }
    std::vector<uint64_t> order() const;  // MRU -> LRU, for tests and diagnostics
    bool check() const;                   // intrusive list / map consistency
    struct Stats {
        uint64_t hits = 0, misses = 0, admits = 0, evictions = 0, bypassed = 0;
    };
    const Stats& stats() const { return st_; }

private:
    struct Slot {
        uint64_t key = 0;
        int32_t prev = -1, next = -1;
        bool used = false;
    };
    void unlink(int32_t s);
    void push_front(int32_t s);

    uint32_t capacity_;
    Phase phase_ = Phase::Decode;
    std::vector<Slot> slots_;
    std::unordered_map<uint64_t, int32_t> map_;
    int32_t head_ = -1, tail_ = -1;
    uint32_t used_ = 0;
    Stats st_;
};

class ExpertStream {
public:
    ExpertStream();
    ~ExpertStream();
    ExpertStream(const ExpertStream&) = delete;
    ExpertStream& operator=(const ExpertStream&) = delete;

    bool open(const std::string& pack_path, uint32_t cache_slots, std::string* err);
    bool open(const std::string& pack_path, uint32_t cache_slots, const expertpack::PackOptions& opt,
              std::string* err);
    bool is_open() const { return reader_.is_open(); }

    void set_phase(Phase p);
    Phase phase() const { return cache_.phase(); }

    // Fills gate/up/down (desc().gate_elems / up_elems / down_elems floats each).
    bool expert(uint32_t layer, uint32_t expert, float* gate, float* up, float* down, std::string* err);

    const expertpack::PackDesc& desc() const { return reader_.desc(); }
    LruCache& cache() { return cache_; }
    const LruCache& cache() const { return cache_; }
    expertpack::PackReader& reader() { return reader_; }
    struct Stats {
        uint64_t pack_reads = 0, cache_hits = 0, cache_misses = 0;
    };
    const Stats& stats() const { return st_; }

private:
    float* slot(int32_t s) { return slots_.data() + size_t(s) * row_elems_; }
    expertpack::PackReader reader_;
    LruCache cache_;
    std::vector<float> slots_;
    uint32_t row_elems_ = 0;
    Stats st_;
};

}  // namespace vegajet::expertstream

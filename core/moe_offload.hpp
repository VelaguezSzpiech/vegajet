// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// MoE expert offload: fixed, page-locked host staging for experts that do not fit in VRAM, a deterministic
// hot/cold split, and one- or two-card VRAM watermarks.
//
// Layout. An ExpertTable names a file holding n_layers * n_experts expert blobs of expert_bytes each
// (expert (l, e) lives at file_offset + (l * n_experts + e) * expert_bytes). The engine needs an expert's bytes on
// the host before its layer runs, and the same expert usually recurs across decode steps, so MoeOffload keeps a
// fixed pool of `staging_slots` page-aligned buffers in one mmap'ed, mlock()ed arena and serves fetch() from there.
// Nothing is allocated per read: the arena, the slot table, the key -> slot map, the read batch (iovecs and
// ReadRequests) and the completion buffer are all sized once in the constructor.
//
// I/O. prefetch() drops experts that are already staged, sorts the rest by expert id and coalesces every run of
// file-adjacent experts into one vectored read (one iovec per slot), so a layer of 256 experts becomes a handful of
// reads. Reads go through an async engine: raw-syscall Linux io_uring (IORING_OP_READV, no liburing dependency)
// when the kernel provides it, and a preadv worker pool otherwise. Both re-read short tails and retry EINTR; a
// runtime io_uring_enter() failure degrades the remaining work to inline preadv instead of losing completions.
// Slots whose read is outstanding are never evicted, and teardown drains every completion before the arena is
// unlocked and unmapped, so no buffer can be written after it is released.
//
// Deterministic hot/cold. An expert is hot iff hot_mask[key] != 0 (when given) or expert < hot_per_layer. The
// classification is a pure function of the config, never of access history, so two runs admit the same experts.
// The VRAM tier is planned deterministically too: hot experts are visited layer-major, expert-major and placed
// first-fit onto a card that is still below its high watermark.
//
// VRAM watermarks. Each CardWatermark has a capacity (expert slots), a high mark (promotion fills up to it) and a
// low mark (once residency exceeds the high mark it is trimmed down to the low mark). Watermarks give hysteresis:
// capacity may be raised and lowered within [low, high] without re-planning. resize_card() and set_watermarks()
// demote deterministically (most recently admitted first) until the low mark is reached.
//
// Release. mlock()ed pages cannot be returned with MADV_DONTNEED (the kernel rejects it for VM_LOCKED), so
// release(), release_all() and the destructor munlock the range, MADV_DONTNEED it (releasing the physical pages)
// and re-lock only what stays pinned.
//
// Threading. MoeOffload is not internally synchronized: call prefetch()/fetch()/poll()/release() from one thread
// (the decode loop). The I/O engine has its own worker threads or ring and is safe across them; a slot with an
// outstanding read is never handed to another caller and the destructor waits for every completion before the
// arena is unlocked and unmapped.
#pragma once

#include <sys/uio.h>
#include "core/offload/policies.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vegajet::moeoffload {

enum class IoKind : uint8_t {
    Auto = 0,    // io_uring when it can be set up, preadv otherwise
    IoUring = 1, // prefer io_uring, still fall back to preadv if setup fails
    Preadv = 2,  // force the worker pool
};

struct ExpertTable {
    int fd = -1;              // opened by the caller, read-only
    uint64_t file_offset = 0; // byte offset of expert (0, 0)
    uint32_t n_layers = 0;
    uint32_t n_experts = 0;
    uint32_t expert_bytes = 0;

    uint64_t expert_offset(uint32_t layer, uint32_t expert) const {
        return file_offset + (static_cast<uint64_t>(layer) * n_experts + expert) * expert_bytes;
    }
};

struct CardWatermark {
    uint32_t capacity = 0; // expert slots on the card
    uint32_t high = 0;     // fill / promote up to (0 means capacity)
    uint32_t low = 0;      // demote down to once residency exceeds high (clamped to high)
};

struct OffloadConfig {
    uint32_t staging_slots = 64;   // fixed pool of pinned host buffers
    uint32_t max_run_rows = 16;    // experts per vectored read (clamped to IOV_MAX)
    uint32_t max_in_flight = 0;    // outstanding read requests; 0 means staging_slots
    uint32_t hot_per_layer = 0;    // deterministic hot prefix per layer
    std::vector<uint8_t> hot_mask; // optional n_layers * n_experts override (1 = hot)
    std::vector<CardWatermark> cards; // 0, 1 or 2 VRAM cards
    uint32_t io_entries = 64;      // io_uring SQ entries to request
    uint32_t threads = 2;          // preadv fallback workers
    IoKind io = IoKind::Auto;
};

struct OffloadStats {
    uint64_t requested = 0;   // rows passed to prefetch()
    uint64_t issued = 0;      // rows actually read
    uint64_t deduped = 0;     // rows skipped because staged or already loading
    uint64_t read_ops = 0;    // vectored reads handed to the engine
    uint64_t bytes = 0;       // bytes read
    uint64_t hits = 0;        // fetch() served a ready staged copy (no I/O)
    uint64_t misses = 0;      // fetch() had to read, wait or read directly
    uint64_t evictions = 0;   // staged rows dropped to make room
    uint64_t deferred = 0;    // rows left unstaged (no slot, queue full, bad id)
    uint64_t errors = 0;      // failed reads and bad arguments
    uint64_t short_reads = 0; // runs that ended at end of file
    uint64_t completions = 0; // reaped read completions
    uint64_t releases = 0;    // release()/release_all() calls that dropped pages
    uint64_t released_bytes = 0;
};

struct Placement {
    uint32_t layer = 0;
    uint32_t expert = 0;
    uint8_t card = 0;
};

// ---- internal batch plumbing (public only because MoeOffload owns preallocated copies)
struct ReadRequest {
    int fd = -1;
    uint64_t offset = 0;
    const iovec* iov = nullptr;
    uint32_t iov_count = 0;
    uint64_t user = 0; // echoed in IoCompletion::user
};

struct IoCompletion {
    uint64_t user = 0;
    int64_t result = 0; // bytes read, or -errno
};

class MoeOffload {
public:
    MoeOffload(ExpertTable table, OffloadConfig cfg, std::string* why = nullptr);
    ~MoeOffload();
    MoeOffload(const MoeOffload&) = delete;
    MoeOffload& operator=(const MoeOffload&) = delete;

    bool ok(std::string* err = nullptr) const;
    const char* backend_name() const;
    bool staging_pinned() const;
    size_t staging_bytes() const;
    const void* staging_base() const { return arena_; } // read-only identity for OS residency diagnostics

    uint32_t n_layers() const { return t_.n_layers; }
    uint32_t n_experts() const { return t_.n_experts; }
    uint32_t expert_bytes() const { return t_.expert_bytes; }

    // ---- deterministic hot / cold classification
    bool is_hot(uint32_t layer, uint32_t expert) const;
    uint32_t hot_per_layer() const { return cfg_.hot_per_layer; }
    uint64_t hot_total() const;

    // ---- asynchronous staging
    // Starts reads for experts that are neither staged nor loading. Returns the number of rows issued.
    size_t prefetch(uint32_t layer, const uint32_t* experts, size_t n);
    void poll();                 // applies finished reads without blocking
    size_t in_flight() const;    // rows with an outstanding read
    bool staged(uint32_t layer, uint32_t expert) const; // resident and ready

    // ---- data
    // Copies expert_bytes to dst, staging the expert first if needed. False on bad id / I/O error.
    bool fetch(uint32_t layer, uint32_t expert, uint8_t* dst, std::string* err = nullptr);

    // HOBBIT: full for row zero/hot experts. Cold draft rows may use an already
    // staged lower-precision table; formats/byte counts belong to their callers.
    // Missing low data falls back to full, never skips an expert. No GPU I/O.
    bool fetch_spec(uint32_t layer, uint32_t expert, uint32_t row, uint32_t rank,
                    double weight, MoeOffload* low, uint8_t* dst, size_t capacity,
                    offload::FetchPrecision* precision, std::string* err = nullptr);

    // ---- release
    bool release(uint32_t layer, uint32_t expert); // munlock + MADV_DONTNEED + lock-on-fault of that slot
    void release_all();                            // drop every staged row

    // ---- VRAM tier
    std::vector<Placement> plan() const; // deterministic hot placement up to the high marks
    void replan();                       // reset residency and apply plan()
    void set_watermarks(const std::vector<CardWatermark>& cards);
    uint32_t resize_card(uint32_t card, uint32_t capacity); // returns rows demoted
    int32_t resident_card(uint32_t layer, uint32_t expert) const; // -1 when not resident
    uint32_t resident_on(uint32_t card) const;
    uint32_t cards() const { return n_cards_; }

    const OffloadStats& stats() const { return st_; }

private:
    class Engine;
    enum class SlotState : uint8_t { Free = 0, Loading = 1, Ready = 2, Error = 3 };
    struct Slot {
        SlotState state = SlotState::Free;
        uint32_t layer = 0;
        uint32_t expert = 0;
        uint32_t run_pos = 0; // position inside its read run
        uint64_t run = 0;     // run id echoed by the engine
        int64_t error = 0;
        int32_t lru_prev = -1;
        int32_t lru_next = -1;
    };

    bool init_arena(std::string* why);
    uint64_t key(uint32_t layer, uint32_t expert) const {
        return static_cast<uint64_t>(layer) * t_.n_experts + expert;
    }
    void lru_unlink(uint32_t slot);
    void lru_push_back(uint32_t slot);
    void lru_touch(uint32_t slot);
    int32_t take_slot(); // free slot or the LRU victim; -1 if every slot is loading
    void drop_slot(uint32_t slot);

    size_t stage(uint32_t layer, const uint32_t* experts, size_t n, bool count_requested);
    void drain(int wait_ms);
    void apply(const IoCompletion& c);
    bool read_direct(uint32_t layer, uint32_t expert, uint8_t* dst, std::string* err);

    void sanitize_cards();
    uint32_t enforce_watermarks(uint32_t card);
    bool admit(uint32_t card, uint64_t k);

    ExpertTable t_;
    OffloadConfig cfg_;
    std::string init_error_;
    std::unique_ptr<Engine> eng_;

    uint64_t nkeys_ = 0;
    size_t slot_stride_ = 0;
    uint8_t* arena_ = nullptr;
    size_t arena_bytes_ = 0;
    bool pin_requested_ = false;
    bool pinned_ = false;

    std::vector<Slot> slots_;
    std::vector<int32_t> key_slot_;
    std::vector<uint32_t> free_;
    int32_t lru_head_ = -1, lru_tail_ = -1;
    size_t in_flight_slots_ = 0;
    uint64_t next_run_ = 1;
    uint64_t outstanding_ops_ = 0;

    std::vector<uint32_t> want_;
    std::vector<iovec> batch_iov_;
    std::vector<ReadRequest> batch_ops_;
    std::vector<IoCompletion> completions_;

    std::vector<int8_t> res_card_;
    std::array<std::vector<uint32_t>, 2> card_keys_;
    std::vector<CardWatermark> wm_;
    uint32_t n_cards_ = 0;

    OffloadStats st_;
};

} // namespace vegajet::moeoffload

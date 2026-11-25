// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Asynchronous row prefetcher for embedding-style tables kept on NVMe / page cache instead of VRAM: per-layer
// embeddings (PLE, one row per token id) and Engram-style hashed n-gram memories (one row per hash bucket).
//
// The engine knows which rows it needs before it needs them (prompt tokens during prefill, draft tokens and the
// hashed n-grams they form during speculative decoding), so it calls prefetch(rows) early and fetch(row) when the
// layer runs. prefetch() drops resident / in-flight rows, sorts the rest, and turns every run of adjacent row ids
// (contiguous in the file) into ONE vectored read whose iovecs scatter the rows into their own cache slots
// (preadv: one syscall per run, no bounce copy). Slots are recycled LRU; rows being read are never evicted.
//
// I/O goes through IoBackend (submit / reap). ReadOp and Completion map 1:1 onto an io_uring IORING_OP_READV SQE
// (fd, offset, iovec array, user_data) and its CQE (user_data, res). Backends: IoUringBackend (raw-syscall
// io_uring), PreadvBackend (thread pool), SyncBackend (inline, for tests and tiny tables); make_io_backend() picks
// io_uring when the kernel allows it and falls back to the thread pool otherwise.
//
// Row formats (all little-endian):
//   F16:    row_elems fp16 values                                        row_bytes = 2 * elems
//   FP8:    fp16 scale, then row_elems E4M3 bytes (value = scale * e4m3)  row_bytes = 2 + elems
//   Q4:     ggml Q4_0 blocks: per 32 elements fp16 d + 16 bytes; byte j holds element j (low nibble) and j + 16
//           (high nibble), value = d * (nibble - 8)                         row_bytes = 18 * elems / 32
// F16 and Q4 are the GGUF F16 / Q4_0 encodings, so a GGUF table tensor of those types is streamed as stored
// (models/model_blueprints.cpp: table_desc()).
#pragma once

#include <sys/uio.h>

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace vegajet::prefetch {

enum class RowFormat : uint8_t { F16 = 0, FP8 = 1, Q4 = 2 };
uint32_t row_bytes_for(RowFormat f, uint32_t elems);  // 0 if elems is invalid for the format
float e4m3_to_f32(uint8_t b);                         // OCP FP8 E4M3 (bias 7, no inf, S.1111.111 = NaN)
float f16_to_f32(uint16_t h);
bool decode_row(RowFormat f, const uint8_t* row, uint32_t elems, float* out);

struct TableDesc {
    int fd = -1;               // opened read-only by the caller (O_DIRECT is the caller's choice)
    uint64_t file_offset = 0;  // byte offset of row 0
    uint64_t n_rows = 0;
    uint32_t row_elems = 0;
    RowFormat fmt = RowFormat::FP8;
    uint32_t row_bytes() const { return row_bytes_for(fmt, row_elems); }
};

// ---- I/O backends
struct ReadOp {
    int fd = -1;
    uint64_t offset = 0;
    std::vector<iovec> iov;    // scatter list (one entry per row of the run)
    uint64_t user_data = 0;
};
struct Completion {
    uint64_t user_data = 0;
    int64_t result = 0;        // bytes read, or -errno
};

class IoBackend {
public:
    virtual ~IoBackend() = default;
    virtual void submit(std::vector<ReadOp> ops) = 0;
    // Moves finished completions into `out`; waits up to wait_ms (0: poll) for at least one when none is ready.
    virtual size_t reap(std::vector<Completion>& out, int wait_ms) = 0;
    virtual const char* name() const = 0;
};

class SyncBackend final : public IoBackend {
public:
    void submit(std::vector<ReadOp> ops) override;
    size_t reap(std::vector<Completion>& out, int wait_ms) override;
    const char* name() const override { return "sync-preadv"; }

private:
    std::vector<Completion> done_;
};

class PreadvBackend final : public IoBackend {
public:
    explicit PreadvBackend(uint32_t threads = 4);
    ~PreadvBackend() override;
    void submit(std::vector<ReadOp> ops) override;
    size_t reap(std::vector<Completion>& out, int wait_ms) override;
    const char* name() const override { return "threadpool-preadv"; }

private:
    void loop();
    std::mutex mu_;
    std::condition_variable work_cv_, done_cv_;
    std::deque<ReadOp> queue_;
    std::vector<Completion> done_;
    std::vector<std::thread> threads_;
    bool stop_ = false;
};

// Linux io_uring (raw syscalls, no liburing): every ReadOp becomes one IORING_OP_READV SQE (at most IOV_MAX iovecs
// per SQE). Short reads are resubmitted for the remainder until the op is complete or hits end of file, -EINTR /
// -EAGAIN completions are resubmitted, and ops beyond the ring capacity wait in a backlog, so in-flight SQEs never
// exceed the CQ size (no CQ overflow). Thread-safe: submit() and reap() may run on different threads; waiting is a
// poll() on the ring fd with the lock released.
// Fallback: if setup fails (kernel without io_uring, io_uring_disabled sysctl, a seccomp profile returning EPERM),
// ok() is false and make_io_backend() picks the thread pool instead. If io_uring_enter() fails later with a non-retry
// error, the backend degrades for good: SQEs the kernel did not take and every later op run inline through
// preadv_full(), and SQEs already in flight still complete through the CQ. A READV completion of -EINVAL /
// -EOPNOTSUPP (fd type without READV support) on an op with nothing read yet is retried through preadv_full() too.
struct IoUringOptions {
    uint32_t entries = 256;            // SQ entries (clamped by the kernel); CQ gets twice as many
    // Test hooks. max_bytes_per_sqe > 0 splits every read into SQEs of at most that many bytes, so the short-read
    // resubmission path runs on every op; simulate_* force the setup / runtime fallbacks.
    uint32_t max_bytes_per_sqe = 0;
    bool simulate_unavailable = false;  // setup fails with ENOSYS
    int simulate_enter_errno = 0;       // the first io_uring_enter() fails with this errno
};

struct IoUringStats {
    uint64_t ops = 0, sqes = 0, cqes = 0, resubmits = 0, retries = 0, enter_calls = 0, inline_ops = 0;
    uint32_t max_in_flight = 0;
    bool degraded = false;
};

class IoUringBackend final : public IoBackend {
public:
    explicit IoUringBackend(IoUringOptions opt = {});
    ~IoUringBackend() override;
    IoUringBackend(const IoUringBackend&) = delete;
    IoUringBackend& operator=(const IoUringBackend&) = delete;

    bool ok(std::string* why = nullptr) const;  // false: setup failed; submit() then runs every op inline
    void submit(std::vector<ReadOp> ops) override;
    size_t reap(std::vector<Completion>& out, int wait_ms) override;
    const char* name() const override;          // "io_uring", or "io_uring-degraded-preadv" after a runtime failure
    IoUringStats stats() const;
    uint32_t sq_entries() const;

private:
    struct Ring;
    struct Pending;
    void pump();            // backlog -> SQEs -> io_uring_enter (lock held)
    void harvest();         // CQEs -> completions / resubmits (lock held)
    void degrade(int err);  // switch to inline preadv (lock held)
    void complete_inline(Pending& p);

    IoUringOptions opt_;
    std::unique_ptr<Ring> ring_;
    std::string error_;
    mutable std::mutex mu_;
    std::deque<std::unique_ptr<Pending>> backlog_;
    std::unordered_map<uint64_t, std::unique_ptr<Pending>> in_flight_;  // tag -> op with one SQE outstanding
    std::vector<Pending*> unsubmitted_;  // SQEs written to the ring but not yet taken by io_uring_enter()
    std::vector<Completion> done_;
    uint64_t next_tag_ = 1;
    IoUringStats st_;
};

enum class BackendKind : uint8_t { Auto = 0, IoUring = 1, ThreadPool = 2, Sync = 3 };
// Auto: io_uring when it can be set up and the environment does not say VJ_IO_URING=0, else the thread pool.
// IoUring without a working io_uring also falls back to the thread pool. `why` names the chosen backend and the
// reason for any fallback.
std::unique_ptr<IoBackend> make_io_backend(BackendKind kind = BackendKind::Auto, uint32_t threads = 4,
                                           std::string* why = nullptr, IoUringOptions opt = {});

// Full read of one ReadOp with preadv, retrying EINTR and short reads; returns bytes or -errno.
int64_t preadv_full(const ReadOp& op);

// ---- prefetcher
struct PrefetchConfig {
    uint32_t slots = 4096;            // resident rows
    uint32_t max_run_rows = 64;       // rows per vectored read (<= IOV_MAX)
};

struct PrefetchStats {
    uint64_t requested = 0;   // rows passed to prefetch()
    uint64_t issued = 0;      // rows actually read
    uint64_t deduped = 0;     // rows skipped because resident or already in flight
    uint64_t read_ops = 0;    // vectored reads (syscalls for preadv backends)
    uint64_t bytes = 0;
    uint64_t hits = 0, misses = 0;  // fetch() found the row resident / had to wait or read
    uint64_t evictions = 0, errors = 0, deferred = 0;  // deferred: no free slot at prefetch time
};

class RowPrefetcher {
public:
    RowPrefetcher(TableDesc table, PrefetchConfig cfg, std::unique_ptr<IoBackend> io);
    ~RowPrefetcher();
    RowPrefetcher(const RowPrefetcher&) = delete;
    RowPrefetcher& operator=(const RowPrefetcher&) = delete;

    bool ok(std::string* err = nullptr) const;
    // Starts reads for rows that are neither resident nor in flight. Returns rows newly issued.
    size_t prefetch(const uint32_t* rows, size_t n);
    // Copies the row's bytes to dst (row_bytes), reading it now if prefetch() did not. False on I/O error / bad id.
    bool fetch(uint32_t row, uint8_t* dst, std::string* err = nullptr);
    bool fetch_f32(uint32_t row, float* dst, std::string* err = nullptr);
    void poll();  // applies finished reads without blocking
    size_t in_flight() const { return in_flight_rows_; }
    const PrefetchStats& stats() const { return st_; }
    const char* backend_name() const { return io_->name(); }

private:
    enum class SlotState : uint8_t { Free, Loading, Ready, Error };
    struct Slot {
        SlotState state = SlotState::Free;
        uint32_t row = 0;
        int64_t error = 0;
        std::list<uint32_t>::iterator lru;  // valid while Ready
    };
    int32_t take_slot();
    void apply(const Completion& c);
    void drain(int wait_ms);

    TableDesc t_;
    PrefetchConfig cfg_;
    std::unique_ptr<IoBackend> io_;
    std::string init_error_;
    uint32_t rb_ = 0;
    std::vector<uint8_t> arena_;
    std::vector<Slot> slots_;
    std::vector<uint32_t> free_;
    std::list<uint32_t> lru_;                     // Ready slots, least recent first
    std::unordered_map<uint32_t, uint32_t> where_;  // row -> slot (Loading, Ready or Error)
    std::unordered_map<uint64_t, std::vector<uint32_t>> runs_;  // user_data -> slots of that read
    uint64_t next_op_ = 1;
    size_t in_flight_rows_ = 0;
    PrefetchStats st_;
};

// Engram-style bucket of the n-gram ending at tokens[n-1] (n tokens), in [0, n_rows). Deterministic per seed.
uint64_t ngram_row(const int32_t* tokens, uint32_t n, uint64_t seed, uint64_t n_rows);

} // namespace vegajet::prefetch

// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Multi-instance swarm router core: N independent workers (each an N=1 `vegajet serve`, one GPU or TP group each),
// one bounded queue, least-busy + prefix-affinity routing, per-request cancellation. Transport-free (the HTTP relay
// in tools/swarm_router.py or an embedding front end drives it); thread-safe.
//
// A request is keyed by chained block hashes of its prompt (block_hashes(): tokens or rendered text, fixed-size
// blocks, h[k] = H(h[k-1], block k)), so two requests share a prefix of k blocks exactly when their first k hashes
// match. Every worker remembers the hashes of the prompts it was given (bounded LRU): those prefixes are warm in its
// KV / prefix cache.
//
// Dispatch (on submit, release, cancel, health change, and while callers wait): queued requests in priority order
// (higher first, then arrival) go to a healthy worker with a free slot: the least-loaded free workers first, the
// longest remembered prefix match among them, then rendezvous hashing (stable spread for cold prefixes); a more
// loaded free worker wins only when it matches at least affinity_min_gain more blocks. A request whose best
// match sits on a busy worker that is strictly better than every free one waits for it up to affinity_wait_ms, then
// takes the best free worker; requests behind it are not held up meanwhile. "Better" means at least
// affinity_min_gain more matched blocks: a short prefix every conversation shares (a system prompt) never makes
// requests queue behind one worker.
// cancel(): a queued request leaves the queue; an assigned one frees its slot at once and cancelled() turns true, so
// the relay can drop the upstream connection (the worker then stops generating). Workers marked unhealthy get no
// new requests and forget their prefixes (a restarted worker has a cold cache).
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace vegajet::swarm {

std::vector<uint64_t> block_hashes(const std::vector<int32_t>& tokens, uint32_t block_tokens = 64);
std::vector<uint64_t> block_hashes(std::string_view text, uint32_t block_bytes = 256);

struct WorkerSpec {
    std::string name;
    uint32_t slots = 1;  // concurrent requests (1 for an N=1 worker)
};

struct RouterConfig {
    size_t max_queue = 64;             // queued (not yet assigned) requests; beyond: rejected
    uint32_t affinity_wait_ms = 250;   // how long a request waits for a busy worker with a better prefix match
    uint32_t affinity_min_gain = 4;    // ... that matches at least this many more blocks than the best free worker
                                       // (a short shared system prompt is not worth waiting for)
    size_t max_prefixes_per_worker = 65536;
};

enum class Status : uint8_t { Unknown = 0, Queued, Assigned, Cancelled, Rejected };

struct WorkerState {
    std::string name;
    uint32_t slots = 1, active = 0;
    bool healthy = true;
    uint64_t served = 0, affinity_hits = 0;
    size_t prefixes = 0;
};

struct RouterStats {
    uint64_t submitted = 0, assigned = 0, affinity_hits = 0, affinity_waits = 0, cancelled = 0, rejected = 0,
             completed = 0;
};

class SwarmRouter {
public:
    using Clock = std::chrono::steady_clock;
    SwarmRouter(std::vector<WorkerSpec> workers, RouterConfig cfg = {});

    // Returns a ticket (> 0), or 0 when rejected (queue full or no healthy worker).
    uint64_t submit(std::vector<uint64_t> prefix_hashes, int32_t priority = 0);
    // Blocks until the ticket is assigned (returns the worker index) or cancelled / unknown / timed out (-1).
    int wait(uint64_t ticket, std::chrono::milliseconds timeout);
    // Non-blocking: dispatches, then reports.
    Status status(uint64_t ticket, int* worker = nullptr);
    bool cancel(uint64_t ticket);
    bool cancelled(uint64_t ticket) const;
    // The request finished (or failed); frees its slot. No-op for cancelled tickets.
    void release(uint64_t ticket);
    void set_healthy(size_t worker, bool healthy);
    // Matched prefix blocks of the ticket's prompt on its worker at assignment.
    uint32_t matched_blocks(uint64_t ticket) const;

    std::vector<WorkerState> workers() const;
    RouterStats stats() const;
    size_t queued() const;
    // active <= slots everywhere; every assigned ticket is counted on exactly one worker.
    bool check_invariants() const;

private:
    struct Req {
        std::vector<uint64_t> hashes;
        int32_t priority = 0;
        uint64_t seq = 0;
        Clock::time_point enqueued;
        Status status = Status::Queued;
        int worker = -1;
        uint32_t matched = 0;
        bool waited = false;
    };
    struct Worker {
        WorkerSpec spec;
        uint32_t active = 0;
        bool healthy = true;
        uint64_t served = 0, affinity_hits = 0;
        std::unordered_set<uint64_t> prefixes;
        std::deque<uint64_t> order;  // insertion order for LRU-ish eviction
    };
    uint32_t match_locked(const Worker& w, const std::vector<uint64_t>& h) const;
    void remember_locked(Worker& w, const std::vector<uint64_t>& h);
    // Assigns what it can; returns the earliest affinity deadline still pending (or max()).
    Clock::time_point dispatch_locked(Clock::time_point now);

    RouterConfig cfg_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::vector<Worker> w_;
    std::unordered_map<uint64_t, Req> reqs_;
    std::vector<uint64_t> queue_;  // queued tickets
    uint64_t next_ = 1, seq_ = 0;
    RouterStats st_;
};

} // namespace vegajet::swarm

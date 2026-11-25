// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Continuous batching scheduler for B = 1..8 concurrent sequences over core/prefix_cache (radix prefix cache,
// refcounted KV blocks, recurrent-state snapshots). Pure CPU: it plans engine steps, the engine executes them.
//
// Each step:
//   1. admission: waiting requests in priority order (higher first, then arrival) take free slots; each attaches the
//      longest cached prefix of its context (blocks shared by reference; hybrid models restore the matching
//      recurrent-state snapshot, otherwise the slot state is reset). A request whose prompt shares at least two blocks
//      with a prompt still being prefilled waits for it (prefix-aware admission). A higher-priority request may
//      preempt the lowest-priority running one when every slot is busy;
//   2. decode first: one input token per sequence whose prompt is done (latency), in priority order;
//   3. chunked prefill: remaining budget goes to prompt chunks (<= prefill_chunk tokens each) in priority order, so
//      long prompts interleave with decode. Each finished chunk publishes its block-aligned prefix to the cache. On
//      hybrid models chunks are cut exactly at snapshot points (every snapshot_interval tokens and the last
//      block-aligned prompt position); the state there is saved into a snapshot (post op) and cached with the blocks;
//   4. KV blocks are allocated per block boundary; when the pool is empty the cache evicts LRU leaves, then the
//      lowest-priority not-yet-planned sequence is preempted (KV released, prompt + generated tokens requeued; its
//      prefix stays cached, so the recompute is mostly hits). A sequence that cannot fit alone ends ContextFull.
// Cancellation (client disconnect) is thread-safe and immediate: a waiting request is dropped at once; a running
// one releases its KV and snapshot references at once, or, while an engine step is executing, as soon as that step
// returns (its results are discarded). Refcounts are audited by check_invariants(): every block reference is held
// by exactly one live sequence table or cache node, so nothing leaks.
#pragma once

#include "core/prefix_cache.hpp"

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace vegajet::batch {

using prefix::BlockId;
using prefix::SnapId;

struct BatchConfig {
    uint32_t max_batch = 8;        // concurrent sequences, 1..8
    uint32_t block_tokens = 16;
    uint32_t kv_blocks = 1024;
    uint32_t max_snapshots = 32;
    bool hybrid = false;           // recurrent layers: a prefix resumes only from a state snapshot
    bool prefix_cache = true;
    uint32_t step_tokens = 512;    // tokens per engine step (decode inputs + prefill chunks)
    uint32_t prefill_chunk = 256;
    bool priority_preemption = true;
    uint32_t snapshot_interval = 512;  // hybrid: also snapshot every this many prompt tokens (block aligned; 0 = only
                                       // at the end of the prompt), so requests diverging inside a shared prefix resume
    bool prefix_aware_admission = true;  // defer a request whose prompt shares >= 2 blocks with a prompt still being
                                         // prefilled, so the shared prefix is computed once
};

struct Request {
    uint64_t id = 0;
    std::vector<int32_t> prompt;
    uint32_t max_new = 16;
    int32_t priority = 0;          // higher runs first
    int32_t eos = -1;
};

enum class End : uint8_t { None = 0, Eos, Length, Cancelled, ContextFull, Rejected };
const char* end_name(End e);

struct Result {
    std::vector<int32_t> tokens;
    End end = End::None;
    uint32_t prefix_hit_tokens = 0;  // prompt tokens served from the cache (all admissions)
    uint32_t prefilled_tokens = 0;   // tokens the engine actually prefilled
    uint32_t preemptions = 0;
};

struct StateOp {
    enum class Kind : uint8_t { Reset, Restore, Save } kind = Kind::Reset;
    uint32_t slot = 0;
    SnapId snap = -1;
    uint32_t tokens = 0;           // Restore / Save: state after this many tokens of the sequence
};

struct SlotWork {
    uint64_t req = 0;
    uint32_t slot = 0;
    uint32_t pos0 = 0;             // position of tokens[0]
    std::vector<int32_t> tokens;   // inputs whose K/V this step writes
    std::vector<BlockId> block_table;  // covers [0, pos0 + tokens.size())
    bool sample = false;           // the engine returns one sampled token for this slot
};

struct StepPlan {
    std::vector<StateOp> pre;      // before the work (Reset / Restore)
    std::vector<SlotWork> work;
    std::vector<StateOp> post;     // after the work (Save)
};

struct StepOut {
    std::vector<std::pair<uint32_t, int32_t>> sampled;  // (slot, token) for every work item with sample = true
};

using Engine = std::function<StepOut(const StepPlan&)>;

struct BatchStats {
    uint64_t steps = 0, prefill_tokens = 0, decode_tokens = 0, prefix_hit_tokens = 0;
    uint64_t preemptions = 0, cancelled = 0, finished = 0;
    uint32_t peak_batch = 0;
};

class ContinuousBatcher {
public:
    explicit ContinuousBatcher(BatchConfig cfg);
    ~ContinuousBatcher();
    // False if the id is in use or the request is empty.
    bool submit(Request r);
    // Thread-safe. True if the request was waiting or running (now cancelled).
    bool cancel(uint64_t id);
    // Plans and runs one engine step; false when there is nothing to do.
    bool step(const Engine& engine);
    void run_until_idle(const Engine& engine, uint64_t max_steps = 1000000);

    bool done(uint64_t id) const;
    Result result(uint64_t id) const;
    uint32_t active() const;
    uint32_t waiting() const;
    BatchStats stats() const;
    prefix::CacheStats cache_stats() const;
    // Every block reference held by a live sequence or the cache, nothing else; snapshot slots likewise.
    bool check_invariants() const;
    // Drops all unpinned cache entries (tests: afterwards an idle batcher holds no block or snapshot).
    void clear_cache();
    uint32_t free_blocks() const;
    uint32_t free_snapshots() const;
    const BatchConfig& config() const { return cfg_; }

private:
    struct Seq;
    void admit_locked(StepPlan& plan);
    bool ensure_blocks_locked(Seq& s, uint32_t tokens, const std::vector<Seq*>& planned);
    void preempt_locked(Seq& s);
    void release_locked(Seq& s, End end);
    void cache_prefix_locked(Seq& s);
    Seq* pick_victim_locked(const Seq& for_seq, const std::vector<Seq*>& planned);
    uint32_t next_snapshot_point(uint32_t kv, uint32_t prompt_end) const;
    bool should_defer_locked(const Seq& cand) const;

    BatchConfig cfg_;
    mutable std::mutex mu_;
    std::unique_ptr<prefix::BlockPool> pool_;
    std::unique_ptr<prefix::RadixCache> cache_;
    std::map<uint64_t, std::unique_ptr<Seq>> seqs_;   // waiting + running
    std::map<uint64_t, Result> results_;              // finished
    std::vector<Seq*> slots_;                         // max_batch entries
    uint64_t arrival_ = 0;
    bool in_step_ = false;
    BatchStats st_;
};

} // namespace vegajet::batch

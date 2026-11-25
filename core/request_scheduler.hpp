// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Continuous-batching request scheduler (pure CPU): many requests share one engine step, KV lives in core/paged_kv
// (prefix-cache reuse), speculative draft depth comes from core/evict_controller.
//
// Each request has a context ctx = prompt + generated tokens and a KV length L (tokens whose K/V are written). Per
// step the scheduler decides, from L alone:
//   decode   generated tokens exist and L == |ctx| - 1: input = [last token] + K draft tokens, the engine returns the
//            accepted drafts + one bonus token; rejected drafts are rolled back with PagedKv::truncate();
//   prefill  otherwise: the next chunk of ctx[L..] (<= prefill_chunk tokens); the chunk that reaches |ctx| samples a
//            token (for a new request the first output token, for a preempted one the next token, exactly what the
//            missed decode step would have produced).
// Step budget: decode inputs first, oldest first (latency), then prefill chunks oldest first (chunked prefill
// interleaved with decode); at most step_token_budget tokens and max_slots active sequences.
// Admission: FCFS from the waiting queue (ordered by arrival, so preempted requests come back first); a new sequence
// first attaches the longest cached block prefix of its ctx (shared system prompts are prefilled once).
// OOM safety: KV is reserved with PagedKv::append() when the step is planned. If blocks run out: drafts are dropped
// first, then the youngest running sequence is preempted (KV released, recomputed later; its full blocks stay in the
// prefix cache, so the recompute is mostly prefix hits). The oldest sequence is never preempted by a younger one, so
// it always makes progress. A sequence that cannot fit even alone ends with Failed (prompt larger than the KV pool)
// or ContextFull (generation filled the pool), never a crash or a livelock.
// Cancellation (client disconnect) is a thread-safe flag applied at the next schedule(): KV released, slot freed.
//
// ServeLoop drives a scheduler + BatchEngine on its own thread and exposes an OpenAiServer GenerateFn; run the
// server with ServerOptions::workers >= max_slots so requests reach the scheduler concurrently.
#pragma once

#include "core/evict_controller.hpp"
#include "core/paged_kv.hpp"
#include "core/server.hpp"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace vegajet::sched {

struct SchedulerConfig {
    uint32_t max_slots = 4;            // concurrently active (KV-holding) sequences
    uint32_t step_token_budget = 256;  // tokens per engine step (prefill chunks + decode inputs + drafts), >= max_slots
    uint32_t prefill_chunk = 128;      // prompt tokens one sequence prefills per step
    uint32_t kv_blocks = 256;
    uint32_t block_tokens = 16;
    bool speculative = true;           // draft depth from the EvictController (needs a drafter)
    uint32_t admit_reserve_blocks = 1; // admit only if this many blocks are free or evictable beyond the first chunk
};

enum class EndReason : uint8_t { None = 0, Eos = 1, Length = 2, ContextFull = 3, Cancelled = 4, Failed = 5 };
const char* end_reason_name(EndReason r);

struct WorkItem {
    uint64_t req = 0;
    paged::SeqId seq = 0;
    uint32_t pos0 = 0;               // ctx position of tokens[0]
    std::vector<int32_t> tokens;     // inputs whose KV this step writes
    uint32_t n_draft = 0;            // decode: trailing draft tokens in `tokens`
    bool sample = false;             // the engine must return sampled tokens (decode, or the last prefill chunk)
    bool decode = false;
};
struct StepPlan {
    std::vector<WorkItem> items;
    uint32_t tokens = 0;
    std::vector<paged::CopyOp> copies;  // KV block copies (COW) the engine applies before the step
};
struct StepOutput {
    uint64_t req = 0;
    std::vector<int32_t> tokens;     // sample: 1 + accepted drafts (accepted drafts first, then the bonus token)
};

struct Delivery {
    uint64_t req = 0;
    std::vector<int32_t> tokens;     // newly generated tokens
    EndReason end = EndReason::None; // != None: the request is finished (and forgotten by the scheduler)
    std::string error;
};

struct SchedulerStats {
    uint64_t steps = 0, prefill_tokens = 0, decode_steps = 0, generated = 0;
    uint64_t drafts_proposed = 0, drafts_accepted = 0, prefix_hit_tokens = 0, preemptions = 0, draft_drops = 0;
    uint64_t finished = 0, cancelled = 0, failed = 0, context_full = 0;
    uint32_t max_active = 0, max_step_tokens = 0, max_decode_batch = 0;
};

// Proposes up to k draft tokens continuing ctx (may return fewer).
using Drafter = std::function<std::vector<int32_t>(uint64_t req, const std::vector<int32_t>& ctx, uint32_t k)>;

class RequestScheduler {
public:
    explicit RequestScheduler(SchedulerConfig cfg, evict::EvictController* ctl = nullptr);

    uint64_t submit(std::vector<int32_t> prompt, uint32_t max_tokens, int32_t eos = -1);
    void cancel(uint64_t id);  // thread-safe; applied at the next schedule()
    // Plans one step. `out` receives deliveries produced while planning (cancellations, failures, context-full).
    StepPlan schedule(const Drafter& drafter, std::vector<Delivery>& out);
    void commit(const StepPlan& plan, const std::vector<StepOutput>& results, std::vector<Delivery>& out);

    bool idle() const { return reqs_.empty(); }
    size_t active() const;
    size_t waiting() const { return waiting_.size(); }
    const SchedulerStats& stats() const { return st_; }
    const paged::PagedKv& kv() const { return kv_; }
    const SchedulerConfig& config() const { return cfg_; }
    bool check_invariants(std::string* why = nullptr) const;

private:
    struct Req {
        uint64_t id = 0;
        std::vector<int32_t> prompt, out;
        uint32_t max_tokens = 0;
        int32_t eos = -1;
        bool has_seq = false;
        paged::SeqId seq = 0;
        uint32_t preempted = 0;
        size_t ctx_len() const { return prompt.size() + out.size(); }
        int32_t at(size_t p) const { return p < prompt.size() ? prompt[p] : out[p - prompt.size()]; }
    };
    void finish(Req& r, EndReason why, std::vector<Delivery>& out, const std::string& err = {});
    void preempt(Req& r, StepPlan& plan);
    // Appends n tokens to r's KV. allow_preempt: on OOM preempt younger sequences (or r itself, or end r if it is
    // alone). False: r got nothing this step (it may have been preempted or finished).
    bool reserve(Req& r, const int32_t* toks, uint32_t n, StepPlan& plan, std::vector<Delivery>& out, bool allow_preempt);
    std::vector<int32_t> ctx_of(const Req& r) const;

    SchedulerConfig cfg_;
    evict::EvictController* ctl_;
    paged::PagedKv kv_;
    std::map<uint64_t, Req> reqs_;    // id order == arrival order (ids increase)
    std::deque<uint64_t> waiting_;    // sorted by id
    std::set<uint64_t> running_;      // ids holding a KV sequence
    uint64_t next_id_ = 1;
    std::mutex cancel_mu_;
    std::set<uint64_t> cancel_;
    int64_t budget_ = 0;  // tokens left in the step being planned
    bool in_step_ = false;  // between schedule() and commit()
    SchedulerStats st_;
};

// One engine step over a plan (the GPU engine, or a CPU mock in tests).
class BatchEngine {
public:
    virtual ~BatchEngine() = default;
    virtual std::vector<StepOutput> step(const StepPlan& plan) = 0;
};

struct ServeHooks {
    std::function<std::vector<int32_t>(const serve::GenerateRequest&)> encode;  // prompt tokens for a request
    std::function<std::string(int32_t)> decode;                                 // text piece of one token
    int32_t eos = -1;
    Drafter drafter;  // optional
};

// Thread that runs schedule -> engine.step -> commit while requests exist; GenerateFn calls block on their mailbox.
class ServeLoop {
public:
    ServeLoop(SchedulerConfig cfg, BatchEngine& engine, ServeHooks hooks, evict::EvictController* ctl = nullptr);
    ~ServeLoop();
    ServeLoop(const ServeLoop&) = delete;
    ServeLoop& operator=(const ServeLoop&) = delete;
    serve::GenerateFn generate_fn();
    SchedulerStats stats();
    bool kv_drained();  // no KV block referenced (all sequences released)

private:
    struct Mailbox {
        std::deque<int32_t> tokens;
        EndReason end = EndReason::None;
        std::string error;
    };
    void loop();
    void post(std::vector<Delivery>& d);
    serve::GenerateResult run(const serve::GenerateRequest& r, const serve::TokenSink& sink);

    RequestScheduler sched_;
    BatchEngine& engine_;
    ServeHooks hooks_;
    std::mutex mu_;  // guards sched_ (the engine step runs unlocked) and mail_
    std::condition_variable work_cv_, mail_cv_;
    std::map<uint64_t, Mailbox> mail_;  // by request id
    bool stop_ = false;
    std::thread th_;
};

}  // namespace vegajet::sched

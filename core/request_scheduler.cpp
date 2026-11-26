// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/request_scheduler.hpp"

#include <algorithm>
#include <exception>

namespace vegajet::sched {

const char* end_reason_name(EndReason r) {
    switch (r) {
        case EndReason::None: return "none";
        case EndReason::Eos: return "eos";
        case EndReason::Length: return "length";
        case EndReason::ContextFull: return "context_full";
        case EndReason::Cancelled: return "cancelled";
        case EndReason::Failed: return "failed";
    }
    return "none";
}

namespace {
SchedulerConfig sanitize(SchedulerConfig c) {
    c.max_slots = std::max<uint32_t>(1, c.max_slots);
    c.step_token_budget = std::max(c.step_token_budget, c.max_slots);
    c.prefill_chunk = std::max<uint32_t>(1, c.prefill_chunk);
    c.kv_blocks = std::max<uint32_t>(1, c.kv_blocks);
    c.block_tokens = std::max<uint32_t>(1, c.block_tokens);
    return c;
}
}  // namespace

RequestScheduler::RequestScheduler(SchedulerConfig cfg, evict::EvictController* ctl)
    : cfg_(sanitize(cfg)), ctl_(ctl), kv_(cfg_.kv_blocks, cfg_.block_tokens) {}

uint64_t RequestScheduler::submit(std::vector<int32_t> prompt, uint32_t max_tokens, int32_t eos) {
    Req r;
    r.id = next_id_++;
    r.prompt = std::move(prompt);
    r.max_tokens = std::max<uint32_t>(1, max_tokens);
    r.eos = eos;
    const uint64_t id = r.id;
    reqs_.emplace(id, std::move(r));
    waiting_.push_back(id);  // ids increase: stays sorted
    return id;
}

void RequestScheduler::cancel(uint64_t id) {
    std::lock_guard<std::mutex> lk(cancel_mu_);
    cancel_.insert(id);
}

size_t RequestScheduler::active() const { return running_.size(); }

std::vector<int32_t> RequestScheduler::ctx_of(const Req& r) const {
    std::vector<int32_t> c(r.prompt);
    c.insert(c.end(), r.out.begin(), r.out.end());
    return c;
}

void RequestScheduler::finish(Req& r, EndReason why, std::vector<Delivery>& out, const std::string& err) {
    if (r.has_seq) kv_.release(r.seq);
    running_.erase(r.id);
    const auto w = std::lower_bound(waiting_.begin(), waiting_.end(), r.id);
    if (w != waiting_.end() && *w == r.id) waiting_.erase(w);
    switch (why) {
        case EndReason::Cancelled: ++st_.cancelled; break;
        case EndReason::Failed: ++st_.failed; break;
        case EndReason::ContextFull: ++st_.context_full; break;
        default: ++st_.finished; break;
    }
    bool merged = false;  // attach the end to this step's token delivery for the request, if there is one
    for (auto it = out.rbegin(); it != out.rend() && !merged; ++it)
        if (it->req == r.id && it->end == EndReason::None) { it->end = why; it->error = err; merged = true; }
    if (!merged) out.push_back({r.id, {}, why, err});
    reqs_.erase(r.id);  // r is dangling from here on
}

void RequestScheduler::preempt(Req& r, StepPlan& plan) {
    for (auto it = plan.items.begin(); it != plan.items.end();) {
        if (it->req == r.id) {
            budget_ += static_cast<int64_t>(it->tokens.size());
            if (it->decode) st_.drafts_proposed -= it->n_draft;
            it = plan.items.erase(it);
        } else {
            ++it;
        }
    }
    kv_.release(r.seq);
    r.has_seq = false;
    running_.erase(r.id);
    waiting_.insert(std::lower_bound(waiting_.begin(), waiting_.end(), r.id), r.id);
    ++r.preempted;
    ++st_.preemptions;
}

bool RequestScheduler::reserve(Req& r, const int32_t* toks, uint32_t n, StepPlan& plan, std::vector<Delivery>& out,
                               bool allow_preempt) {
    if (cfg_.sequence_context && uint64_t(kv_.length(r.seq)) + n > cfg_.sequence_context) {
        finish(r, EndReason::ContextFull, out);
        return false;
    }
    for (;;) {
        if (kv_.append(r.seq, toks, n)) return true;
        if (!allow_preempt) return false;
        uint64_t victim = 0;  // youngest running sequence younger than r
        for (auto it = running_.rbegin(); it != running_.rend(); ++it)
            if (*it > r.id) { victim = *it; break; }
        if (victim) { preempt(reqs_.at(victim), plan); continue; }
        if (running_.size() > 1) { preempt(r, plan); return false; }  // r is the youngest: older ones keep going
        // Alone and still no room: the pool cannot hold this context.
        if (r.out.empty()) finish(r, EndReason::Failed, out, "context does not fit in the KV pool");
        else finish(r, EndReason::ContextFull, out, {});
        return false;
    }
}

StepPlan RequestScheduler::schedule(const Drafter& drafter, std::vector<Delivery>& out) {
    StepPlan plan;
    // 1. cancellations (client disconnects)
    std::set<uint64_t> cancels;
    {
        std::lock_guard<std::mutex> lk(cancel_mu_);
        cancels.swap(cancel_);
    }
    for (uint64_t id : cancels) {
        auto it = reqs_.find(id);
        if (it != reqs_.end()) finish(it->second, EndReason::Cancelled, out);
    }
    // 2. admission, FCFS
    const uint64_t capacity = cfg_.sequence_context ? cfg_.sequence_context : uint64_t(cfg_.kv_blocks) * cfg_.block_tokens;
    while (running_.size() < cfg_.max_slots && !waiting_.empty()) {
        Req& r = reqs_.at(waiting_.front());
        if (r.ctx_len() > capacity) {
            finish(r, r.out.empty() ? EndReason::Failed : EndReason::ContextFull, out,
                   r.out.empty() ? "prompt of " + std::to_string(r.prompt.size()) + " tokens exceeds the KV pool of " +
                                       std::to_string(capacity) + " tokens"
                                 : std::string());
            continue;
        }
        const paged::KvStats ks = kv_.stats();
        // A new request needs room for its first chunk. A preempted one comes back only when its whole context fits
        // (re-admitting it into the same pressure would just preempt it again: thrash).
        const uint32_t first = kv_.blocks_needed(r.preempted ? static_cast<uint32_t>(r.ctx_len()) + 1
                                                             : std::min<uint32_t>(cfg_.prefill_chunk, static_cast<uint32_t>(r.ctx_len())));
        if (!running_.empty() && ks.free + ks.cached < first + cfg_.admit_reserve_blocks) break;
        waiting_.pop_front();
        r.seq = kv_.create();
        r.has_seq = true;
        running_.insert(r.id);
        const std::vector<int32_t> ctx = ctx_of(r);
        const uint32_t covered = cfg_.prefix_reuse ? kv_.attach_prefix(r.seq, ctx.data(), static_cast<uint32_t>(ctx.size())) : 0;
        if (covered == ctx.size() && covered > 0) kv_.truncate(r.seq, covered - 1);  // the last token must be recomputed
        st_.prefix_hit_tokens += kv_.length(r.seq);
    }
    st_.max_active = std::max<uint32_t>(st_.max_active, static_cast<uint32_t>(running_.size()));
    budget_ = cfg_.step_token_budget;
    // 3. decode inputs, oldest first
    std::set<uint64_t> scheduled;
    const std::vector<uint64_t> order(running_.begin(), running_.end());
    uint32_t decode_batch = 0;
    for (uint64_t id : order) {
        if (budget_ <= 0) break;
        auto it = reqs_.find(id);
        if (it == reqs_.end() || !it->second.has_seq) continue;
        Req& r = it->second;
        const uint32_t L = kv_.length(r.seq);
        if (r.out.empty() || L + 1 != r.ctx_len()) continue;
        const uint32_t remaining = r.max_tokens - static_cast<uint32_t>(r.out.size());
        std::vector<int32_t> inputs{r.out.back()};
        if (cfg_.speculative && drafter && ctl_ && remaining > 1 && budget_ > 1) {
            const uint32_t k = std::min<uint32_t>({ctl_->next_depth(), static_cast<uint32_t>(budget_ - 1), remaining - 1});
            std::vector<int32_t> d = drafter(id, ctx_of(r), k);
            if (d.size() > k) d.resize(k);
            inputs.insert(inputs.end(), d.begin(), d.end());
        }
        if (inputs.size() > 1 && !reserve(r, inputs.data(), static_cast<uint32_t>(inputs.size()), plan, out, false)) {
            ++st_.draft_drops;  // memory pressure: verify without drafts rather than preempt anyone
            inputs.resize(1);
        }
        if (inputs.size() == 1 && !reserve(r, inputs.data(), 1, plan, out, true)) continue;
        WorkItem w;
        w.req = id;
        w.seq = r.seq;
        w.pos0 = L;
        w.n_draft = static_cast<uint32_t>(inputs.size() - 1);
        w.tokens = std::move(inputs);
        w.sample = w.decode = true;
        budget_ -= static_cast<int64_t>(w.tokens.size());
        st_.drafts_proposed += w.n_draft;
        plan.items.push_back(std::move(w));
        scheduled.insert(id);
        ++decode_batch;
    }
    // 4. prefill chunks, oldest first
    for (uint64_t id : order) {
        if (budget_ <= 0) break;
        auto it = reqs_.find(id);
        if (it == reqs_.end() || !it->second.has_seq || scheduled.count(id)) continue;
        Req& r = it->second;
        const uint32_t L = kv_.length(r.seq);
        if (!r.out.empty() && L + 1 == r.ctx_len()) continue;  // a decode that did not fit the budget
        const uint32_t n = std::min<uint32_t>({cfg_.prefill_chunk, static_cast<uint32_t>(r.ctx_len() - L),
                                               static_cast<uint32_t>(budget_)});
        std::vector<int32_t> toks(n);
        for (uint32_t i = 0; i < n; ++i) toks[i] = r.at(L + i);
        if (!reserve(r, toks.data(), n, plan, out, true)) continue;
        WorkItem w;
        w.req = id;
        w.seq = r.seq;
        w.pos0 = L;
        w.tokens = std::move(toks);
        w.sample = L + n == r.ctx_len();
        budget_ -= n;
        plan.items.push_back(std::move(w));
        scheduled.insert(id);
    }
    for (uint64_t id : running_) plan.resident.push_back(reqs_.at(id).seq);
    for (const auto& w : plan.items) plan.tokens += static_cast<uint32_t>(w.tokens.size());
    plan.copies = kv_.take_copies();
    in_step_ = !plan.items.empty();
    if (!plan.items.empty()) ++st_.steps;
    st_.max_step_tokens = std::max(st_.max_step_tokens, plan.tokens);
    decode_batch = 0;
    for (const auto& w : plan.items) decode_batch += w.decode;
    st_.max_decode_batch = std::max(st_.max_decode_batch, decode_batch);
    return plan;
}

void RequestScheduler::fail(const StepPlan& plan, const std::string& error, std::vector<Delivery>& out) {
    in_step_=false;
    for (const auto& w:plan.items) {
        auto it=reqs_.find(w.req);
        if (it!=reqs_.end()) finish(it->second,EndReason::Failed,out,error);
    }
}

void RequestScheduler::commit(const StepPlan& plan, const std::vector<StepOutput>& results, std::vector<Delivery>& out) {
    in_step_ = false;
    std::map<uint64_t, const StepOutput*> by;
    for (const auto& r : results) by[r.req] = &r;
    for (const auto& w : plan.items) {
        auto it = reqs_.find(w.req);
        if (it == reqs_.end()) continue;
        Req& r = it->second;
        const uint32_t n = static_cast<uint32_t>(w.tokens.size());
        if (!w.decode) st_.prefill_tokens += n;
        if (!w.sample) continue;
        const auto res = by.find(w.req);
        if (res == by.end() || res->second->tokens.empty()) {
            finish(r, EndReason::Failed, out, "engine returned no token");
            continue;
        }
        const std::vector<int32_t>& got = res->second->tokens;
        std::vector<int32_t> fresh;
        if (w.decode) {
            ++st_.decode_steps;
            uint32_t a = 0;  // accepted drafts: the engine's tokens must reproduce the draft prefix
            while (a < w.n_draft && a + 1 < got.size() && got[a] == w.tokens[1 + a]) ++a;
            kv_.truncate(r.seq, w.pos0 + 1 + a);
            st_.drafts_accepted += a;
            if (ctl_ && w.n_draft > 0) ctl_->observe(w.n_draft, a, 0.0);
            fresh.assign(got.begin(), got.begin() + a + 1);
        } else {
            fresh.push_back(got[0]);
        }
        Delivery d;
        d.req = r.id;
        EndReason end = EndReason::None;
        for (int32_t t : fresh) {
            r.out.push_back(t);
            d.tokens.push_back(t);
            ++st_.generated;
            if (t == r.eos && r.eos >= 0) { end = EndReason::Eos; break; }
            if (r.out.size() >= r.max_tokens) { end = EndReason::Length; break; }
        }
        out.push_back(std::move(d));
        if (end != EndReason::None) finish(r, end, out);
    }
}

bool RequestScheduler::check_invariants(std::string* why) const {
    auto bad = [&](const std::string& m) { if (why) *why = m; return false; };
    if (!kv_.check_invariants()) return bad("paged_kv invariants");
    if (running_.size() > cfg_.max_slots) return bad("more running sequences than slots");
    size_t holders = 0;
    for (const auto& kvp : reqs_) {
        const Req& r = kvp.second;
        holders += r.has_seq;
        if (r.has_seq != (running_.count(r.id) == 1)) return bad("running set out of sync");
        const bool queued = std::binary_search(waiting_.begin(), waiting_.end(), r.id);
        if (r.has_seq == queued) return bad("request both / neither running and waiting");
        // Between steps the KV never runs ahead of the context; during a step it also holds the draft tokens.
        if (r.has_seq && kv_.length(r.seq) > r.ctx_len() + (in_step_ ? cfg_.step_token_budget : 0))
            return bad("KV longer than the context");
        if (r.out.size() > r.max_tokens) return bad("more tokens than max_tokens");
    }
    if (holders != running_.size() || waiting_.size() + running_.size() != reqs_.size()) return bad("request accounting");
    if (!std::is_sorted(waiting_.begin(), waiting_.end())) return bad("waiting queue out of order");
    if (reqs_.empty() && kv_.stats().used != 0) return bad("KV blocks still referenced with no requests");
    return true;
}

// ------------------------------------------------------------------------------------------------ ServeLoop
ServeLoop::ServeLoop(SchedulerConfig cfg, BatchEngine& engine, ServeHooks hooks, evict::EvictController* ctl)
    : sched_(cfg, ctl), engine_(engine), hooks_(std::move(hooks)) {
    th_ = std::thread([this] { loop(); });
}

ServeLoop::~ServeLoop() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_ = true;
    }
    work_cv_.notify_all();
    th_.join();
}

void ServeLoop::post(std::vector<Delivery>& d) {
    for (auto& x : d) {
        auto it = mail_.find(x.req);
        if (it == mail_.end()) continue;
        it->second.tokens.insert(it->second.tokens.end(), x.tokens.begin(), x.tokens.end());
        if (x.end != EndReason::None) { it->second.end = x.end; it->second.error = x.error; }
    }
    if (!d.empty()) mail_cv_.notify_all();
    d.clear();
}

void ServeLoop::loop() {
    std::unique_lock<std::mutex> lk(mu_);
    std::vector<Delivery> d;
    while (!stop_) {
        if (sched_.idle()) { work_cv_.wait(lk, [&] { return stop_ || !sched_.idle(); }); continue; }
        StepPlan plan = sched_.schedule(hooks_.drafter, d);
        post(d);
        if (plan.items.empty()) { work_cv_.wait_for(lk, std::chrono::milliseconds(1)); continue; }
        lk.unlock();
        std::vector<StepOutput> res;
        std::string failure;
        try { res=engine_.step(plan); }
        catch (const std::exception& e) { failure=e.what(); }
        catch (...) { failure="batch engine failed"; }
        lk.lock();
        if (failure.empty()) sched_.commit(plan,res,d);
        else sched_.fail(plan,failure,d);
        post(d);
    }
    for (auto& m : mail_)
        if (m.second.end == EndReason::None) m.second.end = EndReason::Cancelled;
    mail_cv_.notify_all();
}

serve::GenerateResult ServeLoop::run(const serve::GenerateRequest& r, const serve::TokenSink& sink) {
    serve::GenerateResult res;
    const std::vector<int32_t> prompt = hooks_.encode(r);
    res.prompt_tokens = static_cast<uint32_t>(prompt.size());
    std::unique_lock<std::mutex> lk(mu_);
    if (stop_) { res.error = "scheduler stopped"; return res; }
    const uint64_t id = sched_.submit(prompt, static_cast<uint32_t>(std::max(1, r.max_tokens)), hooks_.eos);
    Mailbox& box = mail_[id];
    work_cv_.notify_one();
    bool cancelled = false;
    for (;;) {
        mail_cv_.wait(lk, [&] { return !box.tokens.empty() || box.end != EndReason::None; });
        std::deque<int32_t> toks;
        toks.swap(box.tokens);
        const EndReason end = box.end;
        lk.unlock();
        for (int32_t t : toks) {
            ++res.completion_tokens;
            if (cancelled || (t == hooks_.eos && hooks_.eos >= 0)) continue;
            if (!sink(hooks_.decode(t))) { cancelled = true; sched_.cancel(id); }
        }
        lk.lock();
        if (end != EndReason::None) break;
    }
    const EndReason end = box.end;
    const std::string err = box.error;
    mail_.erase(id);
    switch (end) {
        case EndReason::Eos: res.finish = serve::FinishReason::Stop; break;
        case EndReason::Length:
        case EndReason::ContextFull: res.finish = serve::FinishReason::Length; break;
        case EndReason::Cancelled: res.finish = serve::FinishReason::Cancelled; break;
        case EndReason::Failed: res.finish = serve::FinishReason::Error; res.error = err.empty() ? "failed" : err; break;
        case EndReason::None: break;
    }
    return res;
}

serve::GenerateFn ServeLoop::generate_fn() {
    return [this](const serve::GenerateRequest& r, const serve::TokenSink& sink) { return run(r, sink); };
}

SchedulerStats ServeLoop::stats() {
    std::lock_guard<std::mutex> lk(mu_);
    return sched_.stats();
}

bool ServeLoop::kv_drained() {
    std::lock_guard<std::mutex> lk(mu_);
    return sched_.idle() && sched_.kv().stats().used == 0;
}

}  // namespace vegajet::sched

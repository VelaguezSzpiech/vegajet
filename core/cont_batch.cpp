// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/cont_batch.hpp"

#include <algorithm>
#include <stdexcept>

namespace vegajet::batch {

const char* end_name(End e) {
    switch (e) {
        case End::None: return "none";
        case End::Eos: return "eos";
        case End::Length: return "length";
        case End::Cancelled: return "cancelled";
        case End::ContextFull: return "context_full";
        case End::Rejected: return "rejected";
    }
    return "?";
}

struct ContinuousBatcher::Seq {
    Request req;
    uint64_t arrival = 0;
    std::vector<int32_t> ctx;          // prompt + generated
    std::vector<BlockId> table;        // one reference per entry
    uint32_t kv = 0;                   // tokens whose K/V is written
    int32_t slot = -1;                 // -1: waiting
    bool cancelled = false;            // set during an engine step; released when it returns
    bool has_state_op = false;         // Reset / Restore to emit with this sequence's first work
    StateOp state_op;
    prefix::Match match;               // hybrid: pinned until the Restore has executed
    bool match_held = false;
    uint32_t snap_at = 0;              // hybrid: save the state after this many tokens (0 = none)
    uint32_t prompt_end = 0;           // context length at admission (prefill covers [kv, prompt_end))
    Result res;
};

ContinuousBatcher::ContinuousBatcher(BatchConfig cfg) : cfg_(cfg) {
    cfg_.max_batch = std::max<uint32_t>(1, std::min<uint32_t>(8, cfg_.max_batch));
    cfg_.block_tokens = std::max<uint32_t>(1, cfg_.block_tokens);
    cfg_.step_tokens = std::max(cfg_.step_tokens, cfg_.max_batch);
    cfg_.prefill_chunk = std::max<uint32_t>(1, cfg_.prefill_chunk);
    pool_ = std::make_unique<prefix::BlockPool>(cfg_.kv_blocks);
    cache_ = std::make_unique<prefix::RadixCache>(*pool_, cfg_.block_tokens, cfg_.max_snapshots, cfg_.hybrid);
    slots_.assign(cfg_.max_batch, nullptr);
}

ContinuousBatcher::~ContinuousBatcher() {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& kv : seqs_) {
        Seq& s = *kv.second;
        for (BlockId b : s.table) pool_->unref(b);
        if (s.match_held) cache_->release(s.match);
    }
    seqs_.clear();
}

bool ContinuousBatcher::submit(Request r) {
    std::lock_guard<std::mutex> lk(mu_);
    if (r.prompt.empty() || seqs_.count(r.id) || results_.count(r.id)) return false;
    auto s = std::make_unique<Seq>();
    s->arrival = ++arrival_;
    s->ctx = r.prompt;
    s->req = std::move(r);
    if (s->req.max_new == 0) {
        Result res;
        res.end = End::Length;
        results_[s->req.id] = res;
        return true;
    }
    seqs_[s->req.id] = std::move(s);
    return true;
}

void ContinuousBatcher::release_locked(Seq& s, End end) {
    for (BlockId b : s.table) pool_->unref(b);
    s.table.clear();
    if (s.match_held) { cache_->release(s.match); s.match_held = false; }
    if (s.slot >= 0) slots_[static_cast<size_t>(s.slot)] = nullptr;
    s.res.end = end;
    if (end == End::Cancelled) ++st_.cancelled;
    else ++st_.finished;
    const uint64_t id = s.req.id;
    results_[id] = std::move(s.res);
    seqs_.erase(id);  // `s` is gone after this
}

bool ContinuousBatcher::cancel(uint64_t id) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = seqs_.find(id);
    if (it == seqs_.end() || it->second->cancelled) return false;
    Seq& s = *it->second;
    if (s.slot >= 0 && in_step_) {
        s.cancelled = true;  // the engine may be writing its blocks: released when the step returns
        return true;
    }
    release_locked(s, End::Cancelled);
    return true;
}

void ContinuousBatcher::cache_prefix_locked(Seq& s) {
    if (!cfg_.prefix_cache || cfg_.hybrid) return;  // hybrid prefixes are cached with their snapshots only
    const uint32_t n = s.kv / cfg_.block_tokens * cfg_.block_tokens;
    if (n > 0) cache_->insert(s.ctx.data(), n, s.table.data());
}

void ContinuousBatcher::preempt_locked(Seq& s) {
    cache_prefix_locked(s);
    for (BlockId b : s.table) pool_->unref(b);
    s.table.clear();
    s.kv = 0;
    if (s.match_held) { cache_->release(s.match); s.match_held = false; }
    s.has_state_op = false;
    s.snap_at = 0;
    s.prompt_end = 0;
    slots_[static_cast<size_t>(s.slot)] = nullptr;
    s.slot = -1;
    ++s.res.preemptions;
    ++st_.preemptions;
}

ContinuousBatcher::Seq* ContinuousBatcher::pick_victim_locked(const Seq& f, const std::vector<Seq*>& planned) {
    Seq* v = nullptr;
    for (Seq* s : slots_) {
        if (!s || s == &f || s->cancelled || std::find(planned.begin(), planned.end(), s) != planned.end()) continue;
        const bool lower = s->req.priority < f.req.priority || (s->req.priority == f.req.priority && s->arrival > f.arrival);
        if (!lower) continue;
        if (!v || s->req.priority < v->req.priority || (s->req.priority == v->req.priority && s->arrival > v->arrival)) v = s;
    }
    return v;
}

bool ContinuousBatcher::ensure_blocks_locked(Seq& s, uint32_t tokens, const std::vector<Seq*>& planned) {
    const size_t need = (tokens + cfg_.block_tokens - 1) / cfg_.block_tokens;
    while (s.table.size() < need) {
        BlockId b = pool_->alloc();
        if (b < 0 && cfg_.prefix_cache && cache_->evict_blocks(1) > 0) b = pool_->alloc();
        if (b < 0) {
            Seq* v = pick_victim_locked(s, planned);
            if (!v) return false;
            preempt_locked(*v);
            continue;
        }
        s.table.push_back(b);
    }
    return true;
}

uint32_t ContinuousBatcher::next_snapshot_point(uint32_t kv, uint32_t prompt_end) const {
    if (!cfg_.hybrid || !cfg_.prefix_cache) return 0;
    const uint32_t bt = cfg_.block_tokens;
    uint32_t next = prompt_end / bt * bt;  // last block-aligned prompt position
    const uint32_t iv = cfg_.snapshot_interval / bt * bt;
    if (iv > 0) next = std::min(next, (kv / iv + 1) * iv);
    return next > kv ? next : 0;
}

bool ContinuousBatcher::should_defer_locked(const Seq& cand) const {
    if (!cfg_.prefix_cache || !cfg_.prefix_aware_admission) return false;
    const uint32_t bt = cfg_.block_tokens;
    for (const Seq* r : slots_) {
        if (!r || r->cancelled || r->kv >= r->prompt_end) continue;  // only leaders still prefilling their prompt
        const size_t lim = std::min<size_t>({cand.ctx.size() - 1, r->prompt_end});
        size_t l = 0;
        while (l < lim && cand.ctx[l] == r->ctx[l]) ++l;
        const uint32_t shared = static_cast<uint32_t>(l) / bt * bt;
        if (shared >= 2 * bt && shared > r->kv) return true;  // the leader has not produced (all of) it yet
    }
    return false;
}

void ContinuousBatcher::admit_locked(StepPlan& plan) {
    (void)plan;
    std::vector<const Seq*> deferred;
    for (;;) {
        Seq* best = nullptr;
        for (auto& kv : seqs_) {
            Seq* s = kv.second.get();
            if (s->slot >= 0 || s->cancelled) continue;
            if (std::find(deferred.begin(), deferred.end(), s) != deferred.end()) continue;
            if (!best || s->req.priority > best->req.priority ||
                (s->req.priority == best->req.priority && s->arrival < best->arrival))
                best = s;
        }
        if (!best) return;
        if (should_defer_locked(*best)) {
            deferred.push_back(best);
            continue;
        }
        int32_t slot = -1;
        for (size_t i = 0; i < slots_.size(); ++i)
            if (!slots_[i]) { slot = static_cast<int32_t>(i); break; }
        if (slot < 0) {
            if (!cfg_.priority_preemption) return;
            Seq* v = nullptr;  // lowest-priority runner strictly below the candidate
            for (Seq* s : slots_)
                if (s && !s->cancelled && s->req.priority < best->req.priority &&
                    (!v || s->req.priority < v->req.priority || (s->req.priority == v->req.priority && s->arrival > v->arrival)))
                    v = s;
            if (!v) return;
            preempt_locked(*v);
            continue;
        }
        Seq& s = *best;
        s.slot = slot;
        slots_[static_cast<size_t>(slot)] = &s;
        s.kv = 0;
        s.state_op = StateOp{StateOp::Kind::Reset, static_cast<uint32_t>(slot), -1, 0};
        s.has_state_op = true;
        if (cfg_.prefix_cache && s.ctx.size() > 1) {
            prefix::Match m = cache_->match(s.ctx.data(), static_cast<uint32_t>(s.ctx.size() - 1));
            for (BlockId b : m.blocks) pool_->ref(b);
            s.table = m.blocks;
            s.kv = m.tokens;
            s.res.prefix_hit_tokens += m.tokens;
            st_.prefix_hit_tokens += m.tokens;
            if (cfg_.hybrid && m.snapshot >= 0) {
                s.state_op = StateOp{StateOp::Kind::Restore, static_cast<uint32_t>(slot), m.snapshot, m.tokens};
                s.match = std::move(m);
                s.match_held = true;  // keeps the snapshot from being evicted before the Restore runs
            } else {
                cache_->release(m);
            }
        }
        s.prompt_end = static_cast<uint32_t>(s.ctx.size());
        s.snap_at = next_snapshot_point(s.kv, s.prompt_end);
        uint32_t n_active = 0;
        for (Seq* x : slots_) n_active += x != nullptr;
        st_.peak_batch = std::max(st_.peak_batch, n_active);
    }
}

bool ContinuousBatcher::step(const Engine& engine) {
    std::unique_lock<std::mutex> lk(mu_);
    if (in_step_) return false;
    StepPlan plan;
    admit_locked(plan);
    std::vector<Seq*> order;
    for (Seq* s : slots_)
        if (s) order.push_back(s);
    std::sort(order.begin(), order.end(), [](const Seq* a, const Seq* b) {
        return a->req.priority > b->req.priority || (a->req.priority == b->req.priority && a->arrival < b->arrival);
    });
    std::vector<Seq*> planned;
    struct Save { uint64_t id; SnapId snap; uint32_t tokens; };
    std::vector<Save> saves;
    std::vector<uint64_t> restored;
    uint32_t budget = cfg_.step_tokens;
    auto add_work = [&](Seq& s, uint32_t end) {
        if (s.has_state_op) {
            plan.pre.push_back(s.state_op);
            if (s.state_op.kind == StateOp::Kind::Restore) restored.push_back(s.req.id);
            s.has_state_op = false;
        }
        SlotWork w;
        w.req = s.req.id;
        w.slot = static_cast<uint32_t>(s.slot);
        w.pos0 = s.kv;
        w.tokens.assign(s.ctx.begin() + s.kv, s.ctx.begin() + end);
        w.block_table = s.table;
        w.sample = end == s.ctx.size();
        budget -= end - s.kv;
        planned.push_back(&s);
        if (cfg_.hybrid && s.snap_at == end) {
            const SnapId snap = cache_->alloc_snapshot();
            if (snap >= 0) {
                plan.post.push_back(StateOp{StateOp::Kind::Save, w.slot, snap, end});
                saves.push_back({s.req.id, snap, end});
            }
            s.snap_at = 0;
        }
        plan.work.push_back(std::move(w));
    };
    std::vector<Seq*> stuck;
    for (int pass = 0; pass < 2; ++pass) {  // 0: decode, 1: prefill
        for (Seq* s : order) {
            if (budget == 0) break;
            if (s->slot < 0 || std::find(planned.begin(), planned.end(), s) != planned.end()) continue;  // preempted
            const uint32_t len = static_cast<uint32_t>(s->ctx.size());
            const bool decode = s->kv + 1 == len;
            if ((pass == 0) != decode) continue;
            uint32_t end = decode ? len : std::min(len, s->kv + std::min(cfg_.prefill_chunk, budget));
            if (s->snap_at > s->kv && s->snap_at < end) end = s->snap_at;  // cut exactly at the snapshot point
            if (!ensure_blocks_locked(*s, end, planned)) {
                stuck.push_back(s);
                continue;
            }
            add_work(*s, end);
        }
    }
    // A sequence that cannot get blocks while nothing else holds any can never run: end it instead of livelocking.
    bool ended = false;
    for (Seq* s : stuck) {
        if (s->slot < 0 || !plan.work.empty()) continue;
        bool others = false;
        for (Seq* x : slots_) others = others || (x && x != s);
        if (!others) {
            release_locked(*s, s->res.tokens.empty() ? End::Rejected : End::ContextFull);
            ended = true;
        }
    }
    if (plan.work.empty()) return ended;  // ended: state changed, the next step may admit others
    in_step_ = true;
    lk.unlock();
    StepOut out;
    try {
        out = engine(plan);
    } catch (...) {
        lk.lock();
        in_step_ = false;
        for (const Save& sv : saves) cache_->free_snapshot(sv.snap);
        throw;
    }
    lk.lock();
    in_step_ = false;
    ++st_.steps;
    // 1. KV progress
    for (const SlotWork& w : plan.work) {
        auto it = seqs_.find(w.req);
        if (it == seqs_.end() || it->second->cancelled || it->second->slot != static_cast<int32_t>(w.slot)) continue;
        Seq& s = *it->second;
        s.kv = w.pos0 + static_cast<uint32_t>(w.tokens.size());
        if (w.sample && w.tokens.size() == 1 && !s.res.tokens.empty()) st_.decode_tokens += 1;
        else { st_.prefill_tokens += w.tokens.size(); s.res.prefilled_tokens += static_cast<uint32_t>(w.tokens.size()); }
    }
    // 1b. publish each prompt's block-aligned prefix as soon as it is computed (attention models)
    for (const SlotWork& w : plan.work) {
        auto it = seqs_.find(w.req);
        if (it == seqs_.end() || it->second->cancelled || w.sample) continue;
        cache_prefix_locked(*it->second);
    }
    // 2. snapshots: cached with the prompt blocks they belong to (the cache owns them from here)
    for (const Save& sv : saves) {
        auto it = seqs_.find(sv.id);
        if (it == seqs_.end() || it->second->cancelled) { cache_->free_snapshot(sv.snap); continue; }
        Seq& s = *it->second;
        cache_->insert(s.ctx.data(), sv.tokens, s.table.data(), sv.snap, sv.tokens);
        s.snap_at = next_snapshot_point(s.kv, s.prompt_end);
    }
    // 3. Restores have run: unpin their matches
    for (uint64_t id : restored) {
        auto it = seqs_.find(id);
        if (it != seqs_.end() && it->second->match_held) { cache_->release(it->second->match); it->second->match_held = false; }
    }
    // 4. sampled tokens, completions
    for (const SlotWork& w : plan.work) {
        if (!w.sample) continue;
        auto it = seqs_.find(w.req);
        if (it == seqs_.end() || it->second->cancelled || it->second->slot != static_cast<int32_t>(w.slot)) continue;
        Seq& s = *it->second;
        const auto hit = std::find_if(out.sampled.begin(), out.sampled.end(), [&](const auto& p) { return p.first == w.slot; });
        if (hit == out.sampled.end()) throw std::logic_error("engine returned no token for a sampling slot");
        s.ctx.push_back(hit->second);
        s.res.tokens.push_back(hit->second);
        const bool eos = s.req.eos >= 0 && hit->second == s.req.eos;
        if (eos || s.res.tokens.size() >= s.req.max_new) {
            cache_prefix_locked(s);
            release_locked(s, eos ? End::Eos : End::Length);
        }
    }
    // 5. cancellations that arrived during the step
    std::vector<Seq*> drop;
    for (auto& kv : seqs_)
        if (kv.second->cancelled) drop.push_back(kv.second.get());
    for (Seq* s : drop) release_locked(*s, End::Cancelled);
    return true;
}

void ContinuousBatcher::run_until_idle(const Engine& engine, uint64_t max_steps) {
    for (uint64_t i = 0; i < max_steps && step(engine); ++i) {}
}

bool ContinuousBatcher::done(uint64_t id) const {
    std::lock_guard<std::mutex> lk(mu_);
    return results_.count(id) != 0;
}

Result ContinuousBatcher::result(uint64_t id) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = results_.find(id);
    return it == results_.end() ? Result{} : it->second;
}

uint32_t ContinuousBatcher::active() const {
    std::lock_guard<std::mutex> lk(mu_);
    uint32_t n = 0;
    for (Seq* s : slots_) n += s != nullptr;
    return n;
}

uint32_t ContinuousBatcher::waiting() const {
    std::lock_guard<std::mutex> lk(mu_);
    uint32_t n = 0;
    for (const auto& kv : seqs_) n += kv.second->slot < 0;
    return n;
}

BatchStats ContinuousBatcher::stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    return st_;
}

prefix::CacheStats ContinuousBatcher::cache_stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    return cache_->stats();
}

uint32_t ContinuousBatcher::free_blocks() const {
    std::lock_guard<std::mutex> lk(mu_);
    return pool_->free_count();
}

uint32_t ContinuousBatcher::free_snapshots() const {
    std::lock_guard<std::mutex> lk(mu_);
    return cache_->snapshots_free();
}

void ContinuousBatcher::clear_cache() {
    std::lock_guard<std::mutex> lk(mu_);
    cache_->clear();
}

bool ContinuousBatcher::check_invariants() const {
    std::lock_guard<std::mutex> lk(mu_);
    if (!cache_->check_invariants()) return false;
    std::vector<uint32_t> want = cache_->cache_refs();
    for (const auto& kv : seqs_) {
        const Seq& s = *kv.second;
        if (s.slot < 0 && !s.table.empty()) return false;  // waiting sequences hold no KV
        if (s.slot >= 0 && slots_[static_cast<size_t>(s.slot)] != &s) return false;
        if (s.kv > s.ctx.size() || s.table.size() * cfg_.block_tokens < s.kv) return false;
        for (BlockId b : s.table) ++want[static_cast<size_t>(b)];
    }
    for (size_t i = 0; i < slots_.size(); ++i)
        if (slots_[i] && (slots_[i]->slot != static_cast<int32_t>(i) || !seqs_.count(slots_[i]->req.id))) return false;
    uint32_t free_count = 0;
    for (uint32_t b = 0; b < pool_->capacity(); ++b) {
        if (pool_->refs(static_cast<BlockId>(b)) != want[b]) return false;
        free_count += want[b] == 0;
    }
    if (free_count != pool_->free_count()) return false;
    return cache_->stats().snapshots + cache_->snapshots_free() == cache_->snapshot_capacity();
}

} // namespace vegajet::batch

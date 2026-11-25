// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/swarm_router.hpp"

#include <algorithm>

namespace vegajet::swarm {
namespace {

inline uint64_t mix(uint64_t x) {  // splitmix64 finalizer
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

inline uint64_t fnv(const void* p, size_t n, uint64_t h = 0xCBF29CE484222325ull) {
    const auto* b = static_cast<const unsigned char*>(p);
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 0x100000001B3ull;
    return h;
}

} // namespace

std::vector<uint64_t> block_hashes(const std::vector<int32_t>& tokens, uint32_t block_tokens) {
    const size_t b = std::max<uint32_t>(1, block_tokens);
    std::vector<uint64_t> out;
    uint64_t prev = 0;
    for (size_t i = 0; i < tokens.size(); i += b) {
        const size_t n = std::min(b, tokens.size() - i);
        prev = mix(prev ^ fnv(tokens.data() + i, n * sizeof(int32_t)) ^ (n << 1));
        out.push_back(prev);
    }
    return out;
}

std::vector<uint64_t> block_hashes(std::string_view text, uint32_t block_bytes) {
    const size_t b = std::max<uint32_t>(1, block_bytes);
    std::vector<uint64_t> out;
    uint64_t prev = 0;
    for (size_t i = 0; i < text.size(); i += b) {
        const size_t n = std::min(b, text.size() - i);
        prev = mix(prev ^ fnv(text.data() + i, n) ^ (n << 1));
        out.push_back(prev);
    }
    return out;
}

SwarmRouter::SwarmRouter(std::vector<WorkerSpec> workers, RouterConfig cfg) : cfg_(cfg) {
    for (auto& s : workers) {
        Worker w;
        w.spec = std::move(s);
        if (w.spec.slots == 0) w.spec.slots = 1;
        w_.push_back(std::move(w));
    }
}

uint32_t SwarmRouter::match_locked(const Worker& w, const std::vector<uint64_t>& h) const {
    uint32_t k = 0;
    while (k < h.size() && w.prefixes.count(h[k])) ++k;
    return k;
}

void SwarmRouter::remember_locked(Worker& w, const std::vector<uint64_t>& h) {
    for (uint64_t x : h)
        if (w.prefixes.insert(x).second) w.order.push_back(x);
    while (w.prefixes.size() > cfg_.max_prefixes_per_worker && !w.order.empty()) {
        w.prefixes.erase(w.order.front());
        w.order.pop_front();
    }
}

SwarmRouter::Clock::time_point SwarmRouter::dispatch_locked(Clock::time_point now) {
    Clock::time_point deadline = Clock::time_point::max();
    bool assigned = false;
    for (size_t qi = 0; qi < queue_.size();) {
        Req& r = reqs_.at(queue_[qi]);
        // Candidates: `least` = best match among the least-loaded free workers, `warm` = best match among all free
        // workers. Load wins unless the warm worker matches at least affinity_min_gain more blocks.
        const uint32_t gain = std::max<uint32_t>(1, cfg_.affinity_min_gain);
        int least = -1, warm = -1;
        uint32_t least_score = 0, warm_score = 0, busy_score = 0;
        uint64_t least_rv = 0, warm_rv = 0;
        const uint64_t key = r.hashes.empty() ? mix(r.seq) : r.hashes[0];
        for (size_t i = 0; i < w_.size(); ++i) {
            const Worker& w = w_[i];
            if (!w.healthy) continue;
            const uint32_t s = match_locked(w, r.hashes);
            if (w.active >= w.spec.slots) {
                busy_score = std::max(busy_score, s);
                continue;
            }
            const uint64_t rv = mix(key ^ mix(i + 1));
            const uint32_t la = least < 0 ? 0 : w_[static_cast<size_t>(least)].active;
            if (least < 0 || w.active < la || (w.active == la && (s > least_score || (s == least_score && rv > least_rv)))) {
                least = static_cast<int>(i);
                least_score = s;
                least_rv = rv;
            }
            const uint32_t wa = warm < 0 ? 0 : w_[static_cast<size_t>(warm)].active;
            if (warm < 0 || s > warm_score || (s == warm_score && (w.active < wa || (w.active == wa && rv > warm_rv)))) {
                warm = static_cast<int>(i);
                warm_score = s;
                warm_rv = rv;
            }
        }
        if (least < 0) break;  // no free worker: nothing behind this request can be placed either
        const bool take_warm = warm_score >= least_score + gain;
        const int best = take_warm ? warm : least;
        const uint32_t best_score = take_warm ? warm_score : least_score;
        const auto until = r.enqueued + std::chrono::milliseconds(cfg_.affinity_wait_ms);
        if (busy_score >= best_score + gain && now < until) {
            if (!r.waited) ++st_.affinity_waits;
            r.waited = true;
            deadline = std::min(deadline, until);
            ++qi;
            continue;
        }
        Worker& w = w_[static_cast<size_t>(best)];
        r.status = Status::Assigned;
        r.worker = best;
        r.matched = best_score;
        ++w.active;
        ++w.served;
        if (best_score > 0) {
            ++w.affinity_hits;
            ++st_.affinity_hits;
        }
        remember_locked(w, r.hashes);
        ++st_.assigned;
        queue_.erase(queue_.begin() + static_cast<std::ptrdiff_t>(qi));
        assigned = true;
    }
    if (assigned) cv_.notify_all();
    return deadline;
}

uint64_t SwarmRouter::submit(std::vector<uint64_t> prefix_hashes, int32_t priority) {
    std::lock_guard<std::mutex> lk(mu_);
    ++st_.submitted;
    const bool any = std::any_of(w_.begin(), w_.end(), [](const Worker& w) { return w.healthy; });
    if (!any || queue_.size() >= cfg_.max_queue) {
        ++st_.rejected;
        return 0;
    }
    const uint64_t id = next_++;
    Req r;
    r.hashes = std::move(prefix_hashes);
    r.priority = priority;
    r.seq = seq_++;
    r.enqueued = Clock::now();
    reqs_.emplace(id, std::move(r));
    // keep the queue in (priority desc, arrival) order
    auto pos = std::find_if(queue_.begin(), queue_.end(),
                            [&](uint64_t t) { return reqs_.at(t).priority < priority; });
    queue_.insert(pos, id);
    dispatch_locked(Clock::now());
    return id;
}

int SwarmRouter::wait(uint64_t ticket, std::chrono::milliseconds timeout) {
    std::unique_lock<std::mutex> lk(mu_);
    const auto end = Clock::now() + timeout;
    while (true) {
        auto it = reqs_.find(ticket);
        if (it == reqs_.end()) return -1;
        if (it->second.status == Status::Assigned) return it->second.worker;
        if (it->second.status != Status::Queued) return -1;
        const auto now = Clock::now();
        const auto dl = dispatch_locked(now);
        it = reqs_.find(ticket);
        if (it != reqs_.end() && it->second.status == Status::Assigned) return it->second.worker;
        if (now >= end) return -1;
        cv_.wait_until(lk, std::min(end, dl));
    }
}

Status SwarmRouter::status(uint64_t ticket, int* worker) {
    std::lock_guard<std::mutex> lk(mu_);
    dispatch_locked(Clock::now());
    auto it = reqs_.find(ticket);
    if (it == reqs_.end()) return Status::Unknown;
    if (worker) *worker = it->second.worker;
    return it->second.status;
}

bool SwarmRouter::cancel(uint64_t ticket) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = reqs_.find(ticket);
    if (it == reqs_.end()) return false;
    Req& r = it->second;
    if (r.status == Status::Queued) {
        queue_.erase(std::find(queue_.begin(), queue_.end(), ticket));
        reqs_.erase(it);
    } else if (r.status == Status::Assigned) {
        --w_[static_cast<size_t>(r.worker)].active;
        r.status = Status::Cancelled;
    } else {
        return false;
    }
    ++st_.cancelled;
    dispatch_locked(Clock::now());
    cv_.notify_all();
    return true;
}

bool SwarmRouter::cancelled(uint64_t ticket) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = reqs_.find(ticket);
    return it == reqs_.end() || it->second.status == Status::Cancelled;
}

void SwarmRouter::release(uint64_t ticket) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = reqs_.find(ticket);
    if (it == reqs_.end()) return;
    Req& r = it->second;
    if (r.status == Status::Assigned) {
        --w_[static_cast<size_t>(r.worker)].active;
        ++st_.completed;
    } else if (r.status == Status::Queued) {
        queue_.erase(std::find(queue_.begin(), queue_.end(), ticket));
        ++st_.cancelled;
    }
    reqs_.erase(it);
    dispatch_locked(Clock::now());
    cv_.notify_all();
}

void SwarmRouter::set_healthy(size_t worker, bool healthy) {
    std::lock_guard<std::mutex> lk(mu_);
    if (worker >= w_.size()) return;
    Worker& w = w_[worker];
    w.healthy = healthy;
    if (!healthy) {
        w.prefixes.clear();
        w.order.clear();
    }
    dispatch_locked(Clock::now());
    cv_.notify_all();
}

uint32_t SwarmRouter::matched_blocks(uint64_t ticket) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = reqs_.find(ticket);
    return it == reqs_.end() ? 0 : it->second.matched;
}

std::vector<WorkerState> SwarmRouter::workers() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<WorkerState> out;
    for (const Worker& w : w_)
        out.push_back(WorkerState{w.spec.name, w.spec.slots, w.active, w.healthy, w.served, w.affinity_hits,
                                  w.prefixes.size()});
    return out;
}

RouterStats SwarmRouter::stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    return st_;
}

size_t SwarmRouter::queued() const {
    std::lock_guard<std::mutex> lk(mu_);
    return queue_.size();
}

bool SwarmRouter::check_invariants() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<uint32_t> count(w_.size(), 0);
    size_t queued = 0;
    for (const auto& [id, r] : reqs_) {
        if (r.status == Status::Assigned) {
            if (r.worker < 0 || static_cast<size_t>(r.worker) >= w_.size()) return false;
            ++count[static_cast<size_t>(r.worker)];
        } else if (r.status == Status::Queued) {
            ++queued;
        }
    }
    if (queued != queue_.size()) return false;
    for (size_t i = 0; i < w_.size(); ++i)
        if (count[i] != w_[i].active || w_[i].active > w_[i].spec.slots) return false;
    for (uint64_t t : queue_) {
        auto it = reqs_.find(t);
        if (it == reqs_.end() || it->second.status != Status::Queued) return false;
    }
    return true;
}

} // namespace vegajet::swarm

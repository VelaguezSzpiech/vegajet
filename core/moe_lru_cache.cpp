// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/moe_lru_cache.hpp"

#include <algorithm>
#include <cmath>

namespace vegajet::moecache {

MoeLruCache::MoeLruCache(const CacheConfig& cfg) : cfg_(cfg), layers_(cfg.n_layers) {
    if (cfg_.max_slots == 0) cfg_.max_slots = cfg_.n_experts;
    for (auto& L : layers_) {
        L.expert_slot.assign(cfg_.n_experts, -1);
        L.miss_count.assign(cfg_.n_experts, 0.0f);
    }
    for (uint32_t l = 0; l < cfg_.n_layers; ++l) resize(l, std::min(cfg_.slots_per_layer, cfg_.n_experts));
}

void MoeLruCache::lru_unlink(Layer& L, int32_t s) {
    const int32_t p = L.prev[s], n = L.next[s];
    if (p >= 0) L.next[p] = n; else L.head = n;
    if (n >= 0) L.prev[n] = p; else L.tail = p;
    L.prev[s] = L.next[s] = -1;
}

void MoeLruCache::lru_push_front(Layer& L, int32_t s) {
    L.prev[s] = -1;
    L.next[s] = L.head;
    if (L.head >= 0) L.prev[L.head] = s;
    L.head = s;
    if (L.tail < 0) L.tail = s;
}

void MoeLruCache::rebuild_lru(Layer& L) {
    const size_t n = L.slot_expert.size();
    L.prev.assign(n, -1);
    L.next.assign(n, -1);
    L.head = L.tail = -1;
    for (size_t s = n; s-- > 0;) lru_push_front(L, static_cast<int32_t>(s));
}

int32_t MoeLruCache::pick_victim(Layer& L) const {
    // Empty slots first (any), then the least recently used ready, unpinned slot. Loading slots are never victims.
    for (size_t s = 0; s < L.slot_expert.size(); ++s)
        if (L.slot_expert[s] < 0) return static_cast<int32_t>(s);
    for (int32_t s = L.tail; s >= 0; s = L.prev[s])
        if (L.slot_ready[s] && !L.slot_pinned[s]) return s;
    return -1;
}

bool MoeLruCache::admit(uint32_t layer, uint32_t expert, bool force) {
    Layer& L = layers_[layer];
    if (L.expert_slot[expert] >= 0) return false;
    if (!force && (admits_this_step_ >= cfg_.max_admits_per_step || L.admits_this_step >= cfg_.max_admits_per_layer_step))
        return false;
    const int32_t s = pick_victim(L);
    if (s < 0) return false;
    Admission a;
    a.layer = layer;
    a.expert = expert;
    a.slot = static_cast<uint32_t>(s);
    a.evicted = L.slot_expert[s];
    if (a.evicted >= 0) L.expert_slot[a.evicted] = -1;
    a.ticket = next_ticket_++;
    L.slot_expert[s] = static_cast<int32_t>(expert);
    L.slot_ready[s] = 0;
    L.slot_ticket[s] = a.ticket;
    L.expert_slot[expert] = s;
    L.miss_count[expert] = 0.0f;
    lru_unlink(L, s);
    lru_push_front(L, s);
    pending_.push_back(a);
    if (!force) { ++admits_this_step_; ++L.admits_this_step; }
    return true;
}

void MoeLruCache::route(uint32_t layer, const int32_t* experts, uint32_t n, int32_t* out) {
    Layer& L = layers_[layer];
    for (uint32_t i = 0; i < n; ++i) {
        const int32_t e = experts[i];
        out[i] = -1;
        if (e < 0 || static_cast<uint32_t>(e) >= cfg_.n_experts) continue;
        const int32_t s = L.expert_slot[e];
        if (s >= 0 && L.slot_ready[s]) {
            out[i] = s;
            ++L.stats.hits;
            lru_unlink(L, s);
            lru_push_front(L, s);
            continue;
        }
        ++L.stats.misses;
        L.step_misses += 1.0;
        if (s >= 0) continue;  // loading: counts as a miss, nothing to admit
        L.miss_count[e] += 1.0f;
        if (std::lround(L.miss_count[e]) >= static_cast<long>(cfg_.admit_after_misses)) admit(layer, static_cast<uint32_t>(e), false);
    }
}

void MoeLruCache::end_step() {
    admits_this_step_ = 0;
    for (auto& L : layers_) {
        L.admits_this_step = 0;
        for (auto& c : L.miss_count) c *= static_cast<float>(cfg_.miss_decay);
        L.stats.pressure = 0.9 * L.stats.pressure + 0.1 * L.step_misses;
        L.step_misses = 0.0;
    }
}

std::vector<Admission> MoeLruCache::take_admissions() {
    std::vector<Admission> out;
    out.swap(pending_);
    return out;
}

bool MoeLruCache::commit(const Admission& a) {
    if (a.layer >= layers_.size()) return false;
    Layer& L = layers_[a.layer];
    if (a.slot >= L.slot_expert.size() || L.slot_ticket[a.slot] != a.ticket ||
        L.slot_expert[a.slot] != static_cast<int32_t>(a.expert))
        return false;
    L.slot_ready[a.slot] = 1;
    return true;
}

void MoeLruCache::pin(uint32_t layer, uint32_t expert) {
    Layer& L = layers_[layer];
    if (L.expert_slot[expert] < 0) admit(layer, expert, true);
    const int32_t s = L.expert_slot[expert];
    if (s >= 0) L.slot_pinned[s] = 1;
}

void MoeLruCache::unpin(uint32_t layer, uint32_t expert) {
    Layer& L = layers_[layer];
    const int32_t s = L.expert_slot[expert];
    if (s >= 0) L.slot_pinned[s] = 0;
}

void MoeLruCache::resize(uint32_t layer, uint32_t slots) {
    Layer& L = layers_[layer];
    slots = std::min(slots, cfg_.n_experts);
    const size_t old = L.slot_expert.size();
    if (slots < old) {
        // Keep the most recently used `slots` entries (pinned first), compact them into slots [0, slots).
        std::vector<int32_t> keep;
        for (int32_t s = L.head; s >= 0; s = L.next[s]) if (L.slot_pinned[s] && L.slot_expert[s] >= 0) keep.push_back(s);
        for (int32_t s = L.head; s >= 0; s = L.next[s]) if (!L.slot_pinned[s] && L.slot_expert[s] >= 0) keep.push_back(s);
        if (keep.size() > slots) keep.resize(slots);
        Layer N;
        N.expert_slot.assign(cfg_.n_experts, -1);
        N.miss_count = L.miss_count;
        N.stats = L.stats;
        N.step_misses = L.step_misses;
        N.admits_this_step = L.admits_this_step;
        N.slot_expert.assign(slots, -1);
        N.slot_ready.assign(slots, 0);
        N.slot_pinned.assign(slots, 0);
        N.slot_ticket.assign(slots, 0);
        N.prev.assign(slots, -1);
        N.next.assign(slots, -1);
        // Moved entries keep their ready bit only if they stay in the same slot; a moved slot must be re-uploaded,
        // so it becomes a fresh admission (new ticket).
        std::vector<uint8_t> used(slots, 0);
        std::vector<int32_t> moved;
        for (int32_t s : keep) if (static_cast<uint32_t>(s) < slots) used[s] = 1;
        for (int32_t s : keep) {
            const int32_t e = L.slot_expert[s];
            if (static_cast<uint32_t>(s) < slots) {
                N.slot_expert[s] = e; N.slot_ready[s] = L.slot_ready[s]; N.slot_pinned[s] = L.slot_pinned[s];
                N.slot_ticket[s] = L.slot_ticket[s]; N.expert_slot[e] = s;
            } else {
                moved.push_back(s);
            }
        }
        size_t free_pos = 0;
        for (int32_t s : moved) {
            while (used[free_pos]) ++free_pos;
            used[free_pos] = 1;
            const int32_t e = L.slot_expert[s];
            N.slot_expert[free_pos] = e; N.slot_ready[free_pos] = 0; N.slot_pinned[free_pos] = L.slot_pinned[s];
            N.slot_ticket[free_pos] = next_ticket_; N.expert_slot[e] = static_cast<int32_t>(free_pos);
            pending_.push_back({layer, static_cast<uint32_t>(e), static_cast<uint32_t>(free_pos), -1, next_ticket_++});
        }
        // LRU order: follow the old order over kept entries, then empty slots.
        N.head = N.tail = -1;
        std::vector<int32_t> order;
        for (int32_t s : keep) order.push_back(N.expert_slot[L.slot_expert[s]]);
        for (uint32_t s = 0; s < slots; ++s) if (N.slot_expert[s] < 0) order.push_back(static_cast<int32_t>(s));
        for (size_t i = order.size(); i-- > 0;) lru_push_front(N, order[i]);
        // Pending admissions that pointed at removed slots are now stale: drop them.
        pending_.erase(std::remove_if(pending_.begin(), pending_.end(), [&](const Admission& a) {
                           return a.layer == layer && (a.slot >= slots || N.slot_ticket[a.slot] != a.ticket);
                       }), pending_.end());
        L = std::move(N);
    } else if (slots > old) {
        L.slot_expert.resize(slots, -1);
        L.slot_ready.resize(slots, 0);
        L.slot_pinned.resize(slots, 0);
        L.slot_ticket.resize(slots, 0);
        L.prev.resize(slots, -1);
        L.next.resize(slots, -1);
        if (old == 0) rebuild_lru(L);
        else for (size_t s = old; s < slots; ++s) {  // new empty slots at the LRU tail
            L.prev[s] = L.tail; L.next[s] = -1;
            if (L.tail >= 0) L.next[L.tail] = static_cast<int32_t>(s);
            L.tail = static_cast<int32_t>(s);
            if (L.head < 0) L.head = static_cast<int32_t>(s);
        }
    }
}

void MoeLruCache::rebalance(uint32_t total) {
    const uint32_t n = cfg_.n_layers;
    if (!n) return;
    const uint32_t lo = std::min(cfg_.min_slots, cfg_.n_experts), hi = std::min(cfg_.max_slots, cfg_.n_experts);
    total = std::max(total, lo * n);
    total = std::min(total, hi * n);
    // Hamilton apportionment of (total - n*lo) by pressure, then clamp to hi and redistribute any excess.
    std::vector<uint32_t> give(n, lo);
    uint32_t left = total - lo * n;
    std::vector<double> w(n);
    for (uint32_t l = 0; l < n; ++l) w[l] = layers_[l].stats.pressure + 1e-3;
    while (left > 0) {
        double sum = 0;
        std::vector<uint32_t> open;
        for (uint32_t l = 0; l < n; ++l) if (give[l] < hi) { open.push_back(l); sum += w[l]; }
        if (open.empty()) break;
        std::vector<std::pair<double, uint32_t>> rem;
        uint32_t handed = 0;
        for (uint32_t l : open) {
            const double q = left * w[l] / sum;
            uint32_t add = std::min<uint32_t>(static_cast<uint32_t>(std::floor(q)), hi - give[l]);
            give[l] += add; handed += add;
            rem.push_back({q - std::floor(q), l});
        }
        left -= handed;
        std::stable_sort(rem.begin(), rem.end(), [](auto& a, auto& b) { return a.first > b.first; });
        for (size_t k = 0; k < rem.size() && left > 0; ++k)
            if (give[rem[k].second] < hi) { give[rem[k].second]++; --left; }
        if (handed == 0 && left > 0 && rem.empty()) break;
    }
    for (uint32_t l = 0; l < n; ++l) resize(l, give[l]);
}

std::vector<int32_t> MoeLruCache::slot_table(uint32_t layer) const {
    const Layer& L = layers_[layer];
    std::vector<int32_t> t(L.slot_expert.size(), -1);
    for (size_t s = 0; s < t.size(); ++s) if (L.slot_ready[s]) t[s] = L.slot_expert[s];
    return t;
}

int32_t MoeLruCache::slot_of(uint32_t layer, uint32_t expert) const {
    const Layer& L = layers_[layer];
    const int32_t s = L.expert_slot[expert];
    return (s >= 0 && L.slot_ready[s]) ? s : -1;
}

uint64_t MoeLruCache::total_hits() const { uint64_t h = 0; for (const auto& L : layers_) h += L.stats.hits; return h; }
uint64_t MoeLruCache::total_misses() const { uint64_t m = 0; for (const auto& L : layers_) m += L.stats.misses; return m; }
double MoeLruCache::hit_rate() const {
    const double h = static_cast<double>(total_hits()), m = static_cast<double>(total_misses());
    return h + m > 0 ? h / (h + m) : 0.0;
}

bool MoeLruCache::check_invariants() const {
    for (const auto& L : layers_) {
        const size_t n = L.slot_expert.size();
        std::vector<uint8_t> seen_e(cfg_.n_experts, 0);
        for (size_t s = 0; s < n; ++s) {
            const int32_t e = L.slot_expert[s];
            if (e < 0) continue;
            if (seen_e[e]++ || L.expert_slot[e] != static_cast<int32_t>(s)) return false;
        }
        for (uint32_t e = 0; e < cfg_.n_experts; ++e) {
            const int32_t s = L.expert_slot[e];
            if (s >= 0 && (static_cast<size_t>(s) >= n || L.slot_expert[s] != static_cast<int32_t>(e))) return false;
        }
        size_t cnt = 0;
        int32_t prev = -1;
        for (int32_t s = L.head; s >= 0; s = L.next[s]) {
            if (L.prev[s] != prev || ++cnt > n) return false;
            prev = s;
        }
        if (cnt != n || L.tail != prev) return false;
    }
    return true;
}

} // namespace vegajet::moecache

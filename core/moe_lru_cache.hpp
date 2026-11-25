// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Per-layer routed-expert cache controller (host side, no device types).
//
// Every routed expert always has a base copy in VRAM (e.g. IQ2_S Gate/Up). A cache of `slots` per layer (44 by
// default) holds faster exact copies (e.g. DOT8) of the hottest experts. This class decides which experts occupy
// the slots; the engine uploads / repacks the weights and uploads the per-layer slot table:
//   * route(): per decode step, maps each routed expert to a ready slot (hit) or -1 (miss -> base path);
//   * admission is frequency-gated (an expert needs `admit_after_misses` recent misses) so one-off experts do not
//     thrash the cache, and limited to `max_admits_per_step` uploads per step (copy bandwidth budget);
//   * victims are the least recently used unpinned ready slots; a slot being filled is "loading" and is never a hit
//     until commit() reports the upload done;
//   * resize() / rebalance() move slots between layers ("dynamic"): rebalance() apportions a total slot budget by
//     each layer's recent miss pressure, within [min_slots, max_slots].
#pragma once

#include <cstdint>
#include <vector>

namespace vegajet::moecache {

struct CacheConfig {
    uint32_t n_layers = 0;
    uint32_t n_experts = 0;
    uint32_t slots_per_layer = 44;
    uint32_t max_admits_per_step = 48;      // global upload budget per decode step
    uint32_t max_admits_per_layer_step = 1; // per layer: layers are routed in order, so a global-only budget
                                            // would starve late layers
    uint32_t admit_after_misses = 2;   // misses (within the decay window) before an expert is admitted
    double miss_decay = 0.98;          // per-step decay of the per-expert miss counters
    uint32_t min_slots = 8, max_slots = 0;  // rebalance bounds per layer (max 0 = n_experts)
};

struct Admission {
    uint32_t layer = 0;
    uint32_t expert = 0;
    uint32_t slot = 0;
    int32_t evicted = -1;   // expert that previously held the slot (-1 = slot was empty)
    uint64_t ticket = 0;    // pass to commit()
};

struct LayerStats { uint64_t hits = 0, misses = 0; double pressure = 0.0; };

class MoeLruCache {
public:
    explicit MoeLruCache(const CacheConfig& cfg);

    // Looks up `n` routed experts of `layer` for this step. out_slots[i] = ready slot or -1. Records hits / misses
    // and queues admissions (subject to the gate). Call end_step() once per decode step.
    void route(uint32_t layer, const int32_t* experts, uint32_t n, int32_t* out_slots);
    void end_step();
    // Admissions decided so far (at most max_admits_per_step per step). The slots are "loading" until commit().
    std::vector<Admission> take_admissions();
    bool commit(const Admission& a);   // false if the ticket is stale (slot re-assigned or layer resized)

    void pin(uint32_t layer, uint32_t expert);    // admitted immediately if absent (still needs commit)
    void unpin(uint32_t layer, uint32_t expert);

    void resize(uint32_t layer, uint32_t slots);          // shrink evicts LRU-first
    void rebalance(uint32_t total_slots);                 // Hamilton apportionment by miss pressure

    uint32_t slots(uint32_t layer) const { return static_cast<uint32_t>(layers_[layer].slot_expert.size()); }
    // Device slot table: expert id held by each slot (-1 empty or loading).
    std::vector<int32_t> slot_table(uint32_t layer) const;
    int32_t slot_of(uint32_t layer, uint32_t expert) const;   // ready slot or -1
    const LayerStats& stats(uint32_t layer) const { return layers_[layer].stats; }
    double hit_rate() const;
    uint64_t total_hits() const, total_misses() const;
    bool check_invariants() const;   // tests: no duplicates, maps consistent, LRU list well formed

private:
    struct Layer {
        std::vector<int32_t> slot_expert;   // expert per slot, -1 empty
        std::vector<uint8_t> slot_ready, slot_pinned;
        std::vector<uint64_t> slot_ticket;
        std::vector<int32_t> prev, next;    // LRU list over slots (head = most recent)
        int32_t head = -1, tail = -1;
        std::vector<int32_t> expert_slot;   // slot per expert, -1 absent
        std::vector<float> miss_count;
        LayerStats stats;
        double step_misses = 0.0;
        uint32_t admits_this_step = 0;
    };
    void lru_unlink(Layer& L, int32_t s);
    void lru_push_front(Layer& L, int32_t s);
    int32_t pick_victim(Layer& L) const;
    bool admit(uint32_t layer, uint32_t expert, bool force);
    void rebuild_lru(Layer& L);

    CacheConfig cfg_;
    std::vector<Layer> layers_;
    std::vector<Admission> pending_;
    uint32_t admits_this_step_ = 0;
    uint64_t next_ticket_ = 1;
};

} // namespace vegajet::moecache

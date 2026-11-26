// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/expert_stream.hpp"

#include <cstring>

namespace vegajet::expertstream {

const char* phase_name(Phase p) { return p == Phase::Prefill ? "prefill" : "decode"; }

// ------------------------------------------------------------------------------------------------- LruCache
LruCache::LruCache(uint32_t capacity) : capacity_(capacity) { slots_.resize(capacity); }

void LruCache::unlink(int32_t s) {
    Slot& n = slots_[size_t(s)];
    if (n.prev >= 0) slots_[size_t(n.prev)].next = n.next;
    else head_ = n.next;
    if (n.next >= 0) slots_[size_t(n.next)].prev = n.prev;
    else tail_ = n.prev;
    n.prev = n.next = -1;
}

void LruCache::push_front(int32_t s) {
    Slot& n = slots_[size_t(s)];
    n.prev = -1;
    n.next = head_;
    if (head_ >= 0) slots_[size_t(head_)].prev = s;
    head_ = s;
    if (tail_ < 0) tail_ = s;
}

int32_t LruCache::get(uint64_t key) {
    if (phase_ == Phase::Prefill) {
        ++st_.bypassed;
        return -1;  // prefill bypass: no lookup, no promotion
    }
    auto it = map_.find(key);
    if (it == map_.end()) {
        ++st_.misses;
        return -1;
    }
    ++st_.hits;
    unlink(it->second);
    push_front(it->second);
    return it->second;
}

int32_t LruCache::admit(uint64_t key) {
    if (phase_ == Phase::Prefill || capacity_ == 0) {
        ++st_.bypassed;
        return -1;  // prefill bypass: nothing is inserted or evicted
    }
    auto it = map_.find(key);
    if (it != map_.end()) {
        unlink(it->second);
        push_front(it->second);
        return it->second;
    }
    int32_t s;
    if (used_ < capacity_) {
        s = static_cast<int32_t>(used_++);
    } else {
        s = tail_;  // LRU slot is reused
        if (s < 0) {
            ++st_.bypassed;
            return -1;
        }
        unlink(s);
        map_.erase(slots_[size_t(s)].key);
        ++st_.evictions;
    }
    slots_[size_t(s)].key = key;
    slots_[size_t(s)].used = true;
    push_front(s);
    map_.emplace(key, s);
    ++st_.admits;
    return s;
}

bool LruCache::contains(uint64_t key) const { return map_.find(key) != map_.end(); }

std::vector<uint64_t> LruCache::order() const {
    std::vector<uint64_t> out;
    for (int32_t s = head_; s >= 0; s = slots_[size_t(s)].next) out.push_back(slots_[size_t(s)].key);
    return out;
}

bool LruCache::check() const {
    if (uint32_t(map_.size()) != used_ || map_.size() > capacity_) return false;
    uint32_t n = 0;
    int32_t prev = -1;
    for (int32_t s = head_; s >= 0; s = slots_[size_t(s)].next) {
        const Slot& node = slots_[size_t(s)];
        if (node.prev != prev || !node.used) return false;
        auto it = map_.find(node.key);
        if (it == map_.end() || it->second != s) return false;
        prev = s;
        if (++n > capacity_) return false;
    }
    if (prev != tail_) return false;
    if (used_ == 0 && (head_ != -1 || tail_ != -1)) return false;
    if (used_ != 0 && (head_ < 0 || tail_ < 0)) return false;
    return true;
}

// ------------------------------------------------------------------------------------------------- ExpertStream
ExpertStream::ExpertStream() : cache_(0) {}
ExpertStream::~ExpertStream() = default;

bool ExpertStream::open(const std::string& pack_path, uint32_t cache_slots, std::string* err) {
    expertpack::PackOptions opt;
    return open(pack_path, cache_slots, opt, err);
}

bool ExpertStream::open(const std::string& pack_path, uint32_t cache_slots, const expertpack::PackOptions& opt,
                        std::string* err) {
    if (!reader_.open(pack_path, opt, err)) return false;
    const auto& d = reader_.desc();
    row_elems_ = d.gate_elems + d.up_elems + d.down_elems;
    if (row_elems_ == 0) {
        if (err) *err = "empty expert record";
        return false;
    }
    slots_.assign(size_t(cache_slots) * row_elems_, 0.0f);
    cache_ = LruCache(cache_slots);
    st_ = Stats{};
    return true;
}

void ExpertStream::set_phase(Phase p) { cache_.set_phase(p); }

bool ExpertStream::expert(uint32_t layer, uint32_t expert, float* gate, float* up, float* down,
                          std::string* err) {
    if (!gate || !up || !down) {
        if (err) *err = "null expert buffer";
        return false;
    }
    const auto& d = reader_.desc();
    const uint64_t key = (uint64_t(layer) << 32) | expert;

    if (cache_.phase() == Phase::Decode) {
        int32_t s = cache_.get(key);
        if (s >= 0) {
            ++st_.cache_hits;
            const float* src = slot(s);
            std::memcpy(gate, src, size_t(d.gate_elems) * 4);
            std::memcpy(up, src + d.gate_elems, size_t(d.up_elems) * 4);
            std::memcpy(down, src + d.gate_elems + d.up_elems, size_t(d.down_elems) * 4);
            return true;
        }
        ++st_.cache_misses;
        s = cache_.admit(key);
        if (s >= 0) {
            float* dst = slot(s);
            if (!reader_.read_expert(layer, expert, dst, dst + d.gate_elems, dst + d.gate_elems + d.up_elems, err))
                return false;
            ++st_.pack_reads;
            std::memcpy(gate, dst, size_t(d.gate_elems) * 4);
            std::memcpy(up, dst + d.gate_elems, size_t(d.up_elems) * 4);
            std::memcpy(down, dst + d.gate_elems + d.up_elems, size_t(d.down_elems) * 4);
            return true;
        }
        // capacity 0: fall through to a direct read
    } else {
        cache_.get(key);  // account the bypass
    }
    if (!reader_.read_expert(layer, expert, gate, up, down, err)) return false;
    ++st_.pack_reads;
    return true;
}

}  // namespace vegajet::expertstream

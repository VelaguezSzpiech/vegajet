// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/paged_kv.hpp"

#include <algorithm>
#include <cstring>

namespace vegajet::paged {

namespace {
uint64_t mix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
} // namespace

PagedKv::PagedKv(uint32_t n_blocks, uint32_t block_tokens)
    : bt_(std::max<uint32_t>(1, block_tokens)), blk_(n_blocks), store_(static_cast<size_t>(n_blocks) * bt_, -1) {
    free_.reserve(n_blocks);
    for (uint32_t i = n_blocks; i-- > 0;) free_.push_back(static_cast<BlockId>(i));
}

uint64_t PagedKv::chain(uint64_t prev, const int32_t* t) const {
    uint64_t h = mix(prev ^ 0xA5A5A5A5u);
    for (uint32_t i = 0; i < bt_; ++i) h = mix(h ^ static_cast<uint32_t>(t[i]));
    return h;
}

void PagedKv::lru_remove(BlockId b) {
    Block& k = blk_[b];
    if (!k.in_lru) return;
    if (k.lru_prev >= 0) blk_[k.lru_prev].lru_next = k.lru_next; else lru_head_ = k.lru_next;
    if (k.lru_next >= 0) blk_[k.lru_next].lru_prev = k.lru_prev; else lru_tail_ = k.lru_prev;
    k.lru_prev = k.lru_next = -1;
    k.in_lru = false;
    --lru_size_;
}

void PagedKv::lru_push(BlockId b) {
    Block& k = blk_[b];
    k.lru_prev = -1;
    k.lru_next = lru_head_;
    if (lru_head_ >= 0) blk_[lru_head_].lru_prev = b;
    lru_head_ = b;
    if (lru_tail_ < 0) lru_tail_ = b;
    k.in_lru = true;
    ++lru_size_;
}

BlockId PagedKv::alloc() {
    BlockId b = -1;
    if (!free_.empty()) {
        b = free_.back();
        free_.pop_back();
    } else if (lru_tail_ >= 0) {
        b = lru_tail_;  // least recently released cached block
        lru_remove(b);
        prefix_.erase(blk_[b].hash);
        ++st_.evictions;
    } else {
        return -1;
    }
    Block& k = blk_[b];
    k.ref = 1;
    k.count = 0;
    k.hashed = false;
    k.hash = 0;
    k.last_use = ++tick_;
    return b;
}

void PagedKv::unref(BlockId b) {
    Block& k = blk_[b];
    if (k.ref == 0) return;
    if (--k.ref > 0) return;
    if (k.hashed) lru_push(b);
    else free_.push_back(b);
}

SeqId PagedKv::create() {
    const SeqId id = next_seq_++;
    seqs_[id].live = true;
    return id;
}

uint32_t PagedKv::attach_prefix(SeqId s, const int32_t* t, uint32_t n) {
    Seq& q = seqs_.at(s);
    if (!q.tokens.empty()) return 0;
    uint64_t h = 0;
    for (uint32_t i = 0; i + bt_ <= n; i += bt_) {
        const uint64_t hh = chain(h, t + i);
        const auto it = prefix_.find(hh);
        if (it == prefix_.end()) break;
        const BlockId b = it->second;
        if (std::memcmp(&store_[static_cast<size_t>(b) * bt_], t + i, bt_ * sizeof(int32_t)) != 0) break;  // collision
        Block& k = blk_[b];
        if (k.ref == 0) lru_remove(b);
        ++k.ref;
        k.last_use = ++tick_;
        q.table.push_back(b);
        q.hashes.push_back(hh);
        q.tokens.insert(q.tokens.end(), t + i, t + i + bt_);
        h = hh;
    }
    st_.prefix_hit_tokens += q.tokens.size();
    return static_cast<uint32_t>(q.tokens.size());
}

void PagedKv::seal_full_blocks(Seq& q) {
    while (q.hashes.size() < q.tokens.size() / bt_) {
        const size_t i = q.hashes.size();
        const uint64_t prev = i ? q.hashes[i - 1] : 0;
        const uint64_t h = chain(prev, &q.tokens[i * bt_]);
        q.hashes.push_back(h);
        const BlockId b = q.table[i];
        if (!blk_[b].hashed && !prefix_.count(h)) {
            blk_[b].hashed = true;
            blk_[b].hash = h;
            prefix_.emplace(h, b);
        }
    }
}

bool PagedKv::append(SeqId s, const int32_t* t, uint32_t n) {
    Seq& q = seqs_.at(s);
    const uint32_t have = static_cast<uint32_t>(q.table.size());
    const uint32_t need = blocks_needed(static_cast<uint32_t>(q.tokens.size()) + n);
    if (need > have && need - have > available()) return false;
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t pos = static_cast<uint32_t>(q.tokens.size());
        if (pos % bt_ == 0) q.table.push_back(alloc());  // cannot fail: checked above
        const BlockId b = q.table.back();
        store_[static_cast<size_t>(b) * bt_ + pos % bt_] = t[i];
        blk_[b].count = pos % bt_ + 1;
        blk_[b].last_use = tick_;
        q.tokens.push_back(t[i]);
    }
    seal_full_blocks(q);
    return true;
}

void PagedKv::truncate(SeqId s, uint32_t n) {
    Seq& q = seqs_.at(s);
    if (n >= q.tokens.size()) return;
    const uint32_t keep_blocks = blocks_needed(n);
    while (q.table.size() > keep_blocks) { unref(q.table.back()); q.table.pop_back(); }
    q.tokens.resize(n);
    q.hashes.resize(std::min<size_t>(q.hashes.size(), n / bt_));
    if (n % bt_ == 0 || q.table.empty()) return;
    // The new last block is partial. Sole owner: unregister it and keep writing in place. Shared: copy it (COW).
    const BlockId b = q.table.back();
    if (blk_[b].ref == 1 && blk_[b].hashed) {
        prefix_.erase(blk_[b].hash);
        blk_[b].hashed = false;
    }
    if (blk_[b].ref > 1) {
        BlockId c = alloc();
        if (c < 0) {  // no block to copy into: drop the partial tail instead (caller re-prefills it)
            unref(b);
            q.table.pop_back();
            q.tokens.resize(static_cast<size_t>(q.table.size()) * bt_);
            return;
        }
        std::memcpy(&store_[static_cast<size_t>(c) * bt_], &store_[static_cast<size_t>(b) * bt_], (n % bt_) * sizeof(int32_t));
        blk_[c].count = n % bt_;
        copies_.push_back({b, c, n % bt_});
        ++st_.cow_copies;
        unref(b);
        q.table.back() = c;
    } else {
        blk_[b].count = n % bt_;
    }
}

SeqId PagedKv::fork(SeqId s) {
    const SeqId id = create();
    Seq& src = seqs_.at(s);
    Seq& dst = seqs_.at(id);
    const uint32_t full = static_cast<uint32_t>(src.tokens.size()) / bt_;
    for (uint32_t i = 0; i < full; ++i) { ++blk_[src.table[i]].ref; dst.table.push_back(src.table[i]); }
    dst.hashes = src.hashes;
    dst.tokens.assign(src.tokens.begin(), src.tokens.begin() + static_cast<size_t>(full) * bt_);
    const uint32_t tail = static_cast<uint32_t>(src.tokens.size()) % bt_;
    if (tail) {  // private copy of the partial block (best effort: the tail is dropped if memory is exhausted)
        const BlockId c = alloc();
        if (c >= 0) {
            const BlockId b = src.table.back();
            std::memcpy(&store_[static_cast<size_t>(c) * bt_], &store_[static_cast<size_t>(b) * bt_], tail * sizeof(int32_t));
            blk_[c].count = tail;
            copies_.push_back({b, c, tail});
            ++st_.cow_copies;
            dst.table.push_back(c);
            dst.tokens.insert(dst.tokens.end(), src.tokens.end() - tail, src.tokens.end());
        }
    }
    return id;
}

void PagedKv::release(SeqId s) {
    auto it = seqs_.find(s);
    if (it == seqs_.end()) return;
    for (auto b = it->second.table.rbegin(); b != it->second.table.rend(); ++b) unref(*b);  // tail first: LRU keeps roots
    seqs_.erase(it);
}

std::vector<CopyOp> PagedKv::take_copies() {
    std::vector<CopyOp> c;
    c.swap(copies_);
    return c;
}

KvStats PagedKv::stats() const {
    KvStats s = st_;
    s.free = static_cast<uint32_t>(free_.size());
    s.cached = lru_size_;
    s.used = static_cast<uint32_t>(blk_.size()) - s.free - s.cached;
    return s;
}

std::vector<int32_t> PagedKv::block_contents(BlockId b) const {
    return std::vector<int32_t>(store_.begin() + static_cast<size_t>(b) * bt_,
                                store_.begin() + static_cast<size_t>(b) * bt_ + blk_[b].count);
}

bool PagedKv::check_invariants() const {
    std::vector<uint32_t> refs(blk_.size(), 0);
    for (const auto& [id, q] : seqs_) {
        if (q.table.size() != blocks_needed(static_cast<uint32_t>(q.tokens.size()))) return false;
        for (size_t i = 0; i < q.table.size(); ++i) {
            const BlockId b = q.table[i];
            if (b < 0 || static_cast<size_t>(b) >= blk_.size()) return false;
            refs[b]++;
            const size_t lo = i * bt_, hi = std::min(q.tokens.size(), lo + bt_);
            if (blk_[b].count < hi - lo) return false;
            for (size_t k = lo; k < hi; ++k) if (store_[static_cast<size_t>(b) * bt_ + (k - lo)] != q.tokens[k]) return false;
            if (hi - lo < bt_ && blk_[b].ref != 1) return false;  // partial blocks are private
        }
    }
    uint32_t in_free = 0, in_lru = 0;
    std::vector<uint8_t> is_free(blk_.size(), 0);
    for (BlockId b : free_) { if (is_free[b]++) return false; ++in_free; }
    for (int32_t b = lru_head_; b >= 0; b = blk_[b].lru_next) { if (!blk_[b].in_lru || !blk_[b].hashed || is_free[b]) return false; ++in_lru; }
    if (in_lru != lru_size_) return false;
    for (size_t b = 0; b < blk_.size(); ++b) {
        if (blk_[b].ref != refs[b]) return false;
        if (refs[b] == 0 && !is_free[b] && !blk_[b].in_lru) return false;   // leaked
        if (refs[b] > 0 && (is_free[b] || blk_[b].in_lru)) return false;
    }
    for (const auto& [h, b] : prefix_) if (!blk_[b].hashed || blk_[b].hash != h) return false;
    return in_free + in_lru + static_cast<size_t>(std::count_if(refs.begin(), refs.end(), [](uint32_t r) { return r > 0; })) == blk_.size();
}

} // namespace vegajet::paged

// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/prefix_cache.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace vegajet::prefix {

// ================================================================================================= BlockPool
BlockPool::BlockPool(uint32_t n_blocks) : ref_(n_blocks, 0u) {
    free_.reserve(n_blocks);
    for (uint32_t i = n_blocks; i-- > 0;) free_.push_back(static_cast<BlockId>(i));  // alloc() hands out 0, 1, ...
}

BlockId BlockPool::alloc() {
    if (free_.empty()) return -1;
    const BlockId b = free_.back();
    free_.pop_back();
    ref_[static_cast<size_t>(b)] = 1;
    return b;
}

void BlockPool::ref(BlockId b) {
    auto& r = ref_.at(static_cast<size_t>(b));
    if (r == 0) throw std::logic_error("BlockPool::ref on a free block");
    ++r;
}

void BlockPool::unref(BlockId b) {
    auto& r = ref_.at(static_cast<size_t>(b));
    if (r == 0) throw std::logic_error("BlockPool::unref on a free block");
    if (--r == 0) free_.push_back(b);
}

// ================================================================================================= tree
// A node's edge holds whole blocks; `depth` is the prefix length (tokens) at the end of the edge. A match pins only
// its deepest node: every ancestor then has a child and is never an eviction candidate (only leaves are). sub_pins
// (pins in the subtree) protects snapshots along pinned paths, which a sequence may be about to restore.
struct Node {
    Node* parent = nullptr;
    std::vector<int32_t> toks;
    std::vector<BlockId> blocks;
    std::vector<std::unique_ptr<Node>> kids;
    uint32_t depth = 0;
    uint32_t pins = 0, sub_pins = 0;
    uint64_t last = 0;
    SnapId snap = -1;
};

namespace {

void add_pins(Node* n, int delta) {
    n->pins = static_cast<uint32_t>(static_cast<int>(n->pins) + delta);
    for (Node* p = n; p; p = p->parent) p->sub_pins = static_cast<uint32_t>(static_cast<int>(p->sub_pins) + delta);
}

template <class F> void walk(Node* n, F&& f) {
    std::vector<Node*> st{n};
    while (!st.empty()) {
        Node* x = st.back();
        st.pop_back();
        f(x);
        for (auto& k : x->kids) st.push_back(k.get());
    }
}

} // namespace

RadixCache::RadixCache(BlockPool& pool, uint32_t block_tokens, uint32_t max_snapshots, bool hybrid)
    : pool_(pool), bt_(block_tokens ? block_tokens : 1), hybrid_(hybrid), root_(new Node) {
    snap_free_flag_.assign(max_snapshots, 1u);
    for (uint32_t i = max_snapshots; i-- > 0;) snap_free_.push_back(static_cast<SnapId>(i));
}

RadixCache::~RadixCache() {
    walk(root_.get(), [&](Node* n) {
        for (BlockId b : n->blocks) pool_.unref(b);
    });
}

Node* RadixCache::child_for(Node* n, const int32_t* block) const {
    for (auto& k : n->kids)
        if (std::memcmp(k->toks.data(), block, bt_ * sizeof(int32_t)) == 0) return k.get();
    return nullptr;
}

void RadixCache::touch(Node* n) {
    const uint64_t t = ++clock_;
    for (Node* p = n; p; p = p->parent) p->last = t;
}

// Splits n's edge after `at` blocks: a new upper node takes blocks [0, at); n keeps the rest as its only child.
Node* RadixCache::split(Node* n, uint32_t at) {
    Node* parent = n->parent;
    auto upper = std::make_unique<Node>();
    Node* u = upper.get();
    u->parent = parent;
    u->toks.assign(n->toks.begin(), n->toks.begin() + static_cast<long>(at) * bt_);
    u->blocks.assign(n->blocks.begin(), n->blocks.begin() + at);
    u->depth = n->depth - static_cast<uint32_t>(n->blocks.size() - at) * bt_;
    u->sub_pins = n->sub_pins;
    u->last = n->last;
    n->toks.erase(n->toks.begin(), n->toks.begin() + static_cast<long>(at) * bt_);
    n->blocks.erase(n->blocks.begin(), n->blocks.begin() + at);
    for (auto& k : parent->kids) {
        if (k.get() == n) {
            std::unique_ptr<Node> lower = std::move(k);
            lower->parent = u;
            u->kids.push_back(std::move(lower));
            k = std::move(upper);
            break;
        }
    }
    return u;
}

Match RadixCache::match(const int32_t* toks, uint32_t n) {
    Match m;
    ++st_.lookups;
    Node* cur = root_.get();
    Node* deepest = nullptr;   // deepest node whose edge was entered
    uint32_t pos = 0;
    uint32_t snap_tokens = 0;
    SnapId snap = -1;
    Node* snap_node = nullptr;
    while (pos + bt_ <= n) {
        Node* k = child_for(cur, toks + pos);
        if (!k) break;
        size_t j = 0;
        while (j < k->blocks.size() && pos + bt_ <= n &&
               std::memcmp(k->toks.data() + j * bt_, toks + pos, bt_ * sizeof(int32_t)) == 0) {
            m.blocks.push_back(k->blocks[j]);
            pos += bt_;
            ++j;
        }
        deepest = k;
        if (j < k->blocks.size()) break;  // diverged (or prompt ended) inside the edge
        if (k->snap >= 0) { snap = k->snap; snap_tokens = pos; snap_node = k; }
        cur = k;
    }
    if (hybrid_) {
        m.tokens = snap_tokens;
        m.snapshot = snap;
        m.blocks.resize(snap_tokens / bt_);
        deepest = snap_node;
    } else {
        m.tokens = pos;
    }
    if (m.tokens > 0 && deepest) {
        add_pins(deepest, +1);
        m.pinned.push_back(deepest);
        touch(deepest);
        ++st_.hits;
    }
    st_.hit_tokens += m.tokens;
    st_.miss_tokens += n - m.tokens;
    return m;
}

void RadixCache::release(Match& m) {
    for (Node* p : m.pinned) add_pins(p, -1);
    m.pinned.clear();
}

bool RadixCache::insert(const int32_t* toks, uint32_t n, const BlockId* blocks, SnapId snap, uint32_t snap_tokens) {
    const uint32_t nb = n / bt_;
    bool keep_snap = snap >= 0 && snap_tokens > 0 && snap_tokens % bt_ == 0 && snap_tokens <= nb * bt_;
    Node* cur = root_.get();
    uint32_t b = 0;
    Node* snap_target = nullptr;
    while (b < nb) {
        Node* k = child_for(cur, toks + static_cast<size_t>(b) * bt_);
        if (!k) {
            auto fresh = std::make_unique<Node>();
            fresh->parent = cur;
            fresh->toks.assign(toks + static_cast<size_t>(b) * bt_, toks + static_cast<size_t>(nb) * bt_);
            fresh->blocks.assign(blocks + b, blocks + nb);
            for (BlockId x : fresh->blocks) pool_.ref(x);
            st_.inserted_blocks += nb - b;
            fresh->depth = cur->depth + (nb - b) * bt_;
            Node* f = fresh.get();
            cur->kids.push_back(std::move(fresh));
            cur = f;
            b = nb;
            break;
        }
        uint32_t j = 0;
        while (j < k->blocks.size() && b < nb &&
               std::memcmp(k->toks.data() + static_cast<size_t>(j) * bt_, toks + static_cast<size_t>(b) * bt_,
                           bt_ * sizeof(int32_t)) == 0) {
            ++j;
            ++b;
        }
        if (j < k->blocks.size()) k = split(k, j);
        cur = k;
    }
    if (cur != root_.get()) touch(cur);
    if (keep_snap) {
        // Find (or split out) the node ending exactly at snap_tokens on the path just inserted.
        Node* x = root_.get();
        uint32_t d = 0;
        while (d < snap_tokens) {
            Node* k = child_for(x, toks + d);
            if (!k) { x = nullptr; break; }
            const uint32_t edge = static_cast<uint32_t>(k->blocks.size()) * bt_;
            if (d + edge > snap_tokens) k = split(k, (snap_tokens - d) / bt_);
            d += static_cast<uint32_t>(k->blocks.size()) * bt_;
            x = k;
        }
        snap_target = x;
        if (!snap_target || snap_target->snap >= 0) keep_snap = false;
    }
    if (keep_snap) {
        snap_target->snap = snap;
        touch(snap_target);
        return true;
    }
    if (snap >= 0) free_snapshot(snap);
    return false;
}

SnapId RadixCache::alloc_snapshot() {
    if (snap_free_.empty() && !evict_one_snapshot()) return -1;
    const SnapId s = snap_free_.back();
    snap_free_.pop_back();
    snap_free_flag_[static_cast<size_t>(s)] = 0;
    return s;
}

void RadixCache::free_snapshot(SnapId s) {
    if (s < 0 || static_cast<size_t>(s) >= snap_free_flag_.size() || snap_free_flag_[static_cast<size_t>(s)])
        throw std::logic_error("RadixCache::free_snapshot: invalid or already free snapshot");
    snap_free_flag_[static_cast<size_t>(s)] = 1;
    snap_free_.push_back(s);
}

bool RadixCache::evict_one_snapshot() {
    Node* best = nullptr;
    walk(root_.get(), [&](Node* n) {
        if (n->snap >= 0 && n->sub_pins == 0 && (!best || n->last < best->last)) best = n;
    });
    if (!best) return false;
    free_snapshot(best->snap);
    best->snap = -1;
    ++st_.evicted_snapshots;
    return true;
}

void RadixCache::drop_leaf(Node* n) {
    for (BlockId b : n->blocks) pool_.unref(b);
    st_.evicted_blocks += n->blocks.size();
    if (n->snap >= 0) { free_snapshot(n->snap); ++st_.evicted_snapshots; }
    Node* p = n->parent;
    for (size_t i = 0; i < p->kids.size(); ++i)
        if (p->kids[i].get() == n) { p->kids.erase(p->kids.begin() + static_cast<long>(i)); break; }
}

uint32_t RadixCache::evict_blocks(uint32_t want) {
    const uint32_t before = pool_.free_count();
    while (pool_.free_count() < want) {
        std::vector<Node*> leaves;
        walk(root_.get(), [&](Node* n) {
            if (n != root_.get() && n->kids.empty() && n->sub_pins == 0) leaves.push_back(n);
        });
        if (leaves.empty()) break;
        std::sort(leaves.begin(), leaves.end(), [](const Node* a, const Node* b) { return a->last < b->last; });
        for (Node* l : leaves) {
            drop_leaf(l);
            if (pool_.free_count() >= want) break;
        }
    }
    return pool_.free_count() - before;
}

void RadixCache::clear() {
    for (;;) {
        std::vector<Node*> leaves;
        walk(root_.get(), [&](Node* n) {
            if (n != root_.get() && n->kids.empty() && n->sub_pins == 0) leaves.push_back(n);
        });
        if (leaves.empty()) break;
        for (Node* l : leaves) drop_leaf(l);
    }
}

CacheStats RadixCache::stats() const {
    CacheStats s = st_;
    s.nodes = 0;
    s.cached_blocks = 0;
    s.snapshots = 0;
    walk(root_.get(), [&](Node* n) {
        if (n == root_.get()) return;
        ++s.nodes;
        s.cached_blocks += static_cast<uint32_t>(n->blocks.size());
        s.snapshots += n->snap >= 0;
    });
    return s;
}

std::vector<uint32_t> RadixCache::cache_refs() const {
    std::vector<uint32_t> r(pool_.capacity(), 0u);
    walk(root_.get(), [&](Node* n) {
        for (BlockId b : n->blocks) ++r[static_cast<size_t>(b)];
    });
    return r;
}

bool RadixCache::check_invariants() const {
    bool ok = true;
    std::vector<uint8_t> seen(pool_.capacity(), 0u), snap_seen(snap_free_flag_.size(), 0u);
    walk(root_.get(), [&](Node* n) {
        uint32_t sub = n->pins;
        for (const auto& k : n->kids) {
            sub += k->sub_pins;
            if (k->parent != n || k->blocks.empty() || k->toks.size() != k->blocks.size() * bt_) ok = false;
            if (k->depth != n->depth + k->blocks.size() * bt_) ok = false;
        }
        if (sub != n->sub_pins) ok = false;
        for (size_t a = 0; a < n->kids.size(); ++a)  // children differ in their first block
            for (size_t b = a + 1; b < n->kids.size(); ++b)
                if (std::memcmp(n->kids[a]->toks.data(), n->kids[b]->toks.data(), bt_ * sizeof(int32_t)) == 0) ok = false;
        for (BlockId b : n->blocks) {
            if (b < 0 || static_cast<uint32_t>(b) >= pool_.capacity() || seen[static_cast<size_t>(b)] || pool_.refs(b) == 0) ok = false;
            else seen[static_cast<size_t>(b)] = 1;
        }
        if (n->snap >= 0) {
            if (static_cast<size_t>(n->snap) >= snap_seen.size() || snap_seen[static_cast<size_t>(n->snap)] ||
                snap_free_flag_[static_cast<size_t>(n->snap)])
                ok = false;
            else snap_seen[static_cast<size_t>(n->snap)] = 1;
        }
    });
    return ok;
}

} // namespace vegajet::prefix

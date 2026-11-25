// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Radix-tree prefix cache over refcounted paged-KV blocks, with GDN / SSM recurrent-state snapshots.
//
// Agent workloads resend the same 10K-40K token system prompt and tool schemas with every request. The cache maps
// block-aligned token prefixes to the KV blocks that hold them, so a new request prefills only what follows the
// longest cached prefix:
//   * Radix tree: each edge carries whole blocks (tokens + block ids); children are told apart by their first block.
//     match() walks the prompt once, comparing tokens: O(L) in the prompt length (fan-out scans compare one block).
//     Edges split at block boundaries when prefixes diverge.
//   * Hybrid models (Gated DeltaNet / Mamba layers next to attention) cannot resume from KV alone: the recurrent
//     state after the prefix is needed too. Snapshots are stored at node ends; for a hybrid cache, match() returns
//     the deepest prefix that ends at a snapshot. Snapshot slots come from a fixed pool (device memory per slot).
//   * Refcounts: the cache holds one reference per block it stores; a sequence that reuses cached blocks takes its
//     own references. match() pins the matched path (release() unpins); pinned nodes are never evicted.
//   * Eviction: least recently used unpinned leaves first (a leaf's parent becomes a leaf in turn), until the
//     requested number of blocks is free; snapshots are evicted LRU when the snapshot pool is full.
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace vegajet::prefix {

using BlockId = int32_t;
using SnapId = int32_t;

// Fixed pool of KV blocks with reference counts (block 0..n-1; the engine owns the device memory).
class BlockPool {
public:
    explicit BlockPool(uint32_t n_blocks);
    BlockId alloc();                 // refcount 1, or -1 when none is free
    void ref(BlockId b);
    void unref(BlockId b);           // back to the free list at 0
    uint32_t refs(BlockId b) const { return ref_.at(static_cast<size_t>(b)); }
    uint32_t free_count() const { return static_cast<uint32_t>(free_.size()); }
    uint32_t capacity() const { return static_cast<uint32_t>(ref_.size()); }

private:
    std::vector<uint32_t> ref_;
    std::vector<BlockId> free_;
};

struct Node;

struct Match {
    uint32_t tokens = 0;              // reusable prefix length (block aligned)
    std::vector<BlockId> blocks;      // tokens / block_tokens blocks, in order
    SnapId snapshot = -1;             // hybrid: state after `tokens` tokens; -1 otherwise
    std::vector<Node*> pinned;        // nodes pinned by this match (pass the Match to release())
};

struct CacheStats {
    uint64_t lookups = 0, hits = 0, hit_tokens = 0, miss_tokens = 0;
    uint64_t evicted_blocks = 0, evicted_snapshots = 0, inserted_blocks = 0;
    uint32_t nodes = 0, cached_blocks = 0, snapshots = 0;
};

class RadixCache {
public:
    RadixCache(BlockPool& pool, uint32_t block_tokens, uint32_t max_snapshots, bool hybrid);
    ~RadixCache();
    RadixCache(const RadixCache&) = delete;
    RadixCache& operator=(const RadixCache&) = delete;

    // Longest cached prefix of toks[0, n) (hybrid: ending at a snapshot), pinned until release(). The caller takes its
    // own block references if it keeps the blocks.
    Match match(const int32_t* toks, uint32_t n);
    void release(Match& m);

    // Caches the block-aligned prefix of toks[0, n) whose KV lives in blocks[0, n / block_tokens). New nodes take a
    // reference on their blocks; prefixes already cached keep their blocks. If snap >= 0 it is the recurrent state
    // after snap_tokens tokens (block aligned, <= n): the cache takes ownership (returns false and frees it when the
    // node already has one or the point is not cacheable).
    bool insert(const int32_t* toks, uint32_t n, const BlockId* blocks, SnapId snap = -1, uint32_t snap_tokens = 0);

    // Snapshot slots: alloc evicts the least recently used unpinned snapshot when the pool is full (-1 if impossible).
    SnapId alloc_snapshot();
    void free_snapshot(SnapId s);
    uint32_t snapshot_capacity() const { return static_cast<uint32_t>(snap_free_flag_.size()); }
    uint32_t snapshots_free() const { return static_cast<uint32_t>(snap_free_.size()); }

    // Evicts LRU unpinned leaves until `want` blocks are free in the pool (or nothing evictable is left).
    // Returns the number of blocks that became free.
    uint32_t evict_blocks(uint32_t want);
    void clear();  // drops every unpinned node (tests / shutdown)

    CacheStats stats() const;
    uint32_t block_tokens() const { return bt_; }
    bool hybrid() const { return hybrid_; }
    // Each block's cache-held reference count (0 or 1), for refcount audits.
    std::vector<uint32_t> cache_refs() const;
    bool check_invariants() const;

private:
    Node* child_for(Node* n, const int32_t* block) const;
    Node* split(Node* n, uint32_t at_blocks);
    void touch(Node* n);
    void drop_leaf(Node* n);
    bool evict_one_snapshot();

    BlockPool& pool_;
    uint32_t bt_;
    bool hybrid_;
    std::unique_ptr<Node> root_;
    uint64_t clock_ = 0;
    std::vector<SnapId> snap_free_;
    std::vector<uint8_t> snap_free_flag_;
    CacheStats st_;
};

} // namespace vegajet::prefix

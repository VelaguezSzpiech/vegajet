// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Host-side paged KV block allocator with a prefix-hash cache (the device stores K/V in blocks; this class owns
// which block holds which tokens). No device types: the engine applies the CopyOps and uploads block tables.
//
//  * Blocks hold `block_tokens` positions. A full block is identified by the hash chain h_i = H(h_{i-1}, tokens_i),
//    so equal hashes mean equal prefixes; the token ids are also stored and compared, so a hash collision can never
//    alias KV data.
//  * attach_prefix() maps the longest cached full-block prefix of a prompt into a new sequence (refcount++), so a
//    shared system prompt is prefilled once.
//  * Only full blocks are shared. The last, partial block of a sequence is always private: fork() and truncate()
//    copy it when needed (copy-on-write) and report the copy as a CopyOp for the engine.
//  * Unreferenced full blocks stay cached (evictable, LRU) until an allocation needs them; unreferenced partial
//    blocks go straight to the free list.
//  * append() is all-or-nothing: it fails without side effects when the blocks cannot be found.
#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace vegajet::paged {

using BlockId = int32_t;
using SeqId = uint32_t;

struct CopyOp { BlockId src = -1, dst = -1; uint32_t tokens = 0; };

struct KvStats {
    uint32_t free = 0, cached = 0, used = 0;   // used = referenced by at least one sequence
    uint64_t prefix_hit_tokens = 0, evictions = 0, cow_copies = 0;
};

class PagedKv {
public:
    PagedKv(uint32_t n_blocks, uint32_t block_tokens);

    SeqId create();
    // On an empty sequence: shares the longest cached prefix of whole blocks. Returns the tokens now covered.
    uint32_t attach_prefix(SeqId s, const int32_t* tokens, uint32_t n);
    bool append(SeqId s, const int32_t* tokens, uint32_t n);
    void truncate(SeqId s, uint32_t n_tokens);   // speculative rollback
    SeqId fork(SeqId s);
    void release(SeqId s);

    const std::vector<BlockId>& block_table(SeqId s) const { return seqs_.at(s).table; }
    uint32_t length(SeqId s) const { return static_cast<uint32_t>(seqs_.at(s).tokens.size()); }
    uint32_t block_tokens() const { return bt_; }
    uint32_t blocks_needed(uint32_t tokens) const { return (tokens + bt_ - 1) / bt_; }
    std::vector<CopyOp> take_copies();
    KvStats stats() const;
    // Tokens currently recorded in a block (tests).
    std::vector<int32_t> block_contents(BlockId b) const;
    bool check_invariants() const;

private:
    struct Block {
        uint32_t ref = 0;
        uint32_t count = 0;            // tokens written
        bool hashed = false;
        uint64_t hash = 0;
        uint64_t last_use = 0;
        int32_t lru_prev = -1, lru_next = -1;
        bool in_lru = false;
    };
    struct Seq {
        std::vector<int32_t> tokens;
        std::vector<BlockId> table;
        std::vector<uint64_t> hashes;  // chain hash per full block
        bool live = false;
    };
    uint64_t chain(uint64_t prev, const int32_t* toks) const;
    BlockId alloc();                   // free list first, then LRU eviction; -1 if none
    uint32_t available() const { return static_cast<uint32_t>(free_.size()) + lru_size_; }
    void unref(BlockId b);
    void lru_remove(BlockId b);
    void lru_push(BlockId b);
    void seal_full_blocks(Seq& q);     // registers newly filled blocks in the prefix map

    uint32_t bt_;
    std::vector<Block> blk_;
    std::vector<int32_t> store_;       // n_blocks * bt token ids
    std::vector<BlockId> free_;
    int32_t lru_head_ = -1, lru_tail_ = -1;   // head = most recently released
    uint32_t lru_size_ = 0;
    std::unordered_map<uint64_t, BlockId> prefix_;
    std::unordered_map<SeqId, Seq> seqs_;
    SeqId next_seq_ = 0;
    uint64_t tick_ = 0;
    std::vector<CopyOp> copies_;
    KvStats st_;
};

} // namespace vegajet::paged

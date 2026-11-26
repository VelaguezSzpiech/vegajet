// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once

#include "core/spec_engine.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace vegajet::spec {

constexpr uint32_t kTreeVerifyRows = 7; // anchor + two children + four grandchildren

// Row zero is the already-emitted, not-yet-consumed anchor token. All other
// parents precede children. Depth is a logical RoPE offset, NOT a KV row offset.
struct VerifyTree {
    std::vector<DraftNode> nodes;
    std::vector<uint32_t> depth;
    std::vector<uint32_t> ancestor_mask; // Includes self; committed prefix is always visible.
    // Three raw convolution inputs before each node, oldest first. Negative
    // indices -3,-2,-1 address the immutable pre-verify convolution history.
    std::vector<std::array<int32_t, 3>> conv_inputs;
};

// Convert the first six suffix draft nodes without flattening their ancestry.
// Malformed parents, negative token IDs and duplicate siblings throw.
VerifyTree make_verify_tree(int32_t anchor, const std::vector<DraftNode>& drafts);
// Candidate IDs are ranked per DFlash2 slot, starting at the first draft slot.
// Use the first two distinct valid IDs at each of the first two depths.
VerifyTree make_topk_verify_tree(int32_t anchor, const std::vector<std::vector<int32_t>>& slots);
// Exact "1", matching the existing opt-in env convention; no environment reads.
bool tree_verify_requested(const char* value);

struct TreeAcceptance {
    std::vector<uint32_t> rows; // Root-to-leaf, including anchor, excluding bonus.
    int32_t bonus = -1; // Target argmax after the last accepted row; not consumed yet.
};
// pred[row] must be the full-vocabulary target greedy argmax after that row.
// Unique siblings imply a single greedy path, regardless of draft scores.
TreeAcceptance accept_tree(const VerifyTree& tree, const std::vector<int32_t>& pred,
                           uint32_t max_drafts = kTreeVerifyRows - 1);

struct TreeKvCopy {
    uint32_t src_row = 0, dst_row = 0;
    uint64_t src_byte = 0, dst_byte = 0, bytes = 0;
};
struct TreeCommitPlan {
    // Gather these FP32 k/v/decay/beta records to a disjoint contiguous scratch
    // buffer and replay from the immutable pre-verify spec-GDN snapshot.
    std::vector<uint32_t> replay_rows;
    std::vector<TreeKvCopy> kv_copies; // Execute serially in this order, per layer and GPU.
    std::array<int32_t, 3> conv_tail; // Raw history to install after the accepted path.
    uint32_t next_pos = 0;
    uint32_t next_ring_head = 0;
    uint32_t state_node = 0; // Per-node snapshot to select, if using snapshot commit.
};
// KV rows use physical pos0+node while RoPE uses pos0+depth[node]. Includes
// anchor even on total draft rejection. kv_row_bytes includes all local heads.
// Validates paths, capacities and overflow before producing any copy operations.
TreeCommitPlan plan_tree_commit(const VerifyTree& tree, const std::vector<uint32_t>& path,
                                uint32_t pos0, uint32_t capacity, uint64_t kv_row_bytes,
                                uint32_t ring_head, uint32_t ring_slots = 17);

} // namespace vegajet::spec

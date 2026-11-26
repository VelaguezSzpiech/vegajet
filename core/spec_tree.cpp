// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/spec_tree.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace vegajet::spec {
namespace {

void append_node(VerifyTree& tree, int32_t token, int32_t parent, double score) {
    const auto row = static_cast<uint32_t>(tree.nodes.size());
    if (token < 0 || row >= kTreeVerifyRows || parent < -1 || parent >= static_cast<int32_t>(row) ||
        ((row == 0) != (parent == -1)))
        throw std::invalid_argument("invalid verify tree node");
    for (const auto& node : tree.nodes)
        if (node.parent == parent && node.token == token)
            throw std::invalid_argument("duplicate verify tree sibling");
    std::array<int32_t, 3> history{-3, -2, -1};
    uint32_t depth = 0, mask = 0;
    if (parent >= 0) {
        depth = tree.depth[parent] + 1;
        mask = tree.ancestor_mask[parent];
        const auto& h = tree.conv_inputs[parent];
        history = {h[1], h[2], parent};
    }
    tree.nodes.push_back({token, parent, score});
    tree.depth.push_back(depth);
    tree.ancestor_mask.push_back(mask | (1u << row));
    tree.conv_inputs.push_back(history);
}

// Verify metadata too: a caller must not accidentally pass a stale chain mask.
void validate_tree(const VerifyTree& tree) {
    if (tree.nodes.empty() || tree.nodes.size() > kTreeVerifyRows)
        throw std::invalid_argument("invalid verify tree size");
    VerifyTree expected;
    for (const auto& n : tree.nodes) append_node(expected, n.token, n.parent, n.score);
    if (expected.depth != tree.depth || expected.ancestor_mask != tree.ancestor_mask ||
        expected.conv_inputs != tree.conv_inputs)
        throw std::invalid_argument("inconsistent verify tree metadata");
}

std::vector<int32_t> top_two(const std::vector<int32_t>& slot) {
    std::vector<int32_t> out;
    for (int32_t token : slot) {
        if (token < 0 || std::find(out.begin(), out.end(), token) != out.end()) continue;
        out.push_back(token);
        if (out.size() == 2) break;
    }
    return out;
}

} // namespace

VerifyTree make_verify_tree(int32_t anchor, const std::vector<DraftNode>& drafts) {
    VerifyTree tree;
    append_node(tree, anchor, -1, 1.0);
    const size_t n = std::min<size_t>(drafts.size(), kTreeVerifyRows - 1);
    for (size_t i = 0; i < n; ++i) {
        const auto& d = drafts[i];
        if (d.parent < -1 || d.parent >= static_cast<int32_t>(i))
            throw std::invalid_argument("suffix draft parent must precede child");
        append_node(tree, d.token, d.parent + 1, d.score);
    }
    return tree;
}

VerifyTree make_topk_verify_tree(int32_t anchor, const std::vector<std::vector<int32_t>>& slots) {
    VerifyTree tree = make_verify_tree(anchor, {});
    if (slots.empty()) return tree;
    const auto children = top_two(slots[0]);
    for (int32_t token : children) append_node(tree, token, 0, 0.0);
    if (slots.size() < 2) return tree;
    const auto grandchildren = top_two(slots[1]);
    for (uint32_t parent = 1; parent <= children.size(); ++parent)
        for (int32_t token : grandchildren) append_node(tree, token, parent, 0.0);
    return tree;
}

bool tree_verify_requested(const char* value) {
    return value != nullptr && std::strcmp(value, "1") == 0;
}

TreeAcceptance accept_tree(const VerifyTree& tree, const std::vector<int32_t>& pred,
                           uint32_t max_drafts) {
    validate_tree(tree);
    if (pred.size() != tree.nodes.size() ||
        std::any_of(pred.begin(), pred.end(), [](int32_t token) { return token < 0; }))
        throw std::invalid_argument("one target argmax required per verify node");
    TreeAcceptance result{{0}, pred[0]};
    while (result.rows.size() - 1 < max_drafts) {
        const uint32_t parent = result.rows.back();
        uint32_t match = static_cast<uint32_t>(tree.nodes.size());
        for (uint32_t row = parent + 1; row < tree.nodes.size(); ++row) {
            const auto& node = tree.nodes[row];
            if (node.parent == static_cast<int32_t>(parent) && node.token == pred[parent]) {
                match = row;
                break;
            }
        }
        if (match == tree.nodes.size()) break;
        result.rows.push_back(match);
        result.bonus = pred[match];
    }
    return result;
}

TreeCommitPlan plan_tree_commit(const VerifyTree& tree, const std::vector<uint32_t>& path,
                                uint32_t pos0, uint32_t capacity, uint64_t kv_row_bytes,
                                uint32_t ring_head, uint32_t ring_slots) {
    validate_tree(tree);
    if (path.empty() || path.front() != 0 || path.size() > tree.nodes.size())
        throw std::invalid_argument("commit path must include anchor");
    for (size_t i = 0; i < path.size(); ++i) {
        if (path[i] >= tree.nodes.size() ||
            (i > 0 && tree.nodes[path[i]].parent != static_cast<int32_t>(path[i - 1])))
            throw std::invalid_argument("commit path must follow tree edges");
    }
    if (ring_slots <= path.size() || ring_head >= ring_slots || kv_row_bytes == 0 ||
        pos0 > capacity || tree.nodes.size() > capacity - pos0 ||
        kv_row_bytes > std::numeric_limits<uint64_t>::max() / capacity)
        throw std::invalid_argument("tree commit exceeds KV or snapshot capacity");

    TreeCommitPlan plan;
    plan.replay_rows = path;
    plan.state_node = path.back();
    plan.next_pos = pos0 + static_cast<uint32_t>(path.size());
    plan.next_ring_head = static_cast<uint32_t>((static_cast<uint64_t>(ring_head) + path.size()) % ring_slots);
    const auto& tail = tree.conv_inputs[path.back()];
    plan.conv_tail = {tail[1], tail[2], static_cast<int32_t>(path.back())};
    for (uint32_t i = 0; i < path.size(); ++i) {
        const uint32_t src = pos0 + path[i], dst = pos0 + i;
        // Parent-before-child gives increasing sources and dst <= src. Serial
        // ascending copies cannot clobber any later source (not a parallel memcpy).
        if (src != dst)
            plan.kv_copies.push_back({src, dst, src * kv_row_bytes, dst * kv_row_bytes, kv_row_bytes});
    }
    return plan;
}

} // namespace vegajet::spec

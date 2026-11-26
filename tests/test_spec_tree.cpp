// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/spec_tree.hpp"
#include "models/ornith/ornith_gpu.hpp"
#include "tests/test_common.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace vegajet::spec;

template <class F> void rejects(F f) {
    bool caught = false;
    try { f(); } catch (const std::invalid_argument&) { caught = true; }
    VJ_CHECK(caught);
}

static void construction_and_mask() {
    const auto t = make_topk_verify_tree(10, {{20, 20, -1, 21, 22}, {30, 31, 32}, {40}});
    VJ_CHECK(t.nodes.size() == 7);
    VJ_CHECK((t.depth == std::vector<uint32_t>{0, 1, 1, 2, 2, 2, 2}));
    VJ_CHECK((t.ancestor_mask == std::vector<uint32_t>{1, 3, 5, 11, 19, 37, 69}));
    VJ_CHECK(t.nodes[5].parent == 2 && t.nodes[5].token == 30);
    VJ_CHECK((t.conv_inputs[0] == std::array<int32_t, 3>{-3, -2, -1}));
    VJ_CHECK((t.conv_inputs[6] == std::array<int32_t, 3>{-1, 0, 2}));
    for (uint32_t row = 0; row < t.nodes.size(); ++row) {
        uint32_t expected = 0;
        for (int32_t p = row; p >= 0; p = t.nodes[p].parent) expected |= 1u << p;
        VJ_CHECK(t.ancestor_mask[row] == expected);
        VJ_CHECK((t.ancestor_mask[row] >> (row + 1)) == 0);
    }
    VJ_CHECK(make_topk_verify_tree(10, {}).nodes.size() == 1);
    VJ_CHECK(make_topk_verify_tree(10, {{-1, -2}, {1, 2}}).nodes.size() == 1);
    VJ_CHECK(make_topk_verify_tree(10, {{1}, {2}}).nodes.size() == 3);
    VJ_CHECK(make_topk_verify_tree(10, {{1, 2}}).nodes.size() == 3);
    // Suffix drafts may be deeper than the balanced DFlash2 tree.
    SuffixTree suffix(8);
    suffix.extend({1, 2, 3, 4, 1, 2, 3, 5, 1, 2, 3, 4});
    const auto drafts = suffix.draft({1, 2, 3}, 6, 2, 0.0);
    const auto s = make_verify_tree(3, drafts);
    VJ_CHECK(s.nodes.size() == drafts.size() + 1);
    for (size_t i = 0; i < drafts.size(); ++i) {
        VJ_CHECK(s.nodes[i + 1].token == drafts[i].token);
        VJ_CHECK(s.nodes[i + 1].parent == drafts[i].parent + 1);
    }
    std::vector<DraftNode> chain;
    for (int32_t i = 0; i < 10; ++i) chain.push_back({i + 1, i - 1, 1.0});
    const auto c = make_verify_tree(0, chain);
    VJ_CHECK(c.nodes.size() == 7 && c.depth.back() == 6 && c.ancestor_mask.back() == 127);
    rejects([] { make_verify_tree(-1, {}); });
    rejects([] { make_verify_tree(0, {{1, 0, 1.0}}); });
    rejects([] { make_verify_tree(0, {{1, -2, 1.0}}); });
    rejects([] { make_verify_tree(0, {{-1, -1, 1.0}}); });
    rejects([] { make_verify_tree(0, {{1, -1, 1.0}, {1, -1, 0.5}}); });
}

static void acceptance_and_commit() {
    const auto t = make_topk_verify_tree(10, {{20, 21}, {30, 31}});
    const std::vector<int32_t> pred{21, 30, 31, 90, 91, 92, 99};
    const auto a = accept_tree(t, pred);
    VJ_CHECK((a.rows == std::vector<uint32_t>{0, 2, 6}) && a.bonus == 99);
    // A good-looking off-path prediction must never rescue a rejected parent.
    auto rejected = pred;
    rejected[0] = 100;
    VJ_CHECK((accept_tree(t, rejected).rows == std::vector<uint32_t>{0}));
    VJ_CHECK(accept_tree(t, rejected).bonus == 100);
    VJ_CHECK((accept_tree(t, pred, 0).rows == std::vector<uint32_t>{0}));
    VJ_CHECK((accept_tree(t, pred, 1).rows == std::vector<uint32_t>{0, 2}));
    VJ_CHECK(accept_tree(t, pred, 1).bonus == 31);
    auto middle = pred;
    middle[2] = 77;
    VJ_CHECK((accept_tree(t, middle).rows == std::vector<uint32_t>{0, 2}));
    VJ_CHECK(accept_tree(t, middle).bonus == 77);
    const auto plan = plan_tree_commit(t, a.rows, 100, 107, 64, 16);
    VJ_CHECK(plan.replay_rows == a.rows && plan.state_node == 6);
    VJ_CHECK(plan.next_pos == 103 && plan.next_ring_head == 2);
    VJ_CHECK((plan.conv_tail == std::array<int32_t, 3>{0, 2, 6}));
    VJ_CHECK(plan.kv_copies.size() == 2);
    VJ_CHECK(plan.kv_copies[0].src_row == 102 && plan.kv_copies[0].dst_row == 101);
    VJ_CHECK(plan.kv_copies[0].src_byte == 6528 && plan.kv_copies[0].dst_byte == 6464);
    VJ_CHECK(plan.kv_copies[1].src_row == 106 && plan.kv_copies[1].dst_row == 102);
    VJ_CHECK(plan.kv_copies[1].src_byte == 6784 && plan.kv_copies[1].bytes == 64);
    const auto root = plan_tree_commit(t, {0}, 0, 7, 64, 16);
    VJ_CHECK(root.next_pos == 1 && root.next_ring_head == 0 && root.kv_copies.empty());
    VJ_CHECK((root.conv_tail == std::array<int32_t, 3>{-2, -1, 0}));
    const auto two = plan_tree_commit(t, {0, 2}, 0, 7, 64, 0);
    VJ_CHECK((two.conv_tail == std::array<int32_t, 3>{-1, 0, 2}));
    rejects([&] { accept_tree(t, {}); });
    auto invalid = pred;
    invalid[3] = -1;
    rejects([&] { accept_tree(t, invalid); });
    auto stale = t;
    stale.ancestor_mask[6] |= 1u << 1;
    rejects([&] { accept_tree(stale, pred); });
    rejects([&] { plan_tree_commit(t, {}, 0, 7, 64, 0); });
    rejects([&] { plan_tree_commit(t, {1}, 0, 7, 64, 0); });
    rejects([&] { plan_tree_commit(t, {0, 1, 6}, 0, 7, 64, 0); });
    rejects([&] { plan_tree_commit(t, {0, 2, 7}, 0, 7, 64, 0); });
    rejects([&] { plan_tree_commit(t, {0, 0}, 0, 7, 64, 0); });
    rejects([&] { plan_tree_commit(t, {0}, 101, 107, 64, 0); });
    rejects([&] { plan_tree_commit(t, {0}, 10, 7, 64, 0); });
    rejects([&] { plan_tree_commit(t, {0}, 0, 0, 64, 0); });
    rejects([&] { plan_tree_commit(t, {0}, 0, 7, 0, 0); });
    rejects([&] { plan_tree_commit(t, {0}, 0, 7, 64, 17); });
    rejects([&] { plan_tree_commit(t, a.rows, 0, 7, 64, 0, 3); });
    rejects([&] { plan_tree_commit(t, {0}, 0, 7, std::numeric_limits<uint64_t>::max(), 0); });
    const auto wide = plan_tree_commit(t, a.rows, 0, 7, 1ULL << 33, 0);
    VJ_CHECK(wide.kv_copies.back().src_byte == 6ULL * (1ULL << 33));
}

// Exhaust all 720 parent-before-child layouts with six distinct draft tokens.
// For every possible final node, force that greedy path and check replay/copies.
static void check_layout(const std::vector<DraftNode>& drafts) {
    const auto t = make_verify_tree(10, drafts);
    for (uint32_t leaf = 0; leaf < t.nodes.size(); ++leaf) {
        std::vector<uint32_t> path;
        for (int32_t row = leaf; row >= 0; row = t.nodes[row].parent) path.push_back(row);
        std::reverse(path.begin(), path.end());
        std::vector<int32_t> pred(t.nodes.size(), 999);
        for (size_t i = 1; i < path.size(); ++i) pred[path[i - 1]] = t.nodes[path[i]].token;
        const auto accepted = accept_tree(t, pred);
        VJ_CHECK(accepted.rows == path && accepted.bonus == 999);
        const auto plan = plan_tree_commit(t, path, 3, 10, 8, 16);
        std::vector<uint64_t> kv{900, 901, 902, 100, 101, 102, 103, 104, 105, 106};
        const auto original = kv;
        for (const auto& c : plan.kv_copies) {
            VJ_CHECK(c.src_byte == c.src_row * sizeof(uint64_t));
            VJ_CHECK(c.dst_byte == c.dst_row * sizeof(uint64_t));
            std::memmove(reinterpret_cast<char*>(kv.data()) + c.dst_byte,
                         reinterpret_cast<const char*>(kv.data()) + c.src_byte, c.bytes);
        }
        for (size_t i = 0; i < path.size(); ++i) VJ_CHECK(kv[3 + i] == original[3 + path[i]]);
        VJ_CHECK(kv[0] == 900 && kv[1] == 901 && kv[2] == 902);

        // Scalar FP32 recurrence model: each child must start at its parent's
        // state, not the previously flattened row. This tests the host schedule,
        // not the real device GDN arithmetic or its reductions.
        std::vector<float> states(t.nodes.size());
        const auto update = [](float s, uint32_t row) { return std::fma(0.75f, s, (row + 1) * 0.125f); };
        for (size_t i = 0; i < t.nodes.size(); ++i) {
            const int32_t parent = t.nodes[i].parent;
            states[i] = update(parent < 0 ? 0.5f : states[parent], i);
        }
        float replay = 0.5f;
        for (uint32_t row : plan.replay_rows) replay = update(replay, row);
        VJ_CHECK(replay == states[plan.state_node]);
        std::vector<int32_t> raw{-3, -2, -1};
        for (uint32_t row : path) {
            const std::array<int32_t, 3> h{raw[raw.size()-3], raw[raw.size()-2], raw.back()};
            VJ_CHECK(h == t.conv_inputs[row]);
            raw.push_back(row);
            uint32_t mask = 0;
            for (int32_t p = row; p >= 0; p = t.nodes[p].parent) mask |= 1u << p;
            VJ_CHECK(mask == t.ancestor_mask[row]);
        }
        VJ_CHECK((plan.conv_tail == std::array<int32_t, 3>{raw[raw.size()-3], raw[raw.size()-2], raw.back()}));
    }
}

static void enumerate_layouts(std::vector<DraftNode>& drafts, uint32_t& count) {
    if (drafts.size() == 6) { check_layout(drafts); ++count; return; }
    const int32_t i = static_cast<int32_t>(drafts.size());
    for (int32_t p = -1; p < i; ++p) {
        drafts.push_back({20 + i, p, 0.0});
        enumerate_layouts(drafts, count);
        drafts.pop_back();
    }
}

static void fail_closed_toggle() {
    VJ_CHECK(!tree_verify_requested(nullptr) && !tree_verify_requested(""));
    VJ_CHECK(!tree_verify_requested("0") && !tree_verify_requested("true"));
    VJ_CHECK(!tree_verify_requested("10") && !tree_verify_requested("1extra"));
    VJ_CHECK(tree_verify_requested("1"));
    const char* old = std::getenv("VEGAJET_V12_TREE_VERIFY");
    const bool had_old = old != nullptr;
    const std::string saved = old ? old : "";
    VJ_CHECK(setenv("VEGAJET_V12_TREE_VERIFY", "1", 1) == 0);
    const auto result = vegajet::ornith::run_ornith_gpu("/no/model/should/be/opened", "", {});
    VJ_CHECK(!result.ok && result.error.find("VEGAJET_V12_TREE_VERIFY=1") != std::string::npos);
    if (had_old) VJ_CHECK(setenv("VEGAJET_V12_TREE_VERIFY", saved.c_str(), 1) == 0);
    else VJ_CHECK(unsetenv("VEGAJET_V12_TREE_VERIFY") == 0);
}

int main() {
    construction_and_mask();
    acceptance_and_commit();
    std::vector<DraftNode> drafts;
    uint32_t layouts = 0;
    enumerate_layouts(drafts, layouts);
    VJ_CHECK(layouts == 720);
    fail_closed_toggle();
    std::printf("[PASS] tree verification: construction, masks, greedy acceptance, KV and GDN commit plans (%u layouts)\n", layouts);
}

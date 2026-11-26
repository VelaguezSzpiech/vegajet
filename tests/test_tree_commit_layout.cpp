// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/spec_tree.hpp"
#include "tests/test_common.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace vegajet::spec;

static uint32_t pattern(uint32_t layer, uint32_t row, uint32_t word) {
    return 1u + layer * 1000000u + row * 10000u + word;
}

static void compact_kv(const VerifyTree& tree, const std::vector<uint32_t>& path) {
    constexpr uint32_t row_words = 272u / sizeof(uint32_t);
    for (uint32_t pos : {0u, 1u, 17u, 255u}) {
        const uint32_t capacity = pos + static_cast<uint32_t>(tree.nodes.size());
        const auto plan = plan_tree_commit(tree, path, pos, capacity, 272u, 0u);
        for (uint32_t layer = 0; layer < 12u; ++layer) {
            for (uint32_t rank = 0; rank < 2u; ++rank) {
                for (uint32_t kind = 0; kind < 2u; ++kind) {
                    std::vector<uint32_t> cache((capacity + 1u) * row_words);
                    for (uint32_t row = 0; row <= capacity; ++row)
                        for (uint32_t w = 0; w < row_words; ++w)
                            cache[row * row_words + w] = pattern(layer * 4u + rank * 2u + kind, row, w);
                    const auto original = cache;
                    for (const auto& copy : plan.kv_copies) {
                        VJ_CHECK(copy.src_byte % 4u == 0 && copy.dst_byte % 4u == 0 && copy.bytes == 272u);
                        std::memcpy(cache.data() + copy.dst_byte / 4u,
                                    cache.data() + copy.src_byte / 4u, copy.bytes);
                    }
                    VJ_CHECK(std::equal(cache.begin(), cache.begin() + pos * row_words, original.begin()));
                    for (uint32_t j = 0; j < path.size(); ++j)
                        for (uint32_t w = 0; w < row_words; ++w)
                            VJ_CHECK(cache[(pos + j) * row_words + w] == original[(pos + path[j]) * row_words + w]);
                    VJ_CHECK(std::equal(cache.begin() + capacity * row_words, cache.end(),
                                        original.begin() + capacity * row_words));
                }
            }
        }
    }
}

static void replay_layout(const VerifyTree& tree, const std::vector<uint32_t>& path) {
    // These are the executor's per-layer record strides, including unused rows.
    constexpr uint32_t layers = 36u, stride = 16u, kv_words = 40u * 128u, gate_words = 32u;
    for (uint32_t words : {kv_words, gate_words}) {
        std::vector<uint32_t> source(layers * stride * words), gather(source.size(), 0xdeadbeefu);
        for (uint32_t gi = 0; gi < layers; ++gi)
            for (uint32_t row = 0; row < stride; ++row)
                for (uint32_t w = 0; w < words; ++w)
                    source[(gi * stride + row) * words + w] = pattern(gi, row, w);
        for (uint32_t gi = 0; gi < layers; ++gi)
            for (uint32_t j = 0; j < path.size(); ++j)
                std::copy_n(source.begin() + (gi * stride + path[j]) * words, words,
                            gather.begin() + (gi * stride + j) * words);
        for (uint32_t gi = 0; gi < layers; ++gi) {
            for (uint32_t j = 0; j < stride; ++j)
                for (uint32_t w = 0; w < words; ++w)
                    VJ_CHECK(gather[(gi * stride + j) * words + w] ==
                        (j < path.size() ? pattern(gi, path[j], w) : 0xdeadbeefu));
        }
    }
    for (uint32_t head = 0; head < 17u; ++head) {
        const auto plan = plan_tree_commit(tree, path, 0u, 7u, 272u, head);
        std::array<uint32_t, 17> slots{};
        slots.fill(0xdeadbeefu);
        slots[head] = 99u;
        for (uint32_t row = 0; row < tree.nodes.size(); ++row) slots[(head + row + 1u) % 17u] = row;
        const uint32_t src_slot = (head + path.back() + 1u) % 17u;
        const uint32_t dst_slot = plan.next_ring_head;
        slots[dst_slot] = slots[src_slot];
        VJ_CHECK(slots[dst_slot] == path.back());
        VJ_CHECK(slots[head] == 99u);
        VJ_CHECK(plan.next_pos == path.size());
    }
}

static void catchup_layout(const VerifyTree& tree, const std::vector<uint32_t>& path) {
    for (uint32_t taps : {1u, 3u, 5u}) {
        const uint32_t words = 3072u * taps;
        std::vector<uint32_t> hidden((tree.nodes.size() + 1u) * words);
        for (uint32_t row = 0; row <= tree.nodes.size(); ++row)
            for (uint32_t w = 0; w < words; ++w) hidden[row * words + w] = pattern(0, row, w);
        const auto original = hidden;
        for (uint32_t j = 0; j < path.size(); ++j) {
            if (path[j] == j) continue;
            std::memcpy(hidden.data() + j * words, hidden.data() + path[j] * words, words * 4u);
        }
        for (uint32_t j = 0; j < path.size(); ++j)
            VJ_CHECK(std::equal(hidden.begin() + j * words, hidden.begin() + (j + 1u) * words,
                                original.begin() + path[j] * words));
        VJ_CHECK(std::equal(hidden.begin() + tree.nodes.size() * words, hidden.end(),
                            original.begin() + tree.nodes.size() * words));
    }
    std::vector<int32_t> predictions(tree.nodes.size(), 701);
    for (uint32_t j = 1; j < path.size(); ++j) predictions[path[j - 1u]] = tree.nodes[path[j]].token;
    for (uint32_t rem = 1; rem <= 8u; ++rem) {
        const auto accepted = accept_tree(tree, predictions, rem - 1u);
        VJ_CHECK(accepted.rows.size() <= rem);
        std::vector<int32_t> emitted;
        for (uint32_t j = 1; j < accepted.rows.size(); ++j) emitted.push_back(tree.nodes[accepted.rows[j]].token);
        emitted.push_back(accepted.bonus);
        VJ_CHECK(emitted.size() == accepted.rows.size());
        // The anchor was already emitted; bonus is emitted but not injected.
        std::vector<int32_t> injected;
        for (uint32_t row : accepted.rows) injected.push_back(tree.nodes[row].token);
        VJ_CHECK(injected.front() == tree.nodes[0].token);
        for (uint32_t j = 1; j < injected.size(); ++j) VJ_CHECK(injected[j] == emitted[j - 1u]);
        VJ_CHECK(accepted.bonus == predictions[accepted.rows.back()]);
        if (accepted.rows.size() < path.size()) VJ_CHECK(emitted.back() == tree.nodes[path[accepted.rows.size()]].token);
    }
}

int main() {
    const auto tree = make_topk_verify_tree(9, {{10, 11}, {12, 13}});
    const std::vector<std::vector<uint32_t>> paths{{0}, {0, 1}, {0, 2}, {0, 1, 3}, {0, 1, 4}, {0, 2, 5}, {0, 2, 6}};
    for (const auto& path : paths) {
        compact_kv(tree, path);
        replay_layout(tree, path);
        catchup_layout(tree, path);
    }
    std::printf("[PASS] tree commit physical KV bytes, 36-layer records, accepted taps/catch-up, all leaves and 17 ring heads\n");
}

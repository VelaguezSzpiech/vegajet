// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// B4 route-union evidence: identical top-k / layer / token metadata must not imply an identical routed-expert
// count.  The same-shaped traces below differ only in whether tokens re-read the same expert ids (repeated) or
// touch disjoint ids (disjoint).  The exact producer, bounded failure modes, one-shot feedback, and the
// end-to-end EvictDepth choice are all exercised here.
#include "core/evict_depth.hpp"
#include "core/route_union.hpp"
#include "tests/test_common.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace vegajet;

namespace {

constexpr uint32_t kLayers = 48;
constexpr uint32_t kTokens = 8;
constexpr uint32_t kTopK = 8;
constexpr uint32_t kStride = 16u * 8u;  // the GPU capture stride: 16 tokens * top8

std::vector<int32_t> make_trace(bool repeated) {
    std::vector<int32_t> block(static_cast<size_t>(kLayers) * kStride, 9999);  // poison outside the valid span
    for (uint32_t layer = 0; layer < kLayers; ++layer) {
        for (uint32_t tok = 0; tok < kTokens; ++tok) {
            for (uint32_t j = 0; j < kTopK; ++j) {
                // Repeated: every token routes the same 8 ids in a layer.  Disjoint: token t routes ids 8t..8t+7,
                // so the same 8-token/8-layer/8-wide metadata covers 64 ids per layer instead of 8.
                const int32_t id = repeated ? static_cast<int32_t>(j)
                                            : static_cast<int32_t>(tok * kTopK + j);
                block[static_cast<size_t>(layer) * kStride + tok * kTopK + j] = id;
            }
        }
    }
    return block;
}

EvictDepth warm_full(uint32_t max_depth, uint32_t experts_per_depth) {
    EvictDepthConfig cfg;
    cfg.experts_per_depth = experts_per_depth;
    EvictDepth d(max_depth, cfg);
    for (int i = 0; i < 20; ++i) d.observe(8, 8);
    return d;
}

} // namespace

int main() {
    // ---- same top-k / layers / tokens, different actual route ids
    {
        const auto repeated = make_trace(true);
        const auto disjoint = make_trace(false);
        const auto r_rep = route_union_from_top_ids(repeated.data(), kLayers, kTokens, kTopK, kStride);
        const auto r_dis = route_union_from_top_ids(disjoint.data(), kLayers, kTokens, kTopK, kStride);
        VJ_CHECK(r_rep.ok() && r_dis.ok());
        VJ_CHECK(r_rep.layers == r_dis.layers && r_rep.tokens == r_dis.tokens);
        VJ_CHECK(r_rep.unique_experts == uint64_t(kLayers) * kTopK);           // 384: 8 ids per layer
        VJ_CHECK(r_dis.unique_experts == uint64_t(kLayers) * kTopK * kTokens); // 3072: 64 ids per layer
        VJ_CHECK(r_rep.unique_experts != r_dis.unique_experts);
        VJ_CHECK(r_dis.unique_experts != uint64_t(kLayers) * kTopK);  // top_k*layers is not the routed count
        std::printf("  identical metadata (%u layers, %u tokens, top%u): repeated=%llu disjoint=%llu\n",
                    kLayers, kTokens, kTopK, static_cast<unsigned long long>(r_rep.unique_experts),
                    static_cast<unsigned long long>(r_dis.unique_experts));
    }
    // ---- duplicates inside a token are deduped by the bounded bitset
    {
        std::vector<int32_t> block(static_cast<size_t>(4) * 16u * kTopK, 0);
        for (uint32_t layer = 0; layer < 4; ++layer)
            for (uint32_t tok = 0; tok < 16; ++tok)
                for (uint32_t j = 0; j < kTopK; ++j) block[(layer * 16u + tok) * kTopK + j] = 3;  // same id
        const auto r = route_union_from_top_ids(block.data(), 4, 16, kTopK, 16u * kTopK);
        VJ_CHECK(r.ok() && r.unique_experts == 4);  // one distinct id per layer, not 4*16*8
    }
    // ---- fail-closed bounds: an invalid readback is unknown, never a clamped count
    {
        auto block = make_trace(false);
        block[5] = 256;  // expert id outside [0, 256)
        auto bad = route_union_from_top_ids(block.data(), kLayers, kTokens, kTopK, kStride);
        VJ_CHECK(!bad.ok() && bad.status == RouteUnionStatus::kExpertOutOfRange && bad.unique_experts == 0);

        block = make_trace(false);
        block[6] = -1;
        bad = route_union_from_top_ids(block.data(), kLayers, kTokens, kTopK, kStride);
        VJ_CHECK(!bad.ok() && bad.status == RouteUnionStatus::kExpertOutOfRange && bad.unique_experts == 0);

        VJ_CHECK(route_union_from_top_ids(nullptr, kLayers, kTokens, kTopK, kStride).status ==
                 RouteUnionStatus::kUnavailable);
        VJ_CHECK(route_union_from_top_ids(block.data(), kLayers, 0u, kTopK, kStride).status ==
                 RouteUnionStatus::kUnavailable);

        auto too_many_layers = route_union_from_top_ids(block.data(), kRouteUnionMaxLayers + 1u, 1u, kTopK,
                                                        kStride);
        VJ_CHECK(too_many_layers.status == RouteUnionStatus::kLayerOutOfRange);

        std::vector<int32_t> wide(static_cast<size_t>(2) * 17u * kTopK, 0);
        const auto too_many_tokens = route_union_from_top_ids(wide.data(), 2u, kRouteUnionMaxTokens + 1u, kTopK,
                                                              17u * kTopK);
        VJ_CHECK(too_many_tokens.status == RouteUnionStatus::kTooManyTokens);

        RouteUnion direct;
        const int32_t ids[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        VJ_CHECK(direct.add_layer(0, ids, 7) == RouteUnionStatus::kTooManyIds);
        VJ_CHECK(direct.count() == 0);
    }
    // ---- one-shot, deferred-safe feedback
    {
        const auto block = make_trace(true);
        RouteUnionFeedback fb;
        VJ_CHECK(fb.take() == 0 && !fb.available());  // cold
        VJ_CHECK(record_completed_routes(&fb, block.data(), kLayers, kTokens, kTopK, kStride) ==
                 RouteUnionStatus::kOk);
        VJ_CHECK(fb.available() && fb.unique_experts() == uint64_t(kLayers) * kTopK && fb.tokens() == kTokens);
        VJ_CHECK(fb.take() == uint64_t(kLayers) * kTopK);
        VJ_CHECK(fb.take() == 0);  // consumed: no stale reuse

        fb.observe_count(99, 2);
        VJ_CHECK(record_completed_routes(&fb, nullptr, kLayers, kTokens, kTopK, kStride) ==
                 RouteUnionStatus::kUnavailable);
        VJ_CHECK(!fb.available() && fb.take() == 0);  // deferred readback clears the old value
        VJ_CHECK(!fb.observe_count(0, 8));            // zero is unknown, not an observation
        VJ_CHECK(!fb.observe_count(uint64_t(kRouteUnionMaxLayers) * kRouteUnionMaxExperts + 1, 8));
        VJ_CHECK(!fb.observe_count(8, kRouteUnionMaxTokens + 1));
        VJ_CHECK(!fb.available());
    }
    // ---- end-to-end: measured routed union drives the depth decision
    {
        EvictDepth d = warm_full(8, 256);
        RouteUnionFeedback fb;
        VJ_CHECK(d.choose(8, fb.take()) == 8);  // cold: unknown, prior depth preserved
        VJ_CHECK(d.last_choice().unique_experts == 0);

        const auto repeated = make_trace(true);
        const auto disjoint = make_trace(false);
        VJ_CHECK(record_completed_routes(&fb, repeated.data(), kLayers, kTokens, kTopK, kStride) ==
                 RouteUnionStatus::kOk);
        const uint32_t k_rep = d.choose(8, fb.take());
        VJ_CHECK(k_rep == 8 && d.last_choice().unique_experts == uint64_t(kLayers) * kTopK);
        VJ_CHECK(!d.last_choice().limited_by_experts);

        VJ_CHECK(record_completed_routes(&fb, disjoint.data(), kLayers, kTokens, kTopK, kStride) ==
                 RouteUnionStatus::kOk);
        const uint32_t k_dis = d.choose(8, fb.take());
        VJ_CHECK(d.last_choice().unique_experts == uint64_t(kLayers) * kTopK * kTokens);
        VJ_CHECK(k_dis < k_rep && d.last_choice().limited_by_experts);
        std::printf("  end-to-end: repeated union 384 -> depth %u; disjoint union 3072 -> depth %u\n", k_rep, k_dis);
    }
    std::printf("PASS route_union\n");
    return 0;
}

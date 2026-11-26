// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/route_union.hpp"

#include <algorithm>

namespace vegajet {

const char* route_union_status_name(RouteUnionStatus s) noexcept {
    switch (s) {
        case RouteUnionStatus::kOk: return "ok";
        case RouteUnionStatus::kUnavailable: return "unavailable";
        case RouteUnionStatus::kLayerOutOfRange: return "layer_out_of_range";
        case RouteUnionStatus::kExpertOutOfRange: return "expert_out_of_range";
        case RouteUnionStatus::kTooManyTokens: return "too_many_tokens";
        case RouteUnionStatus::kTooManyIds: return "too_many_ids";
    }
    return "unknown_status";
}

RouteUnion::RouteUnion(RouteUnionConfig cfg) : cfg_(cfg) {
    // The fixed bitset is sized for the hard maxima.  A smaller configured domain is enforced by validation,
    // which keeps every bit index inside the array.
    cfg_.max_layers = std::min(cfg_.max_layers, kRouteUnionMaxLayers);
    cfg_.max_experts = std::min(cfg_.max_experts, kRouteUnionMaxExperts);
    cfg_.max_tokens = std::min(cfg_.max_tokens, kRouteUnionMaxTokens);
    cfg_.top_k = std::min(cfg_.top_k, kRouteUnionMaxExperts);
    reset();
}

void RouteUnion::reset() noexcept {
    bits_.fill(0);
    status_ = RouteUnionStatus::kOk;
    tokens_seen_ = 0;
}

RouteUnionStatus RouteUnion::add_layer(uint32_t layer, const int32_t* ids, uint32_t n) noexcept {
    if (status_ != RouteUnionStatus::kOk) return status_;
    if (!ids || n == 0 || cfg_.top_k == 0) {
        status_ = RouteUnionStatus::kUnavailable;
        return status_;
    }
    if (layer >= cfg_.max_layers) {
        status_ = RouteUnionStatus::kLayerOutOfRange;
        return status_;
    }
    if ((n % cfg_.top_k) != 0) {
        status_ = RouteUnionStatus::kTooManyIds;
        return status_;
    }
    const uint32_t tokens = n / cfg_.top_k;
    if (tokens > cfg_.max_tokens) {
        status_ = RouteUnionStatus::kTooManyTokens;
        return status_;
    }
    // Validate the complete layer first: a partially inserted layer would turn a rejected readback into a
    // plausible-looking but wrong count.
    for (uint32_t i = 0; i < n; ++i) {
        if (ids[i] < 0 || static_cast<uint32_t>(ids[i]) >= cfg_.max_experts) {
            status_ = RouteUnionStatus::kExpertOutOfRange;
            return status_;
        }
    }
    const uint64_t base = static_cast<uint64_t>(layer) * static_cast<uint64_t>(kRouteUnionMaxExperts);
    for (uint32_t i = 0; i < n; ++i) {
        const uint64_t bit = base + static_cast<uint32_t>(ids[i]);
        bits_[bit >> 6] |= (1ULL << (bit & 63u));
    }
    tokens_seen_ = std::max(tokens_seen_, tokens);
    return status_;
}

uint64_t RouteUnion::count() const noexcept {
    if (status_ != RouteUnionStatus::kOk) return 0;
    uint64_t n = 0;
    for (uint64_t w : bits_) n += static_cast<uint64_t>(__builtin_popcountll(w));
    return n;
}

RouteUnionResult route_union_from_top_ids(const int32_t* block, uint32_t layers, uint32_t tokens,
                                          uint32_t top_k, uint32_t layer_stride_ids,
                                          RouteUnionConfig cfg) noexcept {
    RouteUnionResult r;
    r.layers = layers;
    r.tokens = tokens;
    if (!block || layers == 0 || tokens == 0 || top_k == 0) {
        r.status = RouteUnionStatus::kUnavailable;
        return r;
    }
    if (layers > kRouteUnionMaxLayers || layers > cfg.max_layers) {
        r.status = RouteUnionStatus::kLayerOutOfRange;
        return r;
    }
    if (tokens > kRouteUnionMaxTokens || tokens > cfg.max_tokens) {
        r.status = RouteUnionStatus::kTooManyTokens;
        return r;
    }
    const uint64_t need_ids = static_cast<uint64_t>(tokens) * top_k;
    if (need_ids == 0 || need_ids > layer_stride_ids || need_ids > 0xFFFFFFFFull) {
        r.status = RouteUnionStatus::kTooManyIds;
        return r;
    }
    if (top_k > cfg.max_experts) {
        r.status = RouteUnionStatus::kTooManyIds;
        return r;
    }
    cfg.top_k = top_k;
    RouteUnion set(cfg);
    for (uint32_t layer = 0; layer < layers; ++layer) {
        const int32_t* ids = block + static_cast<uint64_t>(layer) * layer_stride_ids;
        const RouteUnionStatus st = set.add_layer(layer, ids, static_cast<uint32_t>(need_ids));
        if (st != RouteUnionStatus::kOk) {
            r.status = st;
            return r;
        }
    }
    const uint64_t n = set.count();
    if (n == 0) {
        r.status = RouteUnionStatus::kUnavailable;
        return r;
    }
    r.status = RouteUnionStatus::kOk;
    r.unique_experts = n;
    r.tokens = set.tokens();
    return r;
}

RouteUnionStatus record_completed_routes(RouteUnionFeedback* fb, const int32_t* block, uint32_t layers,
                                         uint32_t tokens, uint32_t top_k, uint32_t layer_stride_ids,
                                         RouteUnionConfig cfg) noexcept {
    if (!fb) return RouteUnionStatus::kUnavailable;
    // Fail closed: whatever the previous pass measured is no longer the latest completed pass.
    fb->clear();
    const RouteUnionResult r = route_union_from_top_ids(block, layers, tokens, top_k, layer_stride_ids, cfg);
    if (!r.ok() || r.unique_experts == 0) {
        return r.ok() ? RouteUnionStatus::kUnavailable : r.status;
    }
    if (!fb->observe_count(r.unique_experts, r.tokens)) return RouteUnionStatus::kUnavailable;
    return RouteUnionStatus::kOk;
}

} // namespace vegajet

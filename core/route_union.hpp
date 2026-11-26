// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Exact, bounded dedup of routed MoE experts for the EVICT depth controller.
//
// The only admissible source for EvictDepth::choose()'s `unique_experts` argument is the distinct set of
// (layer, routed-expert) keys observed in a COMPLETED target verifier pass.  This file provides:
//   * RouteUnion: a fixed 64x256-bit set (2 KiB) that counts those keys exactly and fails closed on any
//     out-of-range layer/expert id or oversized token block.  It never aliases an invalid id into the set,
//     because aliasing would manufacture a count that the route IDs do not support.
//   * route_union_from_top_ids(): the CPU producer for the existing rank-0 top-ID rows ([layer][token][top_k]).
//   * RouteUnionFeedback: a one-shot lagged signal.  A completed pass publishes one measured count; the next
//     choose() consumes it.  Cold start, a deferred/unavailable readback, a zero-token step, or a consumed
//     signal all mean "unknown" (0), never a metadata estimate such as top_k * layers.
#pragma once

#include <array>
#include <cstdint>

namespace vegajet {

inline constexpr uint32_t kRouteUnionMaxLayers = 64;
inline constexpr uint32_t kRouteUnionMaxExperts = 256;
inline constexpr uint32_t kRouteUnionMaxTokens = 16;

struct RouteUnionConfig {
    uint32_t max_layers = 48;     // MoE trunk layers that carry routed experts
    uint32_t max_experts = 256;   // routed experts per layer (ids are 0..max_experts-1)
    uint32_t max_tokens = 16;     // verified tokens per completed target pass
    uint32_t top_k = 8;           // ids per token and layer
};

enum class RouteUnionStatus : uint8_t {
    kOk = 0,              // exact count available
    kUnavailable,         // no completed routing data (cold, deferred, zero tokens, empty set)
    kLayerOutOfRange,     // layer >= max_layers
    kExpertOutOfRange,    // id < 0 or id >= max_experts; the union is rejected, not clamped
    kTooManyTokens,       // token block exceeds max_tokens (the caller must bound the pass)
    kTooManyIds,          // per-layer id span is not tokens*top_k or exceeds the stride
};

const char* route_union_status_name(RouteUnionStatus s) noexcept;

class RouteUnion {
public:
    explicit RouteUnion(RouteUnionConfig cfg = RouteUnionConfig{});

    // Clears the set and the failure state; the configuration is preserved.
    void reset() noexcept;

    // Adds one layer's routed ids (token-major: token 0's top_k ids, then token 1's, ...).
    // `n` must be tokens*top_k and tokens <= max_tokens.  On any invalid id the whole union is marked failed;
    // count() then returns 0 so a partial set is never mistaken for an exact measurement.
    RouteUnionStatus add_layer(uint32_t layer, const int32_t* ids, uint32_t n) noexcept;

    RouteUnionStatus status() const noexcept { return status_; }
    bool ok() const noexcept { return status_ == RouteUnionStatus::kOk; }
    uint64_t count() const noexcept;  // 0 unless status()==kOk
    uint32_t tokens() const noexcept { return tokens_seen_; }
    const RouteUnionConfig& config() const noexcept { return cfg_; }

private:
    static constexpr uint32_t kWords = (kRouteUnionMaxLayers * kRouteUnionMaxExperts) / 64u;
    RouteUnionConfig cfg_;
    std::array<uint64_t, kWords> bits_{};
    RouteUnionStatus status_ = RouteUnionStatus::kOk;
    uint32_t tokens_seen_ = 0;
};

struct RouteUnionResult {
    RouteUnionStatus status = RouteUnionStatus::kUnavailable;
    uint64_t unique_experts = 0;  // exact distinct (layer, expert) keys, valid only when status==kOk
    uint32_t layers = 0;
    uint32_t tokens = 0;
    bool ok() const noexcept { return status == RouteUnionStatus::kOk; }
};

// Dedups a completed rank-0 top-ID block laid out as [layer][token][top_k].  `layer_stride_ids` is the
// distance between layers in int32 ids (for the GPU capture buffer this is 16*8 = 128).  A null block,
// zero layers/tokens/top_k, or an invalid id yields a non-ok status and no count.
RouteUnionResult route_union_from_top_ids(const int32_t* block, uint32_t layers, uint32_t tokens,
                                          uint32_t top_k, uint32_t layer_stride_ids,
                                          RouteUnionConfig cfg = RouteUnionConfig{}) noexcept;

// One-shot lagged measurement.  record_completed_routes() clears any previous value first, so a deferred or
// unavailable readback cannot leave a stale count for the next proposal.
class RouteUnionFeedback {
public:
    void clear() noexcept {
        unique_ = 0;
        tokens_ = 0;
    }
    // Stores a measured count.  Zero is the unknown sentinel, not an observation; it clears the signal.  The
    // hard bounds reject impossible counts so a caller cannot publish metadata as if it were a routed-key count.
    bool observe_count(uint64_t unique, uint32_t tokens) noexcept {
        if (unique == 0 || tokens == 0 ||
            unique > static_cast<uint64_t>(kRouteUnionMaxLayers) * kRouteUnionMaxExperts ||
            tokens > kRouteUnionMaxTokens) {
            clear();
            return false;
        }
        unique_ = unique;
        tokens_ = tokens;
        ++observations_;
        return true;
    }
    bool available() const noexcept { return unique_ > 0 && tokens_ > 0; }
    uint64_t unique_experts() const noexcept { return unique_; }
    uint32_t tokens() const noexcept { return tokens_; }
    uint64_t observations() const noexcept { return observations_; }
    // Returns the measured count and consumes it; a second call without a new observation returns 0.
    uint64_t take() noexcept {
        const uint64_t u = unique_;
        unique_ = 0;
        tokens_ = 0;
        return u;
    }

private:
    uint64_t unique_ = 0;
    uint32_t tokens_ = 0;
    uint64_t observations_ = 0;
};

// Reads a completed target/draft route-ID block into `fb`.  `block == nullptr`, zero tokens, or a bounded-input
// failure clears `fb` and returns the non-ok status.  On success the exact count and the measured token count are
// published for the next one-shot take().
RouteUnionStatus record_completed_routes(RouteUnionFeedback* fb, const int32_t* block, uint32_t layers,
                                         uint32_t tokens, uint32_t top_k, uint32_t layer_stride_ids,
                                         RouteUnionConfig cfg = RouteUnionConfig{}) noexcept;

} // namespace vegajet

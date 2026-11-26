// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/evict_depth.hpp"

#include <algorithm>
#include <cmath>

namespace vegajet {

EvictDepth::EvictDepth(uint32_t max_depth, EvictDepthConfig cfg) : max_depth_(max_depth), cfg_(cfg) {
    // Keep the configuration inside the domain the controller documents.  A caller-supplied value outside it is
    // clamped rather than silently producing a negative/log-undefined decision.
    cfg_.prior_accept = std::clamp(cfg_.prior_accept, 0.0, 1.0);
    cfg_.min_survival = std::clamp(cfg_.min_survival, 1e-6, 0.999999);
    cfg_.ewma = std::clamp(cfg_.ewma, 1e-6, 1.0);
    if (cfg_.probe_depth == 0) cfg_.probe_depth = 1;
    reset();
}

void EvictDepth::reset() noexcept {
    acceptance_ = cfg_.prior_accept;
    observations_ = 0;
    zero_streak_ = 0;
    quiet_steps_ = 0;
    last_ = EvictDepthChoice{};
    last_.acceptance = acceptance_;
}

double EvictDepth::expert_pressure(uint64_t unique_experts, uint32_t cap) const noexcept {
    // Routing pressure is budgeted per proposal slot, not per configured maximum: a two-token proposal with a
    // huge expert union should not be excused just because max_depth is large.
    if (cfg_.experts_per_depth == 0 || max_depth_ == 0 || cap == 0) return 1.0;
    const double allowance = static_cast<double>(cfg_.experts_per_depth) * static_cast<double>(cap);
    if (!(allowance > 0.0)) return 1.0;
    const double pressure = static_cast<double>(unique_experts) / allowance;
    return pressure > 1.0 ? pressure : 1.0;
}

uint32_t EvictDepth::depth_for(double acceptance) const noexcept {
    // Largest k in [0, max_depth] with acceptance^k >= min_survival.  For acceptance < min_survival even the first
    // drafted position is not worth verifying, so the caller should fall back to a target-only step.
    if (max_depth_ == 0 || !(acceptance > 0.0)) return 0;
    if (acceptance >= 1.0) return max_depth_;
    if (acceptance < cfg_.min_survival) return 0;
    const double k = std::floor(std::log(cfg_.min_survival) / std::log(acceptance));
    if (!std::isfinite(k) || k <= 0.0) return 0;
    return static_cast<uint32_t>(std::min<double>(k, static_cast<double>(max_depth_)));
}

uint32_t EvictDepth::choose(uint32_t available_draft, uint64_t unique_experts) noexcept {
    EvictDepthChoice c;
    c.cap = std::min(available_draft, max_depth_);
    c.unique_experts = unique_experts;
    c.acceptance = acceptance_;
    c.expert_pressure = expert_pressure(unique_experts, c.cap);
    c.acceptance_limit = depth_for(acceptance_);

    if (c.cap == 0) {
        // Nothing to draft (end of request, budget exhausted, or max_depth == 0).  Do not consume the probe
        // cooldown here: no target-only step is being requested by this call.
        c.target_only = true;
        last_ = c;
        return 0;
    }

    const double effective_acceptance = std::clamp(acceptance_ / c.expert_pressure, 0.0, 1.0);
    const uint32_t acceptance_depth = depth_for(effective_acceptance);
    c.limited_by_experts = acceptance_depth < c.acceptance_limit;

    // Two consecutive fully rejected verifies are enough evidence that the drafter is not paying for itself at
    // this point in the trace.  Target-only steps are cheap; probes keep a changed workload (prose -> code) from
    // pinning the controller at zero forever.
    const bool hard_zero = zero_streak_ >= cfg_.hard_zero_streak;
    if (hard_zero || acceptance_depth == 0) {
        if (quiet_steps_ < cfg_.probe_after) {
            ++quiet_steps_;
            c.target_only = true;
            last_ = c;
            return 0;
        }
        quiet_steps_ = 0;
        c.probing = true;
        c.depth = std::min(c.cap, cfg_.probe_depth);
        last_ = c;
        return c.depth;
    }

    quiet_steps_ = 0;
    c.depth = std::min(c.cap, acceptance_depth);
    last_ = c;
    return c.depth;
}

void EvictDepth::observe(uint32_t attempted, uint32_t accepted) noexcept {
    if (attempted == 0) return;  // no target verification happened; target-only steps teach us nothing
    accepted = std::min(accepted, attempted);
    // Standard geometric-chain estimator for a speculative verify: the chain continues until the first mismatch,
    // so `accepted` successes followed by one failure estimate per-position acceptance as accepted/(accepted+1).
    const double sample = accepted == attempted ? 1.0 : static_cast<double>(accepted) / static_cast<double>(accepted + 1u);
    acceptance_ = std::clamp((1.0 - cfg_.ewma) * acceptance_ + cfg_.ewma * sample, 0.0, 1.0);
    ++observations_;
    if (accepted == 0) {
        if (zero_streak_ < UINT32_MAX) ++zero_streak_;
    } else {
        zero_streak_ = 0;
    }
    quiet_steps_ = 0;
}

} // namespace vegajet

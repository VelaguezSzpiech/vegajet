// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/spec_policy.hpp"

#include <algorithm>
#include <cmath>

namespace vegajet::spec_policy {

Chain best_chain(const std::vector<spec::DraftNode>& nodes) {
    Chain c;
    int32_t cur = -1;
    // Best-first order: the first node seen with a given parent is that parent's highest-score child.
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (nodes[i].parent != cur) continue;
        c.tokens.push_back(nodes[i].token);
        c.cum_score.push_back(nodes[i].score);
        cur = static_cast<int32_t>(i);
    }
    return c;
}

std::vector<double> conditional_probs(const Chain& c) {
    std::vector<double> p(c.size());
    double prev = 1.0;
    for (size_t j = 0; j < c.size(); ++j) {
        const double s = std::clamp(c.cum_score[j], 0.0, 1.0);
        p[j] = prev > 0.0 ? std::clamp(s / prev, 0.0, 1.0) : 0.0;
        prev = s;
    }
    return p;
}

// ---------------------------------------------------------------------------------------------------------
AcceptanceGovernor::AcceptanceGovernor(GovernorConfig cfg)
    : cfg_(cfg), k_(std::clamp(cfg.k_init, cfg.k_min, cfg.k_max)) {}

double AcceptanceGovernor::alpha() const {
    if (win_.empty()) return 0.6;
    double acc = 0.0, rej = 0.0;
    for (const auto& [p, a] : win_) {
        acc += a;
        if (a < p) rej += 1.0;
    }
    if (acc + rej <= 0.0) return 0.6;
    return acc / (acc + rej);
}

void AcceptanceGovernor::observe(uint32_t proposed, uint32_t accepted) {
    if (proposed == 0) return;
    win_.push_back({proposed, std::min(accepted, proposed)});
    while (win_.size() > cfg_.window) win_.pop_front();
    if (k_ == 0) since_probe_ = 0;
    if (win_.size() < cfg_.min_samples || since_change_ < cfg_.dwell) return;
    const double a = alpha();
    uint32_t nk = k_;
    if (k_ == 0) {
        if (a >= cfg_.lower_below) nk = cfg_.k_min;           // probe succeeded: re-enable
    } else if (a > cfg_.raise_above && k_ < cfg_.k_max) {
        nk = k_ + 1;
    } else if (a < cfg_.disable_below && k_ == cfg_.k_min) {
        nk = 0;
    } else if (a < cfg_.lower_below && k_ > cfg_.k_min) {
        nk = k_ - 1;
    }
    if (nk != k_) {
        k_ = nk;
        since_change_ = 0;
        ++changes_;
    }
}

void AcceptanceGovernor::tick() {
    ++since_change_;
    if (k_ == 0) ++since_probe_;
}

uint32_t AcceptanceGovernor::k() const {
    if (k_ == 0) return (cfg_.probe_every && since_probe_ >= cfg_.probe_every) ? cfg_.k_min : 0u;
    return k_;
}

// ---------------------------------------------------------------------------------------------------------
double VerifyCostModel::expected_union(uint32_t m) const {
    if (m < union_obs.size() && union_obs[m] > 0.0) return union_obs[m];
    if (n_experts == 0) return 0.0;
    const double p = static_cast<double>(experts_per_token) / n_experts;
    return n_experts * (1.0 - std::pow(1.0 - p, static_cast<double>(m)));
}

double VerifyCostModel::cost_ms(uint32_t m) const {
    if (m == 0) return 0.0;
    if (m < measured_ms.size() && measured_ms[m] > 0.0) return measured_ms[m];
    const double u1 = std::max(1e-9, expected_union(1));
    return dense_ms + expert_ms * expected_union(m) / u1 + per_tok_ms * m;
}

void ExpertUnionTracker::observe(uint32_t m, double distinct, double ema) {
    if (m == 0 || !(distinct > 0.0)) return;
    if (mean_.size() <= m) { mean_.resize(m + 1, 0.0); n_.resize(m + 1, 0); }
    mean_[m] = n_[m] == 0 ? distinct : (1.0 - ema) * mean_[m] + ema * distinct;
    ++n_[m];
}

void ExpertUnionTracker::apply(VerifyCostModel* cm) const {
    if (!cm) return;
    if (cm->union_obs.size() < mean_.size()) cm->union_obs.resize(mean_.size(), 0.0);
    for (size_t m = 0; m < mean_.size(); ++m)
        if (n_[m] > 0) cm->union_obs[m] = mean_[m];
}

EvictDecision plan_evict(const std::vector<double>& cond, const VerifyCostModel& cm, uint32_t min_keep,
                         double draft_ms) {
    EvictDecision best;
    const uint32_t n = static_cast<uint32_t>(cond.size());
    min_keep = std::min(min_keep, n);
    double prefix = 1.0, exp_tok = 1.0;
    best.tokens_per_ms = -1.0;
    for (uint32_t m = 0; m <= n; ++m) {
        if (m > 0) {
            prefix *= std::clamp(cond[m - 1], 0.0, 1.0);
            exp_tok += prefix;
        }
        if (m < min_keep) continue;
        const double cost = cm.cost_ms(m + 1) + draft_ms * m;
        const double rate = cost > 0.0 ? exp_tok / cost : 0.0;
        if (rate > best.tokens_per_ms * (1.0 + 1e-9)) {  // strict: ties keep the shorter prefix
            best.keep = m;
            best.expected_tokens = exp_tok;
            best.tokens_per_ms = rate;
        }
    }
    return best;
}

// ---------------------------------------------------------------------------------------------------------
MeasuredDepthGovernor::MeasuredDepthGovernor(MeasuredDepthConfig cfg) : cfg_(cfg) {
    if (cfg_.min_samples == 0) cfg_.min_samples = 1;
    if (cfg_.k_min == 0) cfg_.k_min = 1;
    ms_.assign(18, 0.0);
    n_.assign(18, 0u);
}

void MeasuredDepthGovernor::observe_verify(uint32_t m, double ms) {
    if (m == 0 || !(ms > 0.0) || m >= 64u) return;
    if (ms_.size() <= m) { ms_.resize(m + 1, 0.0); n_.resize(m + 1, 0u); }
    ms_[m] = (n_[m] == 0) ? ms : (1.0 - cfg_.ema) * ms_[m] + cfg_.ema * ms;
    ++n_[m];
}

void MeasuredDepthGovernor::observe_accept(uint32_t proposed, uint32_t accepted) {
    if (proposed == 0) return;
    win_.emplace_back(proposed, std::min(accepted, proposed));
    while (win_.size() > std::max<uint32_t>(1u, cfg_.alpha_window)) win_.pop_front();
}

double MeasuredDepthGovernor::alpha() const {
    uint64_t prop = 0, acc = 0;
    for (const auto& [p, a] : win_) { prop += p; acc += a; }
    if (prop == 0) return std::clamp(0.5, cfg_.alpha_lo, cfg_.alpha_hi);
    return std::clamp(static_cast<double>(acc) / static_cast<double>(prop), cfg_.alpha_lo, cfg_.alpha_hi);
}

double MeasuredDepthGovernor::measured_ms(uint32_t m) const {
    if (m >= ms_.size() || n_[m] < cfg_.min_samples) return 0.0;
    return ms_[m];
}

double MeasuredDepthGovernor::est_ms(uint32_t m) const {
    if (m == 0) return 0.0;
    const double exact = measured_ms(m);
    if (exact > 0.0) return exact;
    // Nearest trusted measurements on each side.
    uint32_t lo = 0, hi = 0;
    double v_lo = 0.0, v_hi = 0.0;
    for (uint32_t k = 1; k < ms_.size() && k < 64u; ++k) {
        const double v = measured_ms(k);
        if (v <= 0.0) continue;
        if (k <= m) { lo = k; v_lo = v; }
        else if (hi == 0) { hi = k; v_hi = v; }
    }
    const VerifyCostModel param;  // unmeasured-M shape: dense + expert union growth
    if (lo == 0) {
        if (hi == 0) return 0.0;
        // Below the smallest measured M: keep the parametric shape, scaled onto the measurement.
        return v_hi * param.cost_ms(m) / std::max(param.cost_ms(hi), 1e-9);
    }
    if (hi == 0) {
        // Above the largest measured M: extend the last measured slope (>= 0), never extrapolate downward.
        const double base = param.cost_ms(m) - param.cost_ms(lo);
        const double slope = std::max(0.0, (v_lo - param.cost_ms(lo)) / std::max(param.cost_ms(lo), 1e-9));
        return v_lo + base * (1.0 + slope);
    }
    const double t = static_cast<double>(m - lo) / static_cast<double>(hi - lo);
    return v_lo + (v_hi - v_lo) * t;
}

double MeasuredDepthGovernor::time_inflation(uint32_t m) const {
    const double base = est_ms(1);
    if (m == 0 || base <= 0.0) return 0.0;
    const double v = est_ms(m);
    return v > 0.0 ? v / base : 0.0;
}

uint32_t MeasuredDepthGovernor::samples(uint32_t m) const {
    return (m < n_.size()) ? n_[m] : 0u;
}

uint32_t MeasuredDepthGovernor::next_depth(uint32_t cap) const {
    if (cap == 0) return 0;
    bool any = false;
    for (uint32_t k = 1; k < ms_.size() && k < 64u && !any; ++k) any = measured_ms(k) > 0.0;
    if (!any) return 0;  // cold: leave the caller's depth alone

    VerifyCostModel cm;
    cm.measured_ms.assign(static_cast<size_t>(cap) + 2u, 0.0);
    for (uint32_t m = 1; m <= cap + 1u; ++m) cm.measured_ms[m] = est_ms(m);

    std::vector<double> cond(cap, alpha());
    const EvictDecision ev = plan_evict(cond, cm, std::min(cfg_.k_min, cap), 0.0);
    return std::min(std::max(ev.keep, std::min(cfg_.k_min, cap)), cap);
}

// ---------------------------------------------------------------------------------------------------------
DrafterKvSync::DrafterKvSync(SyncConfig cfg) : cfg_(cfg) {
    cfg_.stash_slots = std::max(cfg_.stash_slots, cfg_.catchup_window + 1u);
}

double DrafterKvSync::hole_fraction() const {
    return hole_hist_.empty() ? 0.0 : static_cast<double>(hole_count_) / static_cast<double>(hole_hist_.size());
}

SyncMode DrafterKvSync::effective_mode() const {
    return (cfg_.mode == SyncMode::kDeferred && eager_latched_) ? SyncMode::kEager : cfg_.mode;
}

void DrafterKvSync::note_positions(uint32_t n, uint32_t holes) {
    for (uint32_t i = 0; i < n; ++i) {
        const uint8_t h = i < holes ? 1u : 0u;
        hole_hist_.push_back(h);
        hole_count_ += h;
        holes_total_ += h;
        while (hole_hist_.size() > cfg_.hole_window) {
            hole_count_ -= hole_hist_.front();
            hole_hist_.pop_front();
        }
    }
    const double f = hole_fraction();
    if (f > cfg_.max_hole_frac) eager_latched_ = true;
    else if (f < cfg_.resume_hole_frac) eager_latched_ = false;
}

SyncPlan DrafterKvSync::before_neural() {
    SyncPlan plan;
    for (const Pending& p : pending_)
        plan.catchup.push_back({p.token, p.pos, HSource{HSource::kStash, p.slot}});
    pending_.clear();
    return plan;
}

SyncPlan DrafterKvSync::after_verify(const std::vector<int32_t>& committed, uint32_t pos0,
                                     bool drafter_ingested_first, bool next_cycle_neural_likely) {
    SyncPlan plan;
    const uint32_t first = drafter_ingested_first ? 1u : 0u;
    const uint32_t n = static_cast<uint32_t>(committed.size());
    auto src_of = [](uint32_t j) {
        return j == 0 ? HSource{HSource::kPend, 0u} : HSource{HSource::kTrunkRow, j - 1u};
    };
    // A neural cycle keeps the drafter warm (its next cycle most likely drafts again), so its catch-up runs
    // now; only suffix / plain cycles defer.
    const bool eager = effective_mode() == SyncMode::kEager || next_cycle_neural_likely || drafter_ingested_first;
    if (eager) {
        // Replay older pending positions first (increasing position order), then the new ones directly.
        plan = before_neural();
        for (uint32_t j = first; j < n; ++j) plan.catchup.push_back({committed[j], pos0 + j, src_of(j)});
        note_positions(n > first ? n - first : 0u, 0u);
        return plan;
    }
    uint32_t holes = 0;
    for (uint32_t j = first; j < n; ++j) {
        if (cfg_.catchup_window == 0) { ++holes; continue; }
        while (pending_.size() >= cfg_.catchup_window) {  // oldest pending position becomes a drafter KV hole
            pending_.pop_front();
            ++holes;
        }
        const uint32_t slot = next_slot_;
        next_slot_ = (next_slot_ + 1u) % cfg_.stash_slots;
        plan.stash.push_back({src_of(j), slot});
        pending_.push_back({committed[j], pos0 + j, slot});
    }
    plan.holes_added = holes;
    // History is per committed position; a hole is charged when it is created (at most one per new pending
    // position, since pending_ never exceeds catchup_window before a push).
    const uint32_t n_new = n > first ? n - first : 0u;
    note_positions(n_new, std::min(holes, n_new));
    return plan;
}

// ---------------------------------------------------------------------------------------------------------
SpecPolicy::SpecPolicy(PolicyConfig cfg) : cfg_(cfg), gov_neural_(cfg.neural_gov), gov_suffix_(cfg.suffix_gov) {}

CycleDecision SpecPolicy::decide(const CycleInputs& in) {
    CycleDecision d;
    ++st_.cycles;
    gov_neural_.tick();
    gov_suffix_.tick();
    const uint32_t k_budget = std::min(cfg_.max_draft, in.remaining > 1u ? in.remaining - 1u : 0u);
    if (k_budget == 0) { ++st_.plain; return d; }

    const uint32_t chain_n = static_cast<uint32_t>(std::min<size_t>(in.suffix.size(), k_budget));
    uint32_t conf = 0;
    if (in.suffix_match_len >= cfg_.suffix_min_match)
        while (conf < chain_n && in.suffix.cum_score[conf] >= cfg_.suffix_conf_score) ++conf;
    d.confident = conf;
    const bool suffix_on = gov_suffix_.k() > 0;
    const uint32_t need = std::min(cfg_.suffix_first_min, k_budget);

    if (suffix_on && conf >= need) {
        d.source = DraftSource::kSuffix;
        d.suffix_tokens.assign(in.suffix.tokens.begin(), in.suffix.tokens.begin() + conf);
        ++st_.suffix;
        return d;
    }
    const uint32_t kn = std::min(gov_neural_.k(), k_budget);
    if (in.neural_available && kn > 0) {
        d.source = DraftSource::kNeural;
        d.neural_k = kn;
        ++st_.neural;
        return d;
    }
    if (cfg_.short_suffix_fallback && suffix_on && chain_n > 0) {
        Chain c;
        c.tokens.assign(in.suffix.tokens.begin(), in.suffix.tokens.begin() + chain_n);
        c.cum_score.assign(in.suffix.cum_score.begin(), in.suffix.cum_score.begin() + chain_n);
        std::vector<double> cond = conditional_probs(c);
        const double a = gov_suffix_.alpha();  // calibrate the empirical probabilities by observed acceptance
        for (double& p : cond) p = std::min(p, a);
        const EvictDecision ev = plan_evict(cond, cfg_.cost, 0u, 0.0);
        if (ev.keep > 0) {
            d.source = DraftSource::kSuffix;
            d.suffix_tokens.assign(c.tokens.begin(), c.tokens.begin() + ev.keep);
            ++st_.suffix;
            st_.evicted += chain_n - ev.keep;
            return d;
        }
    }
    ++st_.plain;
    return d;
}

uint32_t SpecPolicy::evict_neural(uint32_t drafted, const std::vector<double>& cond_in) {
    std::vector<double> cond(cond_in);
    if (cond.size() < drafted) cond.resize(drafted, gov_neural_.alpha());
    cond.resize(drafted);
    const EvictDecision ev = plan_evict(cond, cfg_.cost, 0u, 0.0);  // drafting cost is already sunk
    st_.evicted += drafted - ev.keep;
    return ev.keep;
}

void SpecPolicy::observe(DraftSource src, uint32_t proposed, uint32_t accepted) {
    st_.proposed += proposed;
    st_.accepted += std::min(accepted, proposed);
    if (src == DraftSource::kNeural) gov_neural_.observe(proposed, accepted);
    else if (src == DraftSource::kSuffix) gov_suffix_.observe(proposed, accepted);
}

} // namespace vegajet::spec_policy

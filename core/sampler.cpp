// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_set>

namespace vegajet::sampling {

namespace {

// Higher logit first; equal logits -> lower id first. A strict weak order, so std::sort is deterministic.
bool better(const Candidate& a, const Candidate& b) {
    return a.logit > b.logit || (a.logit == b.logit && a.id < b.id);
}

} // namespace

uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

double counter_uniform(uint64_t seed, uint64_t step) {
    return static_cast<double>(splitmix64(seed ^ (step * 0x9E3779B97F4A7C15ull)) >> 11) * 0x1.0p-53;
}

Sampler::Sampler(SamplerParams p) : p_(p) {}

void Sampler::reset() {
    recent_.clear();
    step_ = 0;
}

void Sampler::accept(int32_t id) {
    ++step_;
    if (p_.repeat_last_n == 0) return;
    recent_.push_back(id);
    if (p_.repeat_last_n > 0)
        while (recent_.size() > static_cast<size_t>(p_.repeat_last_n)) recent_.pop_front();
}

void Sampler::apply_penalty(std::vector<Candidate>& c) const {
    if (p_.repeat_penalty == 1.0f || recent_.empty() || p_.repeat_last_n == 0) return;
    const std::unordered_set<int32_t> seen(recent_.begin(), recent_.end());
    for (auto& x : c) {
        if (!seen.count(x.id)) continue;
        x.logit = x.logit > 0.0f ? x.logit / p_.repeat_penalty : x.logit * p_.repeat_penalty;
    }
}

int32_t Sampler::pick(std::vector<Candidate>& c) {
    if (c.empty()) return -1;
    apply_penalty(c);
    // Non-finite logits never win (NaN would break the ordering).
    c.erase(std::remove_if(c.begin(), c.end(), [](const Candidate& x) { return std::isnan(x.logit); }), c.end());
    if (c.empty()) return -1;

    if (p_.temperature <= 0.0f) return std::min_element(c.begin(), c.end(), better)->id;

    if (p_.top_k > 0 && static_cast<size_t>(p_.top_k) < c.size()) {
        std::nth_element(c.begin(), c.begin() + p_.top_k, c.end(), better);
        c.resize(static_cast<size_t>(p_.top_k));
    }
    std::sort(c.begin(), c.end(), better);

    // Softmax at temperature T in double precision, max-subtracted.
    std::vector<double> prob(c.size());
    const double inv_t = 1.0 / static_cast<double>(p_.temperature);
    const double mx = static_cast<double>(c.front().logit);
    double sum = 0.0;
    for (size_t i = 0; i < c.size(); ++i) {
        prob[i] = std::isinf(c[i].logit) && c[i].logit < 0 ? 0.0 : std::exp((static_cast<double>(c[i].logit) - mx) * inv_t);
        sum += prob[i];
    }
    if (!(sum > 0.0)) return c.front().id;  // every candidate at -inf: fall back to the (id-ordered) first
    for (auto& p : prob) p /= sum;

    size_t keep = c.size();
    if (p_.top_p < 1.0f) {
        double cum = 0.0;
        for (size_t i = 0; i < c.size(); ++i) {
            cum += prob[i];
            if (cum >= static_cast<double>(p_.top_p)) { keep = i + 1; break; }
        }
    }
    if (p_.min_p > 0.0f) {
        const double floor_p = static_cast<double>(p_.min_p) * prob[0];
        size_t k = 1;
        while (k < keep && prob[k] >= floor_p) ++k;
        keep = k;
    }
    double kept = 0.0;
    for (size_t i = 0; i < keep; ++i) kept += prob[i];

    const double u = counter_uniform(p_.seed, step_) * kept;
    double cum = 0.0;
    for (size_t i = 0; i < keep; ++i) {
        cum += prob[i];
        if (u < cum) return c[i].id;
    }
    return c[keep - 1].id;  // rounding guard
}

int32_t Sampler::sample(const float* logits, size_t n) {
    if (!logits || n == 0) return -1;
    if (p_.temperature <= 0.0f && (p_.repeat_penalty == 1.0f || recent_.empty())) {
        // Greedy fast path: one pass, no allocation.
        size_t best = n;
        for (size_t i = 0; i < n; ++i)
            if (!std::isnan(logits[i]) && (best == n || logits[i] > logits[best])) best = i;
        return best == n ? -1 : static_cast<int32_t>(best);
    }
    std::vector<Candidate> c(n);
    for (size_t i = 0; i < n; ++i) c[i] = {static_cast<int32_t>(i), logits[i]};
    return pick(c);
}

int32_t Sampler::sample_candidates(std::vector<Candidate> cands) { return pick(cands); }

} // namespace vegajet::sampling

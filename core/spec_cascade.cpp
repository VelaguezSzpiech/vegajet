// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/spec_cascade.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdlib>

namespace vegajet::spec_cascade {

namespace {

bool fail(std::string* err, const std::string& m) {
    if (err) *err = m;
    return false;
}

inline uint64_t mix(uint64_t h, uint64_t v) {
    h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
    h ^= h >> 33;
    h *= 0xFF51AFD7ED558CCDull;
    h ^= h >> 33;
    return h;
}

void append(std::string* note, const std::string& m) {
    if (!note) return;
    if (!note->empty()) *note += "; ";
    *note += m;
}

} // namespace

const char* draft_source_name(DraftSource s) {
    return s == DraftSource::SUFFIX_ONLY ? "SUFFIX_ONLY" : "NEURAL_DRAFTER";
}

CascadeConfig config_from_env(CascadeConfig base, std::string* note) {
    if (note) note->clear();
    if (const char* v = std::getenv("VEGAJET_SPEC_CASCADE")) {  // == kCascadeEnv
        const std::string s(v);
        if (s == "1") base.enabled = true;
        else if (s == "0" || s.empty()) base.enabled = false;
        else append(note, std::string(kCascadeEnv) + "='" + s + "' ignored (use 0 or 1)");
    }
    if (const char* v = std::getenv("VEGAJET_SPEC_CASCADE_MIN")) {  // == kCascadeMinEnv
        char* end = nullptr;
        const long n = std::strtol(v, &end, 10);
        if (end != v && *end == '\0' && n >= 1 && n <= 64) base.min_confident = static_cast<uint32_t>(n);
        else append(note, std::string(kCascadeMinEnv) + "='" + v + "' ignored (use 1..64)");
    }
    return base;
}

ZeroTaxCascade::ZeroTaxCascade(CascadeConfig cfg) : cfg_(cfg) {
    cfg_.support_order = std::min(16u, std::max(1u, cfg_.support_order));
}

void ZeroTaxCascade::reserve(size_t tokens) {
    idx_.reserve(tokens);
    ctx_total_.reserve(tokens + 1);
    ctx_follow_.reserve(tokens + 1);
}

void ZeroTaxCascade::clear() {
    idx_.clear();
    st_ = {};
    ctx_total_.clear();
    ctx_follow_.clear();
}

uint64_t ZeroTaxCascade::ctx_hash(const std::vector<int32_t>& seq, size_t end) const {
    uint64_t h = 0xCBF29CE484222325ull ^ cfg_.support_order;
    for (size_t i = end - cfg_.support_order; i < end; ++i) h = mix(h, static_cast<uint32_t>(seq[i]));
    return h;
}

void ZeroTaxCascade::commit(int32_t token) {
    idx_.push(token);
    const auto& h = idx_.history();
    if (h.size() <= cfg_.support_order) return;
    const uint64_t c = ctx_hash(h, h.size() - 1u);
    ++ctx_total_[c];
    ++ctx_follow_[mix(c, static_cast<uint32_t>(token))];
}

SuffixDraft ZeroTaxCascade::suffix_draft() const {
    SuffixDraft d;
    d.match_length = idx_.longest_match().length;
    if (d.match_length == 0 || cfg_.max_draft == 0) return d;
    d.tokens = idx_.draft(cfg_.max_draft);
    d.degree = idx_.continuation_degrees(d.tokens);
    d.tokens.resize(d.degree.size());  // a draft() chain always walks fully; keep the two in step regardless
    const size_t K = d.tokens.size(), N = cfg_.support_order;
    const auto& h = idx_.history();
    std::vector<int32_t> seq;  // last N history tokens followed by the chain
    if (h.size() >= N) {
        seq.assign(h.end() - static_cast<std::ptrdiff_t>(N), h.end());
        seq.insert(seq.end(), d.tokens.begin(), d.tokens.end());
    }
    d.support.reserve(K);
    d.prob.reserve(K);
    d.confidence.reserve(K);
    double c = 1.0;
    bool run = d.match_length >= cfg_.min_match;
    for (size_t j = 0; j < K; ++j) {
        uint32_t n = 0, f = 0;
        if (!seq.empty()) {
            const uint64_t ch = ctx_hash(seq, N + j);
            const auto it = ctx_total_.find(ch);
            if (it != ctx_total_.end()) {
                n = it->second;
                const auto jt = ctx_follow_.find(mix(ch, static_cast<uint32_t>(d.tokens[j])));
                if (jt != ctx_follow_.end()) f = jt->second;
            }
        }
        const double p = static_cast<double>(f) / static_cast<double>(n + 1u);
        c *= std::min(1.0 / static_cast<double>(std::max(1u, d.degree[j])), p);
        d.support.push_back(n);
        d.prob.push_back(static_cast<float>(p));
        d.confidence.push_back(static_cast<float>(c));
        if (run && c >= cfg_.min_confidence && c > 0.0) ++d.confident;
        else run = false;
    }
    return d;
}

CascadeStep ZeroTaxCascade::peek() const {
    CascadeStep s;
    if (!cfg_.enabled) return s;
    s.suffix = suffix_draft();
    if (s.suffix.confident >= std::max(1u, cfg_.min_confident)) {
        s.source = DraftSource::SUFFIX_ONLY;
        s.draft.assign(s.suffix.tokens.begin(), s.suffix.tokens.begin() + s.suffix.confident);
    }
    return s;
}

CascadeStep ZeroTaxCascade::decide() {
    CascadeStep s = peek();
    ++st_.steps;
    if (s.source == DraftSource::SUFFIX_ONLY) {
        ++st_.suffix_only;
        st_.suffix_tokens += s.draft.size();
    } else {
        ++st_.neural;
    }
    return s;
}

// ---------------------------------------------------------------------------------------------------------
bool expert_union_truncate(const std::vector<Routing>& candidates, uint32_t n_layers, uint32_t n_experts,
                           const UnionBudget& budget, ExpertUnion* out, std::string* err) {
    if (!out) return fail(err, "null output");
    if (n_layers == 0 || n_experts == 0 || n_experts > 65536u) return fail(err, "bad layer / expert count");
    const size_t M = candidates.size();
    for (size_t i = 0; i < M; ++i) {
        if (candidates[i].size() != n_layers)
            return fail(err, "candidate " + std::to_string(i) + " has a routing for " + std::to_string(candidates[i].size()) +
                                 " layers, expected " + std::to_string(n_layers));
        for (const auto& layer : candidates[i])
            for (uint16_t e : layer)
                if (e >= n_experts) return fail(err, "candidate " + std::to_string(i) + " routes to expert " + std::to_string(e));
    }
    const size_t words = (n_experts + 63u) / 64u;
    std::vector<uint64_t> seen(static_cast<size_t>(n_layers) * words, 0);
    std::vector<uint32_t> per_layer(n_layers, 0);
    ExpertUnion u;
    u.unique_after.resize(M);
    u.max_layer_after.resize(M);
    const uint32_t floor_keep = static_cast<uint32_t>(std::min<size_t>(M, budget.min_keep));
    uint64_t total = 0, root = 0;
    uint32_t max_layer = 0;
    bool over = false;
    u.keep = static_cast<uint32_t>(M);
    for (size_t i = 0; i < M; ++i) {
        for (uint32_t l = 0; l < n_layers; ++l) {
            uint64_t* bits = seen.data() + static_cast<size_t>(l) * words;
            for (uint16_t e : candidates[i][l]) {
                const uint64_t m = uint64_t{1} << (e & 63u);
                if (bits[e >> 6] & m) continue;
                bits[e >> 6] |= m;
                ++total;
                max_layer = std::max(max_layer, ++per_layer[l]);
            }
        }
        if (i == 0) root = total;
        u.unique_after[i] = total;
        u.max_layer_after[i] = max_layer;
        const bool exceeds = (budget.max_unique_total && total > budget.max_unique_total) ||
                             (budget.max_unique_per_layer && max_layer > budget.max_unique_per_layer);
        if (exceeds && !over) {
            over = true;
            u.keep = std::max(static_cast<uint32_t>(i), floor_keep);
        }
    }
    u.truncated = u.keep < M;
    // Union of the kept prefix (a second pass: the running sets above may include tokens past the cut).
    std::fill(seen.begin(), seen.end(), 0);
    for (uint32_t i = 0; i < u.keep; ++i)
        for (uint32_t l = 0; l < n_layers; ++l)
            for (uint16_t e : candidates[i][l]) seen[static_cast<size_t>(l) * words + (e >> 6)] |= uint64_t{1} << (e & 63u);
    u.experts.assign(n_layers, {});
    for (uint32_t l = 0; l < n_layers; ++l) {
        for (size_t w = 0; w < words; ++w) {
            uint64_t b = seen[static_cast<size_t>(l) * words + w];
            while (b) {
                const unsigned bit = static_cast<unsigned>(__builtin_ctzll(b));
                u.experts[l].push_back(static_cast<uint16_t>(w * 64u + bit));
                b &= b - 1u;
            }
        }
        u.unique_kept += u.experts[l].size();
    }
    u.fan_out = root ? static_cast<double>(u.unique_kept) / static_cast<double>(root) : 0.0;
    *out = std::move(u);
    return true;
}

std::vector<int32_t> truncate_draft(const std::vector<int32_t>& draft, const ExpertUnion& u) {
    const size_t n = u.keep > 0 ? std::min<size_t>(draft.size(), u.keep - 1u) : 0u;
    return std::vector<int32_t>(draft.begin(), draft.begin() + static_cast<std::ptrdiff_t>(n));
}

} // namespace vegajet::spec_cascade

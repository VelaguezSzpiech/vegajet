// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/sharder.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace vegajet::shard {

std::vector<uint64_t> apportion(uint64_t units, const std::vector<double>& weights) {
    const size_t n = weights.size();
    std::vector<uint64_t> out(n, 0);
    if (n == 0) return out;
    double total = 0;
    for (double w : weights) total += std::max(0.0, w);
    if (total <= 0) { out[0] = units; return out; }
    std::vector<std::pair<double, size_t>> rem;
    uint64_t given = 0;
    for (size_t i = 0; i < n; ++i) {
        const double q = static_cast<double>(units) * std::max(0.0, weights[i]) / total;
        out[i] = static_cast<uint64_t>(std::floor(q));
        given += out[i];
        rem.push_back({q - std::floor(q), i});
    }
    std::stable_sort(rem.begin(), rem.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    for (size_t k = 0; given < units; ++k, ++given) out[rem[k % n].second]++;
    return out;
}

static std::vector<Range> to_ranges(const std::vector<uint64_t>& sizes, uint64_t scale, uint64_t limit) {
    std::vector<Range> r(sizes.size());
    uint64_t pos = 0;
    for (size_t i = 0; i < sizes.size(); ++i) {
        r[i].begin = std::min(pos, limit);
        pos += sizes[i] * scale;
        r[i].end = std::min(pos, limit);
    }
    return r;
}

std::vector<Range> split_rows(uint64_t rows, const std::vector<double>& bw, uint32_t row_unit) {
    const uint64_t u = row_unit ? static_cast<uint64_t>(row_unit) : static_cast<uint64_t>(kRowUnit);
    const uint64_t units = (rows + u - 1) / u;
    return to_ranges(apportion(units, bw), u, rows);
}

std::vector<Range> split_experts(uint64_t n_experts, const std::vector<double>& bw) {
    return to_ranges(apportion(n_experts, bw), 1, n_experts);
}

uint32_t AttnShardPlan::total_kv_heads_stored() const {
    uint32_t s = 0;
    for (const auto& g : gpus) s += static_cast<uint32_t>(g.kv_heads.size());
    return s;
}

AttnShardPlan plan_attention(uint32_t n_q, uint32_t n_kv, const std::vector<double>& bw, bool is_mla) {
    AttnShardPlan p;
    p.n_q_heads = n_q;
    p.n_kv_heads = n_kv;
    p.group = n_kv ? n_q / n_kv : 0;
    const size_t n = bw.size();
    p.gpus.resize(n);
    if (is_mla || (n_kv == 1 && n > 1)) {
        p.mode = AttnShardMode::SequenceSharded;
        double total = 0;
        for (double w : bw) total += w;
        for (size_t i = 0; i < n; ++i) {
            p.gpus[i].q_heads = {0, n_q};
            p.gpus[i].kv_heads = {0, n_kv};
            p.gpus[i].seq_fraction = total > 0 ? bw[i] / total : 1.0 / n;
        }
        return p;
    }
    const auto q = to_ranges(apportion(n_q, bw), 1, n_q);
    std::vector<uint32_t> owners(n_kv, 0);
    for (size_t i = 0; i < n; ++i) {
        p.gpus[i].q_heads = q[i];
        if (q[i].size() == 0 || p.group == 0) continue;
        p.gpus[i].kv_heads = {q[i].begin / p.group, (q[i].end - 1) / p.group + 1};
        for (uint64_t h = p.gpus[i].kv_heads.begin; h < p.gpus[i].kv_heads.end; ++h) owners[h]++;
    }
    for (size_t i = 0; i < n; ++i)
        for (uint64_t h = p.gpus[i].kv_heads.begin; h < p.gpus[i].kv_heads.end; ++h)
            if (owners[h] > 1) p.gpus[i].replicated_kv++;
    return p;
}

std::vector<GdnShard> plan_gdn(uint32_t n_k_groups, uint32_t n_v_heads, const std::vector<double>& bw) {
    const uint32_t per = n_k_groups ? n_v_heads / n_k_groups : 0;
    const auto g = to_ranges(apportion(n_k_groups, bw), 1, n_k_groups);
    std::vector<GdnShard> out(g.size());
    for (size_t i = 0; i < g.size(); ++i) out[i] = {g[i], {g[i].begin * per, g[i].end * per}};
    return out;
}

} // namespace vegajet::shard

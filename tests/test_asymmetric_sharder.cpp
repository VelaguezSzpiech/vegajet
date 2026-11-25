// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Bandwidth-proportional sharding (core/sharder) for N = 1..4, including measured-style 840/972 GB/s (N=2) and
// 840/972/972 GB/s (N=3) ratios:
//  * row splits: b=32 aligned boundaries, contiguous, zero lost / duplicated rows, each share within one unit of quota;
//  * experts and GDN groups: exact partitions, proportional;
//  * GQA: KV ranges are exactly the KV heads of each rank's q heads, boundary heads replicated and counted;
//    single-KV attention falls back to sequence sharding with fractions summing to 1;
//  * N=1: the step planner emits zero AllReduce ops (bypass); N>1 emits exactly two per layer.
#include "core/sharder.hpp"
#include "ops/layer_plan.hpp"
#include "tests/test_common.hpp"

#include <cmath>
#include <cstdio>
#include <numeric>
#include <set>

using namespace vegajet;
using namespace vegajet::shard;

namespace {

std::vector<std::vector<double>> bw_sets() {
    return {
        {840.0},
        {840.0, 972.0},
        {972.0, 840.0},
        {840.0, 972.0, 972.0},
        {840.0, 972.0, 972.0, 840.0},
        {800.0, 800.0, 800.0, 800.0},
    };
}

void check_partition(const std::vector<Range>& r, uint64_t total, bool align32) {
    uint64_t pos = 0;
    for (size_t i = 0; i < r.size(); ++i) {
        VJ_CHECK(r[i].begin == pos);  // contiguous: nothing lost, nothing duplicated
        VJ_CHECK(r[i].end >= r[i].begin);
        if (align32) VJ_CHECK(r[i].begin % kRowUnit == 0);
        if (align32 && r[i].end != total) VJ_CHECK(r[i].size() % kRowUnit == 0);
        pos = r[i].end;
    }
    VJ_CHECK(pos == total);
}

void check_proportional(const std::vector<Range>& r, uint64_t units_total, uint64_t unit, const std::vector<double>& bw) {
    const double sum = std::accumulate(bw.begin(), bw.end(), 0.0);
    uint64_t rows_total = 0;
    for (const auto& x : r) rows_total += x.size();
    for (size_t i = 0; i < r.size(); ++i) {
        const double quota = static_cast<double>(units_total) * bw[i] / sum;
        const double units = std::ceil(static_cast<double>(r[i].size()) / static_cast<double>(unit));
        VJ_CHECK(std::fabs(units - quota) < 1.0 + 1e-9);  // Hamilton: within one unit of the exact quota
    }
    (void)rows_total;
}

ops::ModelDesc tiny_moe(uint32_t layers) {
    ops::ModelDesc m;
    m.arch = "test-moe";
    m.d_model = 1024;
    m.vocab = 32000;
    m.attn = {16, 2, 64};
    m.moe = {64, 4, 256, 256, ops::ExpertShard::SplitFf};
    m.mixer.assign(layers, ops::MixerKind::Gqa);
    m.ffn.assign(layers, ops::FfnKind::Moe);
    return m;
}

} // namespace

int main() {
    uint32_t cases = 0;
    for (const auto& bw : bw_sets()) {
        const size_t n = bw.size();
        // Rows: projections, FFN widths, vocabularies, ragged sizes.
        for (uint64_t rows : {32ull, 96ull, 512ull, 1000ull, 3072ull, 4096ull, 6144ull, 10304ull, 151936ull, 248320ull}) {
            const auto r = split_rows(rows, bw);
            VJ_CHECK(r.size() == n);
            check_partition(r, rows, true);
            check_proportional(r, (rows + kRowUnit - 1) / kRowUnit, kRowUnit, bw);
            ++cases;
        }
        if (n == 2 && bw[0] < bw[1]) {
            const auto r = split_rows(3072, bw);  // 96 units at 840:972 -> 44.5 : 51.5
            VJ_CHECK(r[0].size() < r[1].size());
            VJ_CHECK(r[0].size() == 44 * kRowUnit || r[0].size() == 45 * kRowUnit);
        }
        // Experts (whole-expert parallel) and GDN key groups.
        for (uint64_t ne : {8ull, 64ull, 128ull, 256ull}) {
            const auto e = split_experts(ne, bw);
            check_partition(e, ne, false);
            check_proportional(e, ne, 1, bw);
            ++cases;
        }
        for (auto kv : {std::pair<uint32_t, uint32_t>{16, 32}, {16, 64}, {8, 32}}) {
            const auto g = plan_gdn(kv.first, kv.second, bw);
            uint64_t kpos = 0, vpos = 0;
            for (const auto& s : g) {
                VJ_CHECK(s.k_groups.begin == kpos && s.v_heads.begin == vpos);
                VJ_CHECK(s.v_heads.size() == s.k_groups.size() * (kv.second / kv.first));
                kpos = s.k_groups.end;
                vpos = s.v_heads.end;
            }
            VJ_CHECK(kpos == kv.first && vpos == kv.second);
            ++cases;
        }
        // GQA head parallel with boundary replication; single KV -> sequence sharded.
        for (uint32_t nq : {8u, 16u, 28u, 32u, 40u, 64u}) {
            for (uint32_t nkv : {1u, 2u, 4u, 8u}) {
                if (nq % nkv) continue;
                const auto p = plan_attention(nq, nkv, bw);
                VJ_CHECK(p.gpus.size() == n);
                if (nkv == 1 && n > 1) {
                    VJ_CHECK(p.mode == AttnShardMode::SequenceSharded);
                    double s = 0;
                    for (const auto& g : p.gpus) {
                        s += g.seq_fraction;
                        VJ_CHECK(g.q_heads.size() == nq && g.kv_heads.size() == 1);
                    }
                    VJ_CHECK(std::fabs(s - 1.0) < 1e-12);
                    ++cases;
                    continue;
                }
                VJ_CHECK(p.mode == AttnShardMode::HeadParallel);
                const uint32_t grp = nq / nkv;
                std::vector<uint32_t> owners(nkv, 0);
                uint64_t qpos = 0;
                for (const auto& g : p.gpus) {
                    VJ_CHECK(g.q_heads.begin == qpos);
                    qpos = g.q_heads.end;
                    std::set<uint64_t> need;
                    for (uint64_t h = g.q_heads.begin; h < g.q_heads.end; ++h) need.insert(h / grp);
                    VJ_CHECK(g.kv_heads.size() == need.size());
                    if (!need.empty()) VJ_CHECK(g.kv_heads.begin == *need.begin() && g.kv_heads.end == *need.rbegin() + 1);
                    for (uint64_t h : need) owners[h]++;
                }
                VJ_CHECK(qpos == nq);
                uint32_t extra = 0;
                for (uint32_t h = 0; h < nkv; ++h) {
                    VJ_CHECK(owners[h] >= 1);  // every KV head is stored somewhere
                    extra += owners[h] - 1;
                }
                VJ_CHECK(p.total_kv_heads_stored() == nkv + extra);
                for (size_t i = 0; i < n; ++i) {
                    uint32_t rep = 0;
                    for (uint64_t h = p.gpus[i].kv_heads.begin; h < p.gpus[i].kv_heads.end; ++h) rep += owners[h] > 1;
                    VJ_CHECK(p.gpus[i].replicated_kv == rep);
                }
                ++cases;
            }
        }
    }
    // 840/972 with 16:1 GQA (32 q, 2 kv): the slower rank gets fewer q heads and the straddled KV head is replicated.
    {
        const auto p = plan_attention(32, 2, {840.0, 972.0});
        VJ_CHECK(p.gpus[0].q_heads.size() == 15 && p.gpus[1].q_heads.size() == 17);
        VJ_CHECK(p.gpus[1].kv_heads.size() == 2 && p.gpus[1].replicated_kv == 1 && p.total_kv_heads_stored() == 3);
    }

    // N=1 AllReduce bypass and per-layer AllReduce count for N>1, through the step planner.
    const auto m = tiny_moe(6);
    for (const auto& bw : bw_sets()) {
        ops::TpDesc tp{bw, std::vector<uint32_t>(bw.size(), 60)};
        for (bool prefill : {false, true}) {
            ops::StepPlan plan;
            std::string err;
            VJ_CHECK(ops::build_step_plan(m, tp, {prefill ? 512u : 1u, prefill, 4096}, plan, &err));
            const uint32_t ar = plan.count(ops::OpKind::AllReduce);
            if (bw.size() == 1) VJ_CHECK(ar == 0);
            else VJ_CHECK(ar == 2 * m.n_layers());
            for (const auto& op : plan.ops)
                if (op.op.kind == ops::OpKind::AllReduce) VJ_CHECK(op.op.gpu == (1u << bw.size()) - 1u);
            ++cases;
        }
    }

    std::printf("test_asymmetric_sharder: %u sharding cases (N=1..4, 840/972 and 840/972/972 GB/s), N=1 AllReduce bypass OK\n",
                cases);
    std::printf("PASS\n");
    return 0;
}

// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// core/tp3_plan: pure TP=3 sharding of a hybrid GDN/GQA MoE and a dense GQA descriptor (31.7/34.1/34.1 shares,
// balance, VRAM), power-cap rules, and the direct / ring all-reduce schedules: exact CPU simulation (bit-identical
// ranks, canonical order for direct), chunk coverage, malformed schedules, and the cost model.
#include "core/tp3_plan.hpp"
#include "tests/test_common.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

using namespace vegajet;
using namespace vegajet::tp3;

namespace {

// Synthetic descriptors with the shape of the families involved (numbers are test values, not model specs).
ops::ModelDesc hybrid_moe() {
    ops::ModelDesc m;
    m.arch = "hybrid-gdn-moe";
    m.d_model = 2048;
    m.vocab = 151936;
    m.attn = {16, 2, 256};
    m.gdn = {16, 32, 128, 128, 4};
    m.moe = {512, 10, 512, 512, ops::ExpertShard::SplitFf};
    for (uint32_t l = 0; l < 48; ++l) {
        m.mixer.push_back((l + 1) % 4 == 0 ? ops::MixerKind::Gqa : ops::MixerKind::Gdn);
        m.ffn.push_back(ops::FfnKind::Moe);
    }
    return m;
}

ops::ModelDesc dense_gqa() {
    ops::ModelDesc m;
    m.arch = "dense-gqa";
    m.d_model = 5120;
    m.vocab = 151936;
    m.dense_ff = 17408;
    m.attn = {40, 8, 128};
    m.mixer.assign(64, ops::MixerKind::Gqa);
    m.ffn.assign(64, ops::FfnKind::Dense);
    return m;
}

bool has_fail(const std::vector<Check>& cs, const std::string& name, const std::string& text) {
    for (const auto& c : cs)
        if (c.name == name && !c.ok && c.detail.find(text) != std::string::npos) return true;
    return false;
}

} // namespace

int main() {
    const Tp3Config cfg;
    // ---- sharding. Dense: fine-grained rows, within 1 % of the shares at the default 32-row unit.
    {
        const ops::ModelDesc m = dense_gqa();
        const Plan p = plan_tp3(m, cfg);
        std::printf("  %s, 32-row units:\n%s", m.arch.c_str(), p.summary().c_str());
        VJ_CHECK(p.ok && p.max_share_error < 0.01);
        uint64_t q = 0, rows = 0, vocab = 0;
        for (uint32_t g = 0; g < kGpus; ++g) {
            q += p.gpu[g].q_heads.size();
            rows += p.gpu[g].ffn_rows.size();
            vocab += p.gpu[g].vocab_rows.size();
        }
        VJ_CHECK(q == m.attn.n_q_heads && rows == m.dense_ff && vocab == m.vocab);
        for (uint32_t g = 0; g + 1 < kGpus; ++g)  // contiguous, 32-row aligned splits
            VJ_CHECK(p.gpu[g].vocab_rows.end == p.gpu[g + 1].vocab_rows.begin && p.gpu[g].vocab_rows.end % 32 == 0);
        VJ_CHECK(p.gpu[0].weight_bytes < p.gpu[1].weight_bytes);  // the 31.7 % GPU carries less
    }
    // Hybrid MoE with expert_ff = 512: 32-row units give 16 units -> 5/6/5 and the balance check fails (GPU1 at
    // 37.5 %); 16-row units fix it, and so does whole-expert sharding (below).
    {
        const ops::ModelDesc m = hybrid_moe();
        const Plan p32 = plan_tp3(m, cfg);
        std::printf("  %s, 32-row units:\n%s", m.arch.c_str(), p32.summary().c_str());
        VJ_CHECK(!p32.ok && has_fail(p32.checks, "balance", "row unit 32"));
        VJ_CHECK(p32.gpu[0].expert_rows.size() == 160 && p32.gpu[1].expert_rows.size() == 192 && p32.gpu[2].expert_rows.size() == 160);
        Tp3Config c16 = cfg;
        c16.row_unit = 16;
        const Plan p16 = plan_tp3(m, c16);
        std::printf("  %s, 16-row units:\n%s", m.arch.c_str(), p16.summary().c_str());
        VJ_CHECK(p16.ok && p16.max_share_error < 0.01);
        VJ_CHECK(p16.gpu[0].expert_rows.size() == 160 && p16.gpu[1].expert_rows.size() == 176 && p16.gpu[2].expert_rows.size() == 176);
    }
    {
        // Whole-expert sharding: experts apportioned by share (512 -> 162 / 175 / 175).
        ops::ModelDesc m = hybrid_moe();
        m.moe.shard = ops::ExpertShard::WholeExperts;
        const Plan p = plan_tp3(m, cfg);
        VJ_CHECK(p.ok && p.max_share_error < 0.01 && p.gpu[0].experts.size() == 162 && p.gpu[1].experts.size() == 175 && p.gpu[2].experts.size() == 175);
        // VRAM: an 8 GB card cannot hold its share of the dense descriptor at 8.5 bits (~8.3 GB per GPU).
        Tp3Config small = cfg;
        small.vram_bytes = {8e9, 8e9, 8e9};
        small.bits_dense = 8.5;
        const Plan v = plan_tp3(dense_gqa(), small);
        VJ_CHECK(!v.ok && has_fail(v.checks, "vram.gpu1", "weights"));
        // Invalid inputs fail cleanly.
        Tp3Config bad = cfg;
        bad.share = {0.5, 0.5, 0.0};
        VJ_CHECK(!plan_tp3(hybrid_moe(), bad).ok);
        VJ_CHECK(!plan_tp3(ops::ModelDesc{}, cfg).ok);
    }
    // ---- power rules
    {
        VJ_CHECK(check_power(cfg).size() == 3);
        for (const auto& c : check_power(cfg)) VJ_CHECK(c.ok);
        Tp3Config c = cfg;
        c.power_cap_w = {150, 160, 150};
        VJ_CHECK(has_fail(check_power(c), "power.cap.gpu1", "exceeds the 150 W per-GPU ceiling for 3 GPUs"));
        c.power_cap_w = {40, 150, 150};
        VJ_CHECK(has_fail(check_power(c), "power.cap.gpu0", "below the 50 W floor"));
        c.power_cap_w = {150, 150, 150};
        c.psu_budget_w = 500;
        c.host_w = 120;
        VJ_CHECK(has_fail(check_power(c), "power.psu_budget", "caps 450 W + host 120 W"));
        c.psu_budget_w = 750;
        for (const auto& x : check_power(c)) VJ_CHECK(x.ok);
        c.power_cap_w = {150, 151, 150};
        VJ_CHECK(has_fail(plan_tp3(dense_gqa(), c).checks, "power.cap.gpu1", "ceiling"));
    }
    // ---- all-reduce schedules: exact simulation
    std::mt19937_64 rng(5);
    std::uniform_real_distribution<float> U(-4.0f, 4.0f);
    for (ArAlgo algo : {ArAlgo::Direct, ArAlgo::Ring}) {
        for (uint64_t n : {uint64_t(1), uint64_t(7), uint64_t(63), uint64_t(64), uint64_t(65), uint64_t(1000),
                           uint64_t(2048), uint64_t(4099), uint64_t(100000)}) {
            const ArSchedule s = plan_allreduce(n, 2, algo, cfg);
            VJ_CHECK(s.steps.size() == 12 && s.phases == (algo == ArAlgo::Direct ? 2u : 4u));
            VJ_CHECK(s.chunk[0].begin == 0 && s.chunk[2].end == n);
            for (uint32_t g = 0; g + 1 < kGpus; ++g) VJ_CHECK(s.chunk[g].end == s.chunk[g + 1].begin);
            if (n >= 100000)
                for (uint32_t g = 0; g < kGpus; ++g) VJ_CHECK(std::fabs(double(s.chunk[g].size()) - cfg.share[g] / 0.999 * n) <= 64);  // shares sum to 0.999
            std::array<std::vector<float>, kGpus> b;
            for (auto& v : b) { v.resize(n); for (auto& x : v) x = U(rng); }
            const auto in = b;
            std::string err;
            VJ_CHECK(simulate_allreduce(s, b, &err));
            for (uint64_t i = 0; i < n; ++i) {
                VJ_CHECK(std::memcmp(&b[0][i], &b[1][i], 4) == 0 && std::memcmp(&b[0][i], &b[2][i], 4) == 0);
                const double ref = double(in[0][i]) + in[1][i] + in[2][i];
                VJ_CHECK(std::fabs(b[0][i] - ref) <= 1e-5 * (1 + std::fabs(ref)));
                if (algo == ArAlgo::Direct) {
                    const float canon = (in[0][i] + in[1][i]) + in[2][i];
                    VJ_CHECK(std::memcmp(&b[0][i], &canon, 4) == 0);
                }
            }
        }
    }
    {
        // Malformed schedules are rejected.
        ArSchedule s = plan_allreduce(1000, 2, ArAlgo::Ring, cfg);
        std::array<std::vector<float>, kGpus> b{std::vector<float>(1000), std::vector<float>(1000), std::vector<float>(1000)};
        ArSchedule t = s;
        t.steps[3].end = 1001;
        VJ_CHECK(!simulate_allreduce(t, b));
        t = s;
        t.steps.push_back(t.steps.back());  // second copy into the same range in one phase
        VJ_CHECK(!simulate_allreduce(t, b));
        t = plan_allreduce(1000, 2, ArAlgo::Direct, cfg);
        t.steps.erase(t.steps.begin());     // owner 0 misses a partial
        VJ_CHECK(!simulate_allreduce(t, b));
        std::array<std::vector<float>, kGpus> wrong{std::vector<float>(10), std::vector<float>(1000), std::vector<float>(1000)};
        VJ_CHECK(!simulate_allreduce(s, wrong));
    }
    // ---- cost model
    {
        const uint64_t decode = 2048, prefill = 2048ull * 2048;  // fp16 hidden state: one token / a 2048-token chunk
        const ArSchedule dd = plan_allreduce(decode, 2, ArAlgo::Direct, cfg), dr = plan_allreduce(decode, 2, ArAlgo::Ring, cfg);
        const ArSchedule pd = plan_allreduce(prefill, 2, ArAlgo::Direct, cfg), pr = plan_allreduce(prefill, 2, ArAlgo::Ring, cfg);
        std::printf("  all-reduce, symmetric 12 GB/s links: decode 4 KB direct %.1f us / ring %.1f us; prefill 8 MB "
                    "direct %.0f us / ring %.0f us\n", dd.modeled_us, dr.modeled_us, pd.modeled_us, pr.modeled_us);
        VJ_CHECK(dd.modeled_us < 0.6 * dr.modeled_us);           // latency bound: 2 steps vs 4
        VJ_CHECK(pd.modeled_us <= pr.modeled_us + 1e-9);          // same 4/3 B per GPU, fewer latencies
        VJ_CHECK(choose_allreduce(decode, 2, cfg).algo == ArAlgo::Direct);
        VJ_CHECK(std::fabs(pd.modeled_us - pr.modeled_us) / pr.modeled_us < 0.05);
        // GPU0 behind a Gen1 x16 link (~3.2 GB/s): every message to / from it slows down.
        Tp3Config g1 = cfg;
        for (uint32_t j = 1; j < kGpus; ++j) { g1.link[0][j].gbs = 3.2; g1.link[j][0].gbs = 3.2; }
        const ArSchedule g1d = plan_allreduce(prefill, 2, ArAlgo::Direct, g1), g1r = plan_allreduce(prefill, 2, ArAlgo::Ring, g1);
        std::printf("  GPU0 on a 3.2 GB/s link: prefill direct %.0f us / ring %.0f us -> %s\n", g1d.modeled_us,
                    g1r.modeled_us, algo_name(choose_allreduce(prefill, 2, g1).algo));
        VJ_CHECK(g1d.modeled_us > pd.modeled_us && g1r.modeled_us > pr.modeled_us);
        // A pair without peer access bounces through host memory.
        Tp3Config np = cfg;
        np.link[0][2].p2p = np.link[2][0].p2p = false;
        const ArSchedule nd = plan_allreduce(decode, 2, ArAlgo::Direct, np);
        VJ_CHECK(nd.modeled_us > dd.modeled_us);
        const ArSchedule ch = choose_allreduce(prefill, 2, np);
        VJ_CHECK(ch.modeled_us <= std::min(plan_allreduce(prefill, 2, ArAlgo::Direct, np).modeled_us,
                                           plan_allreduce(prefill, 2, ArAlgo::Ring, np).modeled_us) + 1e-9);
    }
    std::printf("test_tp3_plan: TP=3 sharding, balance, VRAM, power caps, direct/ring all-reduce exact, cost model OK\nPASS\n");
    return 0;
}

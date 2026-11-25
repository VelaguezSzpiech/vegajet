// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// core/vram_planner: the bw_new/bw_old > bpw_new/bpw_old rule (strict), the IQ2_S -> VJ_Q4_A8 M=1 ban, routed
// IQ3_XXS MoE Down promoted first, per-layer partial promotion inside the budget, the >= 2 GiB/GPU reserve, and the
// `vegajet plan` dry-run report. Synthetic hybrid-MoE geometry, two GPUs at 840 / 972 GB/s.
#include "core/vram_planner.hpp"
#include "tests/test_common.hpp"

#include <cmath>
#include <cstdio>

using namespace vegajet::vram;

namespace {

constexpr uint64_t kGiB = 1ull << 30;

std::vector<TensorSpec> model(uint32_t layers) {
    const uint64_t D = 3072, ff = 1024, E = 256;
    std::vector<TensorSpec> t;
    const std::vector<double> half = {0.5, 0.5};
    for (uint32_t L = 0; L < layers; ++L) {
        const std::string p = "blk." + std::to_string(L) + ".";
        t.push_back({p + "attn_qkv", Role::Dense, static_cast<int32_t>(L), Fmt::VJ_Q4_A8, {Fmt::VJ_Q8_DP4A}, D * 8192, 1.0, half});
        t.push_back({p + "ffn_gate_up_exps", Role::RoutedGateUp, static_cast<int32_t>(L), Fmt::IQ2_S,
                     {Fmt::VJ_Q4_A8, Fmt::VJ_Q4_DOT8}, E * 2 * ff * D, 8.0 / E, half});
        t.push_back({p + "ffn_down_exps", Role::RoutedDown, static_cast<int32_t>(L), Fmt::IQ3_XXS, {Fmt::VJ_DOT8_EXACT},
                     E * ff * D, 8.0 / E, half});
    }
    t.push_back({"output", Role::LmHead, -1, Fmt::VJ_Q8_DP4A, {}, D * 248320, 1.0, half});
    return t;
}

uint64_t baseline_bytes_per_gpu(const std::vector<TensorSpec>& t) {
    uint64_t b = 0;
    for (const auto& x : t) b += static_cast<uint64_t>(std::ceil(tensor_bytes(x.base, x.weights) * 0.5));
    return b;
}

} // namespace

int main() {
    // bpw table sanity.
    VJ_CHECK(fmt_bpw(Fmt::IQ2_S) == 2.5625 && fmt_bpw(Fmt::IQ3_XXS) == 3.0625 && fmt_bpw(Fmt::VJ_Q4_A8) == 5.0);
    const BandwidthTable bw = BandwidthTable::gfx906_defaults();
    VJ_CHECK(bw.gbs(Fmt::IQ2_S, 1) == 560 && bw.gbs(Fmt::IQ2_S, 8) == 600 && bw.gbs(Fmt::IQ2_S, 512) == 420);

    const uint32_t layers = 48;
    const auto t = model(layers);
    const uint64_t base = baseline_bytes_per_gpu(t);
    const uint64_t down_extra_per_layer =
        static_cast<uint64_t>(std::ceil((tensor_bytes(Fmt::VJ_DOT8_EXACT, 256ull * 1024 * 3072) -
                                         tensor_bytes(Fmt::IQ3_XXS, 256ull * 1024 * 3072)) * 0.5));
    // Budget: room for exactly 30 Down promotions above the 2 GiB reserve on the tighter GPU.
    const uint64_t vram = 32 * kGiB;
    const uint64_t room = 30 * down_extra_per_layer + down_extra_per_layer / 2;
    GpuBudget g0{vram, 1.0, 0, 0, 0, 0}, g1{vram, 972.0 / 840.0, 0, 0, 0, 0};
    g0.kv_bytes = vram - 2 * kGiB - base - room;
    g1.kv_bytes = g0.kv_bytes - kGiB;  // GPU1 has more headroom; GPU0 binds
    PlannerConfig cfg;
    cfg.m = 1;
    const PlanResult r = plan(t, {g0, g1}, bw, cfg);
    VJ_CHECK(r.ok);
    uint32_t promoted = 0, ban = 0, ratio = 0, budget = 0;
    for (const auto& d : r.decisions) {
        if (d.verdict == Verdict::Promoted) { VJ_CHECK(d.role == Role::RoutedDown && d.to == Fmt::VJ_DOT8_EXACT && d.ms_saved > 0); ++promoted; }
        if (d.verdict == Verdict::RejectedBan) { VJ_CHECK(d.role == Role::RoutedGateUp); ++ban; }
        if (d.verdict == Verdict::RejectedRatio) { VJ_CHECK(d.role == Role::Dense); ++ratio; }
        if (d.verdict == Verdict::RejectedBudget) { VJ_CHECK(d.role == Role::RoutedDown); ++budget; }
    }
    VJ_CHECK(promoted == 30 && budget == layers - 30 && ban == layers && ratio == layers);
    VJ_CHECK(r.decisions[r.promotion_order.front()].role == Role::RoutedDown);
    for (const auto& gp : r.gpus) VJ_CHECK(gp.reserve == 2 * kGiB && gp.headroom() >= 0);
    VJ_CHECK(r.ms_per_token_final < r.ms_per_token_base);
    // IQ2_S Gate/Up: the DOT8 candidate fails the ratio rule and VJ_Q4_A8 is banned; the ban is what is reported.
    for (const auto& d : r.decisions)
        if (d.role == Role::RoutedGateUp) VJ_CHECK(d.why.find("banned") != std::string::npos);

    // IQ3_XXS Down goes first even when another class-1 candidate has a far higher ROI.
    {
        BandwidthTable fast = bw;
        fast.set(Fmt::VJ_Q8_DP4A, 1, 5000);  // dense Q4_A8 -> Q8 now passes the ratio rule with a huge ROI
        const PlanResult q = plan(t, {g0, g1}, fast, cfg);
        VJ_CHECK(q.ok && q.decisions[q.promotion_order.front()].role == Role::RoutedDown);
        uint32_t downs_first = 0;
        for (uint32_t idx : q.promotion_order) { if (q.decisions[idx].role != Role::RoutedDown) break; ++downs_first; }
        VJ_CHECK(downs_first == 30);
        PlannerConfig roi_only = cfg;
        roi_only.iq3_down_first = false;
        const PlanResult z = plan(t, {g0, g1}, fast, roi_only);
        VJ_CHECK(z.decisions[z.promotion_order.front()].role == Role::Dense);
    }
    // The ban holds even when a table claims VJ_Q4_A8 is far faster at M=1; at M=16 the ratio rule decides alone.
    {
        BandwidthTable b2 = bw;
        b2.set(Fmt::VJ_Q4_A8, 1, 2000);
        b2.set(Fmt::VJ_Q4_A8, 16, 2000);
        GpuBudget big{64 * kGiB, 1.0, 0, 0, 0, 0};
        const PlanResult m1 = plan(t, {big, big}, b2, cfg);
        for (const auto& d : m1.decisions) if (d.role == Role::RoutedGateUp) VJ_CHECK(d.verdict == Verdict::RejectedBan);
        PlannerConfig c16 = cfg;
        c16.m = 16;
        const PlanResult m16 = plan(t, {big, big}, b2, c16);
        uint32_t gu = 0;
        for (const auto& d : m16.decisions)
            if (d.role == Role::RoutedGateUp && d.verdict == Verdict::Promoted && d.to == Fmt::VJ_Q4_A8) ++gu;
        VJ_CHECK(gu == layers);
        const PlanResult d16 = plan(t, {big, big}, bw, c16);  // default table: 760/600 < 5.0/2.5625 -> ratio reject
        for (const auto& d : d16.decisions) if (d.role == Role::RoutedGateUp) VJ_CHECK(d.verdict == Verdict::RejectedRatio);
    }
    // Strictness: equal ratios are rejected (no promotion that only breaks even).
    {
        BandwidthTable eq;
        eq.set(Fmt::IQ3_XXS, 1, 3.0625 * 128);
        eq.set(Fmt::VJ_DOT8_EXACT, 1, 4.25 * 128);
        std::vector<TensorSpec> one = {{"d", Role::RoutedDown, 0, Fmt::IQ3_XXS, {Fmt::VJ_DOT8_EXACT}, 1u << 24, 1.0, {1.0}}};
        GpuBudget gb{8 * kGiB, 1.0, 0, 0, 0, 0};
        VJ_CHECK(plan(one, {gb}, eq, cfg).decisions[0].verdict == Verdict::RejectedRatio);
        eq.set(Fmt::VJ_DOT8_EXACT, 1, 4.25 * 128 + 1);
        VJ_CHECK(plan(one, {gb}, eq, cfg).decisions[0].verdict == Verdict::Promoted);
    }
    // Reserve floor and baseline overflow.
    {
        PlannerConfig low = cfg;
        low.reserve_bytes = kGiB / 2;
        GpuBudget gb{8 * kGiB, 1.0, 0, 0, 0, 0};
        std::vector<TensorSpec> one = {{"x", Role::Dense, 0, Fmt::VJ_Q4_A8, {}, 1u << 20, 1.0, {1.0}}};
        VJ_CHECK(plan(one, {gb}, bw, low).gpus[0].reserve == 2 * kGiB);
        GpuBudget tiny{2 * kGiB, 1.0, 0, 0, 0, 0};
        const PlanResult bad = plan(one, {tiny}, bw, low);
        VJ_CHECK(!bad.ok && bad.error.find("does not fit") != std::string::npos);
        VJ_CHECK(!plan(one, {}, bw, low).ok);
    }
    const std::string rep = format_plan(r, cfg);
    VJ_CHECK(rep.find("first promotion: blk.0.ffn_down_exps (IQ3_XXS -> VJ_DOT8_EXACT)") != std::string::npos);
    VJ_CHECK(rep.find("rejected:ban") != std::string::npos && rep.find("x48") != std::string::npos);
    VJ_CHECK(rep.find("promoted         routed_down    IQ3_XXS  -> VJ_DOT8_EXACT x30") != std::string::npos);
    std::printf("%s", rep.c_str());
    std::printf("test_vram_planner: ratio rule, M=1 ban, IQ3 Down first, partial promotion, 2 GiB reserve, report OK\nPASS\n");
    return 0;
}

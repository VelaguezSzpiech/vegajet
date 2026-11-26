// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// v1.3 crossover rule (core/quant_repacker.*): a format promotion old -> new happens only when the
// owner's per-format bandwidth table measures bw_new/bw_old > bpw_new/bpw_old, per tensor class, over
// the total usable VRAM summed across N ranks. Pure CPU: no bandwidth number is compiled in.
#include "core/quant_repacker.hpp"
#include "tests/test_common.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unistd.h>

using namespace vegajet;
using namespace vegajet::quant;

namespace {

constexpr uint64_t GiB = 1ull << 30;

// A routed-expert class as the loader would size it: 48 layers, 256 experts each on one rank set.
PromoClassSpec expert_class(const char* name, GgmlType disk, uint32_t out, uint32_t in) {
    const uint64_t experts = 256, layers = 48;
    PromoClassSpec c;
    c.name = name;
    c.from_format = ggml_type_name(static_cast<uint32_t>(disk));
    c.to_format = "VJ_Q4_DOT8";
    c.role = TensorRole::RoutedMoE;
    c.disk = disk;
    c.old_bytes = VramBudgetPlanner::estimate_bytes(NativeFormat::VJ_IQ_PACKED, disk, out, in, experts) * layers;
    c.new_bytes = VramBudgetPlanner::estimate_bytes(NativeFormat::VJ_Q4_DOT8, disk, out, in, experts) * layers;
    return c;
}

// Bytes the promotion charges to the N-rank pool: the new copy minus the old one it replaces.
int64_t delta_of(const PromoClassSpec& c) { return static_cast<int64_t>(c.new_bytes) - static_cast<int64_t>(c.old_bytes); }

} // namespace

int main() {
    // ------------------------------------------------------------ table parsing, override, coverage
    {
        const std::string path = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") +
                                 "/vj_v13_format_bw_" + std::to_string(::getpid()) + ".json";
        {
            std::ofstream o(path);
            o << "{\n  \"formats\": [\n"
                 "    {\"format\": \"IQ3_XXS\", \"m_max\": 1, \"bw_gbs\": 311.5},\n"
                 "    {\"format\": \"IQ3_XXS\", \"m_max\": 8, \"bw_gbs\": 402.0},\n"
                 "    {\"format\": \"VJ_Q4_DOT8\", \"m_max\": 1, \"bw_gbs\": 555.0},\n"
                 "    {\"format\": \"VJ_Q4_A8\", \"m_max\": 1, \"bw_gbs\": 600.0}\n"
                 "  ]\n}\n";
        }
        FormatBwTable t;
        VJ_CHECK(FormatBwTable::load_json(path, t));
        VJ_CHECK(t.size() == 4);
        VJ_CHECK(t.gbs("IQ3_XXS", 1) == 311.5);   // narrowest entry that covers M=1
        VJ_CHECK(t.gbs("IQ3_XXS", 8) == 402.0);   // the wider entry takes over
        VJ_CHECK(t.gbs("IQ3_XXS", 9) == 0.0);     // uncovered M is unmeasured, never extrapolated
        VJ_CHECK(t.gbs("IQ2_S", 1) == 0.0);       // missing format stays unmeasured
        t.set("IQ2_S", 320.0);                    // owner override in code
        VJ_CHECK(t.gbs("IQ2_S", 1) == 320.0);
        t.set("IQ2_S", 350.0);                    // same (format, m_max): replace, not duplicate
        VJ_CHECK(t.size() == 5 && t.gbs("IQ2_S", 1) == 350.0);
        FormatBwTable missing;
        VJ_CHECK(!FormatBwTable::load_json(path + ".does-not-exist", missing) && missing.empty());
        std::remove(path.c_str());
    }

    // ------------------------------------------------------------ per-class crossover, 2 ranks
    {
        // IQ3_XXS Down: 2.5625.. 4.5 bpw -> bpw ratio 1.469; IQ2_S Gate/Up: -> 1.756.
        PromoClassSpec iq3 = expert_class("IQ3_XXS.ffn_down.routed", GgmlType::IQ3_XXS, 512, 3072);
        PromoClassSpec iq2 = expert_class("IQ2_S.ffn_gate.routed", GgmlType::IQ2_S, 512, 3072);
        VJ_CHECK(iq3.old_bytes > 0 && iq3.new_bytes > iq3.old_bytes);
        VJ_CHECK(double(iq2.new_bytes) / double(iq2.old_bytes) > double(iq3.new_bytes) / double(iq3.old_bytes));

        FormatBwTable t;
        t.set("IQ3_XXS", 300.0);
        t.set("IQ2_S", 320.0);
        t.set("VJ_Q4_DOT8", 500.0);  // 500/300 = 1.667 > 1.469 (IQ3 passes), 500/320 = 1.5625 < 1.756 (IQ2 fails)
        const uint64_t usable = 48 * GiB;  // summed over 2 ranks
        PromoPlan p = VramBudgetPlanner::plan_crossover({iq3, iq2}, usable, t, 1);
        VJ_CHECK(p.promoted_classes == 1);
        VJ_CHECK(p.verdicts[0].promoted);
        VJ_CHECK(p.verdicts[0].bw_ratio > p.verdicts[0].bpw_ratio);
        VJ_CHECK(p.verdicts[0].reason.find("bw ratio") != std::string::npos);
        VJ_CHECK(!p.verdicts[1].promoted);
        VJ_CHECK(p.verdicts[1].reason.find("not worth it") != std::string::npos);
        VJ_CHECK(p.used_bytes == static_cast<uint64_t>(p.verdicts[0].delta_bytes));
        const std::string s = p.summary();
        VJ_CHECK(s.find("PROMOTE IQ3_XXS.ffn_down.routed: IQ3_XXS -> VJ_Q4_DOT8") != std::string::npos);
        VJ_CHECK(s.find("KEEP    IQ2_S.ffn_gate.routed: IQ2_S -> VJ_Q4_DOT8") != std::string::npos);
        VJ_CHECK(s.find("1 of 2 classes promoted") != std::string::npos);

        // Enough VRAM + a table that says the wider format streams faster -> both routed classes move.
        t.set("VJ_Q4_DOT8", 600.0);  // 600/320 = 1.875 > 1.756
        p = VramBudgetPlanner::plan_crossover({iq3, iq2}, usable, t, 1);
        VJ_CHECK(p.promoted_classes == 2 && p.verdicts[0].promoted && p.verdicts[1].promoted);
        VJ_CHECK(p.used_bytes == static_cast<uint64_t>(p.verdicts[0].delta_bytes + p.verdicts[1].delta_bytes));
        VJ_CHECK(p.used_bytes <= usable);

        // Strict law: equality at the crossover point is a refusal.
        PromoClassSpec eq = iq3;
        eq.old_bytes = 1000; eq.new_bytes = 1500;  // bpw ratio 1.5
        t.set("IQ3_XXS", 400.0);
        t.set("VJ_Q4_DOT8", 600.0);                // bw ratio 1.5 exactly
        p = VramBudgetPlanner::plan_crossover({eq}, usable, t, 1);
        VJ_CHECK(!p.verdicts[0].promoted && p.verdicts[0].reason.find("not worth it") != std::string::npos);
        t.set("VJ_Q4_DOT8", 600.001);
        VJ_CHECK(VramBudgetPlanner::plan_crossover({eq}, usable, t, 1).verdicts[0].promoted);

        // Banned regression: IQ2_S -> VJ_Q4_A8 at M=1, whatever the bandwidth says.
        PromoClassSpec ban = iq2;
        ban.to_format = "VJ_Q4_A8";
        t.set("VJ_Q4_A8", 5000.0);
        p = VramBudgetPlanner::plan_crossover({ban}, usable, t, 1);
        VJ_CHECK(!p.verdicts[0].promoted && p.verdicts[0].reason.find("banned") != std::string::npos);
        VJ_CHECK(p.promoted_classes == 0 && p.used_bytes == 0);
        // At M=4 the ban does not apply, but the table has to cover M=4: no extrapolation.
        VJ_CHECK(!VramBudgetPlanner::plan_crossover({ban}, usable, t, 4).verdicts[0].promoted);
        t.set("IQ2_S", 320.0, 4);
        t.set("VJ_Q4_A8", 5000.0, 4);
        p = VramBudgetPlanner::plan_crossover({ban}, usable, t, 4);
        VJ_CHECK(p.verdicts[0].promoted && p.verdicts[0].reason.find("banned") == std::string::npos);

        // No table at all: nothing moves, every class says why.
        FormatBwTable none;
        p = VramBudgetPlanner::plan_crossover({iq3, iq2}, usable, none, 1);
        VJ_CHECK(p.promoted_classes == 0 && p.used_bytes == 0);
        for (const auto& v : p.verdicts) VJ_CHECK(v.reason.find("not measured") != std::string::npos);
    }

    // ------------------------------------------------------------ budget and priority over N ranks
    {
        PromoClassSpec iq3 = expert_class("IQ3_XXS.ffn_down.routed", GgmlType::IQ3_XXS, 512, 3072);
        PromoClassSpec iq2 = expert_class("IQ2_S.ffn_gate.routed", GgmlType::IQ2_S, 512, 3072);
        FormatBwTable t;
        t.set("IQ3_XXS", 300.0);
        t.set("IQ2_S", 300.0);
        t.set("VJ_Q4_DOT8", 900.0);

        // Room for the IQ3 class only. IQ2 is refused as "does not fit" and says how much is missing.
        const uint64_t tight = static_cast<uint64_t>(delta_of(iq3)) + 8 * 1024 * 1024;
        PromoPlan p = VramBudgetPlanner::plan_crossover({iq2, iq3}, tight, t, 1);  // IQ2 listed first
        VJ_CHECK(p.verdicts[1].promoted);                                        // priority: IQ3 first
        VJ_CHECK(!p.verdicts[0].promoted && p.verdicts[0].reason.find("does not fit") != std::string::npos);
        VJ_CHECK(p.used_bytes == static_cast<uint64_t>(delta_of(iq3)) && p.used_bytes <= tight);

        // One rank's worth less than needed: nothing fits, the summary says so.
        p = VramBudgetPlanner::plan_crossover({iq3, iq2}, static_cast<uint64_t>(delta_of(iq3)) - 1, t, 1);
        VJ_CHECK(p.promoted_classes == 0 && p.verdicts[0].reason.find("does not fit") != std::string::npos);
        // The held class is reported, not silently dropped.
        VJ_CHECK(p.verdicts[1].promoted || p.verdicts[1].reason.find("held") != std::string::npos);

        // frees_old = false charges the whole new copy: same budget now fits nothing.
        PromoClassSpec keep = iq3;
        keep.frees_old = false;
        VJ_CHECK(VramBudgetPlanner::plan_crossover({keep}, tight, t, 1).promoted_classes == 0);
        VJ_CHECK(VramBudgetPlanner::plan_crossover({keep}, static_cast<uint64_t>(keep.new_bytes), t, 1).promoted_classes == 1);

        // Pool summed over the ranks: room for IQ3 alone first, then for both once the second rank adds up.
        const uint64_t per_rank = static_cast<uint64_t>(delta_of(iq3)) + 1;
        VJ_CHECK(VramBudgetPlanner::plan_crossover({iq3, iq2}, per_rank, t, 1).promoted_classes == 1);
        const uint64_t two_ranks = per_rank + static_cast<uint64_t>(delta_of(iq2));
        VJ_CHECK(VramBudgetPlanner::plan_crossover({iq3, iq2}, two_ranks, t, 1).promoted_classes == 2);
        VJ_CHECK(VramBudgetPlanner::plan_crossover({iq3, iq2}, two_ranks - 2, t, 1).promoted_classes == 1);
    }

    std::printf("test_v13_dot8_promo: OK\n");
    return 0;
}

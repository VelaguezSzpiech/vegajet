// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/gdn_vshard.hpp"
#include "tests/test_common.hpp"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <set>
using namespace vegajet::shard;

int main() {
    std::vector<GdnShard> p;
    std::string error;
    const std::vector<double> bw{871, 745, 747};
    for (const std::string mode : {"", "0"}) {
        VJ_CHECK(plan_gdn_vshard(16, 64, bw, mode, p, &error));
        const auto base = plan_gdn(16, 64, bw);
        for (size_t r = 0; r < bw.size(); ++r) {
            VJ_CHECK(p[r].k_groups.begin == base[r].k_groups.begin);
            VJ_CHECK(p[r].v_heads.end == base[r].v_heads.end);
        }
    }
    VJ_CHECK(plan_gdn_vshard(16, 64, bw, "20,24,20", p, &error));
    VJ_CHECK(p[1].k_groups.begin == 5 && p[1].k_groups.end == 11);
    // Check the actual group-major Ornith map hv=grp*nk+hk, rather than interpreting
    // the cardinality range as contiguous physical heads. Every head is owned once.
    std::set<uint32_t> physical;
    for (const auto& s : p)
        for (uint32_t grp = 0; grp < 4; ++grp)
            for (uint32_t hk = s.k_groups.begin; hk < s.k_groups.end; ++hk)
                VJ_CHECK(physical.insert(grp * 16 + hk).second);
    VJ_CHECK(physical.size() == 64);
    for (const char* mode : {"0,32,32", "20,20,20", "21,23,20", "20,24", "20,24,20,0",
                             "20,24,20,", "20,24,20junk", "-20,24,60", " 20,24,20", "42949672960,4,4"}) {
        VJ_CHECK(!plan_gdn_vshard(16, 64, bw, mode, p, &error));
        VJ_CHECK(p.empty() && error.find("VEGAJET_GDN_VSHARD") == 0);
    }
    VJ_CHECK(!plan_gdn_vshard(2, 8, bw, "auto", p, &error));
    VJ_CHECK(!plan_gdn_vshard(16, 63, bw, "auto", p, &error));
    VJ_CHECK(!plan_gdn_vshard(16, 64, {871, 0, 747}, "auto", p, &error));
    VJ_CHECK(!plan_gdn_vshard(16, 64, {871, std::numeric_limits<double>::quiet_NaN()}, "auto", p, &error));
    // Independent exhaustive oracle for discrete minimax: every legal composition of nk.
    for (const auto& rates : {bw, std::vector<double>{1000, 800, 800},
                             std::vector<double>{1, 2, 9}, std::vector<double>{7},
                             std::vector<double>{10, 9, 8, 7}}) {
        for (uint32_t nk = rates.size(); nk <= 20; ++nk) {
            VJ_CHECK(plan_gdn_vshard(nk, nk * 4, rates, "auto", p, &error));
            double actual = 0;
            uint64_t pos = 0;
            for (size_t r = 0; r < rates.size(); ++r) {
                VJ_CHECK(p[r].k_groups.begin == pos && p[r].k_groups.size() > 0);
                VJ_CHECK(p[r].v_heads.size() == p[r].k_groups.size() * 4);
                pos = p[r].k_groups.end;
                actual = std::max(actual, p[r].k_groups.size() / rates[r]);
            }
            VJ_CHECK(pos == nk);
            double best = std::numeric_limits<double>::infinity();
            std::function<void(size_t, uint32_t, double)> visit = [&](size_t r, uint32_t left, double worst) {
                if (r + 1 == rates.size()) {
                    if (left) best = std::min(best, std::max(worst, left / rates[r]));
                    return;
                }
                for (uint32_t c = 1; c + rates.size() - r - 1 <= left; ++c)
                    visit(r + 1, left - c, std::max(worst, c / rates[r]));
            };
            visit(0, nk, 0);
            VJ_CHECK(std::fabs(actual - best) < 1e-12);
        }
    }
    std::puts("PASS GDN V-shard validation, group-major coverage and exhaustive minimax oracle");
}

// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/gdn_vshard.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace vegajet::shard {
bool plan_gdn_vshard(uint32_t nk, uint32_t nv, const std::vector<double>& bw,
                     const std::string& mode, std::vector<GdnShard>& out, std::string* error) {
    out.clear();
    if (mode.empty() || mode == "0") { out = plan_gdn(nk, nv, bw); return true; }
    auto fail = [&](const char* why) {
        if (error) *error = std::string("VEGAJET_GDN_VSHARD: ") + why;
        return false;
    };
    if (bw.empty() || nk < bw.size() || nv == 0 || nk == 0 || nv % nk)
        return fail("need integral V/K and at least one K group per rank");
    const uint32_t per = nv / nk;
    std::vector<uint32_t> groups(bw.size(), 1u);
    if (mode == "auto") {
        for (double b : bw) if (!std::isfinite(b) || b <= 0)
            return fail("auto requires finite positive per-rank bandwidth");
        // Merge sorted completion-time streams (1/bw, 2/bw, ...), subject to nonempty ranks.
        // Selecting the smallest next completion minimizes max(groups[r]/bw[r]).
        for (uint32_t used = static_cast<uint32_t>(bw.size()); used < nk; ++used) {
            size_t best = 0;
            for (size_t r = 1; r < bw.size(); ++r)
                if ((groups[r] + 1.0) / bw[r] < (groups[best] + 1.0) / bw[best]) best = r;
            ++groups[best];
        }
        // Prefer the existing partition when it achieves the same optimum, avoiding needless rounding changes.
        const auto base = plan_gdn(nk, nv, bw);
        double old_max = 0, new_max = 0;
        bool nonempty = true;
        for (size_t r = 0; r < bw.size(); ++r) {
            nonempty &= base[r].k_groups.size() != 0;
            old_max = std::max(old_max, base[r].k_groups.size() / bw[r]);
            new_max = std::max(new_max, groups[r] / bw[r]);
        }
        if (nonempty && old_max <= new_max) { out = base; return true; }
    } else {
        size_t pos = 0;
        uint64_t total = 0;
        for (size_t r = 0; r < bw.size(); ++r) {
            if (pos == mode.size() || mode[pos] < '0' || mode[pos] > '9')
                return fail("expected comma-separated positive V-head counts");
            uint64_t count = 0;
            while (pos < mode.size() && mode[pos] >= '0' && mode[pos] <= '9') {
                count = count * 10u + static_cast<unsigned>(mode[pos++] - '0');
                if (count > std::numeric_limits<uint32_t>::max()) return fail("V-head count overflow");
            }
            if (count == 0 || count % per) return fail("counts must be positive multiples of V/K");
            groups[r] = static_cast<uint32_t>(count / per);
            total += count;
            if (r + 1 < bw.size()) {
                if (pos == mode.size() || mode[pos++] != ',') return fail("wrong rank count");
            }
        }
        if (pos != mode.size()) return fail("wrong rank count or trailing characters");
        if (total != nv) return fail("V-head counts must sum to model V heads");
    }
    uint64_t k = 0;
    for (uint32_t count : groups) {
        out.push_back({{k, k + count}, {k * per, (k + count) * per}});
        k += count;
    }
    return true;
}
}

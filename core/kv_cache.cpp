// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/kv_cache.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace vegajet::kv {

const char* tier_name(KvTier t) {
    switch (t) {
        case KvTier::F16: return "F16";
        case KvTier::Q8_0: return "Q8_0";
        case KvTier::Q4_0: return "Q4_0";
        case KvTier::TQ3: return "TQ3";
    }
    return "?";
}

double tier_bytes_per_elem(KvTier t) {
    switch (t) {
        case KvTier::F16: return 2.0;
        case KvTier::Q8_0: return 34.0 / 32.0;
        case KvTier::Q4_0: return 18.0 / 32.0;
        case KvTier::TQ3: return 14.0 / 32.0;  // 3-bit codes (12 B) + FP16 scale per 32
    }
    return 2.0;
}

uint64_t kv_bytes(const KvGeometry& g, uint64_t tokens, KvTier tier) {
    const uint64_t pages = (tokens + kPageTokens - 1) / kPageTokens;
    return static_cast<uint64_t>(std::ceil(static_cast<double>(g.elems_per_token()) * pages * kPageTokens *
                                           tier_bytes_per_elem(tier)));
}

std::string KvPlan::summary() const {
    std::string s;
    char buf[96];
    for (const auto& sp : spans) {
        std::snprintf(buf, sizeof(buf), "%s%s x %llu tok (%.2f GiB)", s.empty() ? "" : " + ", tier_name(sp.tier),
                      static_cast<unsigned long long>(sp.tokens), sp.bytes / 1073741824.0);
        s += buf;
    }
    return s;
}

KvPlan plan_kv(const KvGeometry& g, uint64_t ctx, uint64_t budget, uint64_t recent) {
    KvPlan p;
    const uint64_t r = std::min(ctx, recent);
    const uint64_t hist = ctx - r;
    const uint64_t recent_b = kv_bytes(g, r, KvTier::Q8_0);
    for (KvTier t : {KvTier::Q8_0, KvTier::Q4_0, KvTier::TQ3}) {
        const uint64_t hb = hist ? kv_bytes(g, hist, t) : 0;
        if (recent_b + hb <= budget || t == KvTier::TQ3) {
            p.spans.push_back({KvTier::Q8_0, r, (r + kPageTokens - 1) / kPageTokens, recent_b});
            if (hist) p.spans.push_back({t, hist, (hist + kPageTokens - 1) / kPageTokens, hb});
            p.total_bytes = recent_b + hb;
            p.fits = p.total_bytes <= budget;
            return p;
        }
    }
    return p;
}

} // namespace vegajet::kv

// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include <cstdint>

namespace vegajet::warm {
// Explicit save/check modes preserve a raw code-token oracle across load-time experiment restarts.
// Checking never creates a missing reference. Text files allow independent token-ID inspection.
inline bool raw_token_reference(const std::string& mode, const std::vector<int32_t>& tokens,
                                std::vector<int32_t>& reference, std::string& error) {
    auto fail = [&](const char* why) { error = std::string("VEGAJET_RAW_TOKEN_REFERENCE: ") + why; return false; };
    if (mode.compare(0, 5, "save:") == 0 && mode.size() > 5) {
        std::ofstream out(mode.substr(5));
        if (!out) return fail("cannot save reference");
        out << "VJ_RAW_CODE_TOKENS_V1 " << tokens.size() << '\n';
        for (int32_t t : tokens) out << t << '\n';
        out.close();
        if (!out) return fail("short write saving reference");
        reference = tokens;
        return true;
    }
    if (mode.compare(0, 6, "check:") != 0 || mode.size() <= 6) return fail("expected save:PATH or check:PATH");
    std::ifstream in(mode.substr(6));
    std::string magic;
    size_t count = 0;
    if (!(in >> magic >> count) || magic != "VJ_RAW_CODE_TOKENS_V1" || count != tokens.size())
        return fail("missing/invalid reference or token count mismatch");
    std::vector<int32_t> loaded;
    for (size_t i = 0; i < count; ++i) {
        int64_t t = 0;
        if (!(in >> t) || t < 0 || t > std::numeric_limits<int32_t>::max()) return fail("invalid token ID");
        loaded.push_back(static_cast<int32_t>(t));
    }
    std::string extra;
    if (in >> extra) return fail("trailing reference data");
    reference = std::move(loaded);
    return true;
}
}

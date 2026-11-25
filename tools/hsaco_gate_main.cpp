// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Build-time gate: vj_hsaco_gate <file.hsaco>...   (exit 1 on any violation)
#include "core/hsaco_gate.hpp"

#include <cstdio>

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <file.hsaco>...\n", argv[0]); return 2; }
    int rc = 0;
    for (int i = 1; i < argc; ++i) {
        const auto rep = vegajet::gate::check_code_object_file(argv[i]);
        for (const auto& w : rep.waivers) std::printf("WARNING: hsaco gate waiver: %s\n", w.c_str());
        for (const auto& e : rep.failures) std::fprintf(stderr, "FATAL: hsaco gate: %s [%s]\n", e.c_str(), argv[i]);
        std::printf("hsaco gate %s: %s | %u kernels, %u waived, max vgpr_count=%u, max code=%llu B, 0 B spill\n",
                    rep.ok ? "PASS" : "FAIL", argv[i], rep.kernels, rep.waived, rep.max_vgpr,
                    static_cast<unsigned long long>(rep.max_code_bytes));
        if (!rep.ok) rc = 1;
    }
    return rc;
}

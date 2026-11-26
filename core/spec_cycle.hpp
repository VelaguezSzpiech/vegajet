// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
#include <cstdio>
#include <cstdint>
namespace vegajet::spec {
struct CycleTimes {
    uint64_t cycles = 0;
    double draft = 0, verify = 0, catchup = 0, recompute = 0;
    void add(double d, double v, double c, double r) {
        ++cycles; draft += d; verify += v; catchup += c; recompute += r;
    }
    void print(const char* verifier, const char* source) const {
        if (!cycles) return;
        const double n = static_cast<double>(cycles);
        std::printf("[spec_cycle_ms verifier=%s source=%s cycles=%llu] draft=%.3f verify=%.3f catch-up=%.3f recompute-on-reject=%.3f total=%.3f\n",
            verifier, source, static_cast<unsigned long long>(cycles), draft/n, verify/n,
            catchup/n, recompute/n, (draft+verify+catchup+recompute)/n);
    }
};
}

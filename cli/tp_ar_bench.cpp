// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "cli/tp_ar_bench.hpp"
#include "core/kfd_runtime.hpp"
#include "ops/kernels/vj_ar_tp3_alltoall.hiph"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>
namespace vegajet::cli {
int run_tp_ar_bench(uint32_t ranks, const std::string& hsaco_dir, const tp::CollectiveOptions& options) {
    kfd::Runtime runtime;
    std::string err;
    if (!runtime.init(0.125, 4, &err, ranks) || !runtime.load_hsaco_dir(hsaco_dir, &err)) {
        std::fprintf(stderr, "--bench-ar init: %s\n", err.c_str()); return 3;
    }
    std::vector<uint64_t> partials(ranks), outputs(ranks);
    for (uint32_t r = 0; r < ranks; ++r) {
        partials[r] = runtime.vram_arena(r).alloc(4096 * sizeof(float), 256);
        outputs[r] = runtime.vram_arena(r).alloc(4096 * sizeof(float), 256);
        if (!partials[r] || !outputs[r]) { std::fprintf(stderr, "--bench-ar arena exhausted\n"); return 3; }
    }
    uint64_t verify_errs = 0, verified = 0;
    double elapsed_ms = 0;
    for (uint32_t elements : {1u, 33u, 3072u}) {
        if (!runtime.configure_collective(elements, options, &err)) {
            std::fprintf(stderr, "--bench-ar configure: %s\n", err.c_str()); return 3;
        }
        if (elements == 3072) {
            const auto& topology = runtime.collective_topology();
            std::printf("Runtime sysfs topology (selected order is rank order):\n");
            for (uint32_t r = 0; r < ranks; ++r) std::printf("rank %u identity=%s\n", r, topology.identities[r].c_str());
            std::printf("%s", tp::format_collective_plan(runtime.collective_plan(), tp::plan_topology(topology.peers, options)).c_str());
        }
        std::vector<float> input(elements), expected(elements), output(elements);
        // Refresh every source every epoch: stale/parity/rank-order mistakes
        // cannot pass by repeatedly reducing one constant vector.
        for (uint32_t epoch = 1; epoch <= 64; ++epoch) {
            std::fill(expected.begin(), expected.end(), 0.0f);
            for (uint32_t r = 0; r < ranks; ++r) {
                for (uint32_t i = 0; i < elements; ++i) {
                    const int value = static_cast<int>((uint64_t(i) * 13 + r * 7 + epoch * 11) % 509) - 254;
                    input[i] = static_cast<float>(value) / 19.0f;
                    const float wire = ranks == 1 ? input[i] : vj_tp3::from_half(vj_tp3::to_half(input[i]));
                    expected[i] = r == 0 ? wire : expected[i] + wire;
                }
                if (!runtime.upload_to_vram(r, partials[r], input.data(), elements * sizeof(float))) {
                    std::fprintf(stderr, "--bench-ar input upload failed\n"); return 3;
                }
            }
            const auto start = std::chrono::steady_clock::now();
            if (!runtime.collective_allreduce(partials, outputs, elements, &err)) {
                std::fprintf(stderr, "--bench-ar epoch=%u: %s (discard all outputs)\n", epoch, err.c_str()); return 3;
            }
            elapsed_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            for (uint32_t r = 0; r < ranks; ++r) {
                if (!runtime.download_from_vram(r, output.data(), outputs[r], elements * sizeof(float))) {
                    std::fprintf(stderr, "--bench-ar output download failed\n"); return 3;
                }
                for (uint32_t i = 0; i < elements; ++i) {
                    ++verified;
                    if (!std::isfinite(output[i]) || output[i] != expected[i]) ++verify_errs;
                }
            }
        }
    }
    std::printf("bench-ar TP=%u wire=%s epochs=192 verified=%llu verify_errs=%llu observed_collective_ms=%.3f\n",
                ranks, ranks == 1 ? "none" : "FP16", static_cast<unsigned long long>(verified),
                static_cast<unsigned long long>(verify_errs), elapsed_ms);
    return verify_errs ? 1 : 0;
}
}

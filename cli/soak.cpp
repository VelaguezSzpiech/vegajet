// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// `vegajet soak`: HBM2 arena diagnostic. Fills N x 512 MiB arena chunks per GPU with a deterministic pattern,
// then streams them back through a verify kernel that timestamps every workgroup with s_memrealtime().
//   test A: one 512 MiB chunk read repeatedly          (small VA footprint)
//   test B: every chunk of the arena read in sequence  (full VA footprint)
// For both it reports bandwidth from on-GPU ticks and from host wall time, pattern mismatches, and the umc
// correctable-error delta attributed to each chunk (read from sysfs after every pass over that chunk).
#include "cli/inspect.hpp"

#include "core/host_preflight.hpp"
#include "core/kfd_runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace vegajet::cli {

namespace {

constexpr uint64_t MiB = 1ull << 20;
constexpr uint32_t kSoakWgs = 240;   // 4 WGs/CU on 60 CUs

#pragma pack(push, 1)
struct FillArgs   { uint64_t dst; uint64_t n_vec; uint32_t seed; uint32_t pad; };
struct VerifyArgs { uint64_t src; uint64_t n_vec; uint32_t seed; uint32_t pad; uint64_t err_count; uint64_t ticks; };
#pragma pack(pop)

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string bdf_of(uint32_t location_id) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "0000:%02x:%02x.%u", (location_id >> 8) & 0xFFu, (location_id >> 3) & 0x1Fu, location_id & 7u);
    return buf;
}

uint64_t read_umc_ce(const std::string& bdf) {
    std::ifstream f("/sys/bus/pci/devices/" + bdf + "/ras/umc_err_count");
    std::string k; uint64_t v = 0, ce = 0;
    while (f >> k >> v) if (k == "ce:") ce = v;
    return ce;
}

struct PassStat { uint64_t ticks = 0; double wall = 0.0; uint64_t bytes = 0, errs = 0; uint32_t passes = 0; };

} // namespace

int run_soak(const std::vector<std::string>& args) {
    double seconds = 60.0, gib = 28.5;
    uint64_t chunk_mib = 512;
    bool p2p = false;
    std::string hsaco_dir;
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        auto next = [&]() -> std::string { return i + 1 < args.size() ? args[++i] : std::string(); };
        if (a == "--seconds") seconds = std::atof(next().c_str());
        else if (a == "--gib") gib = std::atof(next().c_str());
        else if (a == "--chunk-mib") chunk_mib = std::strtoull(next().c_str(), nullptr, 10);
        else if (a == "--p2p") p2p = true;
        else if (a == "--hsaco-dir") hsaco_dir = next();
        else {
            std::printf("Usage: vegajet soak [--seconds 60] [--gib 28.5] [--chunk-mib 512] [--hsaco-dir DIR] | soak --p2p [--gib 4]\n"
                        "  --p2p: verified GPU-to-GPU copy over the peer BAR mapping, --gib per direction (GB/s by host wall).\n"
                        "  HBM2 arena pattern soak: 512 MiB vs full-arena streaming bandwidth (s_memrealtime vs host wall),\n"
                        "  pattern mismatches and per-chunk umc CE attribution.\n");
            return (a == "--help" || a == "-h") ? 0 : 2;
        }
    }
    if (!kfd::Runtime::compiled_with_gpu()) { std::fprintf(stderr, "soak: this build has no GPU backend\n"); return 3; }
    if (!(seconds >= 2.0) || !(gib >= 1.0) || gib > 30.0 || chunk_mib < 2 || chunk_mib > 512 || (chunk_mib & 1u)) {
        std::fprintf(stderr, "soak: need --seconds >= 2, 1 <= --gib <= 30, even --chunk-mib in [2, 512]\n");
        return 2;
    }

    kfd::Runtime rt;
    std::string err;
    // The arena follows --gib (default 28.5 -> the historical 30.5 GiB), so 16 GB cards can soak what they hold;
    // --p2p keeps the full arena (its payload slices assume the 2 x 32 GB large-BAR layout).
    const double arena_gib = p2p ? 30.5 : vegajet::preflight::soak_arena_gib(gib);
    if (!rt.init(arena_gib, 64, &err)) { std::fprintf(stderr, "soak: %s\n", err.c_str()); return 3; }
    if (!rt.load_hsaco_dir(kfd::resolve_hsaco_dir(hsaco_dir), &err)) { std::fprintf(stderr, "soak: %s\n", err.c_str()); return 3; }

    if (p2p) {
        // Verified peer copy: GPU `src` writes the soak pattern into the payload slice owned by GPU `dst` through the
        // peer BAR mapping (the bytes cross both host PCIe links), then GPU `dst` verifies its own slice locally.
        if (rt.num_gpus() < 2 || !rt.p2p_payload_slice_va(0) || !rt.p2p_payload_slice_va(1)) {
            std::fprintf(stderr, "soak --p2p: needs 2 GPUs with the PEER_BAR_P2P transport\n");
            return 3;
        }
        const uint64_t total = static_cast<uint64_t>(std::min(gib, 16.0) * 1024.0) * MiB;
        int rc = 0;
        for (size_t src = 0; src < 2; ++src) {
            const size_t dst = 1 - src;
            const kfd::LoadedKernel* fill   = rt.get_kernel(src, "kfd_soak_fill_u32x4");
            const kfd::LoadedKernel* verify = rt.get_kernel(dst, "kfd_soak_verify_u32x4");
            const uint64_t slice_va = rt.p2p_payload_slice_va(dst), slice_bytes = rt.p2p_payload_slice_bytes(dst);
            const uint64_t err_va   = rt.vram_arena(dst).alloc(4096, 4096);
            const uint64_t ticks_va = rt.vram_arena(dst).alloc(2u * kSoakWgs * sizeof(uint64_t), 4096);
            if (!fill || !verify || !err_va || !ticks_va || slice_bytes < 2 * MiB) { std::fprintf(stderr, "soak --p2p: setup failed\n"); return 3; }
            const uint32_t zero = 0;
            if (!rt.upload_to_vram(dst, err_va, &zero, sizeof(zero))) return 3;
            const uint64_t n_vec = slice_bytes / 16u, passes = total / slice_bytes;
            if (passes == 0) { std::fprintf(stderr, "soak --p2p: --gib is smaller than one payload slice\n"); return 2; }
            double t_fill = 0.0, t_best = 1e30;
            const double t_all0 = now_s();
            for (uint64_t ps = 0; ps < passes; ++ps) {
                const uint32_t sd = 0x5eed906u + static_cast<uint32_t>(ps) * 2654435761u + static_cast<uint32_t>(src);
                const FillArgs fa{slice_va, n_vec, sd, 0};
                const double t0 = now_s();
                if (!rt.dispatch_async(src, *fill, kSoakWgs * 256u, 1, 1, 256, 1, 1, &fa, sizeof(fa)) || !rt.wait_idle(src)) { std::fprintf(stderr, "soak --p2p: fill failed\n"); return 3; }
                const double dt = now_s() - t0;
                t_fill += dt; t_best = std::min(t_best, dt);
                const VerifyArgs va{slice_va, n_vec, sd, 0, err_va, ticks_va};
                if (!rt.dispatch_async(dst, *verify, kSoakWgs * 256u, 1, 1, 256, 1, 1, &va, sizeof(va)) || !rt.wait_idle(dst)) { std::fprintf(stderr, "soak --p2p: verify failed\n"); return 3; }
            }
            const double t_all = now_s() - t_all0;
            uint32_t errs = 0;
            if (!rt.download_from_vram(dst, &errs, err_va, sizeof(errs))) return 3;
            const double gb = static_cast<double>(passes * slice_bytes) / 1e9;
            std::printf("[P2P gpu%zu %s -> gpu%zu %s] %.2f GiB in %llu x %llu MiB pushes | push %.3f s = %.2f GB/s mean, %.2f GB/s best pass | "
                        "push+verify %.3f s | mismatched words=%u\n",
                        src, bdf_of(rt.devices()[src].location_id).c_str(), dst, bdf_of(rt.devices()[dst].location_id).c_str(),
                        static_cast<double>(passes * slice_bytes) / (1024.0 * MiB), static_cast<unsigned long long>(passes),
                        static_cast<unsigned long long>(slice_bytes / MiB), t_fill, gb / t_fill,
                        static_cast<double>(slice_bytes) / 1e9 / t_best, t_all, errs);
            if (errs) rc = 1;
        }
        return rc;
    }

    const size_t G = rt.num_gpus();
    const uint64_t chunk_bytes = chunk_mib * MiB;
    const uint64_t n_vec = chunk_bytes / 16u;
    const uint32_t n_chunks = static_cast<uint32_t>(gib * 1024.0 / static_cast<double>(chunk_mib));
    const uint32_t seed = 0x5eed906u;

    struct Gpu {
        const kfd::LoadedKernel* fill = nullptr; const kfd::LoadedKernel* verify = nullptr;
        std::vector<uint64_t> chunks; uint64_t err_va = 0, ticks_va = 0;
        std::string bdf; uint64_t ce0 = 0;
        std::vector<uint64_t> chunk_ce, chunk_errs;
        PassStat small, full;
        uint64_t tick_first = 0, tick_last = 0; double wall_first = 0.0, wall_last = 0.0;
    };
    std::vector<Gpu> gp(G);
    for (size_t g = 0; g < G; ++g) {
        gp[g].fill   = rt.get_kernel(g, "kfd_soak_fill_u32x4");
        gp[g].verify = rt.get_kernel(g, "kfd_soak_verify_u32x4");
        if (!gp[g].fill || !gp[g].verify) { std::fprintf(stderr, "soak: soak kernels missing from vj_kfd_base.hsaco\n"); return 3; }
        gp[g].bdf = bdf_of(rt.devices()[g].location_id);
        gp[g].err_va   = rt.vram_arena(g).alloc(4096, 4096);
        gp[g].ticks_va = rt.vram_arena(g).alloc(2u * kSoakWgs * sizeof(uint64_t), 4096);
        for (uint32_t c = 0; c < n_chunks; ++c) {
            const uint64_t va = rt.vram_arena(g).alloc(chunk_bytes, 512 * MiB >= chunk_bytes ? chunk_bytes : 2 * MiB);
            if (!va) break;
            gp[g].chunks.push_back(va);
        }
        if (!gp[g].err_va || !gp[g].ticks_va || gp[g].chunks.empty()) { std::fprintf(stderr, "soak: arena allocation failed on GPU %zu\n", g); return 3; }
        gp[g].chunk_ce.assign(gp[g].chunks.size(), 0);
        gp[g].chunk_errs.assign(gp[g].chunks.size(), 0);
    }
    std::printf("=== VegaJet HBM2 arena soak: %zu GPU(s), %zu x %llu MiB chunks/GPU (%.2f GiB), %.0f s ===\n", G,
                gp[0].chunks.size(), static_cast<unsigned long long>(chunk_mib),
                static_cast<double>(gp[0].chunks.size() * chunk_bytes) / (1024.0 * MiB), seconds);

    // Fill every chunk (both GPUs in parallel per chunk index).
    for (size_t c = 0; c < gp[0].chunks.size(); ++c) {
        for (size_t g = 0; g < G; ++g) {
            if (c >= gp[g].chunks.size()) continue;
            const FillArgs fa{gp[g].chunks[c], n_vec, seed + static_cast<uint32_t>(c), 0};
            if (!rt.dispatch_async(g, *gp[g].fill, kSoakWgs * 256u, 1, 1, 256, 1, 1, &fa, sizeof(fa))) { std::fprintf(stderr, "soak: fill dispatch failed\n"); return 3; }
        }
        for (size_t g = 0; g < G; ++g) if (!rt.wait_idle(g)) { std::fprintf(stderr, "soak: fill timed out on GPU %zu chunk %zu\n", g, c); return 3; }
    }
    for (size_t g = 0; g < G; ++g) gp[g].ce0 = read_umc_ce(gp[g].bdf);
    std::printf("fill complete\n");

    std::vector<uint64_t> ticks(2u * kSoakWgs);
    // One verify pass over chunk c on every GPU concurrently; accumulates into `st` and per-chunk counters.
    auto verify_pass = [&](size_t c, bool full) -> bool {
        const uint32_t zero = 0;
        std::vector<uint64_t> ce_before(G);
        for (size_t g = 0; g < G; ++g) {
            ce_before[g] = read_umc_ce(gp[g].bdf);
            if (!rt.upload_to_vram(g, gp[g].err_va, &zero, sizeof(zero))) return false;
        }
        const double t0 = now_s();
        for (size_t g = 0; g < G; ++g) {
            const size_t cc = std::min(c, gp[g].chunks.size() - 1);
            const VerifyArgs va{gp[g].chunks[cc], n_vec, seed + static_cast<uint32_t>(cc), 0, gp[g].err_va, gp[g].ticks_va};
            if (!rt.dispatch_async(g, *gp[g].verify, kSoakWgs * 256u, 1, 1, 256, 1, 1, &va, sizeof(va))) return false;
        }
        for (size_t g = 0; g < G; ++g) if (!rt.wait_idle(g)) return false;
        const double wall = now_s() - t0;
        for (size_t g = 0; g < G; ++g) {
            const size_t cc = std::min(c, gp[g].chunks.size() - 1);
            uint32_t errs = 0;
            if (!rt.download_from_vram(g, &errs, gp[g].err_va, sizeof(errs))) return false;
            if (!rt.download_from_vram(g, ticks.data(), gp[g].ticks_va, ticks.size() * sizeof(uint64_t))) return false;
            uint64_t t_min = ~0ull, t_max = 0;
            for (uint32_t w = 0; w < kSoakWgs; ++w) { t_min = std::min(t_min, ticks[2u * w]); t_max = std::max(t_max, ticks[2u * w + 1u]); }
            if (gp[g].tick_first == 0) { gp[g].tick_first = t_min; gp[g].wall_first = t0; }
            gp[g].tick_last = t_max; gp[g].wall_last = t0 + wall;
            PassStat& st = full ? gp[g].full : gp[g].small;
            st.ticks += (t_max > t_min) ? (t_max - t_min) : 0;
            st.wall  += wall;
            st.bytes += chunk_bytes;
            st.errs  += errs;
            ++st.passes;
            gp[g].chunk_errs[cc] += errs;
            gp[g].chunk_ce[cc]   += read_umc_ce(gp[g].bdf) - ce_before[g];
        }
        return true;
    };

    const double t_a_end = now_s() + seconds * 0.5;
    while (now_s() < t_a_end)
        if (!verify_pass(0, false)) { std::fprintf(stderr, "soak: verify pass failed (test A)\n"); return 3; }
    const double t_b_end = now_s() + seconds * 0.5;
    do {
        for (size_t c = 0; c < gp[0].chunks.size(); ++c)
            if (!verify_pass(c, true)) { std::fprintf(stderr, "soak: verify pass failed (test B)\n"); return 3; }
    } while (now_s() < t_b_end);

    // The s_memrealtime tick period is calibrated per GPU against host wall time over the whole run
    // (first workgroup start -> last workgroup end), so dispatch/wait overhead of single passes cancels out.
    int rc = 0;
    for (size_t g = 0; g < G; ++g) {
        const auto& s = gp[g];
        const double kTickS = (s.tick_last > s.tick_first) ? (s.wall_last - s.wall_first) / static_cast<double>(s.tick_last - s.tick_first) : 0.0;
        std::printf("[SOAK gpu%zu %s] s_memrealtime calibrated: %.3f ns/tick (%.2f MHz) over %.1f s\n", g, s.bdf.c_str(),
                    kTickS * 1e9, kTickS > 0.0 ? 1e-6 / kTickS : 0.0, s.wall_last - s.wall_first);
        auto line = [&](const char* name, const PassStat& st) {
            const double gb = static_cast<double>(st.bytes) / 1e9;
            std::printf("[SOAK gpu%zu %s] %-22s passes=%u  %.1f GB/s by s_memrealtime (%.3f ms/pass)  %.1f GB/s by host wall (%.3f ms/pass)  mismatches=%llu\n",
                        g, s.bdf.c_str(), name, st.passes, (st.ticks && kTickS > 0.0) ? gb / (static_cast<double>(st.ticks) * kTickS) : 0.0,
                        st.passes ? static_cast<double>(st.ticks) * kTickS * 1e3 / st.passes : 0.0,
                        st.wall > 0.0 ? gb / st.wall : 0.0, st.passes ? st.wall * 1e3 / st.passes : 0.0,
                        static_cast<unsigned long long>(st.errs));
        };
        line("A: single chunk", s.small);
        line("B: full arena", s.full);
        const uint64_t ce_total = read_umc_ce(s.bdf) - s.ce0;
        std::printf("[SOAK gpu%zu %s] umc CE delta=%llu", g, s.bdf.c_str(), static_cast<unsigned long long>(ce_total));
        for (size_t c = 0; c < s.chunks.size(); ++c)
            if (s.chunk_ce[c] || s.chunk_errs[c])
                std::printf(" | chunk %zu (arena +%.1f GiB): CE=%llu mismatches=%llu", c,
                            static_cast<double>(s.chunks[c] - s.chunks[0]) / (1024.0 * MiB),
                            static_cast<unsigned long long>(s.chunk_ce[c]), static_cast<unsigned long long>(s.chunk_errs[c]));
        std::printf("\n");
        if (s.small.errs || s.full.errs) rc = 1;
    }
    return rc;
}

} // namespace vegajet::cli

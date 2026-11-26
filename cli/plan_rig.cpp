// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "cli/plan_rig.hpp"

#include "core/gguf_reader.hpp"
#include "core/rig_planner.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace vegajet::cli {

namespace {

bool parse_u64(const std::string& s, uint64_t& out) {
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos || s.size() > 12) return false;
    out = std::strtoull(s.c_str(), nullptr, 10);
    return true;
}

void print_rig_usage() {
    std::printf("Usage: vegajet plan <model-file> --gpus N --vram 16|32 [--ctx T] [--kv-bits 16|8] [--cu 60|64] [--matrix]\n"
                "  Rig estimator (GGUF header only, any build): per-GPU VRAM breakdown for TP=N, repacking decisions\n"
                "  (crossover law bw_new/bw_old > bpw_new/bpw_old), KV/context headroom, host-RAM MoE offload need.\n"
                "  --gpus N      GPUs, 1..256 (default 1)\n"
                "  --vram G      GiB per GPU, 16 or 32 (default 32)\n"
                "  --ctx T       context to size the KV cache for (default 8192)\n"
                "  --kv-bits B   KV cache precision, 16 or 8 (default 16)\n"
                "  --cu C        compute units per GPU, 60 (MI50) or 64 (MI60) (default 60)\n"
                "  --matrix      one line per standard rig: 1x16, 2x16, 2x32, 3x32, 4x32, 8x32 GiB\n");
}

} // namespace

bool wants_rig_plan(const std::vector<std::string>& args) {
    for (const auto& a : args)
        if (a == "--gpus" || a == "--vram" || a == "--matrix") return true;
    return false;
}

int run_rig_plan(const std::vector<std::string>& args) {
    rig::RigSpec spec;
    std::string model;
    bool matrix = false;
    for (size_t i = 0; i < args.size(); ++i) {
        const std::string& a = args[i];
        uint64_t v = 0;
        auto value = [&](const char* flag) -> bool {
            if (i + 1 >= args.size() || !parse_u64(args[i + 1], v)) {
                std::fprintf(stderr, "plan: %s needs a number\n", flag);
                return false;
            }
            ++i;
            return true;
        };
        if (a == "--help" || a == "-h") { print_rig_usage(); return 0; }
        if (a == "--matrix") { matrix = true; continue; }
        if (a == "--gpus") {
            if (!value("--gpus")) return 2;
            if (v < 1 || v > 256) { std::fprintf(stderr, "plan: --gpus must be 1..256\n"); return 2; }
            spec.gpus = static_cast<uint32_t>(v);
        } else if (a == "--vram") {
            if (!value("--vram")) return 2;
            if (v != 16 && v != 32) { std::fprintf(stderr, "plan: --vram must be 16 or 32\n"); return 2; }
            spec.vram_gib = static_cast<uint32_t>(v);
        } else if (a == "--ctx") {
            if (!value("--ctx")) return 2;
            if (v < 1 || v > (1u << 24)) { std::fprintf(stderr, "plan: --ctx must be 1..16777216\n"); return 2; }
            spec.ctx = v;
        } else if (a == "--kv-bits") {
            if (!value("--kv-bits")) return 2;
            if (v != 16 && v != 8) { std::fprintf(stderr, "plan: --kv-bits must be 16 or 8\n"); return 2; }
            spec.kv_bits = static_cast<uint32_t>(v);
        } else if (a == "--cu") {
            if (!value("--cu")) return 2;
            if (v != 60 && v != 64) { std::fprintf(stderr, "plan: --cu must be 60 or 64\n"); return 2; }
            spec.n_cu = static_cast<uint32_t>(v);
        } else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "plan: unknown rig option %s (rig mode takes --gpus, --vram, --ctx, --kv-bits, --cu, --matrix)\n", a.c_str());
            return 2;
        } else if (model.empty()) {
            model = a;
        } else {
            std::fprintf(stderr, "plan: more than one model file given\n");
            return 2;
        }
    }
    if (model.empty()) { print_rig_usage(); return 2; }
    GgufFile f;
    if (!f.open(model)) { std::fprintf(stderr, "plan: cannot read %s: %s\n", model.c_str(), f.error().c_str()); return 1; }
    rig::ModelSummary m;
    std::string err;
    if (!rig::summarize_gguf(f, m, &err)) { std::fprintf(stderr, "plan: %s\n", err.c_str()); return 1; }
    if (matrix) {
        const std::string out = rig::format_rig_matrix(m, rig::standard_rigs(spec));
        std::fwrite(out.data(), 1, out.size(), stdout);
        return 0;
    }
    const rig::RigPlan p = rig::plan_rig(m, spec);
    const std::string out = rig::format_rig_plan(m, spec, p);
    std::fwrite(out.data(), 1, out.size(), stdout);
    return p.fits ? 0 : 1;
}

} // namespace vegajet::cli

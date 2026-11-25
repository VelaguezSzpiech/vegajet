// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// `vegajet inspect <file>`: header-only GGUF inspection (metadata, tensor patterns, per-type breakdown,
// projected native VRAM footprint). Reads metadata with pread(); never touches tensor payloads.
#include "cli/inspect.hpp"

#include "core/gguf_reader.hpp"
#include "core/quant_repacker.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

namespace vegajet::cli {

namespace {

std::string pattern_of(const std::string& name) {
    // blk.<n>.foo -> blk.N.foo
    std::string out;
    for (size_t i = 0; i < name.size();) {
        if (std::isdigit(static_cast<unsigned char>(name[i])) && i > 0 && name[i - 1] == '.') {
            size_t j = i;
            while (j < name.size() && std::isdigit(static_cast<unsigned char>(name[j]))) ++j;
            if (j == name.size() || name[j] == '.') { out += 'N'; i = j; continue; }
        }
        out += name[i++];
    }
    return out;
}

double gib(uint64_t b) { return static_cast<double>(b) / (1024.0 * 1024.0 * 1024.0); }

} // namespace

int run_inspect(const std::vector<std::string>& args) {
    std::string path;
    bool all_keys = false;
    for (const auto& a : args) {
        if (a == "--all-keys") all_keys = true;
        else if (a == "--help" || a == "-h") {
            std::printf("Usage: vegajet inspect <model-file> [--all-keys]\n"
                        "  Prints GGUF metadata, tensor name patterns, per-type breakdown and projected\n"
                        "  VJ_Q8_DP4A / VJ_Q4_DOT8 footprints for TP=1..4. Header-only, read-only.\n");
            return 0;
        } else path = a;
    }
    if (path.empty()) { std::fprintf(stderr, "inspect: missing model file argument\n"); return 2; }

    GgufFile f;
    if (!f.open(path)) { std::fprintf(stderr, "inspect: %s\n", f.error().c_str()); return 1; }
    const std::string arch = f.architecture();
    std::printf("== GGUF v%u | %.2f GiB | %zu tensors | %zu KV | alignment %llu | data @ %llu\n", f.version(),
                gib(f.file_size()), f.tensors().size(), f.keys().size(),
                static_cast<unsigned long long>(f.alignment()), static_cast<unsigned long long>(f.data_offset()));
    std::printf("general.architecture = %s\n\n== Metadata\n", arch.c_str());
    for (const auto& k : f.keys()) {
        const bool tok = k.rfind("tokenizer.", 0) == 0;
        if (tok && !all_keys) {
            const GgufValue* v = f.find(k);
            if (v->is_array()) { std::printf("  %-52s [array len %llu]\n", k.c_str(), static_cast<unsigned long long>(v->arr_len)); continue; }
        }
        std::printf("  %-52s %s\n", k.c_str(), f.find(k)->to_string(all_keys ? 64 : 16).c_str());
    }

    // Per-type breakdown and patterns.
    struct Agg { uint64_t count = 0, elems = 0, bytes = 0; };
    std::map<std::string, Agg> by_type;
    std::map<std::string, std::map<std::string, uint64_t>> patterns;  // pattern -> "type shape" -> count
    uint64_t total_bytes = 0, total_elems = 0;
    uint64_t q8_bytes = 0, q4_bytes = 0, unsupported = 0;
    const auto& reg = quant::DecoderRegistry::instance();
    for (const auto& t : f.tensors()) {
        auto& a = by_type[ggml_type_name(t.type)];
        a.count++; a.elems += t.n_elems(); a.bytes += t.n_bytes();
        total_bytes += t.n_bytes(); total_elems += t.n_elems();
        char shape[96];
        std::snprintf(shape, sizeof(shape), "%s [%llu,%llu,%llu,%llu]", ggml_type_name(t.type),
                      (unsigned long long)t.ne[0], (unsigned long long)t.ne[1], (unsigned long long)t.ne[2],
                      (unsigned long long)t.ne[3]);
        patterns[pattern_of(t.name)][shape]++;
        if (!reg.supports(static_cast<quant::DiskQuantType>(t.type))) ++unsupported;
        // Matrices (ne[0] % 32 == 0, >= 2 dims) repack; vectors stay F32 on device.
        const bool matrix = t.n_dims >= 2 && t.ne[0] % 32 == 0;
        const uint64_t e = t.n_elems();
        if (matrix) {
            q8_bytes += e + e / 32 * 2;
            q4_bytes += e / 2 + e / 32 * 2;
        } else {
            q8_bytes += e * 4; q4_bytes += e * 4;
        }
    }
    std::printf("\n== Tensor types\n  %-10s %8s %16s %12s %8s\n", "type", "count", "elements", "GiB", "bpw");
    for (const auto& [name, a] : by_type)
        std::printf("  %-10s %8llu %16llu %12.3f %8.3f\n", name.c_str(), (unsigned long long)a.count,
                    (unsigned long long)a.elems, gib(a.bytes), a.elems ? 8.0 * a.bytes / a.elems : 0.0);
    std::printf("  %-10s %8zu %16llu %12.3f %8.3f\n", "TOTAL", f.tensors().size(), (unsigned long long)total_elems,
                gib(total_bytes), total_elems ? 8.0 * total_bytes / total_elems : 0.0);
    if (unsupported) std::printf("  WARNING: %llu tensors use types without a CPU decoder\n", (unsigned long long)unsupported);

    std::printf("\n== Tensor name patterns (count x type [ne0,ne1,ne2,ne3])\n");
    for (const auto& [pat, shapes] : patterns)
        for (const auto& [shape, n] : shapes)
            std::printf("  %4llu x %-48s %s\n", (unsigned long long)n, pat.c_str(), shape.c_str());

    // Decode read volume per token in disk formats: dense tensors once, routed experts at used/total,
    // one embedding row, MTP (nextn) blocks excluded.
    uint64_t n_exp = 0, n_used = 0, n_block = 0, n_nextn = 0;
    f.get_arch_u64("expert_count", n_exp);
    f.get_arch_u64("expert_used_count", n_used);
    f.get_arch_u64("block_count", n_block);
    f.get_arch_u64("nextn_predict_layers", n_nextn);
    double per_token = 0;
    for (const auto& t : f.tensors()) {
        if (t.name.rfind("blk.", 0) == 0 && n_nextn > 0 &&
            std::strtoull(t.name.c_str() + 4, nullptr, 10) >= n_block - n_nextn) continue;
        if (t.name == "token_embd.weight") per_token += double(t.n_bytes()) / std::max<uint64_t>(1, t.ne[1]);
        else if (t.name.find("_exps.") != std::string::npos && n_exp > 0) per_token += double(t.n_bytes()) * n_used / n_exp;
        else per_token += double(t.n_bytes());
    }
    std::printf("\ndecode bytes/token (disk formats): %.4f GiB\n", per_token / (1024.0 * 1024.0 * 1024.0));

    std::printf("\n== Projected weight footprint (weights only, even split; KV/state/scratch excluded)\n");
    std::printf("  %-28s %10s %10s %10s %10s %10s\n", "layout", "total", "TP=1", "TP=2", "TP=3", "TP=4");
    auto row = [&](const char* label, uint64_t b) {
        std::printf("  %-28s %9.2fG %9.2fG %9.2fG %9.2fG %9.2fG\n", label, gib(b), gib(b), gib(b) / 2, gib(b) / 3, gib(b) / 4);
    };
    row("disk (VJ_IQ_PACKED as-is)", total_bytes);
    row("all VJ_Q4_DOT8", q4_bytes);
    row("all VJ_Q8_DP4A", q8_bytes);
    return 0;
}

} // namespace vegajet::cli

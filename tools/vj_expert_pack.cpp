// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// vj_expert_pack: writes / inspects / verifies a contiguous routed-expert pack (core/expert_pack).
//
//   write:  vj_expert_pack --out pack.vje --layers 48 --experts 256 --gate 4096 --up 4096 --down 4096
//                          [--src raw.f32] [--seed 1] [--verify]
//   info:   vj_expert_pack --info pack.vje
//   selftest: vj_expert_pack --self-test
//
// The raw source layout (and the synthetic weights) is layer-major: for every layer, every expert, gate then up
// then down, all FP32. --verify reads every record back through the same reader the engine uses and compares the
// payload bytes.
#include "core/expert_pack.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace vegajet::expertpack;

namespace {

std::vector<float> synthetic(const PackDesc& d, uint64_t seed) {
    const size_t n = size_t(d.gate_elems + d.up_elems + d.down_elems) * d.n_layers * d.n_experts;
    std::vector<float> v(n);
    uint64_t s = seed ? seed : 1;
    for (size_t i = 0; i < n; ++i) {
        s ^= s >> 33;
        s *= 0xFF51AFD7ED558CCDull;
        s ^= s >> 33;
        v[i] = static_cast<float>(double(s >> 11) / double(1ull << 53) * 2.0 - 1.0) * 0.05f;
    }
    return v;
}

bool verify_pack(const std::string& path, const PackDesc& d, const std::vector<float>& src) {
    PackReader r;
    std::string err;
    if (!r.open(path, &err)) {
        std::fprintf(stderr, "verify: open: %s\n", err.c_str());
        return false;
    }
    const size_t stride = size_t(d.gate_elems + d.up_elems + d.down_elems);
    std::vector<float> g(d.gate_elems), u(d.up_elems), dn(d.down_elems);
    size_t bad = 0;
    for (uint32_t l = 0; l < d.n_layers; ++l)
        for (uint32_t e = 0; e < d.n_experts; ++e) {
            if (!r.read_expert(l, e, g.data(), u.data(), dn.data(), &err)) {
                std::fprintf(stderr, "verify: read (%u,%u): %s\n", l, e, err.c_str());
                return false;
            }
            const float* want = &src[(size_t(l) * d.n_experts + e) * stride];
            if (std::memcmp(g.data(), want, size_t(d.gate_elems) * 4) != 0 ||
                std::memcmp(u.data(), want + d.gate_elems, size_t(d.up_elems) * 4) != 0 ||
                std::memcmp(dn.data(), want + d.gate_elems + d.up_elems, size_t(d.down_elems) * 4) != 0)
                ++bad;
        }
    std::printf("verify: %u layers x %u experts, %zu mismatched records, io=%s\n", d.n_layers, d.n_experts, bad,
                r.io_mode());
    return bad == 0;
}

int self_test() {
    const PackDesc d{2, 3, 64, 64, 64};
    const std::vector<float> src = synthetic(d, 7);
    const std::string path = "./vj-expert-pack-selftest.vje";
    std::string err;
    if (!write_pack(path, d, src.data(), &err)) {
        std::fprintf(stderr, "self test: write: %s\n", err.c_str());
        return 1;
    }
    PackReader r;
    const bool opened = r.open(path, &err);
    bool ok = opened && r.desc().n_layers == d.n_layers && r.desc().n_experts == d.n_experts &&
              r.expert_offset(1, 2) % kAlign == 0 && r.expert_offset(1, 2) > r.expert_offset(0, 0);
    if (!ok) std::fprintf(stderr, "self test: header/index: %s\n", err.c_str());
    if (ok) ok = verify_pack(path, d, src);
    // A truncated copy must be rejected.
    if (ok) {
        std::FILE* in = std::fopen(path.c_str(), "rb");
        std::FILE* out = std::fopen("./vj-expert-pack-selftest-trunc.vje", "wb");
        if (in && out) {
            char buf[8192];
            const size_t n = std::fread(buf, 1, sizeof buf, in);
            std::fwrite(buf, 1, n, out);
        }
        if (in) std::fclose(in);
        if (out) std::fclose(out);
        PackReader t;
        std::string terr;
        const bool rejected = !t.open("./vj-expert-pack-selftest-trunc.vje", &terr);
        if (!rejected) std::fprintf(stderr, "self test: truncated pack accepted\n");
        ok = ok && rejected;
        std::remove("./vj-expert-pack-selftest-trunc.vje");
    }
    std::remove(path.c_str());
    std::printf("vj_expert_pack --self-test: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    std::string out, info, src_path;
    PackDesc d;
    uint32_t seed = 1;
    bool verify = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--self-test") return self_test();
        else if (a == "--out") out = next("--out");
        else if (a == "--info") info = next("--info");
        else if (a == "--src") src_path = next("--src");
        else if (a == "--layers") d.n_layers = static_cast<uint32_t>(std::atoi(next("--layers")));
        else if (a == "--experts") d.n_experts = static_cast<uint32_t>(std::atoi(next("--experts")));
        else if (a == "--gate") d.gate_elems = static_cast<uint32_t>(std::atoi(next("--gate")));
        else if (a == "--up") d.up_elems = static_cast<uint32_t>(std::atoi(next("--up")));
        else if (a == "--down") d.down_elems = static_cast<uint32_t>(std::atoi(next("--down")));
        else if (a == "--seed") seed = static_cast<uint32_t>(std::atoi(next("--seed")));
        else if (a == "--verify") verify = true;
        else {
            std::fprintf(stderr, "unknown argument %s (try --self-test)\n", a.c_str());
            return 2;
        }
    }
    if (!info.empty()) {
        PackReader r;
        std::string err;
        if (!r.open(info, &err)) {
            std::fprintf(stderr, "info: %s\n", err.c_str());
            return 2;
        }
        const auto& pd = r.desc();
        std::printf("layers=%u experts=%u gate=%u up=%u down=%u record=%llu bytes index=%llu bytes io=%s\n",
                    pd.n_layers, pd.n_experts, pd.gate_elems, pd.up_elems, pd.down_elems,
                    static_cast<unsigned long long>(record_bytes(pd)),
                    static_cast<unsigned long long>(index_bytes(pd)), r.io_mode());
        return 0;
    }
    if (out.empty() || !d.n_layers || !d.n_experts || !d.gate_elems || !d.up_elems || !d.down_elems) {
        std::fprintf(stderr,
                     "usage: vj_expert_pack --out P.vje --layers L --experts E --gate G --up U --down D\n"
                     "       [--src raw.f32] [--seed S] [--verify] | --info P.vje | --self-test\n");
        return 2;
    }
    std::vector<float> src;
    if (!src_path.empty()) {
        std::FILE* f = std::fopen(src_path.c_str(), "rb");
        if (!f) {
            std::fprintf(stderr, "cannot open %s\n", src_path.c_str());
            return 2;
        }
        std::fseek(f, 0, SEEK_END);
        const long n = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        const size_t want = size_t(d.n_layers) * d.n_experts * (size_t(d.gate_elems) + d.up_elems + d.down_elems) * 4;
        if (n < 0 || size_t(n) != want) {
            std::fprintf(stderr, "%s: expected %zu bytes, file has %ld\n", src_path.c_str(), want, n);
            std::fclose(f);
            return 2;
        }
        src.resize(want / 4);
        const bool ok = std::fread(src.data(), 4, src.size(), f) == src.size();
        std::fclose(f);
        if (!ok) {
            std::fprintf(stderr, "%s: short read\n", src_path.c_str());
            return 2;
        }
    } else {
        src = synthetic(d, seed);
    }
    std::string err;
    if (!write_pack(out, d, src.data(), &err)) {
        std::fprintf(stderr, "write: %s\n", err.c_str());
        return 1;
    }
    std::printf("wrote %s: %u layers x %u experts, record %llu bytes, index %llu bytes\n", out.c_str(), d.n_layers,
                d.n_experts, static_cast<unsigned long long>(record_bytes(d)),
                static_cast<unsigned long long>(index_bytes(d)));
    if (verify) return verify_pack(out, d, src) ? 0 : 1;
    return 0;
}

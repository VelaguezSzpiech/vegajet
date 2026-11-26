// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// vj_dflash_hidden_diff: per-layer hidden-state diff for one DFlash drafter step.
//
// Reads two raw FP32 dumps of per-token residual states ([tokens][n_states][d], row-major, little-endian, no
// header — the same layout a `--dump-hidden` VJHS file carries after its header) and reports, for every tap layer
// in --layers (default [2, 8, 15, 21, 27, 33, 40, 46]), the max abs and max rel difference, the first tap whose
// relative error exceeds --tol, and — when --fc and --norm are given — the diff of the drafter context
// hidden_norm(fc(concat taps)) built from each side against a dumped --ctx vector. --mask compares a dumped
// [block][pos+block] uint8 mask against the reference mask for --swa (sliding window, strict q - k < W) or a
// full-attention layer (bidirectional block).
// Exit code: 0 clean, 1 a difference above --tol (or mask mismatch), 2 usage/input error.
#include "core/dflash_step_ref.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using vegajet::dflash_step::Config;

bool read_raw(const std::string& path, size_t expect_bytes, std::vector<uint8_t>* out) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (n < 0 || size_t(n) != expect_bytes) {
        std::fprintf(stderr, "%s: expected %zu bytes, file has %ld\n", path.c_str(), expect_bytes, n);
        std::fclose(f);
        return false;
    }
    out->resize(expect_bytes);
    const bool ok = std::fread(out->data(), 1, expect_bytes, f) == expect_bytes;
    std::fclose(f);
    if (!ok) std::fprintf(stderr, "%s: short read\n", path.c_str());
    return ok;
}

bool read_f32(const std::string& path, size_t n, std::vector<double>* out) {
    std::vector<uint8_t> raw;
    if (!read_raw(path, n * 4, &raw)) return false;
    out->resize(n);
    for (size_t i = 0; i < n; ++i) {
        float v;
        std::memcpy(&v, raw.data() + 4 * i, 4);
        (*out)[i] = v;
    }
    return true;
}

std::vector<int> parse_layers(const char* s) {
    std::vector<int> v;
    std::string cur;
    for (const char* p = s;; ++p) {
        if (*p == ',' || *p == '\0') {
            if (!cur.empty()) v.push_back(std::atoi(cur.c_str()));
            cur.clear();
            if (*p == '\0') break;
        } else {
            cur.push_back(*p);
        }
    }
    return v;
}

// ---- self test: identical dumps diff clean, a perturbed tap is named.
int self_test() {
    const int tokens = 2, n_states = 8, d = 4, block = 3, pos = 5, hd = 2;
    Config c;
    c.n_states = n_states; c.d = d; c.hd = hd; c.block = block; c.pos = pos; c.window = 4;
    c.layers = {1, 3};
    const size_t per = size_t(n_states) * d, total = size_t(tokens) * per;
    std::vector<double> st(total);
    for (size_t i = 0; i < total; ++i) st[i] = 0.05 * double((i * 37) % 19) - 0.4;
    const std::string a = "./vj-dflash-diff-a.bin", b = "./vj-dflash-diff-b.bin";
    auto write_f32 = [](const std::string& path, const std::vector<double>& v) {
        std::FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) return false;
        for (double x : v) { float y = static_cast<float>(x); std::fwrite(&y, 4, 1, f); }
        std::fclose(f);
        return true;
    };
    if (!write_f32(a, st) || !write_f32(b, st)) { std::fprintf(stderr, "self test: write failed\n"); return 2; }
    std::vector<double> ref, got;
    if (!read_f32(a, total, &ref) || !read_f32(b, total, &got)) return 2;
    std::vector<double> fc(size_t(d) * c.layers.size() * d, 0.0), nw(d, 1.0);
    for (size_t i = 0; i < fc.size(); ++i) fc[i] = 0.01 * double(i % 9);
    int failures = 0;
    for (int t = 0; t < tokens; ++t) {
        const auto rep = vegajet::dflash_step::diff(c, std::vector<double>(ref.begin() + t * per, ref.begin() + (t + 1) * per),
                                                    std::vector<double>(got.begin() + t * per, got.begin() + (t + 1) * per),
                                                    fc, &nw, nullptr, nullptr, false, 1e-6);
        if (rep.first_bad_layer != -1) { std::fprintf(stderr, "self test: identical dumps differ at layer %d\n", rep.first_bad_layer); ++failures; }
    }
    got[size_t(c.layers[1]) * d + 1] += 0.01;
    const auto bad = vegajet::dflash_step::diff(c, ref, got, fc, &nw, nullptr, nullptr, false, 1e-4);
    if (bad.first_bad_layer != c.layers[1]) { std::fprintf(stderr, "self test: perturbed layer not found\n"); ++failures; }
    std::remove(a.c_str());
    std::remove(b.c_str());
    std::printf("vj_dflash_hidden_diff --self-test: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string ref_path, got_path, fc_path, norm_path, ctx_path, mask_path;
    int tokens = 0, d = 0, n_states = 48, block = 8, pos = 0, hd = 1, window = vegajet::dflash_step::kDefaultWindow;
    double tol = 1e-3;
    bool swa = false;
    std::vector<int> layers(vegajet::dflash_step::kDefaultTaps,
                            vegajet::dflash_step::kDefaultTaps + sizeof vegajet::dflash_step::kDefaultTaps / sizeof(int));

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", what); std::exit(2); }
            return argv[++i];
        };
        if (a == "--self-test") return self_test();
        else if (a == "--ref") ref_path = next("--ref");
        else if (a == "--got") got_path = next("--got");
        else if (a == "--tokens") tokens = std::atoi(next("--tokens"));
        else if (a == "--d") d = std::atoi(next("--d"));
        else if (a == "--n-states") n_states = std::atoi(next("--n-states"));
        else if (a == "--block") block = std::atoi(next("--block"));
        else if (a == "--pos") pos = std::atoi(next("--pos"));
        else if (a == "--window") window = std::atoi(next("--window"));
        else if (a == "--swa") swa = true;
        else if (a == "--layers") layers = parse_layers(next("--layers"));
        else if (a == "--fc") fc_path = next("--fc");
        else if (a == "--norm") norm_path = next("--norm");
        else if (a == "--ctx") ctx_path = next("--ctx");
        else if (a == "--mask") mask_path = next("--mask");
        else if (a == "--tol") tol = std::atof(next("--tol"));
        else { std::fprintf(stderr, "unknown argument %s (try --self-test)\n", a.c_str()); return 2; }
    }
    if (ref_path.empty() || got_path.empty() || tokens <= 0 || d <= 0 || n_states <= 0) {
        std::fprintf(stderr,
                     "usage: vj_dflash_hidden_diff --ref R.f32 --got G.f32 --tokens N --d D [--n-states S]\n"
                     "       [--layers 2,8,15,21,27,33,40,46] [--fc F --norm N --ctx C] [--mask M --pos P --block B\n"
                     "       [--window W] [--swa]] [--tol T] | --self-test\n");
        return 2;
    }
    Config c;
    c.layers = layers; c.n_states = n_states; c.d = d; c.hd = hd; c.block = block; c.pos = pos; c.window = window;
    std::string err;
    if (!vegajet::dflash_step::validate(c, &err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 2; }
    const size_t per = size_t(n_states) * d;
    std::vector<double> ref, got;
    if (!read_f32(ref_path, per * size_t(tokens), &ref) || !read_f32(got_path, per * size_t(tokens), &got)) return 2;
    std::vector<double> fc, nw, ctx;
    if (!fc_path.empty() && !read_f32(fc_path, size_t(d) * layers.size() * d, &fc)) return 2;
    if (!norm_path.empty() && !read_f32(norm_path, size_t(d), &nw)) return 2;
    if (!ctx_path.empty() && !read_f32(ctx_path, size_t(d), &ctx)) return 2;
    std::vector<uint8_t> mask;
    if (!mask_path.empty() && !read_raw(mask_path, size_t(block) * size_t(pos + block), &mask)) return 2;

    int failures = 0;
    for (int t = 0; t < tokens; ++t) {
        const auto rep = vegajet::dflash_step::diff(
            c, std::vector<double>(ref.begin() + t * per, ref.begin() + (t + 1) * per),
            std::vector<double>(got.begin() + t * per, got.begin() + (t + 1) * per),
            fc.empty() ? std::vector<double>() : fc, nw.empty() ? nullptr : &nw, ctx_path.empty() ? nullptr : &ctx,
            mask_path.empty() ? nullptr : &mask, swa, tol);
        std::printf("token %d: %s\n", t, rep.summary.c_str());
        for (size_t k = 0; k < rep.layers.size(); ++k)
            std::printf("  layer %2d: max_abs=%.3e max_rel=%.3e%s\n", rep.layers[k], rep.layer_max_abs[k],
                        rep.layer_max_rel[k], rep.layers[k] == rep.first_bad_layer ? "  <- first over tol" : "");
        if (rep.ctx_compared) std::printf("  context: max_abs=%.3e max_rel=%.3e\n", rep.ctx_max_abs, rep.ctx_max_rel);
        if (!rep.mask_equal) std::printf("  mask: differs (%d)\n", rep.mask_mismatches);
        if (rep.first_bad_layer >= 0 || rep.ctx_max_rel > tol || !rep.mask_equal) ++failures;
    }
    std::printf("%s\n", failures ? "DIFF: FAIL" : "DIFF: PASS");
    return failures ? 1 : 0;
}

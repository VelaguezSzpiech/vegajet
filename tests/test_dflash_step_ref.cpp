// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// core/dflash_step_ref: tap-list validation, sliding-window vs. bidirectional block masks (strict q - k < W),
// hidden_norm(fc(concat taps)) against a hand computation, one attention step (softmax normalisation, masked
// context keys contribute nothing), and the per-layer diff naming the first diverging tap.
#include "core/dflash_step_ref.hpp"
#include "tests/test_common.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace vegajet::dflash_step;

namespace {
Config base_cfg() {
    Config c;
    c.n_states = 8;
    c.layers = {1, 3};
    c.d = 4;
    c.hd = 2;
    c.block = 3;
    c.pos = 6;
    c.window = 4;
    c.eps = 1e-6;
    return c;
}
}  // namespace

int main() {
    Config bad = base_cfg();
    std::string err;
    VJ_CHECK(validate(bad, &err));
    bad.layers = {3, 1};
    VJ_CHECK(!validate(bad, &err));
    bad = base_cfg();
    bad.layers = {1, 8};
    VJ_CHECK(!validate(bad, &err));
    bad = base_cfg();
    bad.layers.clear();
    VJ_CHECK(!validate(bad, &err));

    // ---- masks: full-attention blocks are bidirectional; SWA layers are causal with the strict window.
    const Config c = base_cfg();
    const std::vector<uint8_t> full = build_mask(c, false);
    VJ_CHECK(full.size() == size_t(c.block) * (c.pos + c.block));
    for (uint8_t v : full) VJ_CHECK(v == 1);
    const std::vector<uint8_t> swa = build_mask(c, true);
    // Slot 0 has query position 6: keys 3..6 visible (q - k < 4), key 2 not; later block keys not yet visible.
    VJ_CHECK(mask_allowed(c, true, 0, 6) && mask_allowed(c, true, 0, 3));
    VJ_CHECK(!mask_allowed(c, true, 0, 2));
    VJ_CHECK(!mask_allowed(c, true, 0, 7));  // block key ahead of the query
    VJ_CHECK(swa[size_t(0) * 9 + 6] == 1 && swa[size_t(0) * 9 + 2] == 0 && swa[size_t(0) * 9 + 7] == 0);
    VJ_CHECK(mask_allowed(c, false, 0, 7));  // ... but bidirectional on a full-attention layer
    // Last slot sees the whole block and the newest context keys.
    VJ_CHECK(mask_allowed(c, true, 2, 8) && mask_allowed(c, true, 2, 6) && mask_allowed(c, true, 2, 5));
    VJ_CHECK(!mask_allowed(c, true, 2, 4));

    // ---- hidden_norm(fc(concat taps)) against a hand computation.
    {
        const size_t d = 4;
        std::vector<double> st(size_t(c.n_states) * d, 0.0);
        for (size_t i = 0; i < st.size(); ++i) st[i] = 0.1 * double(i) - 1.0;
        std::vector<double> fc(d * 2 * d, 0.0);  // identity on the first tap, zeros on the second
        for (size_t i = 0; i < d; ++i) fc[i * 2 * d + i] = 1.0;
        std::vector<double> nw(d, 1.0);
        nw[1] = 2.0;
        std::vector<double> ctx;
        VJ_CHECK(build_context(c, st, fc, &nw, &ctx, &err));
        double ss = 0;
        for (size_t i = 0; i < d; ++i) ss += st[size_t(c.layers[0]) * d + i] * st[size_t(c.layers[0]) * d + i];
        const double rms = std::sqrt(ss / double(d) + c.eps);
        for (size_t i = 0; i < d; ++i)
            VJ_CHECK(std::fabs(ctx[i] - st[size_t(c.layers[0]) * d + i] / rms * nw[i]) < 1e-12);
        VJ_CHECK(!build_context(c, st, std::vector<double>(d * d, 0.0), &nw, &ctx, &err));  // wrong fc size
    }

    // ---- one step: with every value vector equal, a full-attention layer returns that value (softmax sums to 1);
    //      on a SWA layer with window 1 only the block's own key is visible, so context values cannot leak.
    {
        std::vector<double> st(size_t(c.n_states) * c.d, 0.0), fc(size_t(c.d) * 2 * c.d, 0.0), nw(c.d, 1.0);
        for (size_t i = 0; i < fc.size(); ++i) fc[i] = 0.01 * double(i % 7);
        std::vector<double> bx(size_t(c.block) * c.d), ck(size_t(c.pos) * c.hd, 0.0),
            cv(size_t(c.pos) * c.hd, 1000.0), wq(size_t(c.hd) * c.d, 0.0), wk(size_t(c.hd) * c.d, 0.0),
            wv(size_t(c.hd) * c.d, 0.0);
        for (size_t i = 0; i < bx.size(); ++i) bx[i] = 0.05 * double(i) - 0.2;
        for (size_t i = 0; i < wq.size(); ++i) wq[i] = 0.03 * double((i * 5) % 11) - 0.1;
        wk = wq;
        for (size_t i = 0; i < wv.size(); ++i) wv[i] = 0.02 * double((i * 3) % 7) - 0.05;
        Step s;
        VJ_CHECK(single_step(c, st, fc, &nw, bx, ck, cv, wq, wk, wv, false, &s, &err));
        VJ_CHECK(s.out.size() == size_t(c.block) * c.hd && s.mask.size() == size_t(c.block) * (c.pos + c.block));
        // Single-visible-key case: a block of one slot at pos 0 on a SWA layer sees only itself, so out = Wv x_0.
        {
            Config one = c;
            one.block = 1;
            one.pos = 0;
            one.window = 1;
            Step o;
            VJ_CHECK(single_step(one, st, fc, &nw, std::vector<double>(bx.begin(), bx.begin() + c.d), {}, {}, wq, wk,
                                 wv, true, &o, &err));
            VJ_CHECK(o.out.size() == size_t(c.hd));
            for (size_t i = 0; i < size_t(c.hd); ++i) {
                double want = 0;
                for (size_t j = 0; j < size_t(c.d); ++j) want += wv[i * c.d + j] * bx[j];
                VJ_CHECK(std::fabs(o.out[i] - want) < 1e-12);
            }
        }
        // Every value zero (context values and Wv) -> the output must be exactly zero on any layer.
        std::vector<double> zero_hd(size_t(c.hd) * c.d, 0.0), zero_ctx(size_t(c.pos) * c.hd, 0.0);
        Step z;
        VJ_CHECK(single_step(c, st, fc, &nw, bx, zero_ctx, zero_ctx, wq, wk, zero_hd, false, &z, &err));
        for (double v : z.out) VJ_CHECK(v == 0.0);
        // SWA layer, window 1: only the query's own key is visible -> context values cannot change the output.
        Config narrow = c;
        narrow.window = 1;
        Step a, b;
        VJ_CHECK(single_step(narrow, st, fc, &nw, bx, ck, cv, wq, wk, wv, true, &a, &err));
        std::vector<double> cv2 = cv;
        for (auto& v : cv2) v = -5000.0;
        VJ_CHECK(single_step(narrow, st, fc, &nw, bx, ck, cv2, wq, wk, wv, true, &b, &err));
        for (size_t i = 0; i < a.out.size(); ++i) VJ_CHECK(a.out[i] == b.out[i]);
        // ... and out_b is exactly Wv x_b.
        for (size_t bi = 0; bi < size_t(c.block); ++bi)
            for (size_t i = 0; i < size_t(c.hd); ++i) {
                double want = 0;
                for (size_t j = 0; j < size_t(c.d); ++j) want += wv[i * c.d + j] * bx[bi * c.d + j];
                VJ_CHECK(std::fabs(a.out[bi * c.hd + i] - want) < 1e-12);
            }
    }

    // ---- diff: identical dumps are clean; a perturbed tap is named with its layer.
    {
        std::vector<double> ref(size_t(c.n_states) * c.d), got, fc(size_t(c.d) * 2 * c.d, 0.0), nw(c.d, 1.0);
        std::mt19937 g(3);
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        for (auto& v : ref) v = u(g);
        for (size_t i = 0; i < fc.size(); ++i) fc[i] = u(g) * 0.1;
        got = ref;
        const DiffReport id = diff(c, ref, got, fc, &nw, nullptr, nullptr, false, 1e-6);
        VJ_CHECK(id.first_bad_layer == -1 && id.layer_max_rel.size() == c.layers.size());
        VJ_CHECK(id.layer_max_abs[0] == 0.0 && id.layer_max_abs[1] == 0.0);
        got[size_t(c.layers[1]) * c.d + 2] += 0.05;
        const DiffReport d2 = diff(c, ref, got, fc, &nw, nullptr, nullptr, false, 1e-3);
        VJ_CHECK(d2.first_bad_layer == c.layers[1]);
        VJ_CHECK(d2.layer_max_rel.size() == 2 && d2.layer_max_rel[1] > d2.layer_max_rel[0]);
        // Context diff and mask diff are reported.
        std::vector<double> ctx;
        VJ_CHECK(build_context(c, ref, fc, &nw, &ctx, nullptr));
        const std::vector<uint8_t> mask = build_mask(c, true);
        const DiffReport d3 = diff(c, ref, ref, fc, &nw, &ctx, &mask, true, 1e-6);
        VJ_CHECK(d3.ctx_compared && d3.ctx_max_rel < 1e-12 && d3.mask_equal && d3.mask_mismatches == 0);
        std::vector<uint8_t> wrong = mask;
        wrong[0] ^= 1u;
        const DiffReport d4 = diff(c, ref, ref, fc, &nw, &ctx, &wrong, true, 1e-6);
        VJ_CHECK(!d4.mask_equal && d4.mask_mismatches == 1);
        std::printf("  %s\n", d2.summary.c_str());
    }

    std::printf("test_dflash_step_ref: taps, masks, hidden_norm/fc, single step, per-layer diff OK\nPASS\n");
    return 0;
}

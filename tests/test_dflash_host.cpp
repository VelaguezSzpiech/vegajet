// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// DFlash2 host diagnostics (core/dflash_host): tap extraction and the off-by-one detector, the four
// fc / hidden_norm orders, block masks around the SWA = 4096 boundary (z-lab causal-on-SWA vs all-bidirectional,
// q - k < W vs <= W), masked attention, the Issue #152 acceptance gate and first-divergence localisation down to
// cycle, draft slot and drafter layer. Synthetic data only.
#include "core/dflash_host.hpp"
#include "tests/test_common.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace vegajet::dflash_host;

namespace {

std::vector<float> randv(size_t n, uint32_t seed, float lo = -1.0f, float hi = 1.0f) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<float> u(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = u(g);
    return v;
}

const std::vector<int> kTaps(std::begin(kDefaultTaps), std::end(kDefaultTaps));

void test_taps() {
    const int T = 6, NS = 49, D = 32;  // 48 target layers -> 49 states (embeddings + 48 layer outputs)
    const auto states = randv(static_cast<size_t>(T) * NS * D, 1);
    const size_t row = kTaps.size() * D;
    std::vector<float> obs0(T * row), obs1(T * row);
    for (int t = 0; t < T; ++t) {
        VJ_CHECK(gather_taps(&states[static_cast<size_t>(t) * NS * D], NS, D, kTaps, 0, &obs0[t * row], nullptr));
        VJ_CHECK(gather_taps(&states[static_cast<size_t>(t) * NS * D], NS, D, kTaps, 1, &obs1[t * row], nullptr));
    }
    // tap 15 -> state 15: the residual stream entering target layer 15.
    VJ_CHECK(obs0[2 * D + 5] == states[15 * D + 5]);
    const ShiftScore s0 = score_tap_shift(states.data(), T, NS, D, kTaps, obs0.data());
    VJ_CHECK(s0.best_shift == 0 && s0.unambiguous && s0.rel_err[1] < 1e-12 && s0.rel_err[0] > 0.5 && s0.rel_err[2] > 0.5);
    const ShiftScore s1 = score_tap_shift(states.data(), T, NS, D, kTaps, obs1.data());
    VJ_CHECK(s1.best_shift == 1 && s1.unambiguous);
    // Out of range: tap 46 + 3 = 49 is past the last state.
    std::string err;
    std::vector<float> tmp(row);
    VJ_CHECK(!gather_taps(states.data(), NS, D, kTaps, 3, tmp.data(), &err) && err.find("state 49") != std::string::npos);
    const ShiftScore s3 = score_tap_shift(states.data(), T, NS, D, kTaps, obs0.data(), {0, 3});
    VJ_CHECK(s3.best_shift == 0 && std::isinf(s3.rel_err[1]));
    std::printf("  taps: shift 0 exact, +1 off-by-one named (rel err %.2f at shift 0)\n", s1.rel_err[1]);
}

void test_norm_orders() {
    const int D = 24, NT = 8, K = NT * D;
    const auto cat = randv(K, 2, -3.0f, 3.0f);
    const auto fc = randv(static_cast<size_t>(D) * K, 3, -0.2f, 0.2f);
    const auto w = randv(D, 4, 0.5f, 1.5f);
    const float eps = 1e-6f;
    // rms_norm by hand.
    {
        const float x[4] = {1.0f, -2.0f, 3.0f, -4.0f};
        const float ww[4] = {1.0f, 2.0f, 1.0f, 0.5f};
        float y[4];
        rms_norm(x, ww, 4, 0.0f, y);
        const double r = std::sqrt((1.0 + 4.0 + 9.0 + 16.0) / 4.0);
        VJ_CHECK(std::fabs(y[1] - (-2.0 / r * 2.0)) < 1e-6 && std::fabs(y[3] - (-4.0 / r * 0.5)) < 1e-6);
    }
    // Unit-weight FcNorm output has RMS 1.
    std::vector<float> y(D);
    project_context(cat.data(), fc.data(), nullptr, D, NT, 0.0f, NormOrder::FcNorm, y.data());
    double ss = 0.0;
    for (float v : y) ss += static_cast<double>(v) * v;
    VJ_CHECK(std::fabs(std::sqrt(ss / D) - 1.0) < 1e-5);
    for (NormOrder o : {NormOrder::FcNorm, NormOrder::NormFc, NormOrder::NormCatFc, NormOrder::Fc}) {
        project_context(cat.data(), fc.data(), w.data(), D, NT, eps, o, y.data());
        const OrderScore sc = classify_norm_order(cat.data(), fc.data(), w.data(), D, NT, eps, y.data());
        VJ_CHECK(sc.best == o);
        for (const auto& [oo, e] : sc.rel_err) VJ_CHECK(oo == o ? e < 1e-6 : e > 1e-3);
    }
    std::printf("  fc/hidden_norm: all 4 orders recovered from their output, the others >= 1e-3 away\n");
}

void test_masks() {
    // Anchor at p = 4100, block of 8: on an SWA layer the oldest context keys fall out of the 4096 window.
    MaskSpec s;
    s.block_start = 4100;
    s.block_len = 8;
    s.swa_layer = true;
    s.window = kDefaultWindow;
    // slot 0 (q = 4100): key 4 is 4096 back -> hidden under q - k < W, visible under <= W; key 5 visible.
    VJ_CHECK(!mask_allowed(s, 0, 4) && mask_allowed(s, 0, 5));
    MaskSpec le = s;
    le.rule = WindowRule::Le;
    VJ_CHECK(mask_allowed(le, 0, 4) && !mask_allowed(le, 0, 3));
    // slot 7 (q = 4107): first visible context key is 12.
    VJ_CHECK(!mask_allowed(s, 7, 11) && mask_allowed(s, 7, 12));
    // z-lab: causal inside the block on SWA layers, bidirectional on full layers.
    VJ_CHECK(mask_allowed(s, 2, 4102) && !mask_allowed(s, 2, 4103));
    MaskSpec full = s;
    full.swa_layer = false;
    VJ_CHECK(mask_allowed(full, 2, 4107) && mask_allowed(full, 0, 0) && mask_allowed(full, 7, 0));
    MaskSpec causal_full = full;
    causal_full.mode = BlockMask::Causal;
    VJ_CHECK(!mask_allowed(causal_full, 2, 4103));
    MaskSpec bidir_swa = s;
    bidir_swa.mode = BlockMask::Bidirectional;
    VJ_CHECK(mask_allowed(bidir_swa, 2, 4107) && !mask_allowed(bidir_swa, 0, 4));  // the window still applies

    const auto m = build_mask(s);
    VJ_CHECK(m.size() == 8u * (4100u + 8u));
    int visible0 = 0;
    for (int k = 0; k < 4108; ++k) visible0 += m[static_cast<size_t>(k)];
    VJ_CHECK(visible0 == 4096);  // keys 5..4100: 4095 context + the anchor itself

    // diff: a port that made the SWA layer bidirectional.
    MaskDiff d = diff_mask(s, build_mask(bidir_swa));
    VJ_CHECK(d.mismatches == 28 && d.first_slot == 0 && d.first_key == 4101);  // 7+6+..+1 hidden block keys
    VJ_CHECK(d.matched_variant && d.variant_mode == BlockMask::Bidirectional && d.variant_rule == WindowRule::Lt);
    // a port with q - k <= W.
    d = diff_mask(s, build_mask(le));
    VJ_CHECK(d.mismatches == 8 && d.first_slot == 0 && d.first_key == 4 && d.variant_rule == WindowRule::Le);
    VJ_CHECK(d.summary.find("q - k <= W") != std::string::npos);
    // exact match, unknown variant, wrong shape.
    VJ_CHECK(diff_mask(s, m).mismatches == 0 && diff_mask(s, m).matched_variant);
    auto odd = m;
    odd[5] ^= 1;
    d = diff_mask(s, odd);
    VJ_CHECK(d.mismatches == 1 && !d.matched_variant && d.summary.find("no known variant") != std::string::npos);
    VJ_CHECK(diff_mask(s, std::vector<uint8_t>(10)).mismatches == -1);
    // Below the window nothing is clipped: p = 100.
    MaskSpec small = s;
    small.block_start = 100;
    VJ_CHECK(mask_allowed(small, 0, 0) && build_mask(small) == build_mask([&] { MaskSpec t = small; t.rule = WindowRule::Le; return t; }()));
    std::printf("  masks: SWA window edge at p=4100 (key 4 vs 5), causal-vs-bidirectional block, variants named\n");
}

// The kernel's visibility for ONE layer (`vj_dflash2_attn_kernel`): context j in [lo, s_pos) with
// lo = (qpos >= kw) ? qpos + 1 - kw : 0 (the kernel window clamp of THIS layer), then block jb in
// [0, blk_end) with blk_end = causal ? slot + 1 : B. Everything the plan does not name is the shipped
// behaviour. kw is the layer's kernel window: the SWA window for a sliding_attention layer, the full layer's
// own ring size for the full layer -- never the global short-context boundary.
std::vector<uint8_t> engine_mask(int s_pos, int B, int kw, bool causal_layer) {
    const int n_keys = s_pos + B;
    std::vector<uint8_t> m(static_cast<size_t>(B) * n_keys, 0);
    for (int i = 0; i < B; ++i) {
        const int qpos = s_pos + i;
        const int lo = (kw > 0 && qpos >= kw) ? qpos - kw + 1 : 0;
        uint8_t* row = &m[static_cast<size_t>(i) * n_keys];
        for (int j = lo; j < s_pos; ++j) row[j] = 1;
        const int blk_end = causal_layer ? i + 1 : B;
        for (int jb = 0; jb < blk_end; ++jb) row[s_pos + jb] = 1;
    }
    return m;
}

// Every layer's engine mask must equal the CPU reference (build_mask) with the mode the plan selected, and the
// selected mode must be the expected one for the layer type (z-lab: causal on SWA, bidirectional on the full one).
constexpr int kFullRingWindow = 32768;  // spec_replay::plan_draft_kv: the full layer's ring after #5

void check_cycle(const DraftBlockMaskPlan& p, int s_pos, int B, int W, uint32_t swa_mask, int n_layer,
                 BlockMaskMode arm_mode) {
    for (int l = 0; l < n_layer; ++l) {
        const bool swa = ((swa_mask >> l) & 1u) != 0u;
        const bool causal = ((p.causal_mask >> l) & 1u) != 0u;
        const bool want_causal = p.arm && arm_mode != BlockMaskMode::Bidir &&
                                 (arm_mode == BlockMaskMode::Causal || swa);
        VJ_CHECK(causal == want_causal);
        MaskSpec want;
        want.block_start = s_pos;
        want.block_len = B;
        want.swa_layer = swa;
        want.window = W;
        // The arm's selected per-layer mode; outside the arm the shipped bidirectional block on every layer.
        want.mode = causal ? BlockMask::Causal : BlockMask::Bidirectional;
        const int kw = swa ? W : kFullRingWindow;  // the launch's own kernel window, per layer
        const MaskDiff d = diff_mask(want, engine_mask(s_pos, B, kw, causal));
        VJ_CHECK(d.mismatches == 0);
        if (causal && swa) {
            // ... and that per-layer mode is exactly the z-lab/SWA rule of the reference, not merely "some causal".
            MaskSpec zlab = want;
            zlab.mode = BlockMask::ZLab;
            VJ_CHECK(build_mask(zlab) == engine_mask(s_pos, B, kw, true));
        }
    }
}

void test_block_mask_plan() {
    BlockMaskMode m = BlockMaskMode::Causal;
    VJ_CHECK(parse_block_mask_mode(nullptr, &m) && m == BlockMaskMode::Bidir);
    VJ_CHECK(parse_block_mask_mode("", &m) && m == BlockMaskMode::Bidir);
    VJ_CHECK(parse_block_mask_mode("bidir", &m) && m == BlockMaskMode::Bidir);
    VJ_CHECK(parse_block_mask_mode("zlab", &m) && m == BlockMaskMode::ZLab);
    VJ_CHECK(parse_block_mask_mode("causal", &m) && m == BlockMaskMode::Causal);
    VJ_CHECK(!parse_block_mask_mode("ZLab", &m) && !parse_block_mask_mode("all", &m) &&
             !parse_block_mask_mode("1", &m) && !parse_block_mask_mode(nullptr, nullptr));
    VJ_CHECK(std::string(block_mask_mode_name(BlockMaskMode::ZLab)) == "zlab");

    // Mixed-layout reference fixture, not deployed metadata: 5 SWA layers + 1 full layer, B=8, W=4096.
    // The deployed six-SWA W=2048/B=16 case is covered separately below.
    const int n_layer = 6, W = 4096, B = 8;
    const uint32_t swa = 0x1Fu;
    DraftBlockMaskConfig base{};
    base.swa_layer_mask = swa;
    base.n_layer = static_cast<uint32_t>(n_layer);
    base.block_len = static_cast<uint32_t>(B);
    base.swa_window = static_cast<uint32_t>(W);

    // Default: the shipped bidirectional block on every layer, pattern not even read.
    DraftBlockMaskConfig c = base;
    c.mode = BlockMaskMode::Bidir;
    DraftBlockMaskPlan p = plan_draft_block_mask(c);
    VJ_CHECK(p.ok && !p.arm && p.causal_mask == 0u && p.summary.find("bidir") != std::string::npos);
    check_cycle(p, 100, B, W, swa, n_layer, BlockMaskMode::Bidir);

    // A masked mode without the per-layer pattern fails closed (no "every layer is SWA" guess).
    for (BlockMaskMode mode : {BlockMaskMode::ZLab, BlockMaskMode::Causal}) {
        c = base;
        c.mode = mode;
        c.pattern_known = false;
        p = plan_draft_block_mask(c);
        VJ_CHECK(!p.ok && p.error.find("fail closed") != std::string::npos &&
                 p.error.find("dflash.attention.sliding_window_pattern") != std::string::npos);
    }

    // z-lab inside the short-context region: causal block on the 5 SWA layers, bidirectional on the full layer.
    c = base;
    c.mode = BlockMaskMode::ZLab;
    c.pattern_known = true;
    c.block_start = 200;
    p = plan_draft_block_mask(c);
    VJ_CHECK(p.ok && p.arm && p.causal_mask == swa);
    VJ_CHECK(p.summary.find("5 swa") != std::string::npos && p.summary.find("1 full") != std::string::npos);
    check_cycle(p, 200, B, W, swa, n_layer, BlockMaskMode::ZLab);
    // The boundary is the GLOBAL SWA window: the last anchor whose whole block still fits it is W - B ...
    c.block_start = W - B;
    p = plan_draft_block_mask(c);
    VJ_CHECK(p.ok && p.arm && p.causal_mask == swa);
    check_cycle(p, W - B, B, W, swa, n_layer, BlockMaskMode::ZLab);
    // ... one token later the arm is off for every layer, and W + 1 (past the window) is still unmasked.
    for (int s_pos : {W - B + 1, W + 1}) {
        c.block_start = static_cast<uint32_t>(s_pos);
        p = plan_draft_block_mask(c);
        VJ_CHECK(p.ok && !p.arm && p.causal_mask == 0u);
        VJ_CHECK(((p.causal_mask >> 5) & 1u) == 0u);  // the full layer is never causal
        check_cycle(p, s_pos, B, W, swa, n_layer, BlockMaskMode::ZLab);
        // At W + 1 the full layer attends block slot 7 from slot 0: no masking at all, explicitly.
        MaskSpec full;
        full.block_start = s_pos;
        full.block_len = B;
        full.swa_layer = false;
        full.window = W;
        full.mode = BlockMask::ZLab;  // z-lab is bidirectional on a full layer
        VJ_CHECK(mask_allowed(full, 0, s_pos + B - 1));
    }

    // The all-causal control mode: every layer causal inside the arm, none outside it.
    c = base;
    c.mode = BlockMaskMode::Causal;
    c.pattern_known = true;
    c.block_start = 200;
    p = plan_draft_block_mask(c);
    VJ_CHECK(p.ok && p.arm && p.causal_mask == 0x3Fu);
    check_cycle(p, 200, B, W, swa, n_layer, BlockMaskMode::Causal);
    c.block_start = W + 1;
    p = plan_draft_block_mask(c);
    VJ_CHECK(p.ok && !p.arm && p.causal_mask == 0u);
    check_cycle(p, W + 1, B, W, swa, n_layer, BlockMaskMode::Causal);

    // Deployed all-SWA geometry: every layer is causal only inside the explicit short-context arm.
    c = base;
    c.pattern_known = true;
    c.swa_layer_mask = 0x3Fu;
    c.swa_window = 2048u;
    c.block_len = 16u;
    for (BlockMaskMode mode : {BlockMaskMode::Bidir, BlockMaskMode::ZLab, BlockMaskMode::Causal}) {
        c.mode = mode;
        for (uint32_t pos : {200u, 2032u, 2033u, 2049u}) {
            c.block_start = pos;
            p = plan_draft_block_mask(c);
            const bool active = mode != BlockMaskMode::Bidir && pos <= 2032u;
            VJ_CHECK(p.ok && p.arm == active && p.causal_mask == (active ? 0x3Fu : 0u));
            check_cycle(p, static_cast<int>(pos), 16, 2048, 0x3Fu, 6, mode);
        }
    }

    // Invalid or degenerate configurations are refused, never silently coerced.
    c = base;
    c.mode = BlockMaskMode::ZLab;
    c.pattern_known = true;
    c.swa_window = 0u;
    VJ_CHECK(!plan_draft_block_mask(c).ok);
    c = base;
    c.mode = BlockMaskMode::ZLab;
    c.pattern_known = true;
    for (uint32_t bad : {0u, 33u}) {
        c.block_len = bad;
        VJ_CHECK(!plan_draft_block_mask(c).ok);
    }
    c.block_len = static_cast<uint32_t>(B);
    for (uint32_t bad : {0u, 33u}) {
        c.n_layer = bad;
        VJ_CHECK(!plan_draft_block_mask(c).ok);
    }
    c.n_layer = static_cast<uint32_t>(n_layer);
    c.swa_layer_mask = 0x40u;  // a pattern bit for a layer that does not exist
    VJ_CHECK(!plan_draft_block_mask(c).ok);
    c.swa_layer_mask = swa;
    // A drafter without an SWA layer under z-lab is a no-op arm, not an error.
    c.swa_layer_mask = 0u;
    p = plan_draft_block_mask(c);
    VJ_CHECK(p.ok && p.arm && p.causal_mask == 0u && p.summary.find("0 swa") != std::string::npos);
    // 32 layers: the known-layer mask covers the top bit without shifting into undefined behaviour.
    c.n_layer = 32u;
    c.swa_layer_mask = 0xFFFFFFFFu;
    p = plan_draft_block_mask(c);
    VJ_CHECK(p.ok && p.arm && p.causal_mask == 0xFFFFFFFFu);
    // Bidir stays the shipped mask even with an unknown pattern and a zero window.
    c = base;
    c.mode = BlockMaskMode::Bidir;
    c.pattern_known = false;
    c.swa_window = 0u;
    p = plan_draft_block_mask(c);
    VJ_CHECK(p.ok && !p.arm && p.causal_mask == 0u);
    std::printf("  block mask: 5 SWA + 1 full, arm inside W - B only, full layer bidirectional, unknown pattern refused\n");
}

void test_masked_attention() {
    const int B = 4, P = 6, NK = P + B, HD = 8;
    const auto q = randv(static_cast<size_t>(B) * HD, 5);
    const auto k = randv(static_cast<size_t>(NK) * HD, 6);
    const auto v = randv(static_cast<size_t>(NK) * HD, 7);
    MaskSpec zl{P, B, true, 0, BlockMask::ZLab, WindowRule::Lt};
    MaskSpec bd = zl;
    bd.mode = BlockMask::Bidirectional;
    std::vector<float> oz(B * HD), ob(B * HD);
    masked_attention(q.data(), k.data(), v.data(), build_mask(zl), B, NK, HD, oz.data());
    masked_attention(q.data(), k.data(), v.data(), build_mask(bd), B, NK, HD, ob.data());
    // The last slot sees every block key under both masks; slot 0 does not.
    VJ_CHECK(rel_l2(&oz[(B - 1) * HD], &ob[(B - 1) * HD], HD) < 1e-7);
    VJ_CHECK(rel_l2(&oz[0], &ob[0], HD) > 1e-3);
    // One visible key: the output is that key's value.
    std::vector<uint8_t> one(static_cast<size_t>(B) * NK, 0);
    for (int i = 0; i < B; ++i) one[static_cast<size_t>(i) * NK + 3] = 1;
    masked_attention(q.data(), k.data(), v.data(), one, B, NK, HD, oz.data());
    for (int i = 0; i < B; ++i)
        for (int c = 0; c < HD; ++c) VJ_CHECK(std::fabs(oz[i * HD + c] - v[3 * HD + c]) < 1e-6);
    // Brute-force softmax for slot 1 under the z-lab mask.
    const auto mz = build_mask(zl);
    masked_attention(q.data(), k.data(), v.data(), mz, B, NK, HD, oz.data());
    double den = 0.0, num[HD] = {};
    for (int j = 0; j < NK; ++j) {
        if (!mz[static_cast<size_t>(NK) + j]) continue;
        double dot = 0.0;
        for (int c = 0; c < HD; ++c) dot += static_cast<double>(q[HD + c]) * k[j * HD + c];
        const double w = std::exp(dot / std::sqrt(static_cast<double>(HD)));
        den += w;
        for (int c = 0; c < HD; ++c) num[c] += w * v[j * HD + c];
    }
    for (int c = 0; c < HD; ++c) VJ_CHECK(std::fabs(oz[HD + c] - num[c] / den) < 1e-5);
}

Cycle cyc(int pos, std::vector<int32_t> d, std::vector<int32_t> t) { return Cycle{pos, std::move(d), std::move(t)}; }

void test_acceptance() {
    VJ_CHECK(gate_of(4.0, 1) == Gate::Pass && gate_of(3.99, 1) == Gate::Marginal && gate_of(3.5, 1) == Gate::Marginal);
    VJ_CHECK(gate_of(3.49, 1) == Gate::Abort && gate_of(9.0, 0) == Gate::NoData);
    VJ_CHECK(accepted_prefix(cyc(0, {1, 2, 3}, {1, 2, 4})) == 2 && accepted_prefix(cyc(0, {1, 2, 3}, {1})) == 1);
    VJ_CHECK(accepted_prefix(cyc(0, {}, {1})) == 0);

    // B = 8: 20 cycles over a target stream; the reference drafter accepts 5 per cycle, the engine matches it until
    // cycle 6, then its draft slot 2 goes wrong (accepts 2).
    std::vector<Cycle> ref, eng;
    std::mt19937 g(9);
    for (int c = 0; c < 20; ++c) {
        const int pos = 100 + 6 * c;
        std::vector<int32_t> tgt(7), rd(7);
        for (int i = 0; i < 7; ++i) tgt[i] = static_cast<int32_t>(g() % 1000);
        rd = tgt;
        rd[5] = tgt[5] + 1;  // reference: 5 accepted
        auto ed = rd;
        if (c >= 6) ed[2] = tgt[2] + 7;  // engine: wrong from slot 2 on
        ref.push_back(cyc(pos, rd, tgt));
        eng.push_back(cyc(pos, ed, tgt));
    }
    ref.push_back(cyc(1000, {1, 2, 3}, {1, 2}));   // partial window: not in the means
    eng.push_back(cyc(2000, {1, 2, 3, 4, 5, 6, 7}, {9, 9, 9, 9, 9, 9, 9}));  // no reference cycle at this pos
    const AcceptanceReport r = compare_cycles(eng, ref, 8);
    VJ_CHECK(r.reference_full == 20 && std::fabs(r.reference_mean - 5.0) < 1e-12 && r.reference_gate == Gate::Pass);
    VJ_CHECK(r.engine_full == 21);
    const double want = (6 * 5.0 + 14 * 2.0 + 0.0) / 21.0;  // 2.762
    VJ_CHECK(std::fabs(r.engine_mean - want) < 1e-12 && r.engine_gate == Gate::Abort && r.flagged);
    VJ_CHECK(r.aligned == 20 && r.first_divergent_cycle == 6);
    VJ_CHECK(r.cycles[6].pos == 136 && r.cycles[6].first_slot == 2 && r.cycles[6].engine_accepted == 2 &&
             r.cycles[6].reference_accepted == 5);
    VJ_CHECK(r.engine_hist[5] == 6 && r.engine_hist[2] == 14 && r.engine_hist[0] == 1);
    VJ_CHECK(r.summary.find("first divergence at pos 136") != std::string::npos);
    VJ_CHECK(r.summary.find("FLAG") != std::string::npos);
    std::printf("  acceptance: %s\n", r.summary.c_str());

    const AcceptanceReport same = compare_cycles(ref, ref, 8);
    VJ_CHECK(same.first_divergent_cycle == -1 && !same.flagged && same.summary.find("equal the reference") != std::string::npos);
    const AcceptanceReport none = compare_cycles({}, ref, 8);
    VJ_CHECK(none.engine_gate == Gate::NoData && !none.flagged && none.aligned == 0);
    VJ_CHECK(compare_cycles(ref, ref, 4).summary.find("defined for B=8") != std::string::npos);
}

void test_layer_divergence() {
    const int L = 5, B = 8, D = 16;
    const auto ref = randv(static_cast<size_t>(L) * B * D, 11);
    auto eng = ref;
    for (int c = 0; c < D; ++c) eng[(static_cast<size_t>(3) * B + 6) * D + c] *= 1.05f;  // layer 3, slot 6: +5 %
    for (int c = 0; c < D; ++c) eng[(static_cast<size_t>(4) * B + 1) * D + c] += 0.3f;   // and everything after
    const LayerDivergence d = first_layer_divergence(eng.data(), ref.data(), L, B, D, 1e-3);
    VJ_CHECK(d.layer == 3 && d.slot == 6 && std::fabs(d.rel_err - 0.05) < 1e-6);
    VJ_CHECK(d.layer_max[0] == 0.0 && d.layer_max[2] == 0.0 && d.layer_max[4] > 0.1);
    VJ_CHECK(first_layer_divergence(eng.data(), ref.data(), L, B, D, 10.0).layer == -1);
    VJ_CHECK(first_layer_divergence(ref.data(), ref.data(), L, B, D, 0.0).layer == -1);
}

}  // namespace

int main() {
    test_taps();
    test_norm_orders();
    test_masks();
    test_block_mask_plan();
    test_masked_attention();
    test_acceptance();
    test_layer_divergence();
    std::printf("test_dflash_host: PASS\n");
    return 0;
}

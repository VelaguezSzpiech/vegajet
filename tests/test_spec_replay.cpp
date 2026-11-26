// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Deterministic draft -> verify -> commit replay (core/spec_replay) for DFlash2 B = 8, K = 7:
//   * first-token rejection, partial and full acceptance, the p_min early stop and the n_gen tail clamp, on code
//     and prose streams, with the GDN ring and with --spec-gdn, with and without the suffix cascade: engine and
//     z-lab reference agree cycle by cycle and the output is the target's greedy continuation;
//   * retained failing fixture (host defect, fixed): d51c19a's drafter KV layout (every layer a W-slot ring with
//     the window applied) drops context keys of the full-attention layer once the anchor passes W - B; the replay
//     names that cache-state divergence, the per-layer plan removes it up to the 32K limit;
//   * > 32K context: the source plan switches to blk.48 + FR-Spec at the limit; without the switch the replay
//     names the cache divergence at the first anchor past the limit;
//   * FR-Spec 32K / 64K shortlists (striped over 2 ranks, tail pinned) with full-vocabulary verification, and a
//     corrupted row -> id map named as a shortlist-mapping divergence;
//   * block mask: the current kernel's all-bidirectional mask is named against z-lab (kernel-side, Codex).
#include "core/fr_spec.hpp"
#include "core/spec_replay.hpp"
#include "tests/test_common.hpp"

#include <cstdio>
#include <vector>

using namespace vegajet::spec_replay;

namespace {

constexpr uint32_t kVocab = 248320u;
const std::vector<uint8_t> kPattern = {1, 1, 1, 1, 1, 0};  // 5 sliding-window layers + 1 full layer (z-lab)

DraftKvConfig kv_cfg(uint32_t max_ctx = 65536u) {
    DraftKvConfig c;
    c.n_layer = 6; c.swa_pattern = kPattern; c.swa_window = 4096; c.block_len = 8; c.max_ctx = max_ctx; c.full_ctx_cap = 32768;
    return c;
}

ReplayConfig base_cfg(const DraftKvPlan& kv, uint32_t n_gen = 128) {
    ReplayConfig r;
    r.k_draft = 7; r.n_gen = n_gen; r.prefill_chunk = 1024; r.ring_slots = 17; r.suffix_max_draft = 11;
    r.kv = kv;
    r.causal_layer = kPattern;  // z-lab masks: SWA causal, full bidirectional
    SourceConfig sc; sc.has_dflash2 = true; sc.has_mtp = true; sc.fr_spec_rows = 0; sc.dflash_max_anchor = kv.max_anchor_pos;
    r.sources = plan_sources(sc, 0, 0);
    return r;
}

Script make_script(bool code, uint32_t prompt_len, uint32_t n_gen, std::vector<uint32_t> plan, uint64_t seed = 7) {
    Script s;
    s.prompt_len = prompt_len;
    s.vocab = kVocab;
    const uint32_t n = prompt_len + n_gen + 32u;
    s.truth = code ? synth_code_stream(n, kVocab, seed) : synth_prose_stream(n, kVocab, seed);
    s.accept_plan = std::move(plan);
    return s;
}

Divergence run_pair(const Script& s, const ReplayConfig& e_cfg, const ReplayConfig& r_cfg, Trace* et = nullptr, Trace* rt = nullptr) {
    const Trace e = replay_engine(s, e_cfg);
    const Trace r = replay_reference(s, r_cfg);
    if (!e.error.empty()) std::fprintf(stderr, "engine replay: %s\n", e.error.c_str());
    if (!r.error.empty()) std::fprintf(stderr, "reference replay: %s\n", r.error.c_str());
    VJ_CHECK(e.error.empty() && r.error.empty());
    if (et) *et = e;
    if (rt) *rt = r;
    return first_divergence(e, r);
}

void test_kv_plan() {
    const DraftKvPlan fixed = plan_draft_kv(kv_cfg());
    VJ_CHECK(fixed.ok && fixed.pattern_known && fixed.layers.size() == 6);
    for (int l = 0; l < 5; ++l) VJ_CHECK(fixed.layers[l].kind == LayerKind::kSwa && fixed.layers[l].slots == 4096 && fixed.layers[l].kernel_win == 4096);
    VJ_CHECK(fixed.layers[5].kind == LayerKind::kFull && fixed.layers[5].slots == 32768 && fixed.layers[5].ring_mask == 32767);
    VJ_CHECK(fixed.max_anchor_pos == 32768u - 8u);
    // 5 x 4096 + 32768 slots x 8 heads x 128 x FP16 x (K, V)
    VJ_CHECK(fixed.total_bytes == (5ULL * 4096ULL + 32768ULL) * 8ULL * 128ULL * 2ULL * 2ULL);
    // --max-ctx below the cap sizes the full layer to the next power of two of max_ctx (never below W)
    const DraftKvPlan small = plan_draft_kv(kv_cfg(5000));
    VJ_CHECK(small.ok && small.layers[5].slots == 8192 && small.max_anchor_pos == 8192u - 8u);
    const DraftKvPlan legacy = legacy_draft_kv(kv_cfg());
    VJ_CHECK(legacy.ok && legacy.layers[5].slots == 4096 && legacy.max_anchor_pos == 4096u - 8u);
    // unknown pattern: every layer SWA (the legacy layout, unbounded anchor), flagged
    DraftKvConfig unk = kv_cfg(); unk.swa_pattern.clear();
    const DraftKvPlan u = plan_draft_kv(unk);
    VJ_CHECK(u.ok && !u.pattern_known && u.max_anchor_pos == UINT32_MAX && u.layers[5].slots == 4096);
    DraftKvConfig bad = kv_cfg(); bad.swa_window = 3000;
    VJ_CHECK(!plan_draft_kv(bad).ok);
    bad = kv_cfg(); bad.swa_pattern.pop_back();
    VJ_CHECK(!plan_draft_kv(bad).ok);
    std::printf("  kv plan: %s | legacy: %s\n", fixed.summary.c_str(), legacy.summary.c_str());
}

void test_source_plan() {
    SourceConfig c; c.has_dflash2 = true; c.has_mtp = true; c.fr_spec_rows = 32768; c.dflash_max_anchor = 32760;
    SourcePlan p = plan_sources(c, 1000, 128);
    VJ_CHECK(p.initial == Source::kDFlash2 && p.switch_at_pos == UINT32_MAX && !p.mtp_kv_from_prefill);
    p = plan_sources(c, 32700, 512);
    VJ_CHECK(p.initial == Source::kDFlash2 && p.after_switch == Source::kMtpFrSpec && p.switch_at_pos == 32761 && p.mtp_kv_from_prefill);
    VJ_CHECK(source_at(p, 32760) == Source::kDFlash2 && source_at(p, 32761) == Source::kMtpFrSpec);
    p = plan_sources(c, 40000, 128);
    VJ_CHECK(p.initial == Source::kMtpFrSpec && p.mtp_kv_from_prefill);
    c.fr_spec_rows = 0;
    VJ_CHECK(plan_sources(c, 40000, 128).initial == Source::kMtp);
    c.has_mtp = false;
    VJ_CHECK(plan_sources(c, 40000, 128).initial == Source::kSuffixOnly);
    // the 32K long-context threshold applies even when the cache could go further
    c.has_mtp = true; c.dflash_max_anchor = UINT32_MAX; c.long_ctx = 32768;
    VJ_CHECK(plan_sources(c, 32760, 128).switch_at_pos == 32769);
    c.has_dflash2 = false;
    VJ_CHECK(plan_sources(c, 10, 10).initial == Source::kMtp);
}

// Acceptance patterns on short context: engine and reference identical, output lossless.
void test_acceptance_patterns() {
    const DraftKvPlan fixed = plan_draft_kv(kv_cfg());
    const DraftKvPlan legacy = legacy_draft_kv(kv_cfg());
    struct Case { const char* name; std::vector<uint32_t> plan; };
    const Case cases[] = {
        {"first-token rejection", {0}},
        {"partial (3 of 7)", {3}},
        {"full acceptance", {7}},
        {"mixed", {0, 7, 3, 1, 7, 6, 0, 2}},
    };
    for (bool code : {true, false}) {
        for (const Case& cs : cases) {
            for (bool spec_gdn : {false, true}) {
                for (const DraftKvPlan* kv : {&fixed, &legacy}) {  // < W: the legacy layout is still exact
                    const Script s = make_script(code, 300, 128, cs.plan);
                    ReplayConfig cfg = base_cfg(*kv);
                    cfg.spec_gdn = spec_gdn;
                    Trace e, r;
                    const Divergence d = run_pair(s, cfg, cfg, &e, &r);
                    VJ_CHECK(d.field == Field::kNone);
                    VJ_CHECK(e.output.size() == 128u);
                    // mean accepted drafts per neural cycle follows the plan (the tail clamp shortens the last ones)
                    if (cs.plan.size() == 1u) {
                        const uint32_t want = cs.plan[0];
                        uint32_t full_rows = 0;
                        for (const auto& row : e.rows) {
                            if (row.k == 7u) { VJ_CHECK(row.accepted == want); ++full_rows; }
                            VJ_CHECK(row.n_commit == row.accepted + 1u);
                            VJ_CHECK(row.inject_pos.size() == row.n_commit && row.inject_pos.front() == row.anchor_pos);
                            VJ_CHECK(row.gdn_state_pos == row.anchor_pos + row.n_commit);
                        }
                        VJ_CHECK(full_rows > 0u);
                        if (want == 0u) VJ_CHECK(e.neural_accepted == 0u && e.rows.size() >= 127u);
                        if (want == 7u) VJ_CHECK(e.rows.size() <= 17u);
                    }
                }
            }
        }
        std::printf("  %s: 4 acceptance patterns x {ring, spec-gdn} x {fixed, legacy} identical to reference\n", code ? "code" : "prose");
    }
}

void test_suffix_pmin_tail() {
    const DraftKvPlan fixed = plan_draft_kv(kv_cfg());
    // code + CascadeHybrid: suffix cycles bypass the drafter, still lossless and identical
    {
        const Script s = make_script(true, 600, 256, {2, 7, 0});
        ReplayConfig cfg = base_cfg(fixed, 256);
        cfg.hybrid_suffix = true;
        Trace e, r;
        VJ_CHECK(run_pair(s, cfg, cfg, &e, &r).field == Field::kNone);
        VJ_CHECK(e.suffix_cycles > 0u && e.neural_cycles > 0u);
        uint32_t sfx_ge4 = 0;
        for (const auto& row : e.rows) if (row.suffix) { VJ_CHECK(row.shortlist_rows.empty()); sfx_ge4 += row.k >= 4u; }
        VJ_CHECK(sfx_ge4 > 0u);
        std::printf("  code + suffix cascade: %llu suffix / %llu neural cycles, %llu/%llu drafts accepted\n",
                    static_cast<unsigned long long>(e.suffix_cycles), static_cast<unsigned long long>(e.neural_cycles),
                    static_cast<unsigned long long>(e.accepted), static_cast<unsigned long long>(e.drafted));
    }
    // p_min stop after 2 drafts
    {
        const Script s = make_script(false, 300, 64, {7});
        ReplayConfig cfg = base_cfg(fixed, 64);
        cfg.pmin_stop_at = 2;
        Trace e;
        VJ_CHECK(run_pair(s, cfg, cfg, &e).field == Field::kNone);
        for (const auto& row : e.rows) VJ_CHECK(row.k <= 2u);
    }
    // tail clamp: n_gen = 10 with full acceptance -> k_cur = min(7, rem - 1), then one k = 0 single step (rem = 1)
    {
        const Script s = make_script(false, 300, 10, {7});
        ReplayConfig cfg = base_cfg(fixed, 10);
        Trace e;
        VJ_CHECK(run_pair(s, cfg, cfg, &e).field == Field::kNone);
        VJ_CHECK(e.output.size() == 10u);
        VJ_CHECK(e.rows.size() == 2u && e.rows[0].k == 7u && e.rows[1].k == 0u && e.rows[1].n_commit == 1u);
    }
}

// Retained failing fixture: d51c19a's legacy layout past the window.
void test_legacy_ring_fixture() {
    const DraftKvPlan fixed = plan_draft_kv(kv_cfg());
    const DraftKvPlan legacy = legacy_draft_kv(kv_cfg());
    for (bool code : {true, false}) {
        const Script s = make_script(code, 4200, 48, {7, 3, 0});
        ReplayConfig ref = base_cfg(fixed, 48);
        ReplayConfig eng = base_cfg(legacy, 48);
        eng.sources = ref.sources;  // DFlash2 throughout
        Trace e, r;
        const Divergence d = run_pair(s, eng, ref, &e, &r);
        VJ_CHECK(d.field == Field::kCache && d.cycle == 0);
        VJ_CHECK(e.rows[0].view_diffs.front().layer == 5u);           // the full-attention layer
        VJ_CHECK(e.rows[0].view_diffs.front().first_missing == 0);    // oldest context key evicted by the ring
        VJ_CHECK(e.neural_accepted == 0u && r.neural_accepted > 0u); // acceptance collapses, output still lossless
        std::printf("  legacy fixture (%s, prompt 4200): first divergence %s @ cycle %d: %s | mean acc %.2f vs reference %.2f\n",
                    code ? "code" : "prose", field_name(d.field), d.cycle, d.detail.c_str(), e.mean_accepted(), r.mean_accepted());
        // fixed layout: identical
        VJ_CHECK(run_pair(s, ref, ref).field == Field::kNone);
    }
    // exactly at the legacy boundary: anchor W - B is still exact, W - B + 1 is not
    {
        const Script s = make_script(false, 4096 - 8, 9, {7});  // one neural cycle: anchor = prompt length, K = 7
        ReplayConfig eng = base_cfg(legacy, 9);
        ReplayConfig ref = base_cfg(fixed, 9);
        eng.sources = ref.sources;
        VJ_CHECK(run_pair(s, eng, ref).field == Field::kNone);
        const Script s2 = make_script(false, 4096 - 7, 9, {7});
        VJ_CHECK(run_pair(s2, eng, ref).field == Field::kCache);
    }
}

// > 32K: blk.48 + FR-Spec takes over at the limit.
void test_long_context_fallback() {
    const DraftKvPlan fixed = plan_draft_kv(kv_cfg());
    std::vector<double> scores = vegajet::fr_spec::TokenFrequencyModel(kVocab).scores();
    Shortlist sl;
    std::string err;
    VJ_CHECK(build_shortlist(scores, 32768, 2, 512, &sl, &err));
    const Script s = make_script(true, 32740, 96, {7, 5, 0, 7});
    SourceConfig sc; sc.has_dflash2 = true; sc.has_mtp = true; sc.fr_spec_rows = 32768; sc.dflash_max_anchor = fixed.max_anchor_pos;
    ReplayConfig cfg = base_cfg(fixed, 96);
    cfg.sources = plan_sources(sc, s.prompt_len, 96);
    cfg.shortlist = &sl;
    VJ_CHECK(cfg.sources.switch_at_pos == 32761u && cfg.sources.mtp_kv_from_prefill);
    Trace e, r;
    VJ_CHECK(run_pair(s, cfg, cfg, &e, &r).field == Field::kNone);
    uint32_t n_df = 0, n_mtp = 0;
    for (const auto& row : e.rows) {
        if (row.anchor_pos <= 32760u) { VJ_CHECK(row.source == Source::kDFlash2); ++n_df; }
        else { VJ_CHECK(row.source == Source::kMtpFrSpec); ++n_mtp; }
    }
    VJ_CHECK(n_df > 0u && n_mtp > 0u);
    // without the switch, DFlash2 past the limit loses context on the full layer
    ReplayConfig no_switch = cfg;
    no_switch.sources = plan_sources(SourceConfig{true, true, 32768, UINT32_MAX, UINT32_MAX}, s.prompt_len, 96);
    Trace e2;
    const Divergence d = run_pair(s, no_switch, cfg, &e2);
    VJ_CHECK(d.field == Field::kCache && e2.rows[static_cast<size_t>(d.cycle)].anchor_pos > 32760u);
    std::printf("  > 32K: %u DFlash2 cycles then %u MTP+FR-Spec cycles, lossless; without the switch: %s @ anchor %u\n",
                n_df, n_mtp, field_name(d.field), e2.rows[static_cast<size_t>(d.cycle)].anchor_pos);
}

void test_shortlists() {
    std::vector<double> scores = vegajet::fr_spec::TokenFrequencyModel(kVocab).scores();
    for (uint32_t rows : {32768u, 65536u}) {
        Shortlist sl;
        std::string err;
        VJ_CHECK(build_shortlist(scores, rows, 2, 512, &sl, &err));
        VJ_CHECK(sl.rows_per_rank == rows / 2u);
        VJ_CHECK(check_shortlist(sl, kVocab, 512, &err));
        // striping: global entry i lives on rank i % 2; the 512 pinned tail ids come first
        VJ_CHECK(sl.tok_map[0][0] == static_cast<int32_t>(kVocab - 512u) && sl.tok_map[1][0] == static_cast<int32_t>(kVocab - 511u));
        // prose (ids spread over the vocabulary) under FR-Spec: off-list truth tokens are rejected, the full-vocabulary
        // verify keeps the output exact
        const DraftKvPlan fixed = plan_draft_kv(kv_cfg());
        const Script s = make_script(false, 400, 128, {7});
        ReplayConfig cfg = base_cfg(fixed, 128);
        cfg.shortlist = &sl;
        Trace e, r;
        VJ_CHECK(run_pair(s, cfg, cfg, &e, &r).field == Field::kNone);
        ReplayConfig full = base_cfg(fixed, 128);
        Trace ef;
        VJ_CHECK(run_pair(s, full, full, &ef).field == Field::kNone);
        VJ_CHECK(e.output == ef.output);
        uint32_t off_list = 0;
        for (uint32_t i = 0; i < 128; ++i) off_list += !sl.contains(s.truth[400 + i]);
        VJ_CHECK(off_list > 0u ? e.neural_accepted < ef.neural_accepted : e.neural_accepted == ef.neural_accepted);
        std::printf("  FR-Spec %u rows: %u/GPU, %u of 128 greedy tokens off-list, mean acc %.2f (full vocab %.2f), output identical\n",
                    rows, sl.rows_per_rank, off_list, e.mean_accepted(), ef.mean_accepted());
        // corrupted map fixture: rank 0 rows 0/1 swapped in the engine's copy only
        Shortlist bad = sl;
        std::swap(bad.tok_map[0][0], bad.tok_map[0][1]);
        // force drafts through the swapped rows: a script whose truth tokens are those ids
        Script s2 = s;
        for (uint32_t i = 0; i < 64; ++i) s2.truth[400 + i] = (i % 2u) ? sl.tok_map[0][0] : sl.tok_map[0][1];
        ReplayConfig ce = cfg; ce.shortlist = &bad;
        const Trace te = replay_engine(s2, ce);
        const Trace tr = replay_reference(s2, cfg);
        VJ_CHECK(te.error.empty() && tr.error.empty());
        const Divergence d = first_divergence(te, tr);
        VJ_CHECK(d.field == Field::kShortlist || d.field == Field::kTokenId);
        VJ_CHECK(d.cycle == 0);
        if (rows == 32768u) std::printf("  corrupted map fixture: %s @ cycle %d: %s\n", field_name(d.field), d.cycle, d.detail.c_str());
    }
    Shortlist sl;
    std::string err;
    VJ_CHECK(!build_shortlist(scores, 1000, 2, 512, &sl, &err));
}

// The kernel's mask (all layers bidirectional) against z-lab: named, kernel-side.
void test_mask_fixture() {
    const DraftKvPlan fixed = plan_draft_kv(kv_cfg());
    const Script s = make_script(false, 300, 32, {3});
    ReplayConfig eng = base_cfg(fixed, 32);
    eng.causal_layer.assign(6, 0);  // vj_dflash2_attn_kernel: block keys 0..B-1 on every layer
    ReplayConfig ref = base_cfg(fixed, 32);
    const Divergence d = run_pair(s, eng, ref);
    VJ_CHECK(d.field == Field::kMask && d.cycle == 0 && d.slot == 0);
    std::printf("  mask fixture: %s @ cycle %d: %s\n", field_name(d.field), d.cycle, d.detail.c_str());
}

}  // namespace

int main() {
    test_kv_plan();
    test_source_plan();
    test_acceptance_patterns();
    test_suffix_pmin_tail();
    test_legacy_ring_fixture();
    test_long_context_fallback();
    test_shortlists();
    test_mask_fixture();
    std::printf("test_spec_replay: PASS\n");
    return 0;
}

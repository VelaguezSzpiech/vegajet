// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Host side of the blk.48 MTP draft path (core/mtp_draft): chained launch descriptors (one host wait per K steps),
// FR-Spec 32K / 64K argmax-id mapping against the engine's formula, the MTP cache ledger around verify (first-token
// rejection, partial, full acceptance, suffix cycles), the >= 4-token suffix bypass and the bench statistics line.
// A replayed generation with full-vocabulary verification proves the drafted ids never change the output.
#include "core/fr_spec.hpp"
#include "core/mtp_draft.hpp"
#include "tests/test_common.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace vegajet;
using namespace vegajet::mtp_draft;

namespace {

constexpr uint32_t kVocab = 248320u;

void test_chain() {
    ChainConfig c;
    c.vocab = kVocab; c.voc_rows = {kVocab / 2u, kVocab / 2u}; c.max_ctx = 4096;
    for (uint32_t k = 1; k <= kMaxChain; ++k) {
        c.k = k;
        const ChainPlan p = plan_chain(c, 100);
        std::string err;
        VJ_CHECK(check_chain(p, c, 100, &err));
        VJ_CHECK(p.steps.back().host_read == 1u && p.host_waits == 1u);
        for (const auto& d : p.steps) VJ_CHECK(d.lm_mode == static_cast<uint32_t>(LmMode::kFullVocab) && d.lm_rows == kVocab / 2u);
    }
    c.k = 0; VJ_CHECK(!plan_chain(c, 0).ok);
    c.k = 17; VJ_CHECK(!plan_chain(c, 0).ok);
    c.k = 4; VJ_CHECK(!plan_chain(c, 4093).ok && plan_chain(c, 4092).ok);
    // tampered plans are rejected
    ChainPlan p = plan_chain(c, 10);
    std::string err;
    ChainPlan t = p; t.steps[2].tok_slot = 0; VJ_CHECK(!check_chain(t, c, 10, &err));
    t = p; t.steps[2].h_out_slot = t.steps[2].h_slot; VJ_CHECK(!check_chain(t, c, 10, &err));
    t = p; t.steps[1].host_read = 1; VJ_CHECK(!check_chain(t, c, 10, &err));
    t = p; t.steps[3].attend_hi = t.steps[3].pos + 2u; VJ_CHECK(!check_chain(t, c, 10, &err));
    t = p; t.steps[0].tok_src = static_cast<uint32_t>(TokSrc::kDeviceArgmax); VJ_CHECK(!check_chain(t, c, 10, &err));
    std::printf("  chain: K = 1..16 valid with one host wait; 5 tampered layouts rejected\n");
}

void test_fr_ids() {
    const auto scores = fr_spec::TokenFrequencyModel(kVocab).scores();
    for (uint32_t rows : {32768u, 65536u}) {
        spec_replay::Shortlist sl;
        std::string err;
        VJ_CHECK(spec_replay::build_shortlist(scores, rows, 2, 512, &sl, &err));
        ChainConfig c;
        c.vocab = kVocab; c.shortlist = &sl; c.k = 4;
        const uint32_t lr = sl.rows_per_rank;
        for (uint32_t g = 0; g < 2; ++g) {
            for (uint32_t r = 0; r < lr; ++r) {
                const int32_t kid = static_cast<int32_t>(g * lr + r);  // kernel: ro = g * lr, id = ro + row
                const int32_t id = resolve_draft_id(kid, c);
                // engine formula (step_mtp): g = idx >= lr ? 1 : 0, local = idx - g * lr
                const uint32_t ge = static_cast<uint32_t>(kid) >= lr ? 1u : 0u;
                VJ_CHECK(id == sl.tok_map[ge][static_cast<uint32_t>(kid) - ge * lr]);
                VJ_CHECK(sl.rank_of[static_cast<size_t>(id)] == static_cast<int32_t>(g) && sl.row_of[static_cast<size_t>(id)] == static_cast<int32_t>(r));
            }
        }
        VJ_CHECK(resolve_draft_id(static_cast<int32_t>(2u * lr), c) == -1 && resolve_draft_id(-1, c) == -1);
        const ChainPlan p = plan_chain(c, 0);
        VJ_CHECK(p.ok && p.steps[0].lm_mode == static_cast<uint32_t>(LmMode::kShortlist) && p.steps[0].lm_rows == lr);
        std::printf("  FR-Spec %uK: %u kernel ids per rank resolve to their vocabulary ids (engine formula)\n", rows / 1024u, lr);
    }
    ChainConfig full; full.vocab = kVocab;
    VJ_CHECK(resolve_draft_id(12345, full) == 12345 && resolve_draft_id(static_cast<int32_t>(kVocab), full) == -1);
}

void test_ledger() {
    MtpLedger L(256);
    std::string err;
    VJ_CHECK(L.prefill(20, &err) && L.exact_prefix() == 20u);
    VJ_CHECK(!L.draft(21, 4, &err));  // row 20 not written yet
    // neural cycle at anchor 20, K = 4: first-token rejection
    VJ_CHECK(L.draft(20, 4, &err));
    VJ_CHECK(L.row(20) == Row::kExact && L.row(21) == Row::kProvisional && L.row(23) == Row::kProvisional);
    std::vector<int32_t> vt = {5, 6, 7, 8, 9};
    VerifyPlan v = L.after_verify(vt, 20, 0, true);
    VJ_CHECK(v.catchup.empty() && v.next_anchor == 21u && v.h_pend_trunk_row == 0u);
    L.apply(v);
    VJ_CHECK(L.exact_prefix() == 21u);
    // partial: 2 of 4
    VJ_CHECK(L.draft(21, 4, &err));
    v = L.after_verify(vt, 21, 2, true);
    VJ_CHECK(v.catchup.size() == 2u && v.catchup[0].pos == 22u && v.catchup[0].trunk_row == 0u && v.catchup[1].token == 7);
    L.apply(v);
    VJ_CHECK(L.exact_prefix() == 24u && v.next_anchor == 24u && v.h_pend_trunk_row == 2u);
    // full: 4 of 4
    VJ_CHECK(L.draft(24, 4, &err));
    v = L.after_verify(vt, 24, 4, true);
    L.apply(v);
    VJ_CHECK(L.exact_prefix() == 29u && v.next_anchor == 29u);
    // suffix cycle (no chain): the anchor row is caught up from h_pend, then the accepted tokens
    v = L.after_verify(vt, 29, 3, false);
    VJ_CHECK(v.catchup.size() == 4u && v.catchup[0].pos == 29u && v.catchup[0].trunk_row == UINT32_MAX);
    L.apply(v);
    VJ_CHECK(L.exact_prefix() == 33u);
    // skipping the catch-up breaks the next chain
    MtpLedger L2(64);
    VJ_CHECK(L2.prefill(10, &err) && L2.draft(10, 4, &err));
    (void) L2.after_verify(vt, 10, 3, true);  // not applied
    VJ_CHECK(!L2.draft(14, 4, &err));
}

void test_bypass() {
    for (uint32_t depth : {0u, 1u, 2u, 4u, 6u, 16u}) {
        const uint32_t bounded = depth == 0u ? 1u : (depth < 4u ? depth : 4u);
        VJ_CHECK(suffix_min_chain(depth, false, false) == 1u);
        VJ_CHECK(suffix_min_chain(depth, true, false) == bounded);
        VJ_CHECK(suffix_min_chain(depth, false, true) == bounded);
        VJ_CHECK(suffix_min_chain(depth, true, true) == bounded);
    }
    SuffixBypass b(11);
    const std::vector<int32_t> phrase = {40, 41, 42, 43, 44, 45, 46, 47, 48, 49};
    b.commit({1, 2, 3});
    b.commit(phrase);
    b.commit({7, 8});
    VJ_CHECK(!b.decide().bypass);
    b.commit({40, 41, 42});  // 3-token match: below the bypass minimum
    VJ_CHECK(!b.decide().bypass);
    b.commit(43);            // 4-token match, chain 44..49 + 7, 8, 40.. -> bypass
    const BypassDecision d = b.decide();
    VJ_CHECK(d.bypass && d.match_len >= 4u && d.draft.size() >= 4u && d.draft[0] == 44 && d.draft.size() <= 11u);
    VJ_CHECK(b.hits == 1u && b.misses == 2u);
}

void test_stats() {
    SpecStats s;
    CycleSample a; a.depth = 4; a.accepted = 4; a.draft_ms = 2.0; a.verify_ms = 6.0;
    a.experts = {{1, 2, 3, 3, 4, 5, 6, 7}, {9, 9, 9, 9, 9, 9, 9, 9}};  // 7 and 1 distinct -> 4.0
    CycleSample b; b.suffix = true; b.depth = 8; b.accepted = 2; b.verify_ms = 8.0;
    s.add(a); s.add(b);
    s.wall_s = 0.5;
    VJ_CHECK(s.cycles == 2u && s.neural == 1u && s.suffix == 1u && s.committed == 8u);
    VJ_CHECK(s.accepted_per_cycle() == 3.0 && s.tokens_per_cycle() == 4.0 && s.mean_depth() == 6.0 && s.suffix_hit_rate() == 0.5);
    VJ_CHECK(s.unique_experts() == 4.0 && s.accepted_tok_s() == 16.0);
    const std::string line = format_stats("MTP K=4 code", s);
    VJ_CHECK(line == "[SPEC-STATS MTP K=4 code] cycles=2 acc/cyc=3.000 tok/cyc=4.000 depth=6.00 sfx_hit=50.0% uniq_exp=4.0 draft_ms=1.000 verify_ms=7.000 acc_tok_s=16.00");
    s.add_single();
    VJ_CHECK(s.committed == 9u && s.tokens_per_cycle() == 4.0 && s.cycles == 2u);
    SpecStats empty;
    VJ_CHECK(format_stats("x", empty).find("uniq_exp=n/a") != std::string::npos);
}

// A replayed MTP+FR-Spec generation: chained drafts resolved through the shortlist, full-vocabulary verify, ledger
// catch-up, suffix bypass. The output must be the greedy continuation whatever the shortlist drops.
void test_generation() {
    const auto scores = fr_spec::TokenFrequencyModel(kVocab).scores();
    spec_replay::Shortlist sl;
    std::string err;
    VJ_CHECK(spec_replay::build_shortlist(scores, 32768, 2, 512, &sl, &err));
    for (bool code : {true, false}) {
        const uint32_t prompt = 200, n_gen = 160, K = 4;
        // code: repetitive low ids (mostly on-list); prose: ids spread over the vocabulary
        std::vector<int32_t> truth = code ? spec_replay::synth_code_stream(prompt + n_gen + 32, 4096, 3)
                                          : spec_replay::synth_prose_stream(prompt + n_gen + 32, kVocab, 3);
        ChainConfig c; c.vocab = kVocab; c.shortlist = &sl; c.k = K; c.max_ctx = 4096;
        MtpLedger L(4096);
        VJ_CHECK(L.prefill(prompt, &err));
        SuffixBypass bp(11);
        bp.commit(std::vector<int32_t>(truth.begin(), truth.begin() + prompt));
        std::vector<int32_t> out{truth[prompt]};
        bp.commit(out.back());
        SpecStats st;
        uint32_t pos = prompt, cyc = 0;
        while (out.size() < n_gen) {
            const uint32_t rem = n_gen - static_cast<uint32_t>(out.size());
            CycleSample cs;
            std::vector<int32_t> drafts;
            const BypassDecision bd = bp.decide();
            if (bd.bypass) {
                drafts.assign(bd.draft.begin(), bd.draft.begin() + std::min<size_t>(bd.draft.size(), rem > 1 ? rem - 1 : 0));
                cs.suffix = true;
            }
            if (!cs.suffix) {
                const uint32_t k = std::min<uint32_t>(K, rem > 1 ? rem - 1 : 0);
                if (k > 0) {
                    const ChainPlan cp = plan_chain(ChainConfig{k, c.max_ctx, kVocab, 2, &sl, {}}, pos);
                    VJ_CHECK(check_chain(cp, ChainConfig{k, c.max_ctx, kVocab, 2, &sl, {}}, pos, &err));
                    VJ_CHECK(L.draft(pos, k, &err));
                    for (uint32_t j = 0; j < k; ++j) {
                        // drafter: right on even cycles while the token is on the list, else a fixed list row
                        const int32_t want = truth[pos + 1 + j];
                        const bool right = (cyc % 3u != 2u) && sl.contains(want);
                        const int32_t intended = right ? want : (sl.tok_map[1][7] == want ? sl.tok_map[1][8] : sl.tok_map[1][7]);
                        const int32_t kid = sl.rank_of[static_cast<size_t>(intended)] * static_cast<int32_t>(sl.rows_per_rank) + sl.row_of[static_cast<size_t>(intended)];
                        drafts.push_back(resolve_draft_id(kid, c));
                    }
                }
            }
            if (drafts.empty()) {  // single target step
                out.push_back(truth[pos + 1]);
                bp.commit(out.back());
                L.apply(L.after_verify({truth[pos]}, pos, 0, false));
                ++pos;
                continue;
            }
            // full-vocabulary verify: target greedy = truth
            std::vector<int32_t> vt{out.back()};
            vt.insert(vt.end(), drafts.begin(), drafts.end());
            uint32_t acc = 0;
            while (acc < drafts.size() && drafts[acc] == truth[pos + 1 + acc]) { out.push_back(drafts[acc]); bp.commit(drafts[acc]); ++acc; }
            if (out.size() < n_gen) { out.push_back(truth[pos + 1 + acc]); bp.commit(out.back()); }
            const VerifyPlan vp = L.after_verify(vt, pos, acc, !cs.suffix);
            L.apply(vp);
            VJ_CHECK(L.exact_prefix() >= vp.next_anchor);
            cs.depth = static_cast<uint32_t>(drafts.size()); cs.accepted = acc;
            st.add(cs);
            pos = vp.next_anchor;
            ++cyc;
        }
        for (uint32_t i = 0; i < n_gen; ++i) VJ_CHECK(out[i] == truth[prompt + i]);
        std::printf("  %s: %s\n", code ? "code " : "prose", format_stats(code ? "MTP K=4 FR32K code" : "MTP K=4 FR32K prose", st).c_str());
        VJ_CHECK(code ? st.suffix > 0u : true);
    }
}

}  // namespace

int main() {
    test_chain();
    test_fr_ids();
    test_ledger();
    test_bypass();
    test_stats();
    test_generation();
    std::printf("test_mtp_draft: PASS\n");
    return 0;
}

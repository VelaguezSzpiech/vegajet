// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// FR-Spec (HEAD core/fr_spec) at 32K / 64K rows on a 248,320-token, 2-rank split:
//  * every shortlist is padded to a multiple of 256 rows per rank and in total, must-include tokens are present,
//    row <-> token maps are exact inverses; a non-multiple row count is refused;
//  * lossless target-head verification: the argmax kernels are emulated block by block (wave scan with strict '>',
//    wave merge, host merge). With heavy exact-value ties, the shortlist pick equals the full-vocab argmax restricted
//    to the shortlist, so whenever the target token is in the shortlist the draft equals it; a speculative loop
//    (draft on the shortlist, verify on the full head) reproduces plain greedy decoding token for token.
#include "core/fr_spec.hpp"
#include "tests/test_common.hpp"

#include <cmath>
#include <cstdio>
#include <random>

using namespace vegajet::fr_spec;

namespace {

// Emulates one argmax dispatch: rows [0, nrows) of a layout; value(pos), id(pos). Writes G block winners.
template <class V, class I>
void dispatch(uint32_t nrows, const ArgmaxLayout& lay, V value, I id, std::vector<float>& bv, std::vector<int32_t>& bi) {
    const uint32_t G = lay.grid_blocks, W = lay.waves_per_block;
    bv.assign(G, -1e30f);
    bi.assign(G, -1);
    for (uint32_t b = 0; b < G; ++b) {
        for (uint32_t w = 0; w < W; ++w) {
            float best = -1e30f;
            int32_t bid = -1;
            for (uint64_t pos = uint64_t(b) * W + w; pos < nrows; pos += uint64_t(G) * W) {
                const float v = value(static_cast<uint32_t>(pos));
                if (v > best) { best = v; bid = id(static_cast<uint32_t>(pos)); }
            }
            if (best > bv[b]) { bv[b] = best; bi[b] = bid; }
        }
    }
}

int32_t full_argmax_kernel(const std::vector<float>& logit, const VocabSplit& sp, const ArgmaxLayout& lay) {
    std::vector<std::vector<float>> bv(sp.n_ranks);
    std::vector<std::vector<int32_t>> bi(sp.n_ranks);
    for (uint32_t r = 0; r < sp.n_ranks; ++r)
        dispatch(sp.rows_per_rank(), lay, [&](uint32_t p) { return logit[sp.token_of(r, p)]; },
                 [&](uint32_t p) { return sp.token_of(r, p); }, bv[r], bi[r]);
    std::vector<const float*> pv;
    std::vector<const int32_t*> pi;
    for (uint32_t r = 0; r < sp.n_ranks; ++r) { pv.push_back(bv[r].data()); pi.push_back(bi[r].data()); }
    return merge_full_vocab_argmax(pv.data(), pi.data(), sp.n_ranks, lay.grid_blocks);
}

ArgmaxPick shortlist_argmax_kernel(const std::vector<float>& logit, const Shortlist& sl) {
    std::vector<std::vector<float>> bv(sl.split.n_ranks);
    std::vector<std::vector<int32_t>> bi(sl.split.n_ranks);
    for (uint32_t r = 0; r < sl.split.n_ranks; ++r) {
        const uint32_t base = sl.rank_base[r];
        dispatch(sl.rank_rows[r], sl.draft_layout, [&](uint32_t p) { return logit[sl.row_to_token[base + p]]; },
                 [&](uint32_t p) { return static_cast<int32_t>(base + p); }, bv[r], bi[r]);
    }
    std::vector<const float*> pv;
    std::vector<const int32_t*> pi;
    for (uint32_t r = 0; r < sl.split.n_ranks; ++r) { pv.push_back(bv[r].data()); pi.push_back(bi[r].data()); }
    return merge_shortlist_argmax(sl, pv.data(), pi.data(), sl.draft_layout.grid_blocks);
}

// Reference: argmax by (value desc, full-vocab tie key asc) over `allowed` tokens.
int32_t ref_argmax(const std::vector<float>& logit, const VocabSplit& sp, const ArgmaxLayout& lay, const Shortlist* only) {
    int32_t best = -1;
    for (uint32_t t = 0; t < sp.scored_vocab(); ++t) {
        if (only && !only->contains(static_cast<int32_t>(t))) continue;
        if (best < 0 || logit[t] > logit[best] ||
            (logit[t] == logit[best] && full_vocab_tie_key(static_cast<int32_t>(t), sp, lay) < full_vocab_tie_key(best, sp, lay)))
            best = static_cast<int32_t>(t);
    }
    return best;
}

} // namespace

int main() {
    const VocabSplit sp{248320, 2};
    TokenFrequencyModel model(sp.vocab);
    std::mt19937 rng(5);
    for (int i = 0; i < 20000; ++i) model.observe(static_cast<int32_t>(std::min<uint32_t>(sp.vocab - 1, static_cast<uint32_t>(std::pow(rng() % 4000, 1.5)))));
    AlwaysInclude inc;
    inc.must = {248044, 248045, 248046, 0, 1};  // control tokens at the top of the vocab + low ids
    for (int i = 0; i < 300; ++i) inc.prefer.push_back(static_cast<int32_t>(rng() % sp.vocab));

    for (uint32_t rows : {32768u, 65536u}) {
        for (Balance bal : {Balance::kEqualPerRank, Balance::kGlobalTopN}) {
            ShortlistConfig cfg;
            cfg.rows = rows;
            cfg.balance = bal;
            Shortlist sl;
            std::string err;
            VJ_CHECK(build_shortlist(sp, cfg, model.scores(), inc, &sl, &err));
            VJ_CHECK(sl.rows == rows && sl.rows % kRowGranule == 0);
            uint32_t sum = 0;
            for (uint32_t r = 0; r < sp.n_ranks; ++r) { VJ_CHECK(sl.rank_rows[r] % kRowGranule == 0); sum += sl.rank_rows[r]; }
            VJ_CHECK(sum == rows);
            for (int32_t m : inc.must) VJ_CHECK(sl.contains(m));
            for (uint32_t row = 0; row < sl.rows; ++row) {
                const int32_t t = sl.row_to_token[row];
                VJ_CHECK(sp.scored(t) && sl.token_to_row[t] == static_cast<int32_t>(row));
                VJ_CHECK(sp.rank_of(t) == (row < sl.rank_base[1] ? 0u : 1u));  // rows live on the rank owning the token
            }
            // Lossless checks with heavy ties.
            uint32_t in_sl = 0, steps = 0;
            for (int s = 0; s < 12; ++s) {
                std::vector<float> logit(sp.vocab);
                for (auto& v : logit) v = std::round(static_cast<float>(rng() % 4000) / 400.0f * 4.0f) / 4.0f;  // 41 levels
                if (s % 3 == 0) for (int k = 0; k < 64; ++k) logit[sl.row_to_token[rng() % sl.rows]] = 50.0f;  // ties at the top
                const int32_t full = full_argmax_kernel(logit, sp, sl.full_layout);
                VJ_CHECK(full == ref_argmax(logit, sp, sl.full_layout, nullptr));
                const ArgmaxPick d = shortlist_argmax_kernel(logit, sl);
                VJ_CHECK(d.token == ref_argmax(logit, sp, sl.full_layout, &sl));
                if (sl.contains(full)) { VJ_CHECK(d.token == full); ++in_sl; }
                ++steps;
            }
            std::printf("  rows=%u %s: rank rows %u+%u, target in shortlist %u/%u steps, draft == target whenever present\n",
                        rows, bal == Balance::kEqualPerRank ? "equal  " : "global ", sl.rank_rows[0], sl.rank_rows[1], in_sl, steps);
        }
    }
    // A row count that is not a multiple of 256 is refused.
    {
        ShortlistConfig cfg;
        cfg.rows = 32768 + 100;
        Shortlist sl;
        std::string err;
        VJ_CHECK(!build_shortlist(sp, cfg, model.scores(), inc, &sl, &err) && !err.empty());
    }
    // Speculative loop on a small vocab: draft greedily on the shortlist, verify on the full head -> plain greedy.
    {
        const VocabSplit small{8192, 2};
        TokenFrequencyModel m2(small.vocab);
        ShortlistConfig cfg;
        cfg.rows = 1024;
        cfg.full_layout = {16, 4};
        cfg.draft_layout = {8, 4};
        Shortlist sl;
        std::string err;
        VJ_CHECK(build_shortlist(small, cfg, m2.scores(), {}, &sl, &err));
        auto logits_at = [&](int32_t prev, uint32_t pos) {  // deterministic "model": logits from (prev token, position)
            std::mt19937 g(static_cast<uint32_t>(prev * 7919 + pos));
            std::vector<float> l(small.vocab);
            for (auto& v : l) v = static_cast<float>(g() % 64) / 8.0f;
            return l;
        };
        std::vector<int32_t> greedy{1}, spec{1};
        for (uint32_t i = 0; i < 300; ++i) greedy.push_back(full_argmax_kernel(logits_at(greedy.back(), i), small, sl.full_layout));
        uint32_t accepted = 0, drafted = 0;
        while (spec.size() < greedy.size()) {
            std::vector<int32_t> draft;
            int32_t prev = spec.back();
            for (uint32_t k = 0; k < 4; ++k) {
                const auto pick = shortlist_argmax_kernel(logits_at(prev, static_cast<uint32_t>(spec.size() - 1 + k)), sl);
                draft.push_back(pick.token);
                prev = pick.token;
            }
            drafted += 4;
            prev = spec.back();
            for (uint32_t k = 0; k <= draft.size() && spec.size() < greedy.size(); ++k) {
                const int32_t t = full_argmax_kernel(logits_at(prev, static_cast<uint32_t>(spec.size() - 1)), small, sl.full_layout);
                spec.push_back(t);
                prev = t;
                if (k == draft.size() || t != draft[k]) break;
                ++accepted;
            }
        }
        VJ_CHECK(spec == greedy);
        std::printf("  speculative loop: 300 tokens identical to greedy; %u/%u shortlist drafts accepted\n", accepted, drafted);
    }
    std::printf("test_fr_spec_lossless: 32K/64K shortlists, 256-row padding, tie-exact lossless verification OK\nPASS\n");
    return 0;
}

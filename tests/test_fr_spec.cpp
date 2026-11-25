// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// FR-Spec draft shortlist: 256-row granularity, coverage, determinism, remap round trip, TP split, the
// exact full-vocab argmax tie rule (CPU simulation of ornith_vocab_argmax_q8_kernel + host merge), the
// gather plan, and a draft that misses because the true argmax is outside the shortlist.
// Single-matrix draft head: 256-row round-up alignment + VEGAJET_FR_SPEC_ROWS, short_id <-> full_id round trips
// at the 248,320-token vocab, byte-exact VJ_Q8_DP4A / VJ_Q4_A8 row slicing (odd-256 padding and XOR swizzle),
// the frequency-table loader / fallback, and a lossless greedy speculative loop driven by the sliced head.
#include "core/fr_spec.hpp"
#include "core/quant_repacker.hpp"
#include "tests/test_common.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>

using namespace vegajet::fr_spec;
namespace quant = vegajet::quant;

namespace {

// Bit-faithful CPU model of the argmax kernel's selection logic: wave w of block b scans rows
// b*W + w + k*G*W with strict '>', then waves 0..W-1 are merged with strict '>'.
void sim_kernel(const std::vector<float>& vals, uint32_t local_rows, uint32_t row_off, const ArgmaxLayout& lay,
                std::vector<float>& bv, std::vector<int32_t>& bi) {
    const uint32_t G = lay.grid_blocks, W = lay.waves_per_block;
    bv.assign(G, -1e30f);
    bi.assign(G, -1);
    for (uint32_t b = 0; b < G; ++b) {
        float wbest_v = -1e30f;
        int32_t wbest_i = -1;
        for (uint32_t w = 0; w < W; ++w) {
            float v = -1e30f;
            int32_t id = -1;
            for (uint32_t row = b * W + w; row < local_rows; row += G * W)
                if (vals[row] > v) { v = vals[row]; id = static_cast<int32_t>(row_off + row); }
            if (w == 0 || v > wbest_v) { wbest_v = v; wbest_i = id; }
        }
        bv[b] = wbest_v;
        bi[b] = wbest_i;
    }
}

int32_t brute_argmax(const std::vector<float>& logits, const VocabSplit& split, const ArgmaxLayout& lay,
                     const Shortlist* restrict_to) {
    int32_t best = -1;
    float bv = -1e30f;
    uint64_t bk = ~0ULL;
    for (uint32_t t = 0; t < split.scored_vocab(); ++t) {
        if (restrict_to && !restrict_to->contains(static_cast<int32_t>(t))) continue;
        const float v = logits[t];
        const uint64_t k = full_vocab_tie_key(static_cast<int32_t>(t), split, lay);
        if (v > bv || (best >= 0 && v == bv && k < bk)) { bv = v; best = static_cast<int32_t>(t); bk = k; }
    }
    return best;
}

int32_t full_pipeline(const std::vector<float>& logits, const VocabSplit& split, const ArgmaxLayout& lay) {
    std::vector<std::vector<float>> bv(split.n_ranks);
    std::vector<std::vector<int32_t>> bi(split.n_ranks);
    std::vector<const float*> pv;
    std::vector<const int32_t*> pi;
    const uint32_t per = split.rows_per_rank();
    for (uint32_t r = 0; r < split.n_ranks; ++r) {
        std::vector<float> loc(logits.begin() + r * per, logits.begin() + (r + 1) * per);
        sim_kernel(loc, per, r * per, lay, bv[r], bi[r]);
    }
    for (uint32_t r = 0; r < split.n_ranks; ++r) { pv.push_back(bv[r].data()); pi.push_back(bi[r].data()); }
    return merge_full_vocab_argmax(pv.data(), pi.data(), split.n_ranks, lay.grid_blocks);
}

ArgmaxPick shortlist_pipeline(const std::vector<float>& logits, const Shortlist& sl) {
    const uint32_t R = sl.split.n_ranks;
    std::vector<std::vector<float>> bv(R);
    std::vector<std::vector<int32_t>> bi(R);
    std::vector<const float*> pv;
    std::vector<const int32_t*> pi;
    for (uint32_t r = 0; r < R; ++r) {
        std::vector<float> sub(sl.rank_rows[r]);
        for (uint32_t p = 0; p < sl.rank_rows[r]; ++p) sub[p] = logits[sl.split.token_of(r, sl.device_local_rows[r][p])];
        sim_kernel(sub, sl.rank_rows[r], sl.rank_base[r], sl.draft_layout, bv[r], bi[r]);
    }
    for (uint32_t r = 0; r < R; ++r) { pv.push_back(bv[r].data()); pi.push_back(bi[r].data()); }
    return merge_shortlist_argmax(sl, pv.data(), pi.data(), sl.draft_layout.grid_blocks);
}

void check_structure(const Shortlist& sl) {
    const uint32_t R = sl.split.n_ranks;
    VJ_CHECK(sl.rows % kRowGranule == 0);
    uint32_t sum = 0;
    for (uint32_t r = 0; r < R; ++r) {
        VJ_CHECK(sl.rank_rows[r] % kRowGranule == 0);
        VJ_CHECK(sl.rank_base[r] == sum);
        sum += sl.rank_rows[r];
        VJ_CHECK(sl.sorted_local_rows[r].size() == sl.rank_rows[r]);
        VJ_CHECK(sl.device_local_rows[r].size() == sl.rank_rows[r]);
        VJ_CHECK(std::is_sorted(sl.sorted_local_rows[r].begin(), sl.sorted_local_rows[r].end()));
        VJ_CHECK(std::adjacent_find(sl.sorted_local_rows[r].begin(), sl.sorted_local_rows[r].end()) ==
                 sl.sorted_local_rows[r].end());
        std::vector<uint32_t> dev(sl.device_local_rows[r]);
        std::sort(dev.begin(), dev.end());
        VJ_CHECK(dev == sl.sorted_local_rows[r]);  // device order is a permutation of the sorted list
        for (uint32_t p = 0; p < sl.rank_rows[r]; ++p) {
            const uint32_t row = sl.rank_base[r] + p;
            const int32_t tok = sl.row_to_token[row];
            VJ_CHECK(sl.split.scored(tok));
            VJ_CHECK(sl.split.rank_of(tok) == r);                       // per-GPU split correctness
            VJ_CHECK(sl.split.local_row(tok) == sl.device_local_rows[r][p]);
            VJ_CHECK(sl.token_to_row[tok] == static_cast<int32_t>(row));  // remap round trip
        }
    }
    VJ_CHECK(sum == sl.rows);
    uint32_t n_in = 0;
    for (size_t t = 0; t < sl.token_to_row.size(); ++t)
        if (sl.token_to_row[t] >= 0) {
            ++n_in;
            VJ_CHECK(sl.row_to_token[sl.token_to_row[t]] == static_cast<int32_t>(t));
        }
    VJ_CHECK(n_in == sl.rows);
}

// ---------------------------------------------------------------------------------------------------------
// Single-matrix draft head checks.
// ---------------------------------------------------------------------------------------------------------
bool same_bytes(const uint8_t* a, const uint8_t* b, size_t n) { return std::memcmp(a, b, n) == 0; }
bool same_float(float a, float b) { return std::memcmp(&a, &b, sizeof(float)) == 0; }

int32_t first_max(const std::vector<float>& v, const IdMap* restrict_to) {
    int32_t bi = -1;
    for (size_t i = 0; i < v.size(); ++i) {
        if (restrict_to && restrict_to->to_short(static_cast<int32_t>(i)) < 0) continue;
        if (bi < 0 || v[i] > v[static_cast<size_t>(bi)]) bi = static_cast<int32_t>(i);
    }
    return bi;
}

// Deterministic hidden state of a toy target for context (a, b): N(0,1) entries, plus optional drafter noise.
std::vector<float> hidden(int32_t a, int32_t b, uint32_t D, float noise) {
    std::mt19937 g(static_cast<uint32_t>(a) * 2654435761u ^ static_cast<uint32_t>(b) * 40503u ^ 0x9E3779B9u);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> h(D);
    for (auto& v : h) v = nd(g);
    if (noise > 0.0f) {
        std::mt19937 g2(static_cast<uint32_t>(a) * 31u + static_cast<uint32_t>(b) * 7919u + 17u);
        for (auto& v : h) v += noise * nd(g2);
    }
    return h;
}

int vocab_head_checks(std::mt19937& rng) {
    int n = 0;
    const VocabSplit split{248320u, 2u};

    // --- 256-row alignment: presets, round-up, clamp, env toggle ---
    {
        VJ_CHECK(round_up_rows(0) == 0 && round_up_rows(1) == 256 && round_up_rows(256) == 256);
        VJ_CHECK(round_up_rows(257) == 512 && round_up_rows(32769) == 33024 && round_up_rows(kRowsPreset64K) == 65536);
        VJ_CHECK(row_granule(split, Balance::kEqualPerRank) == 512 && row_granule(split, Balance::kGlobalTopN) == 256);
        std::string note;
        VJ_CHECK(resolve_rows(kRowsPreset32K, split, Balance::kEqualPerRank, &note) == 32768 && note.empty());
        VJ_CHECK(resolve_rows(kRowsPreset64K, split, Balance::kEqualPerRank, &note) == 65536 && note.empty());
        VJ_CHECK(resolve_rows(1000, split, Balance::kEqualPerRank, &note) == 1024 && !note.empty());
        VJ_CHECK(resolve_rows(33000, split, Balance::kGlobalTopN, &note) == 33024);
        VJ_CHECK(resolve_rows(33000, split, Balance::kEqualPerRank, &note) == 33280);
        VJ_CHECK(resolve_rows(300000, split, Balance::kEqualPerRank, &note) == 248320 && note.find("clamped") != std::string::npos);
        VJ_CHECK(resolve_rows(0, split, Balance::kEqualPerRank, &note) == 0 && note.empty());
        VJ_CHECK(resolve_rows(5, VocabSplit{300u, 2u}, Balance::kEqualPerRank, &note) == 0 && !note.empty());
        uint64_t r = 7;
        std::string err;
        VJ_CHECK(parse_rows_spec("32k", &r, &err) && r == 32768);
        VJ_CHECK(parse_rows_spec(" 64K ", &r, &err) && r == 65536);
        VJ_CHECK(parse_rows_spec("65536", &r, &err) && r == 65536);
        VJ_CHECK(parse_rows_spec("off", &r, &err) && r == 0);
        VJ_CHECK(parse_rows_spec("", &r, &err) && r == 0);
        for (const char* bad : {"-5", "12x", "1e5", "k", "9999999999"}) VJ_CHECK(!parse_rows_spec(bad, &r, &err) && !err.empty());
        VJ_CHECK(unsetenv(kRowsEnv) == 0 && rows_from_env(split, Balance::kEqualPerRank, &note) == 0 && note.empty());
        VJ_CHECK(setenv(kRowsEnv, "1000", 1) == 0 && rows_from_env(split, Balance::kEqualPerRank, &note) == 1024 && !note.empty());
        VJ_CHECK(setenv(kRowsEnv, "64k", 1) == 0 && rows_from_env(split, Balance::kEqualPerRank, &note) == 65536 && note.empty());
        VJ_CHECK(setenv(kRowsEnv, "0", 1) == 0 && rows_from_env(split) == 0);
        VJ_CHECK(setenv(kRowsEnv, "lots", 1) == 0 && rows_from_env(split, Balance::kEqualPerRank, &note) == 0 && !note.empty());
        VJ_CHECK(unsetenv(kRowsEnv) == 0);
        // Whatever is requested, the resolved size is accepted by the shortlist builder (no unaligned path).
        TokenFrequencyModel prior(split.vocab);
        const auto sc = prior.scores();
        for (uint64_t req : {uint64_t{1}, uint64_t{4097}, uint64_t{32768}, uint64_t{40000 + rng() % 20000}, uint64_t{999999}}) {
            for (Balance bal : {Balance::kEqualPerRank, Balance::kGlobalTopN}) {
                const uint32_t rows = resolve_rows(req, split, bal);
                VJ_CHECK(rows >= req || rows == 248320u);
                ShortlistConfig cfg;
                cfg.rows = rows;
                cfg.balance = bal;
                Shortlist sl;
                VJ_CHECK(build_shortlist(split, cfg, sc, {}, &sl, &err));
                for (uint32_t rr : sl.rank_rows) VJ_CHECK(rr % kRowGranule == 0);
            }
        }
        n += 6;
    }

    // --- short_id <-> full_id round trips at the 248,320-token vocab ---
    {
        const uint32_t V = split.vocab;
        const FrequencyTable fb = fallback_frequency_table(V);
        VJ_CHECK(!fb.from_file && fb.ranked.size() == V && fb.ranked[0] == 0 && fb.ranked[V - 1] == static_cast<int32_t>(V - 1));
        const auto sc = table_scores(fb, V);
        const std::vector<int32_t> must = {248044, 248045, 248319};
        for (uint32_t rows : {kRowsPreset32K, kRowsPreset64K}) {
            std::vector<int32_t> plain, with_must;
            std::string err;
            VJ_CHECK(select_draft_vocab(sc, V, rows, {}, &plain, &err));
            VJ_CHECK(plain.size() == rows);
            for (uint32_t i = 0; i < rows; ++i) VJ_CHECK(plain[i] == static_cast<int32_t>(i));  // low-ID prefix
            VJ_CHECK(select_draft_vocab(sc, V, rows, must, &with_must, &err));
            VJ_CHECK(with_must.size() == rows && std::is_sorted(with_must.begin(), with_must.end()));
            IdMap m;
            VJ_CHECK(make_id_map(with_must, V, &m, &err));
            VJ_CHECK(m.rows() == rows && m.vocab() == V);
            uint32_t mapped = 0;
            for (uint32_t f = 0; f < V; ++f) {
                const int32_t s = m.to_short(static_cast<int32_t>(f));
                if (s >= 0) { ++mapped; VJ_CHECK(m.to_full(s) == static_cast<int32_t>(f)); }
            }
            for (uint32_t s = 0; s < rows; ++s) VJ_CHECK(m.to_short(m.to_full(static_cast<int32_t>(s))) == static_cast<int32_t>(s));
            VJ_CHECK(mapped == rows);
            for (int32_t t : must) VJ_CHECK(m.to_short(t) >= 0);
            VJ_CHECK(m.to_short(-1) == -1 && m.to_short(static_cast<int32_t>(V)) == -1 && m.to_full(static_cast<int32_t>(rows)) == -1);
            ++n;
        }
        IdMap bad;
        std::string err;
        VJ_CHECK(!make_id_map({1, 2, 1}, 10, &bad, &err) && !err.empty());
        VJ_CHECK(!make_id_map({1, 10}, 10, &bad, &err));
        // TP shortlist order as a map.
        ShortlistConfig cfg;
        Shortlist sl;
        VJ_CHECK(build_shortlist(split, cfg, TokenFrequencyModel(V).scores(), {}, &sl, &err));
        const IdMap tm = id_map_of(sl);
        for (uint32_t s = 0; s < tm.rows(); ++s) VJ_CHECK(tm.to_short(tm.to_full(static_cast<int32_t>(s))) == static_cast<int32_t>(s));
        n += 2;
    }

    // --- byte-exact slicing: VJ_Q8_DP4A at D = 3072 (odd-256 padded qs, dense FP16 d) ---
    {
        const uint32_t V = 4096, D = 3072;
        const HeadPlanes P = head_planes_q8_dp4a(V, D);
        VJ_CHECK(P.qs.stride_bytes == quant::select_tcc_layout(D).stride_bytes && P.qs.stride_bytes == 3328u);
        std::string err;
        VJ_CHECK(validate_planes(P, &err));
        std::vector<uint8_t> qs(P.qs_bytes(), 0), scl(P.scale_bytes());
        for (uint32_t r = 0; r < V; ++r)
            for (uint32_t c = 0; c < D; ++c) qs[static_cast<size_t>(r) * P.qs.stride_bytes + c] = static_cast<uint8_t>(rng() % 255u);
        for (size_t i = 0; i + 1 < scl.size(); i += 2) {
            const uint16_t h = quant::f32_to_fp16(0.001f + 0.0001f * static_cast<float>(rng() % 97u));
            std::memcpy(&scl[i], &h, 2);
        }
        std::vector<double> sc(V);
        for (auto& v : sc) v = static_cast<double>(rng() % 100000u);
        std::vector<int32_t> ids;
        VJ_CHECK(select_draft_vocab(sc, V, 1000, {7, 4095}, &ids, &err));
        VJ_CHECK(ids.size() == 1024);  // 1000 rounded up to 256
        IdMap m;
        VJ_CHECK(make_id_map(ids, V, &m, &err));
        DraftHead h;
        VJ_CHECK(slice_draft_head(P, qs.data(), scl.data(), m, &h, &err));
        VJ_CHECK(h.planes.rows == 1024 && h.qs.size() == 1024u * 3328u && h.scale.size() == 1024u * 192u);
        for (uint32_t s = 0; s < 1024; ++s) {
            const size_t f = static_cast<size_t>(m.to_full(static_cast<int32_t>(s)));
            VJ_CHECK(same_bytes(&h.qs[s * 3328u], &qs[f * 3328u], 3328u));
            VJ_CHECK(same_bytes(&h.scale[s * 192u], &scl[f * 192u], 192u));
        }
        std::vector<float> x(D), full(V), draft(1024);
        for (auto& v : x) v = static_cast<float>(static_cast<int>(rng() % 2001u) - 1000) * 1e-3f;
        head_logits_host(P, qs.data(), scl.data(), x.data(), full.data());
        head_logits_host(h.planes, h.qs.data(), h.scale.data(), x.data(), draft.data());
        for (uint32_t s = 0; s < 1024; ++s) VJ_CHECK(same_float(draft[s], full[static_cast<size_t>(m.to_full(static_cast<int32_t>(s)))]));
        VJ_CHECK(draft_argmax_full_id(draft.data(), m) == first_max(full, &m));
        // Refusals: unaligned draft size, map built for another vocab.
        IdMap m300;
        VJ_CHECK(make_id_map(std::vector<int32_t>(ids.begin(), ids.begin() + 300), V, &m300, &err));
        VJ_CHECK(!slice_draft_head(P, qs.data(), scl.data(), m300, &h, &err));
        IdMap mv;
        VJ_CHECK(make_id_map(ids, V + 1, &mv, &err));
        VJ_CHECK(!slice_draft_head(P, qs.data(), scl.data(), mv, &h, &err));
        HeadPlanes broken = P;
        broken.qs.raw_row_bytes = D / 2u;
        VJ_CHECK(!validate_planes(broken, &err));
        n += 3;
    }

    // --- byte-exact slicing: VJ_Q4_A8 at D = 3072 from the real packer (odd-256 qs and dm planes) ---
    {
        const uint32_t V = 1024, D = 3072;
        std::vector<float> w(static_cast<size_t>(V) * D);
        std::normal_distribution<float> nd(0.0f, 0.02f);
        for (auto& v : w) v = nd(rng);
        const quant::PackedVjQ4A8 pk = quant::pack_f32_to_vj_q4_a8(w.data(), V, D, true);
        const HeadPlanes P = head_planes_q4_a8(V, D, true);
        VJ_CHECK(P.qs.stride_bytes == pk.qs_row_stride_bytes && P.scale.stride_bytes == pk.dm_row_stride_u32 * 4u);
        VJ_CHECK(P.qs_bytes() == pk.qs.size() && P.scale_bytes() == pk.dm.size() * 4u);
        const auto* dm = reinterpret_cast<const uint8_t*>(pk.dm.data());
        std::vector<int32_t> ids;
        std::string err;
        std::vector<double> sc(V);
        for (auto& v : sc) v = static_cast<double>(rng() % 1000u);
        VJ_CHECK(select_draft_vocab(sc, V, 256, {}, &ids, &err) && ids.size() == 256);
        IdMap m;
        VJ_CHECK(make_id_map(ids, V, &m, &err));
        DraftHead h;
        VJ_CHECK(slice_draft_head(P, pk.qs.data(), dm, m, &h, &err));
        const uint32_t qsb = P.qs.stride_bytes, dmb = P.scale.stride_bytes;
        for (uint32_t s = 0; s < 256; ++s) {
            const size_t f = static_cast<size_t>(m.to_full(static_cast<int32_t>(s)));
            VJ_CHECK(same_bytes(&h.qs[s * qsb], &pk.qs[f * qsb], qsb));
            VJ_CHECK(same_bytes(&h.scale[s * dmb], dm + f * dmb, dmb));
        }
        // Reference GEMV agrees with the packer's own dequantization, and sliced rows give identical logits.
        std::vector<float> deq(w.size()), x(D), full(V), draft(256);
        quant::dequantize_vj_q4_a8(pk, deq.data());
        for (auto& v : x) v = nd(rng) * 50.0f;
        head_logits_host(P, pk.qs.data(), dm, x.data(), full.data());
        for (uint32_t r = 0; r < V; r += 97) {
            double ref = 0.0, mag = 0.0;
            for (uint32_t c = 0; c < D; ++c) { ref += static_cast<double>(deq[static_cast<size_t>(r) * D + c]) * x[c]; mag += std::fabs(deq[static_cast<size_t>(r) * D + c] * x[c]); }
            VJ_CHECK(std::fabs(full[r] - ref) <= 1e-4 * mag + 1e-6);
        }
        head_logits_host(h.planes, h.qs.data(), h.scale.data(), x.data(), draft.data());
        for (uint32_t s = 0; s < 256; ++s) VJ_CHECK(same_float(draft[s], full[static_cast<size_t>(m.to_full(static_cast<int32_t>(s)))]));
        n += 2;
    }

    // --- XOR-swizzled plane (16 whole tiles per row): tiles are re-permuted for the new row index ---
    {
        const uint32_t V = 64, D = 4096;
        const quant::TccRowLayout lay = quant::select_tcc_layout(D);
        VJ_CHECK(lay.mode == quant::TccLayoutMode::XorTileSwizzle && lay.stride_bytes == D);
        std::vector<uint8_t> nat(static_cast<size_t>(V) * D), scl(static_cast<size_t>(V) * (D / 16u));
        for (auto& b : nat) b = static_cast<uint8_t>(rng());
        for (size_t i = 0; i + 1 < scl.size(); i += 2) { const uint16_t h = quant::f32_to_fp16(0.002f); std::memcpy(&scl[i], &h, 2); }
        const std::vector<uint8_t> swz = quant::pack_tcc_deswizzled(nat.data(), V, lay);
        HeadPlanes Pn = head_planes_q8_dp4a(V, D);
        Pn.qs.stride_bytes = D;  // dense natural image of the same matrix
        HeadPlanes Ps = Pn;
        Ps.qs.xor_swizzled = true;
        std::string err;
        VJ_CHECK(validate_planes(Ps, &err));
        std::vector<uint32_t> rows;
        for (int i = 0; i < 48; ++i) rows.push_back(static_cast<uint32_t>(rng() % V));
        rows[0] = 5; rows[1] = 21; rows[2] = 0;  // same (row & 15) as the destination for none of these
        HeadPlanes Pd;
        std::vector<uint8_t> dq, ds;
        VJ_CHECK(slice_rows(Ps, swz.data(), scl.data(), rows, &Pd, &dq, &ds, &err));
        VJ_CHECK(Pd.rows == 48 && Pd.qs.xor_swizzled);
        const std::vector<uint8_t> back = quant::unpack_tcc_deswizzled(dq.data(), 48, lay);
        uint32_t naive_wrong = 0;
        for (uint32_t s = 0; s < 48; ++s) {
            VJ_CHECK(same_bytes(&back[static_cast<size_t>(s) * D], &nat[static_cast<size_t>(rows[s]) * D], D));
            if ((rows[s] & 15u) != (s & 15u) &&
                !same_bytes(&dq[static_cast<size_t>(s) * D], &swz[static_cast<size_t>(rows[s]) * D], D)) ++naive_wrong;
        }
        VJ_CHECK(naive_wrong > 0);  // a plain row memcpy would have scrambled these rows
        std::vector<float> x(D), ln(V), ls(V), ld(48);
        for (auto& v : x) v = static_cast<float>(static_cast<int>(rng() % 201u) - 100) * 0.01f;
        head_logits_host(Pn, nat.data(), scl.data(), x.data(), ln.data());
        head_logits_host(Ps, swz.data(), scl.data(), x.data(), ls.data());
        head_logits_host(Pd, dq.data(), ds.data(), x.data(), ld.data());
        for (uint32_t r = 0; r < V; ++r) VJ_CHECK(same_float(ln[r], ls[r]));
        for (uint32_t s = 0; s < 48; ++s) VJ_CHECK(same_float(ld[s], ln[rows[s]]));
        HeadPlanes odd = Ps;
        odd.qs.raw_row_bytes = odd.qs.stride_bytes = 3072u;
        odd.cols = 3072u;
        odd.scale.raw_row_bytes = odd.scale.stride_bytes = 192u;
        VJ_CHECK(!validate_planes(odd, &err));  // 12 tiles cannot be XOR swizzled
        std::printf("[fr_spec] XOR-swizzled slice: 48/48 rows exact after re-permutation (%u would be scrambled by a plain copy)\n",
                    naive_wrong);
        n += 2;
    }

    // --- frequency tables: parser, file loader, fallback ---
    {
        const std::string text =
            "# vegajet fr-spec frequency table v1\n# vocab 1000\n# tokens 26 distinct 4\n"
            "17 10\n3 10\n999 5\n42 1\n17 1\n\n";
        FrequencyTable t;
        std::string err;
        VJ_CHECK(parse_frequency_table(text, 1000, &t, &err));
        VJ_CHECK(t.from_file && (t.ranked == std::vector<int32_t>{17, 3, 999, 42}));  // 17 = 11, 3 = 10
        VJ_CHECK((t.counts == std::vector<double>{11.0, 10.0, 5.0, 1.0}));
        VJ_CHECK(parse_frequency_table("5\n2\n5\n9\n", 10, &t, &err) && (t.ranked == std::vector<int32_t>{5, 2, 9}) && t.counts.empty());
        VJ_CHECK(!parse_frequency_table("# vocab 999\n1 2\n", 1000, &t, &err) && err.find("vocab") != std::string::npos);
        VJ_CHECK(!parse_frequency_table("1 2\n3\n", 1000, &t, &err));          // mixed
        VJ_CHECK(!parse_frequency_table("1000 2\n", 1000, &t, &err));          // out of range
        VJ_CHECK(!parse_frequency_table("-1 2\n", 1000, &t, &err));
        VJ_CHECK(!parse_frequency_table("1 -2\n", 1000, &t, &err));
        VJ_CHECK(!parse_frequency_table("1 2 3\n", 1000, &t, &err));
        VJ_CHECK(!parse_frequency_table("# only comments\n", 1000, &t, &err));
        const auto sc = table_scores(FrequencyTable{}, 1000);  // empty table == pure prior
        VJ_CHECK(sc.size() == 1000 && sc[0] > sc[999]);
        VJ_CHECK(parse_frequency_table(text, 1000, &t, &err));
        const auto st = table_scores(t, 1000);
        std::vector<int32_t> top;
        VJ_CHECK(select_draft_vocab(st, 1000, 256, {}, &top, &err));
        for (int32_t id : {17, 3, 999, 42}) VJ_CHECK(std::binary_search(top.begin(), top.end(), id));
        VJ_CHECK(st[17] > st[3] && st[3] > st[999] && st[999] > st[42]);
        char path[] = "/tmp/vj_fr_spec_tableXXXXXX";
        const int fd = mkstemp(path);
        VJ_CHECK(fd >= 0);
        VJ_CHECK(write(fd, text.data(), text.size()) == static_cast<ssize_t>(text.size()));
        close(fd);
        FrequencyTable lf;
        VJ_CHECK(load_frequency_table(path, 1000, &lf, &err) && lf.ranked == t.ranked && lf.counts == t.counts);
        std::string note;
        const FrequencyTable wrong = load_frequency_table_or_fallback(path, 2000, &note);  // vocab mismatch -> fallback
        VJ_CHECK(!wrong.from_file && !note.empty() && wrong.ranked.size() == 2000);
        VJ_CHECK(unlink(path) == 0);
        const FrequencyTable gone = load_frequency_table_or_fallback(path, 1000, &note);
        VJ_CHECK(!gone.from_file && gone.source == "fallback:low-id-prefix" && !note.empty());
        const FrequencyTable none = load_frequency_table_or_fallback("", 1000, &note);
        VJ_CHECK(!none.from_file && note.empty());
        n += 3;
    }

    // --- lossless greedy speculation with the sliced head (ties included) ---
    {
        const uint32_t V = 4096, D = 256, K = 4;
        const HeadPlanes P = head_planes_q8_dp4a(V, D);
        std::vector<uint8_t> qs(P.qs_bytes(), 0), scl(P.scale_bytes());
        for (uint32_t r = 0; r < V; ++r)
            for (uint32_t c = 0; c < D; ++c) qs[static_cast<size_t>(r) * P.qs.stride_bytes + c] = static_cast<uint8_t>(rng() % 255u);
        for (uint32_t r = 0; r < V; ++r)
            for (uint32_t b = 0; b < D / 32u; ++b) {
                // Low ids get larger scales: a skewed (Zipf-like) token distribution, like a real LM head.
                const uint16_t h = quant::f32_to_fp16(0.004f + 0.02f / (1.0f + static_cast<float>(r) / 64.0f));
                std::memcpy(&scl[static_cast<size_t>(r) * P.scale.stride_bytes + b * 2u], &h, 2);
            }
        // Exact duplicates -> exact logit ties between a lower and a higher token id.
        for (int i = 0; i < 300; ++i) {
            const uint32_t a = rng() % V, b = rng() % V;
            if (a == b) continue;
            const uint32_t lo = std::min(a, b), hi = std::max(a, b);
            std::memcpy(&qs[static_cast<size_t>(hi) * P.qs.stride_bytes], &qs[static_cast<size_t>(lo) * P.qs.stride_bytes], P.qs.stride_bytes);
            std::memcpy(&scl[static_cast<size_t>(hi) * P.scale.stride_bytes], &scl[static_cast<size_t>(lo) * P.scale.stride_bytes], P.scale.stride_bytes);
        }
        auto target_next = [&](int32_t a, int32_t b) {
            const auto h = hidden(a, b, D, 0.0f);
            std::vector<float> lg(V);
            head_logits_host(P, qs.data(), scl.data(), h.data(), lg.data());
            return first_max(lg, nullptr);
        };
        const int N = 400;
        std::vector<int32_t> plain = {1, 2};
        while (plain.size() < static_cast<size_t>(N) + 2u) plain.push_back(target_next(plain[plain.size() - 2], plain.back()));

        // Frequency table from a corpus (the first half of the target's own output) in the tool's text format;
        // tokens that only show up later are missing from the draft vocab unless the prior picks them.
        std::vector<uint64_t> cnt(V, 0);
        for (size_t i = 0; i < plain.size() / 2u; ++i) ++cnt[static_cast<size_t>(plain[i])];
        std::string text = "# vegajet fr-spec frequency table v1\n# vocab 4096\n";
        for (uint32_t t = 0; t < V; ++t)
            if (cnt[t]) text += std::to_string(t) + " " + std::to_string(cnt[t]) + "\n";
        FrequencyTable ft;
        std::string err;
        VJ_CHECK(parse_frequency_table(text, V, &ft, &err));
        std::vector<int32_t> ids;
        VJ_CHECK(select_draft_vocab(table_scores(ft, V), V, 1000, {}, &ids, &err) && ids.size() == 1024);
        // Force a miss: drop the target's most frequent token of the second half from the draft vocab.
        {
            std::vector<uint32_t> late(V, 0);
            for (size_t i = plain.size() / 2u; i < plain.size(); ++i) ++late[static_cast<size_t>(plain[i])];
            const int32_t hot = static_cast<int32_t>(std::max_element(late.begin(), late.end()) - late.begin());
            int32_t cold = static_cast<int32_t>(V) - 1;
            while (cnt[static_cast<size_t>(cold)] || late[static_cast<size_t>(cold)] || std::binary_search(ids.begin(), ids.end(), cold)) --cold;
            auto it = std::lower_bound(ids.begin(), ids.end(), hot);
            if (it != ids.end() && *it == hot) *it = cold;
            else ids.back() = cold;  // already absent: keep the size, still a valid distinct id
            std::sort(ids.begin(), ids.end());
        }
        IdMap m;
        VJ_CHECK(make_id_map(ids, V, &m, &err));
        DraftHead dh;
        VJ_CHECK(slice_draft_head(P, qs.data(), scl.data(), m, &dh, &err));

        // Same hidden state: the draft pick is exactly the full-vocab first-max argmax restricted to the draft vocab.
        uint32_t ties_seen = 0;
        for (int it = 0; it < 200; ++it) {
            const auto h = hidden(static_cast<int32_t>(rng() % V), static_cast<int32_t>(rng() % V), D, 0.0f);
            std::vector<float> full(V), dl(1024);
            head_logits_host(P, qs.data(), scl.data(), h.data(), full.data());
            head_logits_host(dh.planes, dh.qs.data(), dh.scale.data(), h.data(), dl.data());
            const int32_t want = first_max(full, &m);
            VJ_CHECK(draft_argmax_full_id(dl.data(), m) == want);
            for (uint32_t s = 0; s < 1024; ++s)
                if (m.to_full(static_cast<int32_t>(s)) > want && same_float(dl[s], full[static_cast<size_t>(want)])) ++ties_seen;
        }

        // Greedy speculative loop: noisy drafter on the sliced head, verifier on the full head.
        std::vector<int32_t> out = {1, 2};
        uint64_t drafted = 0, accepted = 0, outside = 0;
        while (out.size() < static_cast<size_t>(N) + 2u) {
            std::vector<int32_t> ctx = out, draft;
            for (uint32_t k = 0; k < K; ++k) {
                const auto h = hidden(ctx[ctx.size() - 2], ctx.back(), D, 0.35f);
                std::vector<float> dl(1024);
                head_logits_host(dh.planes, dh.qs.data(), dh.scale.data(), h.data(), dl.data());
                const int32_t d = draft_argmax_full_id(dl.data(), m);
                VJ_CHECK(m.to_short(d) >= 0);
                draft.push_back(d);
                ctx.push_back(d);
            }
            drafted += K;
            for (uint32_t k = 0; k <= K && out.size() < static_cast<size_t>(N) + 2u; ++k) {
                const int32_t v = target_next(out[out.size() - 2], out.back());
                out.push_back(v);  // the verifier's token is always what gets committed
                if (k == K || draft[k] != v) { if (k < K && m.to_short(v) < 0) ++outside; break; }
                ++accepted;
            }
        }
        VJ_CHECK(out == plain);
        VJ_CHECK(accepted > 0 && outside > 0);
        std::printf("[fr_spec] lossless spec loop: %d tokens identical to plain greedy; %llu/%llu drafts accepted, "
                    "%llu rejections with the target token outside the 1024-row draft vocab; %u exact ties resolved to the lower id\n",
                    N, static_cast<unsigned long long>(accepted), static_cast<unsigned long long>(drafted),
                    static_cast<unsigned long long>(outside), ties_seen);
        n += 2;
    }
    return n;
}

} // namespace

int main() {
    const VocabSplit split{248320u, 2u};
    const ArgmaxLayout lay{240u, 4u};
    std::mt19937 rng(1234);
    int n_checks = 0;

    // --- tie key == today's kernel + host merge order (derivation check on a small split) ---
    {
        const VocabSplit sm{2u * 3000u, 2u};
        const ArgmaxLayout sl_lay{7u, 4u};
        for (int it = 0; it < 200; ++it) {
            std::vector<float> lg(sm.vocab);
            for (auto& v : lg) v = static_cast<float>(rng() % 3u);  // massive ties
            VJ_CHECK(full_pipeline(lg, sm, sl_lay) == brute_argmax(lg, sm, sl_lay, nullptr));
            ++n_checks;
        }
    }

    // --- invariants: multiple of 256, both balance modes, both sizes ---
    TokenFrequencyModel prior(split.vocab);
    {
        for (uint32_t rows : {32768u, 65536u}) {
            for (Balance bal : {Balance::kEqualPerRank, Balance::kGlobalTopN}) {
                ShortlistConfig cfg;
                cfg.rows = rows;
                cfg.balance = bal;
                Shortlist sl;
                std::string err;
                std::vector<double> sc = prior.scores();
                for (int i = 0; i < 3000; ++i) sc[130000u + rng() % 100000u] += 1e-3;  // skew mass to rank 1
                VJ_CHECK(build_shortlist(split, cfg, sc, {}, &sl, &err));
                check_structure(sl);
                if (bal == Balance::kEqualPerRank) VJ_CHECK(sl.rank_rows[0] == rows / 2 && sl.rank_rows[1] == rows / 2);
                ++n_checks;
            }
        }
        ShortlistConfig bad;
        bad.rows = 32768u + 100u;
        Shortlist sl;
        std::string err;
        VJ_CHECK(!build_shortlist(split, bad, prior.scores(), {}, &sl, &err) && !err.empty());
        bad.rows = 256u;  // multiple of 256 but not of 512 -> equal mode refuses, global mode accepts
        VJ_CHECK(!build_shortlist(split, bad, prior.scores(), {}, &sl, &err));
        bad.balance = Balance::kGlobalTopN;
        VJ_CHECK(build_shortlist(split, bad, prior.scores(), {}, &sl, &err));
        check_structure(sl);
        n_checks += 3;
    }

    // --- odd vocab: the unscored tail token never enters a shortlist ---
    {
        const VocabSplit odd{2u * 1024u + 1u, 2u};
        ShortlistConfig cfg;
        cfg.rows = 1024u;
        std::vector<double> sc(odd.vocab, 0.0);
        sc[odd.vocab - 1] = 100.0;
        Shortlist sl;
        std::string err;
        AlwaysInclude inc;
        inc.must = {static_cast<int32_t>(odd.vocab - 1)};
        VJ_CHECK(build_shortlist(odd, cfg, sc, inc, &sl, &err));
        VJ_CHECK(!sl.contains(static_cast<int32_t>(odd.vocab - 1)));
        check_structure(sl);
        ++n_checks;
    }

    // --- coverage, must / prefer sets, online counts ---
    {
        ShortlistConfig cfg;
        AlwaysInclude inc;
        inc.must = {248044, 248045, 248046, 248319};  // special tokens live at the top of the id range
        for (int i = 0; i < 500; ++i) inc.prefer.push_back(static_cast<int32_t>(150000u + i * 37u));
        Shortlist sl;
        std::string err;
        VJ_CHECK(build_shortlist(split, cfg, prior.scores(), inc, &sl, &err));
        check_structure(sl);
        for (int32_t t : inc.must) VJ_CHECK(sl.contains(t));
        for (int32_t t : inc.prefer) VJ_CHECK(sl.contains(t));
        VJ_CHECK(sl.n_must == 4 && sl.n_prefer_taken == 500 && sl.n_prefer_dropped == 0);
        const double cov = coverage(sl, prior);
        ShortlistConfig gcfg;
        gcfg.balance = Balance::kGlobalTopN;
        Shortlist slg;
        VJ_CHECK(build_shortlist(split, gcfg, prior.scores(), inc, &slg, &err));
        check_structure(slg);
        const double cov_g = coverage(slg, prior);
        std::printf("[fr_spec] prior-only coverage, 32768 rows: equal-per-rank %.3f, global top-N %.3f (ranks %u/%u)\n",
                    cov, cov_g, slg.rank_rows[0], slg.rank_rows[1]);
        VJ_CHECK(cov > 0.70 && cov_g >= cov);

        // Prefer budget: 25% of a 16384 quota = 4096 per rank.
        AlwaysInclude big;
        for (int i = 0; i < 10000; ++i) big.prefer.push_back(static_cast<int32_t>(200000u + i));
        VJ_CHECK(build_shortlist(split, cfg, prior.scores(), big, &sl, &err));
        VJ_CHECK(sl.n_prefer_taken == 4096u && sl.n_prefer_dropped == 10000u - 4096u);
        VJ_CHECK(sl.contains(200000) && sl.contains(200000 + 4095) && !sl.contains(200000 + 4096) && !sl.contains(200000 + 9999));  // priority order respected

        // Must overflow fails loudly.
        ShortlistConfig tiny;
        tiny.rows = 512u;
        AlwaysInclude over;
        for (int i = 0; i < 300; ++i) over.must.push_back(i);
        VJ_CHECK(!build_shortlist(split, tiny, prior.scores(), over, &sl, &err));

        // Online manager: text drifts to high ids -> misses -> rebuild -> covered.
        ManagerConfig mc;
        mc.rebuild_min_tokens = 1000;
        ShortlistManager mgr(split, mc);
        mgr.set_must({248044});
        std::vector<int32_t> prompt;
        for (int i = 0; i < 200; ++i) prompt.push_back(static_cast<int32_t>(rng() % 20000u));
        VJ_CHECK(mgr.init(prompt, &err));
        check_structure(mgr.current());
        const uint64_t gen0 = mgr.current().generation;
        std::vector<int32_t> hot;
        for (int i = 0; i < 64; ++i) hot.push_back(static_cast<int32_t>(200000u + i * 101u));
        std::vector<int32_t> ctx = prompt, stream;
        for (int i = 0; i < 1200; ++i) stream.push_back(hot[rng() % hot.size()]);
        for (int32_t h : hot) VJ_CHECK(!mgr.current().contains(h));
        mgr.observe_verified(stream);
        ctx.insert(ctx.end(), stream.begin(), stream.end());
        VJ_CHECK(mgr.window_miss_rate() > 0.9 && mgr.should_rebuild());
        VJ_CHECK(mgr.rebuild(ctx, &err));
        VJ_CHECK(mgr.current().generation == gen0 + 1u);
        for (int32_t h : hot) VJ_CHECK(mgr.current().contains(h));
        VJ_CHECK(mgr.current().contains(248044));
        VJ_CHECK(!mgr.should_rebuild());
        n_checks += 4;
    }

    // --- determinism (including heavy score ties -> lower id wins) ---
    {
        ShortlistConfig cfg;
        std::vector<double> sc(split.vocab, 1.0);
        Shortlist a, b;
        std::string err;
        VJ_CHECK(build_shortlist(split, cfg, sc, {}, &a, &err));
        VJ_CHECK(build_shortlist(split, cfg, sc, {}, &b, &err));
        VJ_CHECK(a.row_to_token == b.row_to_token && a.device_local_rows == b.device_local_rows);
        VJ_CHECK(a.sorted_local_rows[0].back() == 16383u && a.sorted_local_rows[1].back() == 16383u);
        std::vector<double> sc2 = prior.scores();
        VJ_CHECK(build_shortlist(split, cfg, sc2, {}, &a, &err));
        VJ_CHECK(build_shortlist(split, cfg, sc2, {}, &b, &err));
        VJ_CHECK(a.row_to_token == b.row_to_token);
        n_checks += 2;
    }

    // --- exact tie rule on the real shape: shortlist pick == full argmax restricted to the shortlist ---
    {
        for (uint32_t draft_grid : {240u, 60u, 64u}) {
            ShortlistConfig cfg;
            cfg.draft_layout.grid_blocks = draft_grid;
            std::vector<double> sc = prior.scores();
            for (int i = 0; i < 5000; ++i) sc[rng() % split.vocab] += 1e-4;
            Shortlist sl;
            std::string err;
            VJ_CHECK(build_shortlist(split, cfg, sc, {}, &sl, &err));
            int in_hits = 0, mismatch_without_layout = 0;
            ShortlistConfig cfg_plain = cfg;
            cfg_plain.tie_exact_layout = false;
            Shortlist slp;
            VJ_CHECK(build_shortlist(split, cfg_plain, sc, {}, &slp, &err));
            for (int it = 0; it < 12; ++it) {
                std::vector<float> lg(split.vocab);
                for (auto& v : lg) v = static_cast<float>(rng() % 4u);  // thousands of tied maxima
                const int32_t full = full_pipeline(lg, split, lay);
                VJ_CHECK(full == brute_argmax(lg, split, lay, nullptr));
                const int32_t want = brute_argmax(lg, split, lay, &sl);
                const ArgmaxPick got = shortlist_pipeline(lg, sl);
                VJ_CHECK(got.token == want);
                VJ_CHECK(got.value == lg[want]);
                if (sl.contains(full)) { VJ_CHECK(got.token == full); ++in_hits; }
                if (shortlist_pipeline(lg, slp).token != brute_argmax(lg, split, lay, &slp)) ++mismatch_without_layout;
                ++n_checks;
            }
            std::printf("[fr_spec] draft grid %u: exact tie match 12/12 (full argmax in shortlist %d/12); "
                        "plain sorted layout mismatches %d/12\n", draft_grid, in_hits, mismatch_without_layout);
        }
    }

    // --- true argmax outside the shortlist: draft is wrong, verification rejects it ---
    {
        ShortlistConfig cfg;
        Shortlist sl;
        std::string err;
        VJ_CHECK(build_shortlist(split, cfg, prior.scores(), {}, &sl, &err));
        std::vector<float> lg(split.vocab);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto& v : lg) v = nd(rng);
        const int32_t outside = 230000;
        VJ_CHECK(!sl.contains(outside));
        lg[outside] = 50.0f;
        const int32_t verified = full_pipeline(lg, split, lay);
        const ArgmaxPick draft = shortlist_pipeline(lg, sl);
        VJ_CHECK(verified == outside);
        VJ_CHECK(draft.token >= 0 && sl.contains(draft.token) && draft.token != verified);
        VJ_CHECK(draft.token == brute_argmax(lg, split, lay, &sl));
        // Greedy verification: accepted prefix stops at the first mismatch, the bonus token is the verifier's.
        const bool accepted = draft.token == verified;
        VJ_CHECK(!accepted);
        n_checks += 1;
    }

    // --- gather plan: device sub-matrix rows equal the selected source rows ---
    {
        const VocabSplit sm{2u * 4096u, 2u};
        ShortlistConfig cfg;
        cfg.rows = 1024u;
        cfg.draft_layout.grid_blocks = 8u;
        cfg.full_layout.grid_blocks = 8u;
        std::vector<double> sc(sm.vocab);
        for (uint32_t t = 0; t < sm.vocab; ++t) sc[t] = 1.0 / (t + 1.0);
        for (int i = 0; i < 100; ++i) sc[rng() % sm.vocab] = 10.0;
        for (bool exact : {true, false}) {
            cfg.tie_exact_layout = exact;
            Shortlist sl;
            std::string err;
            VJ_CHECK(build_shortlist(sm, cfg, sc, {}, &sl, &err));
            RowPlanes pl{48u, 6u, 8u};  // padded qs stride, 3 fp16 scales per row in an 8-byte-strided plane
            auto plan = make_gather_plan(sl, pl);
            VJ_CHECK(plan.size() == 2);
            for (uint32_t r = 0; r < 2; ++r) {
                const uint32_t per = sm.rows_per_rank();
                std::vector<uint8_t> sq(per * pl.qs_row_stride_bytes), ss(per * pl.scale_row_stride);
                for (size_t i = 0; i < sq.size(); ++i) sq[i] = static_cast<uint8_t>((i * 131u + r) ^ (i >> 7));
                for (size_t i = 0; i < ss.size(); ++i) ss[i] = static_cast<uint8_t>(i * 17u + 3u * r);
                std::vector<uint8_t> dq(plan[r].dst_qs_bytes, 0xEE), ds(plan[r].dst_scale_bytes, 0xEE);
                apply_gather_host(plan[r], sq.data(), ss.data(), dq.data(), ds.data());
                VJ_CHECK(dq.size() == sl.rank_rows[r] * pl.qs_row_stride_bytes && ds.size() == sl.rank_rows[r] * 6u);
                for (uint32_t p = 0; p < sl.rank_rows[r]; ++p) {
                    const uint32_t src = sl.device_local_rows[r][p];
                    VJ_CHECK(std::equal(dq.begin() + p * 48u, dq.begin() + (p + 1) * 48u, sq.begin() + src * 48u));
                    VJ_CHECK(std::equal(ds.begin() + p * 6u, ds.begin() + (p + 1) * 6u, ss.begin() + src * 8u));
                }
                if (!exact) VJ_CHECK(plan[r].qs.size() < sl.rank_rows[r] / 2);  // sorted rows coalesce into runs
            }
            ++n_checks;
        }
    }

    n_checks += vocab_head_checks(rng);

    std::printf("test_fr_spec: all %d checks passed\n", n_checks);
    return 0;
}

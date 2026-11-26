// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/spec_replay.hpp"

#include "core/suffix_trie.hpp"

#include <algorithm>
#include <cstdio>
#include <numeric>

namespace vegajet::spec_replay {

namespace {

bool is_pow2(uint32_t v) { return v != 0u && (v & (v - 1u)) == 0u; }

uint32_t next_pow2(uint32_t v) {
    uint32_t p = 1u;
    while (p < v && p < (1u << 31)) p <<= 1;
    return p;
}

uint64_t mix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// A token that is never `tok` (deterministic in (tok, salt)).
int32_t off_truth(int32_t tok, uint64_t salt, uint32_t vocab) {
    const uint32_t v = std::max<uint32_t>(vocab, 2u);
    const uint32_t d = 1u + static_cast<uint32_t>(mix64(salt ^ static_cast<uint64_t>(static_cast<uint32_t>(tok))) % (v - 1u));
    return static_cast<int32_t>((static_cast<uint32_t>(tok) + d) % v);
}

std::string fmt(const char* f, long long a, long long b = 0, long long c = 0, long long d = 0) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), f, a, b, c, d);
    return buf;
}

}  // namespace

const char* layer_kind_name(LayerKind k) { return k == LayerKind::kFull ? "full" : "swa"; }

const char* source_name(Source s) {
    switch (s) {
        case Source::kDFlash2: return "dflash2";
        case Source::kMtpFrSpec: return "mtp+fr-spec";
        case Source::kMtp: return "mtp";
        case Source::kSuffixOnly: return "suffix-only";
        case Source::kRaw: return "raw";
    }
    return "?";
}

const char* field_name(Field f) {
    switch (f) {
        case Field::kNone: return "none";
        case Field::kTokenId: return "token-id";
        case Field::kPosition: return "position";
        case Field::kShortlist: return "shortlist-mapping";
        case Field::kCache: return "cache-state";
        case Field::kMask: return "block-mask";
        case Field::kLength: return "trace-length";
    }
    return "?";
}

// ---- drafter KV plan -------------------------------------------------------------------------------------------
static DraftKvPlan finish_plan(DraftKvPlan p, const DraftKvConfig& cfg) {
    p.total_bytes = 0;
    p.max_anchor_pos = UINT32_MAX;
    uint32_t n_full = 0;
    for (const auto& l : p.layers) {
        p.total_bytes += l.bytes;
        if (l.kind == LayerKind::kFull) {
            ++n_full;
            const uint32_t lim = (l.slots >= cfg.block_len) ? (l.slots - cfg.block_len) : 0u;
            p.max_anchor_pos = std::min(p.max_anchor_pos, lim);
        }
    }
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%u layers (%u full, %u swa), pattern %s, %.1f MiB, max anchor %s",
                  cfg.n_layer, n_full, cfg.n_layer - n_full, p.pattern_known ? "known" : "unknown",
                  static_cast<double>(p.total_bytes) / 1048576.0,
                  p.max_anchor_pos == UINT32_MAX ? "unbounded" : std::to_string(p.max_anchor_pos).c_str());
    p.summary = buf;
    p.ok = true;
    return p;
}

static bool check_cfg(const DraftKvConfig& cfg, DraftKvPlan* p) {
    if (cfg.n_layer == 0u) { p->error = "n_layer = 0"; return false; }
    if (!is_pow2(cfg.swa_window)) { p->error = "sliding window must be a power of two (kernel ring mask)"; return false; }
    if (cfg.block_len < 2u || cfg.block_len > cfg.swa_window) { p->error = "block length out of range"; return false; }
    if (!cfg.swa_pattern.empty() && cfg.swa_pattern.size() != cfg.n_layer) {
        p->error = "swa_pattern has " + std::to_string(cfg.swa_pattern.size()) + " entries, need " + std::to_string(cfg.n_layer);
        return false;
    }
    return true;
}

DraftKvPlan plan_draft_kv(const DraftKvConfig& cfg) {
    DraftKvPlan p;
    if (!check_cfg(cfg, &p)) return p;
    p.pattern_known = !cfg.swa_pattern.empty();
    const uint32_t full_need = std::max(cfg.swa_window, next_pow2(std::min(std::max(cfg.max_ctx, 1u), std::max(cfg.full_ctx_cap, 1u))));
    for (uint32_t l = 0; l < cfg.n_layer; ++l) {
        DraftKvLayer L;
        L.kind = (p.pattern_known && cfg.swa_pattern[l] == 0u) ? LayerKind::kFull : LayerKind::kSwa;
        L.slots = (L.kind == LayerKind::kFull) ? full_need : cfg.swa_window;
        L.kernel_win = L.slots;
        L.ring_mask = L.slots - 1u;
        L.bytes = 2ULL * L.slots * cfg.n_kv * cfg.head_dim * 2ULL;
        p.layers.push_back(L);
    }
    return finish_plan(std::move(p), cfg);
}

DraftKvPlan legacy_draft_kv(const DraftKvConfig& cfg) {
    DraftKvPlan p;
    if (!check_cfg(cfg, &p)) return p;
    p.pattern_known = !cfg.swa_pattern.empty();
    for (uint32_t l = 0; l < cfg.n_layer; ++l) {
        DraftKvLayer L;
        L.kind = (p.pattern_known && cfg.swa_pattern[l] == 0u) ? LayerKind::kFull : LayerKind::kSwa;
        L.slots = cfg.swa_window;
        L.kernel_win = cfg.swa_window;
        L.ring_mask = cfg.swa_window - 1u;
        L.bytes = 2ULL * L.slots * cfg.n_kv * cfg.head_dim * 2ULL;
        p.layers.push_back(L);
    }
    return finish_plan(std::move(p), cfg);
}

// ---- source policy ----------------------------------------------------------------------------------------------
SourcePlan plan_sources(const SourceConfig& cfg, uint32_t prompt_len, uint32_t max_new) {
    SourcePlan p;
    const Source mtp_src = cfg.has_mtp ? (cfg.fr_spec_rows > 0u ? Source::kMtpFrSpec : Source::kMtp) : Source::kSuffixOnly;
    if (!cfg.has_dflash2) {
        p.initial = p.after_switch = mtp_src;
        p.mtp_kv_from_prefill = cfg.has_mtp;
        return p;
    }
    // DFlash2 drafts with the anchor at position pos; it stays exact while pos <= limit.
    const uint32_t limit = std::min(cfg.dflash_max_anchor, cfg.long_ctx);
    const uint64_t last_anchor = static_cast<uint64_t>(prompt_len) + (max_new > 0u ? max_new - 1u : 0u);
    if (prompt_len > limit) {
        p.initial = p.after_switch = mtp_src;
        p.mtp_kv_from_prefill = cfg.has_mtp;
        return p;
    }
    p.initial = Source::kDFlash2;
    p.after_switch = Source::kDFlash2;
    if (last_anchor > limit) {
        p.after_switch = mtp_src;
        p.switch_at_pos = limit + 1u;
        p.mtp_kv_from_prefill = cfg.has_mtp;  // the MTP cache must already hold every committed position
    }
    return p;
}

Source source_at(const SourcePlan& p, uint32_t anchor_pos) {
    return anchor_pos >= p.switch_at_pos ? p.after_switch : p.initial;
}

// ---- FR-Spec shortlist ------------------------------------------------------------------------------------------
bool build_shortlist(const std::vector<double>& scores, uint32_t rows, uint32_t n_ranks, uint32_t pinned_tail,
                     Shortlist* out, std::string* err) {
    const uint32_t vocab = static_cast<uint32_t>(scores.size());
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };
    if (n_ranks == 0u) return fail("n_ranks = 0");
    if (rows == 0u || rows > vocab) return fail("rows must be in 1..vocab");
    if (rows % (256u * n_ranks) != 0u) return fail("rows must be a multiple of 256 per rank");
    std::vector<uint8_t> picked(vocab, 0);
    std::vector<int32_t> selected;
    selected.reserve(rows);
    const uint32_t tail_start = vocab > pinned_tail ? vocab - pinned_tail : 0u;
    for (uint32_t t = tail_start; t < vocab && selected.size() < rows; ++t) { picked[t] = 1; selected.push_back(static_cast<int32_t>(t)); }
    std::vector<int32_t> rest;
    rest.reserve(vocab);
    for (uint32_t i = 0; i < vocab; ++i) if (!picked[i]) rest.push_back(static_cast<int32_t>(i));
    const size_t need = rows - selected.size();
    // Same comparator as the engine plus an id tie-break, so equal scores give one deterministic order.
    std::partial_sort(rest.begin(), rest.begin() + static_cast<std::ptrdiff_t>(need), rest.end(), [&](int32_t a, int32_t b) {
        return scores[static_cast<size_t>(a)] != scores[static_cast<size_t>(b)] ? scores[static_cast<size_t>(a)] > scores[static_cast<size_t>(b)] : a < b;
    });
    for (size_t i = 0; i < need; ++i) selected.push_back(rest[i]);
    Shortlist s;
    s.rows = rows; s.n_ranks = n_ranks; s.rows_per_rank = rows / n_ranks;
    s.tok_map.assign(n_ranks, {});
    s.rank_of.assign(vocab, -1);
    s.row_of.assign(vocab, -1);
    for (uint32_t i = 0; i < rows; ++i) {
        const uint32_t r = i % n_ranks;
        const int32_t tok = selected[i];
        s.rank_of[static_cast<size_t>(tok)] = static_cast<int32_t>(r);
        s.row_of[static_cast<size_t>(tok)] = static_cast<int32_t>(s.tok_map[r].size());
        s.tok_map[r].push_back(tok);
    }
    *out = std::move(s);
    return true;
}

bool check_shortlist(const Shortlist& s, uint32_t vocab, uint32_t pinned_tail, std::string* err) {
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };
    if (s.rows_per_rank % 256u != 0u) return fail("rows per rank not a multiple of 256");
    if (s.tok_map.size() != s.n_ranks) return fail("rank count mismatch");
    std::vector<uint8_t> seen(vocab, 0);
    for (uint32_t r = 0; r < s.n_ranks; ++r) {
        if (s.tok_map[r].size() != s.rows_per_rank) return fail("rank " + std::to_string(r) + " holds " + std::to_string(s.tok_map[r].size()) + " rows");
        for (uint32_t row = 0; row < s.tok_map[r].size(); ++row) {
            const int32_t t = s.tok_map[r][row];
            if (t < 0 || static_cast<uint32_t>(t) >= vocab) return fail("id out of range");
            if (seen[static_cast<size_t>(t)]) return fail("duplicate id " + std::to_string(t));
            seen[static_cast<size_t>(t)] = 1;
            if (s.rank_of[static_cast<size_t>(t)] != static_cast<int32_t>(r) || s.row_of[static_cast<size_t>(t)] != static_cast<int32_t>(row))
                return fail("inverse map mismatch at id " + std::to_string(t));
        }
    }
    const uint32_t tail = std::min(pinned_tail, s.rows);
    for (uint32_t t = vocab - std::min(tail, vocab); t < vocab; ++t)
        if (!seen[t]) return fail("pinned tail id " + std::to_string(t) + " missing");
    return true;
}

// ---- replay -----------------------------------------------------------------------------------------------------
namespace {

struct Slot { int64_t pos = -1; int32_t tok = -1; };

struct Sim {
    const Script& s;
    const ReplayConfig& cfg;
    bool reference;
    std::vector<std::vector<Slot>> cache;  // engine: [layer][slot]
    Trace tr;

    Sim(const Script& sc, const ReplayConfig& c, bool ref) : s(sc), cfg(c), reference(ref) {
        if (!reference)
            for (const auto& l : cfg.kv.layers) cache.emplace_back(l.slots);
    }

    int32_t truth_at(uint64_t p) const { return p < s.truth.size() ? s.truth[p] : -1; }

    // Target greedy prediction after the token at position p, given that positions [0, p] hold `prefix_ok`
    // ? truth : something else.
    int32_t target_pred(uint32_t p, bool prefix_ok) const {
        const int32_t t = truth_at(static_cast<uint64_t>(p) + 1u);
        return prefix_ok ? t : off_truth(t, 0x7A11ULL + p, s.vocab);
    }

    void inject(uint32_t pos_beg, const std::vector<int32_t>& toks, CycleRow* row) {
        for (size_t m = 0; m < toks.size(); ++m) {
            const uint32_t p = pos_beg + static_cast<uint32_t>(m);
            if (row) { row->inject_pos.push_back(p); row->inject_tok.push_back(toks[m]); }
            if (reference) continue;
            for (size_t l = 0; l < cache.size(); ++l) {
                auto& sl = cache[l][p & cfg.kv.layers[l].ring_mask];
                sl.pos = p; sl.tok = toks[m];
            }
        }
    }

    // Key set check for one drafter layer / query slot. Returns the first reference key the engine missed or read
    // with the wrong token (-1 when identical); fills counts.
    int64_t view_check(size_t l, uint32_t p, uint32_t i, uint32_t* expected, uint32_t* seen) const {
        const auto& L = cfg.kv.layers[l];
        const uint32_t q = p + i;
        // z-lab reference window: SWA layers see q - k < W (W = the configured sliding window), full layers all.
        uint32_t lo_r = 0u;
        if (L.kind == LayerKind::kSwa) {
            const uint32_t W = L.slots;  // an SWA layer's ring is exactly its window in both plans
            lo_r = (q >= W) ? q - W + 1u : 0u;
        }
        const uint32_t win = L.kernel_win;
        const uint32_t lo_e = (q >= win) ? q - win + 1u : 0u;
        *expected = p > lo_r ? p - lo_r : 0u;
        *seen = p > lo_e ? p - lo_e : 0u;
        int64_t first = -1;
        if (lo_e > lo_r) first = lo_r;
        const uint32_t from = std::max(lo_e, lo_r);
        for (uint32_t j = from; j < p && first < 0; ++j) {
            const Slot& sl = cache[l][j & L.ring_mask];
            if (sl.pos != static_cast<int64_t>(j) || sl.tok != truth_at(j)) first = j;
        }
        return first;
    }

    std::vector<int32_t> neural_drafts(Source src, uint32_t p, uint32_t k, uint64_t cycle, CycleRow* row) {
        uint32_t good = k;
        if (!s.accept_plan.empty()) good = std::min<uint32_t>(k, s.accept_plan[cycle % s.accept_plan.size()]);
        // Drafter view (DFlash2 only: the MTP head runs over the full-length target-position cache).
        if (src == Source::kDFlash2 && !reference) {
            const uint32_t B = cfg.k_draft + 1u;
            int bad_slot = -1;
            for (size_t l = 0; l < cache.size(); ++l) {
                for (uint32_t i = 0; i < B; ++i) {
                    uint32_t e = 0, sn = 0;
                    const int64_t f = view_check(l, p, i, &e, &sn);
                    if (f >= 0) {
                        LayerView v; v.layer = static_cast<uint32_t>(l); v.slot = i; v.keys_expected = e; v.keys_seen = sn; v.first_missing = f;
                        row->view_diffs.push_back(v);
                        if (bad_slot < 0 || static_cast<int>(i) < bad_slot) bad_slot = static_cast<int>(i);
                    }
                }
                const bool want_causal = cfg.kv.layers[l].kind == LayerKind::kSwa;
                const bool is_causal = l < cfg.causal_layer.size() ? cfg.causal_layer[l] != 0u : want_causal;
                if (is_causal != want_causal) row->mask_diff_layers.push_back(static_cast<uint32_t>(l));
            }
            // Draft t comes from block slot t + 1; a wrong view at slot i reaches every slot through the
            // bidirectional full layer, so everything from slot max(i - 1, 0) on is lost.
            if (bad_slot >= 0) good = std::min<uint32_t>(good, bad_slot > 0 ? static_cast<uint32_t>(bad_slot - 1) : 0u);
        }
        std::vector<int32_t> d;
        for (uint32_t t = 0; t < k; ++t) {
            const int32_t want = truth_at(static_cast<uint64_t>(p) + 1u + t);
            int32_t tok = (t < good) ? want : off_truth(want, 0xD8AFULL + cycle * 131u + t, s.vocab);
            int32_t row_code = -1;
            const bool fr = cfg.shortlist && (src == Source::kDFlash2 || src == Source::kMtpFrSpec);
            if (fr) {
                const Shortlist& sl = *cfg.shortlist;
                // The drafter can only emit shortlist rows: an off-list token becomes the off-list fallback row.
                int32_t intended = tok;
                if (!sl.contains(intended)) {
                    const uint64_t h = mix64(static_cast<uint64_t>(cycle) * 977u + t);
                    const uint32_t r = static_cast<uint32_t>(h % sl.n_ranks);
                    intended = sl.tok_map[r][static_cast<size_t>((h >> 8) % sl.tok_map[r].size())];
                    if (intended == want) intended = sl.tok_map[r][0] == want ? sl.tok_map[r][1] : sl.tok_map[r][0];
                }
                const int32_t rk = sl.rank_of[static_cast<size_t>(intended)], rw = sl.row_of[static_cast<size_t>(intended)];
                row_code = (rk << 24) | rw;
                // Engine: the kernel writes tok_map[rank][row]; reference: the intended vocabulary id.
                tok = reference ? intended : sl.tok_map[static_cast<size_t>(rk)][static_cast<size_t>(rw)];
            }
            if (row) row->shortlist_rows.push_back(row_code);
            d.push_back(tok);
            // p_min gate: the engine stops early but always keeps at least one draft.
            if (cfg.pmin_stop_at > 0u && d.size() >= cfg.pmin_stop_at) break;
        }
        return d;
    }

    void run() {
        const uint32_t B = cfg.k_draft + 1u;
        if (s.prompt_len == 0u || s.truth.size() < static_cast<size_t>(s.prompt_len) + cfg.n_gen + B) {
            tr.error = "script truth shorter than prompt + n_gen + B";
            return;
        }
        if (!reference && cfg.kv.layers.empty()) { tr.error = "engine replay needs a drafter KV plan"; return; }
        sfx::CascadeConfig ccfg;
        ccfg.min_match = 4u; ccfg.min_chain = 4u;
        ccfg.max_draft = std::min<uint32_t>(cfg.suffix_max_draft, cfg.ring_slots - 2u);
        sfx::CascadeHybrid cascade(ccfg);

        // GDN state bookkeeping, mirrored from the engine: ring of `ring_slots` states (or two --spec-gdn slots),
        // each holding "tokens absorbed".
        std::vector<int64_t> ring(cfg.ring_slots, -1);
        uint32_t ring_head = 0u;
        int64_t spec_slot[2] = {0, -1};
        uint32_t spec_active = 0u;
        ring[0] = 0;

        uint32_t pos = 0u;
        std::vector<int32_t> prompt(s.truth.begin(), s.truth.begin() + s.prompt_len);
        for (uint32_t off = 0; off < s.prompt_len; off += cfg.prefill_chunk) {
            const uint32_t cnt = std::min(cfg.prefill_chunk, s.prompt_len - off);
            std::vector<int32_t> chunk(prompt.begin() + off, prompt.begin() + off + cnt);
            inject(pos, chunk, nullptr);
            pos += cnt;
            if (cfg.spec_gdn) spec_slot[spec_active] = pos; else ring[ring_head] = pos;
        }
        if (cfg.hybrid_suffix) cascade.commit(prompt);
        std::vector<int32_t> gen{target_pred(pos - 1u, true)};
        if (cfg.hybrid_suffix) cascade.commit(gen.back());
        uint64_t neural_idx = 0;
        uint32_t cyc = 0;
        auto gdn_pos = [&]() -> int64_t { return cfg.spec_gdn ? spec_slot[spec_active] : ring[ring_head]; };

        while (gen.size() < cfg.n_gen) {
            const uint32_t rem = cfg.n_gen - static_cast<uint32_t>(gen.size());
            const Source src = source_at(cfg.sources, pos);
            const uint32_t k_cur = std::min<uint32_t>({cfg.k_draft, B - 1u, cfg.ring_slots - 2u, rem > 1u ? rem - 1u : 0u});
            const uint32_t k_sfx = std::min<uint32_t>({cfg.suffix_max_draft, cfg.ring_slots - 2u, rem > 1u ? rem - 1u : 0u});
            const bool committed_ok = (gen.back() == truth_at(pos));
            CycleRow row;
            row.cycle = cyc++;
            row.source = src;
            row.anchor_pos = pos;
            row.anchor_tok = gen.back();
            if (gdn_pos() != static_cast<int64_t>(pos)) {
                tr.error = fmt("cycle %lld: GDN state holds %lld tokens, anchor at %lld", row.cycle, gdn_pos(), pos);
                return;
            }
            if (k_cur == 0u || src == Source::kRaw) {
                const int32_t pred = target_pred(pos, committed_ok);
                inject(pos, {gen.back()}, &row);
                if (cfg.spec_gdn) spec_slot[spec_active] = pos + 1; else ring[ring_head] = pos + 1;  // in-place M=1 update
                ++pos;
                gen.push_back(pred);
                if (cfg.hybrid_suffix) cascade.commit(pred);
                row.preds = {pred}; row.bonus = pred; row.n_commit = 1; row.gdn_state_pos = pos;
                tr.rows.push_back(row);
                continue;
            }
            std::vector<int32_t> drafts;
            if (cfg.hybrid_suffix && k_sfx > 0u) {
                const auto dec = cascade.decide_and_count();
                if (dec.source == sfx::DraftSource::kSuffix && !dec.draft.empty()) {
                    drafts.assign(dec.draft.begin(), dec.draft.begin() + std::min<size_t>(dec.draft.size(), k_sfx));
                    row.suffix = true;
                }
            }
            if (!row.suffix) {
                drafts = neural_drafts(src, pos, k_cur, neural_idx++, &row);
                ++tr.neural_cycles;
            } else {
                ++tr.suffix_cycles;
            }
            const uint32_t dk = static_cast<uint32_t>(drafts.size());
            std::vector<int32_t> vtoks{gen.back()};
            vtoks.insert(vtoks.end(), drafts.begin(), drafts.end());
            // Verify: preds[i] is the target's greedy token after vtoks[0..i] at positions pos..pos+i.
            std::vector<int32_t> preds;
            bool ok = committed_ok;
            for (uint32_t i = 0; i <= dk; ++i) {
                ok = ok && vtoks[i] == truth_at(static_cast<uint64_t>(pos) + i);
                preds.push_back(target_pred(pos + i, ok));
            }
            // GDN: the verify pass writes the state after each verified token into the next ring slots / other slot.
            if (dk + 1u > cfg.ring_slots - 1u) { tr.error = "verify batch overruns the GDN ring"; return; }
            for (uint32_t i = 0; i <= dk; ++i) ring[(ring_head + 1u + i) % cfg.ring_slots] = static_cast<int64_t>(pos) + 1 + i;
            spec_slot[1u - spec_active] = static_cast<int64_t>(pos) + dk + 1;
            uint32_t acc = 0;
            while (acc < dk && preds[acc] == drafts[acc]) { gen.push_back(drafts[acc]); if (cfg.hybrid_suffix) cascade.commit(drafts[acc]); ++acc; }
            if (gen.size() < cfg.n_gen) { gen.push_back(preds[acc]); if (cfg.hybrid_suffix) cascade.commit(preds[acc]); row.bonus = preds[acc]; }
            const uint32_t n_commit = acc + 1u;
            // commit_spec_gdn(acc, dk) / ring_head advance
            if (cfg.spec_gdn) {
                if (acc == dk) spec_active ^= 1u; else spec_slot[spec_active] = static_cast<int64_t>(pos) + n_commit;
            } else {
                // The advanced head slot holds the state the verify pass wrote after vtoks[acc]; the next cycle's
                // gdn_pos() check proves it.
                ring_head = (ring_head + n_commit) % cfg.ring_slots;
            }
            inject(pos, std::vector<int32_t>(vtoks.begin(), vtoks.begin() + n_commit), &row);
            tr.drafted += dk; tr.accepted += acc;
            if (!row.suffix) tr.neural_accepted += acc;
            row.k = dk; row.drafts = drafts; row.preds = preds; row.accepted = acc; row.n_commit = n_commit;
            pos += n_commit;
            row.gdn_state_pos = static_cast<uint32_t>(gdn_pos());
            tr.rows.push_back(std::move(row));
        }
        tr.output = gen;
        // Lossless check: the committed output equals the target's greedy continuation.
        for (size_t i = 0; i < gen.size(); ++i)
            if (gen[i] != truth_at(s.prompt_len + i)) { tr.error = fmt("output token %lld differs from the greedy continuation", static_cast<long long>(i)); return; }
    }
};

}  // namespace

Trace replay_engine(const Script& s, const ReplayConfig& cfg) { Sim sim(s, cfg, false); sim.run(); return sim.tr; }
Trace replay_reference(const Script& s, const ReplayConfig& cfg) { Sim sim(s, cfg, true); sim.run(); return sim.tr; }

Divergence first_divergence(const Trace& e, const Trace& r) {
    Divergence d;
    const size_t n = std::min(e.rows.size(), r.rows.size());
    auto vec_diff = [](const std::vector<int32_t>& a, const std::vector<int32_t>& b) -> int {
        const size_t m = std::min(a.size(), b.size());
        for (size_t i = 0; i < m; ++i) if (a[i] != b[i]) return static_cast<int>(i);
        return a.size() == b.size() ? -1 : static_cast<int>(m);
    };
    for (size_t c = 0; c < n; ++c) {
        const CycleRow& a = e.rows[c];
        const CycleRow& b = r.rows[c];
        d.cycle = static_cast<int>(c);
        // Pipeline order: the cycle's inputs (anchor), the drafter's cache view, its block mask, the shortlist
        // row -> id mapping, the drafted ids, the verify result, and the commit (positions written back).
        if (a.anchor_pos != b.anchor_pos) { d.field = Field::kPosition; d.detail = fmt("anchor position %lld vs reference %lld", a.anchor_pos, b.anchor_pos); return d; }
        if (a.anchor_tok != b.anchor_tok) { d.field = Field::kTokenId; d.detail = fmt("anchor token %lld vs reference %lld at position %lld", a.anchor_tok, b.anchor_tok, a.anchor_pos); return d; }
        if (!a.view_diffs.empty()) {
            const LayerView& v = a.view_diffs.front();
            d.field = Field::kCache; d.slot = static_cast<int>(v.slot);
            d.detail = fmt("drafter layer %lld query slot %lld: key position %lld missing or stale (%lld keys seen", v.layer, v.slot, v.first_missing, v.keys_seen) +
                       fmt(", reference %lld) at anchor %lld", v.keys_expected, a.anchor_pos);
            return d;
        }
        if (!a.mask_diff_layers.empty()) {
            d.field = Field::kMask; d.slot = static_cast<int>(a.mask_diff_layers.front());
            d.detail = fmt("drafter layer %lld block mask differs from z-lab (%lld layers) at anchor %lld", a.mask_diff_layers.front(),
                           static_cast<long long>(a.mask_diff_layers.size()), a.anchor_pos);
            return d;
        }
        if (a.shortlist_rows != b.shortlist_rows) {
            d.field = Field::kShortlist; d.slot = vec_diff(a.shortlist_rows, b.shortlist_rows);
            d.detail = "shortlist rows differ";
            return d;
        }
        if (int i = vec_diff(a.drafts, b.drafts); i >= 0) {
            const bool same_rows = !a.shortlist_rows.empty() && a.shortlist_rows == b.shortlist_rows;
            d.field = same_rows ? Field::kShortlist : Field::kTokenId; d.slot = i;
            d.detail = same_rows
                ? fmt("draft slot %lld: row (rank %lld, row %lld) maps to a different vocabulary id", i,
                      a.shortlist_rows[static_cast<size_t>(i)] >> 24, a.shortlist_rows[static_cast<size_t>(i)] & 0xFFFFFF)
                : fmt("draft slot %lld token differs at anchor %lld", i, a.anchor_pos);
            if (same_rows) d.detail += fmt(": engine %lld, reference %lld", a.drafts[static_cast<size_t>(i)], b.drafts[static_cast<size_t>(i)]);
            return d;
        }
        if (int i = vec_diff(a.preds, b.preds); i >= 0) { d.field = Field::kTokenId; d.slot = i; d.detail = fmt("verify prediction %lld differs at anchor %lld", i, a.anchor_pos); return d; }
        if (a.accepted != b.accepted || a.bonus != b.bonus) { d.field = Field::kTokenId; d.detail = fmt("accepted %lld vs %lld", a.accepted, b.accepted); return d; }
        if (a.n_commit != b.n_commit || a.inject_pos != b.inject_pos || a.gdn_state_pos != b.gdn_state_pos) {
            d.field = Field::kPosition; d.detail = fmt("commit: n_commit %lld vs %lld, gdn state %lld vs %lld", a.n_commit, b.n_commit, a.gdn_state_pos, b.gdn_state_pos);
            return d;
        }
        if (a.inject_tok != b.inject_tok) { d.field = Field::kTokenId; d.detail = "drafter cache injected a different token"; return d; }
    }
    if (e.rows.size() != r.rows.size()) {
        d.field = Field::kLength; d.cycle = static_cast<int>(n);
        d.detail = fmt("engine ran %lld cycles, reference %lld", static_cast<long long>(e.rows.size()), static_cast<long long>(r.rows.size()));
        return d;
    }
    d.cycle = -1;
    return d;
}

// ---- synthetic streams ------------------------------------------------------------------------------------------
std::vector<int32_t> synth_code_stream(uint32_t n, uint32_t vocab, uint64_t seed) {
    // Lines of a small "language": keyword, identifier, punctuation, repeated with few identifier changes, so
    // long suffix matches are common (like source code).
    std::vector<int32_t> out;
    out.reserve(n);
    const uint32_t v = std::max<uint32_t>(vocab, 1024u);
    uint64_t st = seed;
    std::vector<int32_t> idents;
    for (int i = 0; i < 6; ++i) idents.push_back(static_cast<int32_t>(300u + mix64(seed + static_cast<uint64_t>(i)) % (v / 2u)));
    const int32_t kw[4] = {101, 102, 103, 104}, punct[4] = {11, 12, 13, 14};
    while (out.size() < n) {
        st = mix64(st);
        const int32_t id = idents[st % idents.size()];
        const int32_t line[8] = {kw[(st >> 8) % 4], id, punct[0], idents[(st >> 16) % idents.size()], punct[1], id, punct[2], 10};
        for (int32_t t : line) if (out.size() < n) out.push_back(t);
    }
    return out;
}

std::vector<int32_t> synth_prose_stream(uint32_t n, uint32_t vocab, uint64_t seed) {
    // Zipf-like word choice (few repeats of long spans).
    std::vector<int32_t> out;
    out.reserve(n);
    const uint32_t v = std::max<uint32_t>(vocab, 1024u);
    uint64_t st = seed ^ 0x5EEDULL;
    while (out.size() < n) {
        st = mix64(st);
        const double u = static_cast<double>(st >> 11) / 9007199254740992.0;  // [0, 1)
        const uint32_t rank = static_cast<uint32_t>(1.0 / (1.0 - u * 0.999));   // heavy tail
        out.push_back(static_cast<int32_t>(500u + (rank * 2654435761u) % (v - 600u)));
    }
    return out;
}

}  // namespace vegajet::spec_replay

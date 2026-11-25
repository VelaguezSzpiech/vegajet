// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/fr_spec.hpp"

#include "core/quant_repacker.hpp"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>

namespace vegajet::fr_spec {

uint64_t tie_key(uint32_t local_row, uint32_t rank, const ArgmaxLayout& lay) {
    const uint32_t W = std::max(1u, lay.waves_per_block);
    const uint32_t G = std::max(1u, lay.grid_blocks);
    const uint64_t wave  = local_row % W;
    const uint64_t block = (local_row / W) % G;
    const uint64_t iter  = local_row / (static_cast<uint64_t>(W) * G);
    // (block, rank, wave, iter) lexicographic; field widths: block < 2^24, rank/wave < 2^8, iter < 2^24.
    return (block << 40) | (static_cast<uint64_t>(rank & 0xFFu) << 32) | ((wave & 0xFFu) << 24) | (iter & 0xFFFFFFu);
}

uint64_t full_vocab_tie_key(int32_t tok, const VocabSplit& split, const ArgmaxLayout& lay) {
    return tie_key(split.local_row(tok), split.rank_of(tok), lay);
}

// ---------------------------------------------------------------------------------------------------------
TokenFrequencyModel::TokenFrequencyModel(uint32_t vocab, PriorConfig cfg)
    : cfg_(cfg), rank_(vocab), counts_(vocab, 0.0) {
    std::iota(rank_.begin(), rank_.end(), 0u);
    double norm = 0.0;
    for (uint32_t r = 0; r < vocab; ++r) norm += 1.0 / std::pow(r + cfg_.rank_offset, cfg_.zipf_s);
    prior_norm_ = norm > 0.0 ? norm : 1.0;
}

void TokenFrequencyModel::set_ranked_prior(const std::vector<int32_t>& ranked_ids) {
    const uint32_t V = vocab();
    std::vector<uint8_t> listed(V, 0);
    uint32_t next = 0;
    for (int32_t t : ranked_ids) {
        if (t < 0 || static_cast<uint32_t>(t) >= V || listed[t]) continue;
        listed[t] = 1;
        rank_[t] = next++;
    }
    for (uint32_t t = 0; t < V; ++t)
        if (!listed[t]) rank_[t] = next++;
}

void TokenFrequencyModel::observe(int32_t tok, double w) {
    if (tok < 0 || static_cast<uint32_t>(tok) >= vocab() || !(w > 0.0)) return;
    counts_[tok] += w;
    total_ += w;
}

void TokenFrequencyModel::observe(const std::vector<int32_t>& toks, double w) {
    for (int32_t t : toks) observe(t, w);
}

void TokenFrequencyModel::decay(double f) {
    if (!(f > 0.0) || f >= 1.0) return;
    total_ = 0.0;
    for (double& c : counts_) { c *= f; total_ += c; }
}

double TokenFrequencyModel::score(int32_t tok) const {
    if (tok < 0 || static_cast<uint32_t>(tok) >= vocab()) return 0.0;
    const double prior = 1.0 / (std::pow(rank_[tok] + cfg_.rank_offset, cfg_.zipf_s) * prior_norm_);
    if (total_ <= 0.0) return prior;
    const double lam = total_ / (total_ + cfg_.tau_tokens);
    return (1.0 - lam) * prior + lam * (counts_[tok] / total_);
}

std::vector<double> TokenFrequencyModel::scores() const {
    std::vector<double> s(vocab());
    for (uint32_t t = 0; t < vocab(); ++t) s[t] = score(static_cast<int32_t>(t));
    return s;
}

// ---------------------------------------------------------------------------------------------------------
namespace {

bool fail(std::string* err, const std::string& m) {
    if (err) *err = m;
    return false;
}

// Device position order for a rank: positions sorted by the draft dispatch's tie key; tokens sorted by the
// full-vocab key; the i-th token goes to the i-th position.
std::vector<uint32_t> tie_exact_order(const std::vector<uint32_t>& sorted_local, uint32_t rank,
                                      const ArgmaxLayout& full, const ArgmaxLayout& draft) {
    const size_t n = sorted_local.size();
    std::vector<uint32_t> toks(sorted_local);
    std::sort(toks.begin(), toks.end(), [&](uint32_t a, uint32_t b) {
        return tie_key(a, rank, full) < tie_key(b, rank, full);
    });
    std::vector<uint32_t> pos(n);
    std::iota(pos.begin(), pos.end(), 0u);
    std::sort(pos.begin(), pos.end(), [&](uint32_t a, uint32_t b) {
        return tie_key(a, rank, draft) < tie_key(b, rank, draft);
    });
    std::vector<uint32_t> dev(n);
    for (size_t i = 0; i < n; ++i) dev[pos[i]] = toks[i];
    return dev;
}

} // namespace

bool build_shortlist(const VocabSplit& split, const ShortlistConfig& cfg, const std::vector<double>& scores,
                     const AlwaysInclude& inc, Shortlist* out, std::string* err) {
    const uint32_t R = split.n_ranks;
    if (!out) return fail(err, "null output");
    if (R == 0 || split.rows_per_rank() == 0) return fail(err, "empty vocab split");
    if (cfg.rows == 0 || cfg.rows % kRowGranule != 0) return fail(err, "rows must be a non-zero multiple of 256");
    if (cfg.rows > split.scored_vocab()) return fail(err, "rows exceed the scored vocabulary");
    if (scores.size() < split.vocab) return fail(err, "scores must have one entry per token");
    const uint32_t V = split.scored_vocab();
    const uint32_t per = split.rows_per_rank();

    std::vector<uint8_t> sel(V, 0);
    std::vector<uint32_t> cnt(R, 0), must_cnt(R, 0), pref_cnt(R, 0);
    uint32_t n_must = 0;
    for (int32_t t : inc.must) {
        if (!split.scored(t) || sel[t]) continue;
        sel[t] = 1;
        ++cnt[split.rank_of(t)];
        ++must_cnt[split.rank_of(t)];
        ++n_must;
    }

    // Global score order (desc, ties -> lower id), shared by both balance modes.
    std::vector<int32_t> order(V);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        const double sa = std::isfinite(scores[a]) ? scores[a] : -1e300;
        const double sb = std::isfinite(scores[b]) ? scores[b] : -1e300;
        return sa > sb;
    });

    std::vector<uint32_t> quota(R, 0);
    if (cfg.balance == Balance::kEqualPerRank) {
        if (cfg.rows % (kRowGranule * R) != 0) return fail(err, "kEqualPerRank needs rows % (256 * n_ranks) == 0");
        std::fill(quota.begin(), quota.end(), cfg.rows / R);
        if (cfg.rows / R > per) return fail(err, "per-rank quota exceeds rows per rank");
    } else {
        // Ideal counts from the global top-N (forced tokens first), rounded to granules by largest remainder.
        // The prefer set counts toward the ideal (capped at the budget of a rows-sized quota).
        std::vector<uint32_t> ideal(cnt);
        std::vector<uint8_t> tmp(sel);
        uint32_t taken = n_must;
        const uint64_t pref_cap = static_cast<uint64_t>(cfg.rows) * cfg.prefer_budget_pct / 100u;
        uint64_t pref_seen = 0;
        for (int32_t t : inc.prefer) {
            if (taken >= cfg.rows || pref_seen >= pref_cap) break;
            if (!split.scored(t) || tmp[t]) continue;
            tmp[t] = 1;
            ++ideal[split.rank_of(t)];
            ++taken;
            ++pref_seen;
        }
        for (size_t i = 0; i < order.size() && taken < cfg.rows; ++i) {
            const int32_t t = order[i];
            if (tmp[t]) continue;
            ++ideal[split.rank_of(t)];
            ++taken;
        }
        const uint32_t granules = cfg.rows / kRowGranule;
        const uint32_t max_g = per / kRowGranule;
        uint32_t used = 0;
        std::vector<std::pair<uint32_t, uint32_t>> rem;  // (remainder, rank)
        for (uint32_t r = 0; r < R; ++r) {
            const uint32_t g = std::min(ideal[r] / kRowGranule, max_g);
            quota[r] = g;
            used += g;
            rem.push_back({ideal[r] % kRowGranule, r});
        }
        std::stable_sort(rem.begin(), rem.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        for (size_t k = 0; used < granules && k < 4u * R + granules; ++k) {
            const uint32_t r = rem[k % R].second;
            if (quota[r] < max_g) { ++quota[r]; ++used; }
        }
        if (used != granules) return fail(err, "cannot distribute granules across ranks");
        // Guarantee room for the must set: move granules from the rank with the most spare granules.
        for (uint32_t r = 0; r < R; ++r) {
            const uint32_t need = (must_cnt[r] + kRowGranule - 1u) / kRowGranule;
            while (quota[r] < need) {
                uint32_t donor = R;
                uint32_t best_spare = 0;
                for (uint32_t d = 0; d < R; ++d) {
                    const uint32_t d_need = (must_cnt[d] + kRowGranule - 1u) / kRowGranule;
                    const uint32_t spare = quota[d] > d_need ? quota[d] - d_need : 0u;
                    if (d != r && spare > best_spare) { best_spare = spare; donor = d; }
                }
                if (donor == R || quota[r] >= max_g) return fail(err, "must-include tokens exceed the shortlist");
                --quota[donor];
                ++quota[r];
            }
        }
        for (uint32_t r = 0; r < R; ++r) quota[r] *= kRowGranule;
    }
    for (uint32_t r = 0; r < R; ++r)
        if (must_cnt[r] > quota[r]) return fail(err, "must-include tokens exceed a rank's quota");

    uint32_t n_pref = 0, n_drop = 0;
    for (int32_t t : inc.prefer) {
        if (!split.scored(t) || sel[t]) continue;
        const uint32_t r = split.rank_of(t);
        const uint64_t budget = static_cast<uint64_t>(quota[r]) * cfg.prefer_budget_pct / 100u;
        if (pref_cnt[r] >= budget || cnt[r] >= quota[r]) { ++n_drop; continue; }
        sel[t] = 1;
        ++cnt[r];
        ++pref_cnt[r];
        ++n_pref;
    }
    for (int32_t t : order) {
        if (sel[t]) continue;
        const uint32_t r = split.rank_of(t);
        if (cnt[r] >= quota[r]) continue;
        sel[t] = 1;
        ++cnt[r];
    }
    for (uint32_t r = 0; r < R; ++r)
        if (cnt[r] != quota[r]) return fail(err, "internal: rank fill mismatch");

    Shortlist sl;
    sl.split = split;
    sl.full_layout = cfg.full_layout;
    sl.draft_layout = cfg.draft_layout;
    sl.rows = cfg.rows;
    sl.rank_rows = quota;
    sl.rank_base.assign(R, 0);
    for (uint32_t r = 1; r < R; ++r) sl.rank_base[r] = sl.rank_base[r - 1] + quota[r - 1];
    sl.sorted_local_rows.assign(R, {});
    for (uint32_t t = 0; t < V; ++t)
        if (sel[t]) sl.sorted_local_rows[split.rank_of(static_cast<int32_t>(t))].push_back(split.local_row(static_cast<int32_t>(t)));
    sl.device_local_rows.resize(R);
    sl.row_to_token.assign(cfg.rows, -1);
    sl.token_to_row.assign(split.vocab, -1);
    for (uint32_t r = 0; r < R; ++r) {
        sl.device_local_rows[r] = cfg.tie_exact_layout
            ? tie_exact_order(sl.sorted_local_rows[r], r, cfg.full_layout, cfg.draft_layout)
            : sl.sorted_local_rows[r];
        for (uint32_t p = 0; p < quota[r]; ++p) {
            const int32_t tok = split.token_of(r, sl.device_local_rows[r][p]);
            const uint32_t row = sl.rank_base[r] + p;
            sl.row_to_token[row] = tok;
            sl.token_to_row[tok] = static_cast<int32_t>(row);
        }
    }
    sl.n_must = n_must;
    sl.n_prefer_taken = n_pref;
    sl.n_prefer_dropped = n_drop;
    sl.generation = out->generation + 1u;
    *out = std::move(sl);
    return true;
}

double coverage(const Shortlist& sl, const TokenFrequencyModel& model) {
    double in = 0.0, all = 0.0;
    for (uint32_t t = 0; t < model.vocab(); ++t) {
        const double s = model.score(static_cast<int32_t>(t));
        all += s;
        if (sl.contains(static_cast<int32_t>(t))) in += s;
    }
    return all > 0.0 ? in / all : 0.0;
}

// ---------------------------------------------------------------------------------------------------------
std::vector<RankGather> make_gather_plan(const Shortlist& sl, const RowPlanes& planes) {
    const uint64_t qs_b = planes.qs_row_stride_bytes;
    const uint64_t sc_b = planes.scale_row_bytes;
    const uint64_t sc_s = planes.scale_row_stride ? planes.scale_row_stride : sc_b;
    std::vector<RankGather> plan(sl.device_local_rows.size());
    for (size_t r = 0; r < plan.size(); ++r) {
        RankGather& g = plan[r];
        g.src_rows = sl.device_local_rows[r];
        const uint64_t n = g.src_rows.size();
        g.dst_qs_bytes = n * qs_b;
        g.dst_scale_bytes = n * sc_b;
        for (uint64_t p = 0; p < n; ++p) {
            const uint64_t row = g.src_rows[p];
            const bool contig = p > 0 && row == static_cast<uint64_t>(g.src_rows[p - 1]) + 1u;
            if (contig && !g.qs.empty()) g.qs.back().bytes += qs_b;
            else g.qs.push_back({row * qs_b, p * qs_b, qs_b});
            if (contig && sc_s == sc_b && !g.scale.empty()) g.scale.back().bytes += sc_b;
            else g.scale.push_back({row * sc_s, p * sc_b, sc_b});
        }
    }
    return plan;
}

void apply_gather_host(const RankGather& g, const uint8_t* src_qs, const uint8_t* src_scale, uint8_t* dst_qs,
                       uint8_t* dst_scale) {
    for (const auto& s : g.qs) std::copy(src_qs + s.src_off, src_qs + s.src_off + s.bytes, dst_qs + s.dst_off);
    for (const auto& s : g.scale)
        std::copy(src_scale + s.src_off, src_scale + s.src_off + s.bytes, dst_scale + s.dst_off);
}

// ---------------------------------------------------------------------------------------------------------
ArgmaxPick merge_shortlist_argmax(const Shortlist& sl, const float* const* blk_val, const int32_t* const* blk_id,
                                  uint32_t n_blocks) {
    ArgmaxPick best;
    uint64_t best_key = ~0ULL;
    for (uint32_t r = 0; r < sl.split.n_ranks; ++r) {
        for (uint32_t b = 0; b < n_blocks; ++b) {
            const int32_t id = blk_id[r][b];
            const float v = blk_val[r][b];
            if (id < 0 || static_cast<uint32_t>(id) >= sl.rows) continue;
            const int32_t tok = sl.row_to_token[id];
            if (tok < 0) continue;
            const uint64_t key = full_vocab_tie_key(tok, sl.split, sl.full_layout);
            if (v > best.value || (best.token >= 0 && v == best.value && key < best_key)) {
                best.value = v;
                best.token = tok;
                best.shortlist_row = id;
                best_key = key;
            }
        }
    }
    return best;
}

int32_t merge_full_vocab_argmax(const float* const* blk_val, const int32_t* const* blk_id, uint32_t n_ranks,
                                uint32_t n_blocks, float* best_value) {
    float best_v = -1e30f;
    int32_t best_i = -1;
    for (uint32_t b = 0; b < n_blocks; ++b)
        for (uint32_t r = 0; r < n_ranks; ++r)
            if (blk_val[r][b] > best_v) { best_v = blk_val[r][b]; best_i = blk_id[r][b]; }
    if (best_value) *best_value = best_v;
    return best_i;
}

// ---------------------------------------------------------------------------------------------------------
ShortlistManager::ShortlistManager(const VocabSplit& split, ManagerConfig cfg)
    : split_(split), cfg_(cfg), model_(split.vocab, cfg.prior) {}

bool ShortlistManager::init(const std::vector<int32_t>& prompt, std::string* err) {
    model_.observe(prompt);
    return rebuild(prompt, err);
}

void ShortlistManager::observe_verified(const std::vector<int32_t>& toks) {
    for (int32_t t : toks) {
        if (!sl_.contains(t)) ++miss_since_rebuild_;
        ++since_rebuild_;
        model_.observe(t);
    }
}

double ShortlistManager::window_miss_rate() const {
    return since_rebuild_ ? static_cast<double>(miss_since_rebuild_) / static_cast<double>(since_rebuild_) : 0.0;
}

bool ShortlistManager::should_rebuild() const {
    return since_rebuild_ >= cfg_.rebuild_min_tokens && window_miss_rate() > cfg_.rebuild_miss_rate;
}

bool ShortlistManager::rebuild(const std::vector<int32_t>& context, std::string* err) {
    AlwaysInclude inc;
    inc.must = must_;
    const size_t n = context.size();
    const size_t w = std::min<size_t>(n, cfg_.prefer_window);
    inc.prefer.reserve(w);
    for (size_t i = 0; i < w; ++i) inc.prefer.push_back(context[n - 1 - i]);  // most recent first
    Shortlist next;
    next.generation = sl_.generation;
    if (!build_shortlist(split_, cfg_.shortlist, model_.scores(), inc, &next, err)) return false;
    sl_ = std::move(next);
    since_rebuild_ = 0;
    miss_since_rebuild_ = 0;
    if (model_.observed_mass() > 0.0) model_.decay(cfg_.decay_on_rebuild);
    return true;
}

// ---------------------------------------------------------------------------------------------------------
// Row counts.
// ---------------------------------------------------------------------------------------------------------
uint32_t row_granule(const VocabSplit& split, Balance balance) {
    return balance == Balance::kEqualPerRank ? kRowGranule * std::max(1u, split.n_ranks) : kRowGranule;
}

uint32_t round_up_rows(uint64_t requested, uint32_t granule) {
    if (requested == 0 || granule == 0) return 0;
    const uint64_t g = granule;
    const uint64_t cap = (static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) / g) * g;
    const uint64_t up = requested > cap ? cap : ((requested + g - 1u) / g) * g;
    return static_cast<uint32_t>(std::min(up, cap));
}

uint32_t resolve_rows(uint64_t requested, const VocabSplit& split, Balance balance, std::string* note) {
    if (note) note->clear();
    if (requested == 0) return 0;
    const uint32_t g = row_granule(split, balance);
    const uint32_t cap = (split.scored_vocab() / g) * g;
    if (cap == 0) {
        if (note) *note = "vocabulary smaller than one " + std::to_string(g) + "-row granule; FR-Spec off";
        return 0;
    }
    uint32_t rows = round_up_rows(requested, g);
    std::string msg;
    if (rows != requested)
        msg = "rounded " + std::to_string(requested) + " up to " + std::to_string(rows) + " rows (multiple of " +
              std::to_string(g) + ")";
    if (rows > cap) {
        rows = cap;
        msg += (msg.empty() ? "" : "; ") + std::string("clamped to ") + std::to_string(cap) + " rows (scored vocabulary)";
    }
    if (note) *note = msg;
    return rows;
}

bool parse_rows_spec(const std::string& text, uint64_t* rows, std::string* err) {
    if (!rows) return fail(err, "null output");
    std::string t;
    for (char c : text)
        if (!std::isspace(static_cast<unsigned char>(c))) t.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    if (t.empty() || t == "0" || t == "off" || t == "none") { *rows = 0; return true; }
    uint64_t mul = 1;
    if (t.back() == 'k') { mul = 1024; t.pop_back(); }
    if (t.empty() || t.size() > 9) return fail(err, "expected 0, off, 32k, 64k or a row count");
    uint64_t v = 0;
    for (char c : t) {
        if (c < '0' || c > '9') return fail(err, "expected 0, off, 32k, 64k or a row count");
        v = v * 10u + static_cast<uint64_t>(c - '0');
    }
    *rows = v * mul;
    return true;
}

uint32_t rows_from_env(const VocabSplit& split, Balance balance, std::string* note) {
    if (note) note->clear();
    const char* v = std::getenv("VEGAJET_FR_SPEC_ROWS");  // == kRowsEnv (literal for the CLI docs drift test)
    if (!v) return 0;
    uint64_t req = 0;
    std::string err;
    if (!parse_rows_spec(v, &req, &err)) {
        if (note) *note = std::string(kRowsEnv) + "='" + v + "': " + err + "; FR-Spec off";
        return 0;
    }
    std::string adj;
    const uint32_t rows = resolve_rows(req, split, balance, &adj);
    if (note && !adj.empty()) *note = std::string(kRowsEnv) + ": " + adj;
    return rows;
}

// ---------------------------------------------------------------------------------------------------------
// Id maps.
// ---------------------------------------------------------------------------------------------------------
bool make_id_map(const std::vector<int32_t>& short_to_full, uint32_t vocab, IdMap* out, std::string* err) {
    if (!out) return fail(err, "null output");
    IdMap m;
    m.full_to_short.assign(vocab, -1);
    m.short_to_full = short_to_full;
    for (size_t s = 0; s < short_to_full.size(); ++s) {
        const int32_t f = short_to_full[s];
        if (f < 0 || static_cast<uint32_t>(f) >= vocab) return fail(err, "draft id " + std::to_string(s) + " maps outside the vocabulary");
        if (m.full_to_short[f] >= 0) return fail(err, "token " + std::to_string(f) + " appears twice in the draft vocabulary");
        m.full_to_short[f] = static_cast<int32_t>(s);
    }
    *out = std::move(m);
    return true;
}

IdMap id_map_of(const Shortlist& sl) {
    IdMap m;
    m.short_to_full = sl.row_to_token;
    m.full_to_short = sl.token_to_row;
    return m;
}

// ---------------------------------------------------------------------------------------------------------
// Draft-head slicing.
// ---------------------------------------------------------------------------------------------------------
HeadPlanes head_planes_q8_dp4a(uint32_t rows, uint32_t cols) {
    HeadPlanes p;
    p.format = HeadFormat::kQ8Dp4a;
    p.rows = rows;
    p.cols = cols;
    p.qs = {cols, quant::select_tcc_layout(cols).stride_bytes, false};
    p.scale = {cols / 32u * 2u, cols / 32u * 2u, false};
    return p;
}

HeadPlanes head_planes_q4_a8(uint32_t rows, uint32_t cols, bool tcc_pad_dm) {
    HeadPlanes p;
    p.format = HeadFormat::kQ4A8;
    p.rows = rows;
    p.cols = cols;
    p.qs = {cols / 2u, quant::select_tcc_odd256_layout(cols / 2u).stride_bytes, false};
    const uint32_t dm = cols / 32u * 4u;
    p.scale = {dm, tcc_pad_dm ? quant::select_tcc_odd256_layout(dm).stride_bytes : dm, false};
    return p;
}

namespace {

bool check_plane(const PlaneLayout& l, uint32_t want_raw, const char* name, std::string* err) {
    if (l.raw_row_bytes != want_raw) return fail(err, std::string(name) + " plane: raw row bytes do not match the format");
    if (l.stride_bytes < l.raw_row_bytes) return fail(err, std::string(name) + " plane: stride smaller than a row");
    if (l.xor_swizzled) {
        const uint32_t T = quant::kTccTileBytes;
        if (l.raw_row_bytes % T != 0 || (l.raw_row_bytes / T) % quant::kTccChannels != 0)
            return fail(err, std::string(name) + " plane: XOR swizzle needs a multiple of 16 whole 256-B tiles");
    }
    return true;
}

// Byte `off` (logical, < raw_row_bytes) of row `row` inside a plane.
inline uint64_t plane_off(const PlaneLayout& l, uint64_t row, uint32_t off) {
    uint64_t o = off;
    if (l.xor_swizzled) {
        const uint32_t T = quant::kTccTileBytes;
        o = static_cast<uint64_t>((off / T) ^ static_cast<uint32_t>(row & (quant::kTccChannels - 1u))) * T + off % T;
    }
    return row * l.stride_bytes + o;
}

void copy_row(const PlaneLayout& l, const uint8_t* src, uint64_t src_row, uint8_t* dst, uint64_t dst_row) {
    const uint8_t* s = src + src_row * l.stride_bytes;
    uint8_t* d = dst + dst_row * l.stride_bytes;
    std::memcpy(d, s, l.stride_bytes);
    if (!l.xor_swizzled || (src_row & 15u) == (dst_row & 15u)) return;
    const uint32_t T = quant::kTccTileBytes, tiles = l.raw_row_bytes / T;
    const uint32_t ms = static_cast<uint32_t>(src_row & 15u), md = static_cast<uint32_t>(dst_row & 15u);
    for (uint32_t kt = 0; kt < tiles; ++kt)
        std::memcpy(d + static_cast<size_t>(kt ^ md) * T, s + static_cast<size_t>(kt ^ ms) * T, T);
}

} // namespace

bool validate_planes(const HeadPlanes& p, std::string* err) {
    if (p.cols == 0 || p.cols % 32u != 0) return fail(err, "cols must be a non-zero multiple of 32");
    const bool q8 = p.format == HeadFormat::kQ8Dp4a;
    if (!check_plane(p.qs, q8 ? p.cols : p.cols / 2u, "qs", err)) return false;
    return check_plane(p.scale, p.cols / 32u * (q8 ? 2u : 4u), "scale", err);
}

bool slice_rows(const HeadPlanes& src, const uint8_t* src_qs, const uint8_t* src_scale,
                const std::vector<uint32_t>& src_rows, HeadPlanes* dst, std::vector<uint8_t>* dst_qs,
                std::vector<uint8_t>* dst_scale, std::string* err) {
    if (!dst || !dst_qs || !dst_scale) return fail(err, "null output");
    if (!validate_planes(src, err)) return false;
    if (!src_rows.empty() && (!src_qs || !src_scale)) return fail(err, "null source plane");
    for (uint32_t r : src_rows)
        if (r >= src.rows) return fail(err, "source row " + std::to_string(r) + " out of range");
    HeadPlanes d = src;
    d.rows = static_cast<uint32_t>(src_rows.size());
    dst_qs->assign(d.qs_bytes(), 0);
    dst_scale->assign(d.scale_bytes(), 0);
    for (size_t s = 0; s < src_rows.size(); ++s) {
        copy_row(src.qs, src_qs, src_rows[s], dst_qs->data(), s);
        copy_row(src.scale, src_scale, src_rows[s], dst_scale->data(), s);
    }
    *dst = d;
    return true;
}

bool slice_draft_head(const HeadPlanes& src, const uint8_t* src_qs, const uint8_t* src_scale, const IdMap& ids,
                      DraftHead* out, std::string* err) {
    if (!out) return fail(err, "null output");
    if (ids.rows() == 0 || ids.rows() % kRowGranule != 0) return fail(err, "draft rows must be a non-zero multiple of 256");
    if (ids.vocab() != src.rows) return fail(err, "id map vocabulary does not match the head's row count");
    std::vector<uint32_t> rows(ids.rows());
    for (uint32_t s = 0; s < ids.rows(); ++s) {
        const int32_t f = ids.short_to_full[s];
        if (f < 0 || ids.to_short(f) != static_cast<int32_t>(s)) return fail(err, "id map is not a bijection");
        rows[s] = static_cast<uint32_t>(f);
    }
    DraftHead h;
    if (!slice_rows(src, src_qs, src_scale, rows, &h.planes, &h.qs, &h.scale, err)) return false;
    h.ids = ids;
    *out = std::move(h);
    return true;
}

void head_logits_host(const HeadPlanes& p, const uint8_t* qs, const uint8_t* scale, const float* x, float* out) {
    const uint32_t nb = p.cols / 32u;
    for (uint32_t r = 0; r < p.rows; ++r) {
        float sum = 0.0f;
        for (uint32_t b = 0; b < nb; ++b) {
            const float* xb = x + static_cast<size_t>(b) * 32u;
            if (p.format == HeadFormat::kQ8Dp4a) {
                uint16_t dh;
                std::memcpy(&dh, scale + plane_off(p.scale, r, b * 2u), 2);
                float acc = 0.0f;
                for (uint32_t i = 0; i < 32u; ++i)
                    acc += static_cast<float>(static_cast<int8_t>(qs[plane_off(p.qs, r, b * 32u + i)])) * xb[i];
                sum += quant::fp16_to_f32(dh) * acc;
            } else {
                uint32_t dm;
                std::memcpy(&dm, scale + plane_off(p.scale, r, b * 4u), 4);
                float acc = 0.0f, xs = 0.0f;
                for (uint32_t i = 0; i < 16u; ++i) {
                    const uint8_t byte = qs[plane_off(p.qs, r, b * 16u + i)];
                    acc += static_cast<float>(byte & 15u) * xb[i] + static_cast<float>(byte >> 4) * xb[i + 16u];
                    xs += xb[i] + xb[i + 16u];
                }
                sum += quant::fp16_to_f32(static_cast<uint16_t>(dm & 0xFFFFu)) * acc +
                       quant::fp16_to_f32(static_cast<uint16_t>(dm >> 16)) * xs;
            }
        }
        out[r] = sum;
    }
}

int32_t draft_argmax_full_id(const float* short_logits, const IdMap& ids, float* best) {
    int32_t bi = -1;
    float bv = -std::numeric_limits<float>::infinity();
    for (uint32_t s = 0; s < ids.rows(); ++s) {
        const float v = short_logits[s];
        if (bi < 0 ? !std::isnan(v) : v > bv) { bv = v; bi = static_cast<int32_t>(s); }
    }
    if (best) *best = bv;
    return ids.to_full(bi);
}

// ---------------------------------------------------------------------------------------------------------
// Frequency tables.
// ---------------------------------------------------------------------------------------------------------
bool parse_frequency_table(const std::string& text, uint32_t vocab, FrequencyTable* out, std::string* err) {
    if (!out) return fail(err, "null output");
    if (vocab == 0) return fail(err, "empty vocabulary");
    std::vector<double> cnt(vocab, 0.0);
    std::vector<uint8_t> seen(vocab, 0);
    std::vector<int32_t> order;
    int mode = -1;  // 0 = ranked ids, 1 = id + count
    std::istringstream in(text);
    std::string line;
    size_t ln = 0;
    while (std::getline(in, line)) {
        ++ln;
        const std::string where = "line " + std::to_string(ln) + ": ";
        std::istringstream ls(line);
        std::string a, b, c;
        ls >> a;
        if (a.empty()) continue;
        if (a[0] == '#') {
            std::istringstream hs(line.substr(line.find('#') + 1));
            std::string key;
            hs >> key;
            if (key == "vocab") {
                unsigned long long hv = 0;
                if (!(hs >> hv)) return fail(err, where + "malformed vocab header");
                if (hv != vocab) return fail(err, where + "table vocab " + std::to_string(hv) + " != target vocab " + std::to_string(vocab));
            }
            continue;
        }
        ls >> b >> c;
        if (!c.empty()) return fail(err, where + "expected '<id>' or '<id> <count>'");
        char* end = nullptr;
        errno = 0;
        const long long id = std::strtoll(a.c_str(), &end, 10);
        if (errno || *end != '\0' || a[0] == '+' || a[0] == '-') return fail(err, where + "bad token id");
        if (id < 0 || static_cast<unsigned long long>(id) >= vocab) return fail(err, where + "token id outside the vocabulary");
        const int m = b.empty() ? 0 : 1;
        if (mode >= 0 && m != mode) return fail(err, where + "mixes ranked ids and counted ids");
        mode = m;
        double w = 0.0;
        if (m == 1) {
            end = nullptr;
            w = std::strtod(b.c_str(), &end);
            if (*end != '\0' || !std::isfinite(w) || w < 0.0) return fail(err, where + "bad count");
        }
        if (!seen[id]) { seen[id] = 1; order.push_back(static_cast<int32_t>(id)); }
        cnt[id] += w;
    }
    if (order.empty()) return fail(err, "frequency table has no entries");
    FrequencyTable t;
    if (mode == 1) {
        std::stable_sort(order.begin(), order.end(), [&](int32_t x, int32_t y) {
            return cnt[x] != cnt[y] ? cnt[x] > cnt[y] : x < y;
        });
        t.counts.reserve(order.size());
        for (int32_t id : order) t.counts.push_back(cnt[id]);
    }
    t.ranked = std::move(order);
    t.from_file = true;
    t.source = "file";
    *out = std::move(t);
    return true;
}

bool load_frequency_table(const std::string& path, uint32_t vocab, FrequencyTable* out, std::string* err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return fail(err, "cannot open the frequency table");
    std::ostringstream ss;
    ss << f.rdbuf();
    if (f.bad()) return fail(err, "cannot read the frequency table");
    return parse_frequency_table(ss.str(), vocab, out, err);
}

FrequencyTable fallback_frequency_table(uint32_t vocab) {
    FrequencyTable t;
    t.ranked.resize(vocab);
    std::iota(t.ranked.begin(), t.ranked.end(), 0);
    t.source = "fallback:low-id-prefix";
    return t;
}

FrequencyTable load_frequency_table_or_fallback(const std::string& path, uint32_t vocab, std::string* note) {
    if (note) note->clear();
    if (path.empty()) return fallback_frequency_table(vocab);
    FrequencyTable t;
    std::string err;
    if (load_frequency_table(path, vocab, &t, &err)) return t;
    if (note) *note = "frequency table unusable (" + err + "); using the low-ID prefix fallback";
    return fallback_frequency_table(vocab);
}

std::vector<double> table_scores(const FrequencyTable& t, uint32_t vocab, PriorConfig prior) {
    TokenFrequencyModel m(vocab, prior);
    m.set_ranked_prior(t.ranked);
    for (size_t i = 0; i < t.counts.size() && i < t.ranked.size(); ++i) m.observe(t.ranked[i], t.counts[i]);
    return m.scores();
}

bool select_draft_vocab(const std::vector<double>& scores, uint32_t vocab, uint32_t rows,
                        const std::vector<int32_t>& must, std::vector<int32_t>* out, std::string* err) {
    if (!out) return fail(err, "null output");
    const VocabSplit one{vocab, 1u};
    const uint32_t r = resolve_rows(rows, one, Balance::kEqualPerRank);
    if (r == 0) return fail(err, "no draft rows (FR-Spec off or vocabulary too small)");
    ShortlistConfig cfg;
    cfg.rows = r;
    cfg.balance = Balance::kEqualPerRank;
    cfg.tie_exact_layout = false;
    AlwaysInclude inc;
    inc.must = must;
    Shortlist sl;
    if (!build_shortlist(one, cfg, scores, inc, &sl, err)) return false;
    out->assign(sl.sorted_local_rows[0].begin(), sl.sorted_local_rows[0].end());
    return true;
}

} // namespace vegajet::fr_spec

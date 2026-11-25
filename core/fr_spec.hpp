// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// FR-Spec draft-vocabulary shortlist (model-agnostic, host only).
//
// The neural drafter (e.g. a nextn/MTP head) pays a full-vocab LM head pass per draft step. FR-Spec scores
// only a frequency-ranked "hot" subset of the output-embedding rows while DRAFTING; verification still
// scores every row, so the accepted output is unchanged (a draft whose true argmax lies outside the
// shortlist is simply wrong and gets rejected by the verifier).
//
// What this module produces:
//   * per-rank (per-GPU) shortlists under a contiguous TP vocab split (rank r owns local rows
//     [r*rows_per_rank, (r+1)*rows_per_rank) of the output matrix);
//   * total row count an exact multiple of 256 (each rank's count is also a multiple of 256);
//   * the device layout order of the gathered sub-matrix and a gather plan (row stride / scale plane are
//     parameters) to build it from the rows already resident on each GPU;
//   * the shortlist-row -> token map and the inverse;
//   * the argmax merge rule. The engine's full-vocab argmax (ornith_vocab_argmax_q8_kernel + the host loop
//     in step_batch/step_mtp) resolves exact-value ties with a fixed order; full_vocab_tie_key() reproduces
//     that order as an integer key (smaller key wins a tie). With `tie_exact_layout` the gathered rows are
//     permuted so the SAME kernel, launched on the sub-matrix, resolves intra-block ties in that same order,
//     and merge_shortlist_argmax() compares block winners by (value desc, full key asc). Per-row dot
//     products are computed identically on the sub-matrix (same kernel, same row bytes), so the shortlist
//     pick is exactly "the full-vocab argmax restricted to the shortlist", ties included.
//
// Frequency source (no external data): a tokenizer-rank prior (byte-level BPE ids are assigned in merge
// order, which tracks corpus frequency, so a Zipf law over the id/rank is a usable cold-start prior), or a
// caller-supplied ranked id list, blended with online counts of verified/accepted tokens as they arrive.
//
// Single-matrix helpers (end of this header): 32K / 64K presets with round-up 256-row alignment and the
// VEGAJET_FR_SPEC_ROWS toggle, byte-exact (XOR-swizzle aware) draft-head row slicing for VJ_Q8_DP4A and
// VJ_Q4_A8 head images, bidirectional short_id <-> full_id maps, and a corpus frequency-table loader with a
// deterministic low-ID fallback (tables come from tools/build_fr_spec_table.py).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vegajet::fr_spec {

constexpr uint32_t kRowGranule = 256;

// Contiguous TP split of the output rows, matching the engine: half_voc = vocab / n_ranks, rank r scores
// tokens [r*half_voc, (r+1)*half_voc). Tokens >= n_ranks*rows_per_rank are never scored by the full head
// either (only possible for a vocab not divisible by n_ranks) and are never put in a shortlist.
struct VocabSplit {
    uint32_t vocab   = 0;
    uint32_t n_ranks = 2;
    uint32_t rows_per_rank() const { return n_ranks ? vocab / n_ranks : 0; }
    uint32_t scored_vocab() const { return rows_per_rank() * n_ranks; }
    bool     scored(int32_t tok) const { return tok >= 0 && static_cast<uint32_t>(tok) < scored_vocab(); }
    uint32_t rank_of(int32_t tok) const { return static_cast<uint32_t>(tok) / rows_per_rank(); }
    uint32_t local_row(int32_t tok) const { return static_cast<uint32_t>(tok) % rows_per_rank(); }
    int32_t  token_of(uint32_t rank, uint32_t local) const { return static_cast<int32_t>(rank * rows_per_rank() + local); }
};

// Launch shape of the argmax kernel: grid_blocks WGs, waves_per_block waves; wave w of block b takes rows
// b*W + w, then strides by grid_blocks*W (ornith_vocab_argmax_q8_kernel: kVocabBlocks=240, W=4).
struct ArgmaxLayout {
    uint32_t grid_blocks     = 240;
    uint32_t waves_per_block = 4;
};

// Tie order of the full-vocab argmax as an integer (smaller wins an exact-value tie). Derivation:
//   kernel: each wave scans its rows in increasing iteration order with strict '>', then wave 0..W-1 are
//   merged with strict '>'; host: for b = 0..G-1, for rank = 0..R-1 (GPU0 before GPU1 at the same b), strict
//   '>'. Nested first-wins reductions give lexicographic order on (block, rank, wave, iter).
uint64_t tie_key(uint32_t local_row, uint32_t rank, const ArgmaxLayout& lay);
uint64_t full_vocab_tie_key(int32_t tok, const VocabSplit& split, const ArgmaxLayout& lay);

// ---------------------------------------------------------------------------------------------------------
// Frequency model: cold-start prior + online counts.
// ---------------------------------------------------------------------------------------------------------
struct PriorConfig {
    double zipf_s       = 1.0;    // p_prior(rank) ~ 1 / (rank + rank_offset)^s
    double rank_offset  = 10.0;
    double tau_tokens   = 4096.0; // blend weight of the online counts: lambda = n / (n + tau)
};

class TokenFrequencyModel {
public:
    explicit TokenFrequencyModel(uint32_t vocab, PriorConfig cfg = {});
    // Optional ranked prior (most frequent first). Ids not listed keep their tokenizer rank but are placed
    // after every listed id. Out-of-range / duplicate ids are ignored.
    void set_ranked_prior(const std::vector<int32_t>& ranked_ids);
    void observe(int32_t tok, double weight = 1.0);
    void observe(const std::vector<int32_t>& toks, double weight = 1.0);
    // Multiplies all online counts by factor in (0, 1] (exponential forgetting for drifting prompts).
    void decay(double factor);
    double score(int32_t tok) const;  // blended probability estimate (deterministic)
    std::vector<double> scores() const;
    double observed_mass() const { return total_; }
    uint32_t vocab() const { return static_cast<uint32_t>(rank_.size()); }

private:
    PriorConfig cfg_;
    std::vector<uint32_t> rank_;
    std::vector<double> counts_;
    double total_ = 0.0;
    double prior_norm_ = 1.0;
};

// ---------------------------------------------------------------------------------------------------------
// Shortlist construction.
// ---------------------------------------------------------------------------------------------------------
enum class Balance {
    kEqualPerRank,  // every rank gets rows / n_ranks (balanced GPU time; the default)
    kGlobalTopN,    // global top-N by score, per-rank counts rounded to multiples of 256 (largest remainder)
};

struct ShortlistConfig {
    uint32_t rows = 32768;               // total; must be a multiple of 256 (and of 256*n_ranks for kEqual)
    Balance  balance = Balance::kEqualPerRank;
    bool     tie_exact_layout = true;    // permute device order so the kernel's intra-block tie order matches
    ArgmaxLayout full_layout;            // launch shape of the full-vocab argmax (defines the tie rule)
    ArgmaxLayout draft_layout;           // launch shape used for the shortlist argmax dispatch
    uint32_t prefer_budget_pct = 25;     // max share of a rank's quota taken by the `prefer` set
};

struct AlwaysInclude {
    std::vector<int32_t> must;    // special / control / EOS tokens: build fails if they do not fit
    std::vector<int32_t> prefer;  // prompt / suffix-window tokens in priority order (most important first)
};

struct Shortlist {
    VocabSplit   split;
    ArgmaxLayout full_layout, draft_layout;
    uint32_t rows = 0;                                   // total, multiple of 256
    std::vector<uint32_t> rank_rows;                     // per rank, each a multiple of 256
    std::vector<uint32_t> rank_base;                     // first global shortlist row of each rank
    std::vector<std::vector<uint32_t>> sorted_local_rows; // per rank, ascending local row ids
    std::vector<std::vector<uint32_t>> device_local_rows; // per rank: device position -> source local row
    std::vector<int32_t> row_to_token;                   // global shortlist row (rank_base[r]+pos) -> token
    std::vector<int32_t> token_to_row;                   // token -> global shortlist row, -1 if absent
    uint32_t n_must = 0, n_prefer_taken = 0, n_prefer_dropped = 0;
    uint64_t generation = 0;
    bool contains(int32_t tok) const {
        return tok >= 0 && static_cast<size_t>(tok) < token_to_row.size() && token_to_row[tok] >= 0;
    }
};

// scores: one value per token id (size >= split.vocab); higher = more likely. Ties broken by lower id.
bool build_shortlist(const VocabSplit& split, const ShortlistConfig& cfg, const std::vector<double>& scores,
                     const AlwaysInclude& inc, Shortlist* out, std::string* err);

// Probability mass of `model` covered by the shortlist (diagnostic).
double coverage(const Shortlist& sl, const TokenFrequencyModel& model);

// ---------------------------------------------------------------------------------------------------------
// Gather plan: build the [rank_rows[r], row_stride] qs plane and [rank_rows[r], scale_row_bytes] scale
// plane on each GPU from that GPU's resident full-half planes. Offsets are bytes relative to the plane base.
// ---------------------------------------------------------------------------------------------------------
struct RowPlanes {
    uint64_t qs_row_stride_bytes = 0;  // e.g. Q8WeightDev::row_stride_bytes
    uint64_t scale_row_bytes     = 0;  // e.g. (cols / 32) * 2 for FP16 block-32 scales
    uint64_t scale_row_stride    = 0;  // 0 = scale_row_bytes (dense)
};
struct CopySegment { uint64_t src_off = 0, dst_off = 0, bytes = 0; };
struct RankGather {
    std::vector<uint32_t> src_rows;       // == device_local_rows[r] (for an index-driven gather kernel)
    std::vector<CopySegment> qs, scale;   // coalesced byte runs (for a copy-engine / memcpy path)
    uint64_t dst_qs_bytes = 0, dst_scale_bytes = 0;
};
std::vector<RankGather> make_gather_plan(const Shortlist& sl, const RowPlanes& planes);
// CPU reference of the gather (tests / host-side rebuild): src planes of one rank -> dst planes.
void apply_gather_host(const RankGather& g, const uint8_t* src_qs, const uint8_t* src_scale,
                       uint8_t* dst_qs, uint8_t* dst_scale);

// ---------------------------------------------------------------------------------------------------------
// Argmax merge.
// ---------------------------------------------------------------------------------------------------------
struct ArgmaxPick {
    int32_t token = -1;
    float   value = -1e30f;
    int32_t shortlist_row = -1;
};
// blk_val[r] / blk_id[r]: the n_blocks per-block winners of rank r's shortlist dispatch, launched with
// row_off = sl.rank_base[r] and local_rows = sl.rank_rows[r] (ids are global shortlist rows, -1 = empty).
// Comparison: value desc, then full_vocab_tie_key(token) asc. Values <= -1e30 never win (host init).
ArgmaxPick merge_shortlist_argmax(const Shortlist& sl, const float* const* blk_val, const int32_t* const* blk_id,
                                  uint32_t n_blocks);
// Reference copy of today's full-vocab host merge (b-major, rank-minor, strict '>').
int32_t merge_full_vocab_argmax(const float* const* blk_val, const int32_t* const* blk_id, uint32_t n_ranks,
                                uint32_t n_blocks, float* best_value = nullptr);

// ---------------------------------------------------------------------------------------------------------
// Online manager: blends counts from verified tokens, tracks out-of-shortlist misses, decides rebuilds.
// ---------------------------------------------------------------------------------------------------------
struct ManagerConfig {
    ShortlistConfig shortlist;
    PriorConfig     prior;
    uint32_t rebuild_min_tokens  = 2048;  // never rebuild more often than this many verified tokens
    double   rebuild_miss_rate   = 0.02;  // rebuild when the windowed out-of-shortlist rate exceeds this
    uint32_t prefer_window       = 4096;  // most recent context tokens offered as `prefer`
    double   decay_on_rebuild    = 0.5;   // forget old counts at each rebuild
};

class ShortlistManager {
public:
    ShortlistManager(const VocabSplit& split, ManagerConfig cfg);
    TokenFrequencyModel& model() { return model_; }
    const Shortlist& current() const { return sl_; }
    void set_must(std::vector<int32_t> must) { must_ = std::move(must); }
    // Initial build: prior + prompt (prompt tokens are counted and offered as `prefer`).
    bool init(const std::vector<int32_t>& prompt, std::string* err);
    // Feed every token the verifier committed (accepted drafts + the bonus token).
    void observe_verified(const std::vector<int32_t>& toks);
    bool should_rebuild() const;
    double window_miss_rate() const;
    // Rebuilds from the counts and the last `prefer_window` tokens of `context`; bumps generation.
    bool rebuild(const std::vector<int32_t>& context, std::string* err);

private:
    VocabSplit split_;
    ManagerConfig cfg_;
    TokenFrequencyModel model_;
    Shortlist sl_;
    std::vector<int32_t> must_;
    uint64_t since_rebuild_ = 0, miss_since_rebuild_ = 0;
};

// ---------------------------------------------------------------------------------------------------------
// Row-count presets, 256-row alignment and the VEGAJET_FR_SPEC_ROWS toggle.
// A shortlist size that is not a multiple of the 256-row granule would send the draft GEMV / argmax down the
// unaligned fallback path, so any requested size is ROUNDED UP to the granule (256 for a single matrix or
// Balance::kGlobalTopN, 256 * n_ranks for Balance::kEqualPerRank so every rank gets a multiple of 256), then
// clamped down to the largest granule multiple that fits the scored vocabulary.
// ---------------------------------------------------------------------------------------------------------
constexpr uint32_t kRowsPreset32K = 32768u;
constexpr uint32_t kRowsPreset64K = 65536u;
constexpr const char* kRowsEnv  = "VEGAJET_FR_SPEC_ROWS";   // unset / "0" / "off" = FR-Spec off (default)
constexpr const char* kTableEnv = "VEGAJET_FR_SPEC_TABLE";  // reserved for a corpus-table loader; not read yet

uint32_t row_granule(const VocabSplit& split, Balance balance);
// Smallest multiple of `granule` >= requested (0 stays 0; saturates at the largest multiple below 2^32).
uint32_t round_up_rows(uint64_t requested, uint32_t granule = kRowGranule);
// 0 -> 0 (off). Otherwise round up to row_granule(), clamp to the scored vocabulary. *note (optional) says
// what was adjusted ("" when the request was used as is).
uint32_t resolve_rows(uint64_t requested, const VocabSplit& split, Balance balance, std::string* note = nullptr);
// Accepts "", "0", "off" (-> 0), "32k" / "64k" (any "<n>k" = n * 1024) or a positive decimal row count.
bool parse_rows_spec(const std::string& text, uint64_t* rows, std::string* err);
// VEGAJET_FR_SPEC_ROWS parsed and resolved; an invalid value disables FR-Spec and explains why in *note.
uint32_t rows_from_env(const VocabSplit& split, Balance balance = Balance::kEqualPerRank, std::string* note = nullptr);

// ---------------------------------------------------------------------------------------------------------
// Bidirectional short_id <-> full_id maps of a draft vocabulary.
// ---------------------------------------------------------------------------------------------------------
struct IdMap {
    std::vector<int32_t> short_to_full;  // [rows]  draft row -> full token id
    std::vector<int32_t> full_to_short;  // [vocab] full token id -> draft row, -1 = not in the draft vocabulary
    uint32_t rows() const { return static_cast<uint32_t>(short_to_full.size()); }
    uint32_t vocab() const { return static_cast<uint32_t>(full_to_short.size()); }
    int32_t to_full(int32_t short_id) const {
        return short_id >= 0 && static_cast<size_t>(short_id) < short_to_full.size() ? short_to_full[short_id] : -1;
    }
    int32_t to_short(int32_t full_id) const {
        return full_id >= 0 && static_cast<size_t>(full_id) < full_to_short.size() ? full_to_short[full_id] : -1;
    }
};
// Fails on an out-of-range or duplicate id.
bool make_id_map(const std::vector<int32_t>& short_to_full, uint32_t vocab, IdMap* out, std::string* err);
// The rank-major global shortlist order of a TP Shortlist (row_to_token / token_to_row) as an IdMap.
IdMap id_map_of(const Shortlist& sl);

// ---------------------------------------------------------------------------------------------------------
// Draft-head row slicing for a quantized output.weight held as one host matrix (full vocabulary or one TP
// rank's rows). Byte-exact: every draft row is the source row's qs and scale bytes, padding included. When a
// plane is stored XOR tile swizzled (phys_tile = logical_tile ^ (row & 15)), the 256-B tiles are re-permuted
// for the row's NEW index so the unchanged kernel addressing still reads the same logical bytes.
// ---------------------------------------------------------------------------------------------------------
enum class HeadFormat : uint8_t {
    kQ8Dp4a = 0,  // quant::NativeFormat::VJ_Q8_DP4A: int8 qs [cols] + FP16 d per 32
    kQ4A8   = 1,  // quant::NativeFormat::VJ_Q4_A8: u4 qs [cols/2] + (FP16 d | FP16 m << 16) per 32
};
struct PlaneLayout {
    uint32_t raw_row_bytes = 0;  // data bytes per row
    uint32_t stride_bytes  = 0;  // row pitch (>= raw_row_bytes; padding bytes are copied as they are)
    bool     xor_swizzled  = false;
};
struct HeadPlanes {
    HeadFormat format = HeadFormat::kQ8Dp4a;
    uint32_t rows = 0, cols = 0;
    PlaneLayout qs, scale;
    uint64_t qs_bytes() const { return static_cast<uint64_t>(rows) * qs.stride_bytes; }
    uint64_t scale_bytes() const { return static_cast<uint64_t>(rows) * scale.stride_bytes; }
};
// The layout the engine's VJ_Q8_DP4A head uploader writes (qs pitch = quant::select_tcc_layout(cols).stride_bytes,
// data stored unswizzled; FP16 scales dense).
HeadPlanes head_planes_q8_dp4a(uint32_t rows, uint32_t cols);
// The layout quant::pack_f32_to_vj_q4_a8 produces (odd-256-B padded qs and, when tcc_pad_dm, dm planes).
HeadPlanes head_planes_q4_a8(uint32_t rows, uint32_t cols, bool tcc_pad_dm = true);
bool validate_planes(const HeadPlanes& p, std::string* err);

struct DraftHead {
    HeadPlanes planes;               // rows = draft rows (a multiple of 256)
    std::vector<uint8_t> qs, scale;  // [rows * stride] each
    IdMap ids;
};
// Gathers rows ids.short_to_full[s] of `src` (row index == full token id) into draft row s.
bool slice_draft_head(const HeadPlanes& src, const uint8_t* src_qs, const uint8_t* src_scale, const IdMap& ids,
                      DraftHead* out, std::string* err);
// Same for explicit source rows (e.g. one TP rank's local rows); dst row s = src row src_rows[s].
bool slice_rows(const HeadPlanes& src, const uint8_t* src_qs, const uint8_t* src_scale,
                const std::vector<uint32_t>& src_rows, HeadPlanes* dst, std::vector<uint8_t>* dst_qs,
                std::vector<uint8_t>* dst_scale, std::string* err);
// CPU reference GEMV over a head image (fixed per-row accumulation order, so a sliced row produces the
// bit-identical logit of its source row). out has `p.rows` entries.
void head_logits_host(const HeadPlanes& p, const uint8_t* qs, const uint8_t* scale, const float* x, float* out);
// First-max argmax over the draft logits, returned as a FULL token id (-1 for rows == 0). With ascending
// short_to_full (select_draft_vocab) the first-max tie rule equals the full-vocab first-max rule restricted to
// the draft vocabulary.
int32_t draft_argmax_full_id(const float* short_logits, const IdMap& ids, float* best = nullptr);

// ---------------------------------------------------------------------------------------------------------
// Frequency tables. Text format written by tools/build_fr_spec_table.py:
//     # vegajet fr-spec frequency table v1
//     # vocab 248320                  (optional; must match the target vocab when present)
//     <token_id> <count>              (one per line; or "<token_id>" alone for a ranked list, not mixed)
// Counts are summed over duplicate ids; ranked lists keep the first occurrence. Order of the result: count
// desc, id asc (deterministic). Without a table the fallback is the low-ID BPE prefix (byte-level BPE ids
// follow merge order, which tracks corpus frequency).
// ---------------------------------------------------------------------------------------------------------
struct FrequencyTable {
    std::vector<int32_t> ranked;  // most frequent first
    std::vector<double>  counts;  // parallel to `ranked`, empty for a ranked-only table / the fallback
    bool from_file = false;
    std::string source;           // "file" / "fallback:low-id-prefix"
};
bool parse_frequency_table(const std::string& text, uint32_t vocab, FrequencyTable* out, std::string* err);
bool load_frequency_table(const std::string& path, uint32_t vocab, FrequencyTable* out, std::string* err);
FrequencyTable fallback_frequency_table(uint32_t vocab);
// Empty path -> fallback. Unreadable / invalid file -> fallback, reason in *note.
FrequencyTable load_frequency_table_or_fallback(const std::string& path, uint32_t vocab, std::string* note);
// Scores for build_shortlist / select_draft_vocab: ranked prior from the table blended with its counts.
std::vector<double> table_scores(const FrequencyTable& t, uint32_t vocab, PriorConfig prior = {});
// Single-matrix draft vocabulary: top `rows` ids by score (rows rounded up to 256), `must` ids always included,
// returned ASCENDING (the order that keeps the first-max tie rule of the full head).
bool select_draft_vocab(const std::vector<double>& scores, uint32_t vocab, uint32_t rows,
                        const std::vector<int32_t>& must, std::vector<int32_t>* out, std::string* err);

} // namespace vegajet::fr_spec

// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Deterministic host replay of the speculative draft -> verify -> commit cycle (v0.1.1 DFlash2 acceptance track).
//
// The engine's DFlash2 loop (models/ornith/ornith_gpu.cpp, run_greedy_dflash2) keeps five pieces of host state in
// step: the committed token stream, the anchor position, the drafter's context KV cache (one ring per drafter
// layer, written by dflash_inject_kv after every commit), the GDN recurrent state slot / ring head, and the FR-Spec
// shortlist row -> vocabulary map that turns a drafted row back into a token id. This module replays that loop
// with a scripted target and a scripted drafter, records one trace row per cycle, and replays the same script
// against the z-lab reference semantics (tools/dflash_ref). first_divergence() then names the first field that
// differs: token ids, positions, shortlist mapping, cache state or block mask.
//
// The replay also carries the host-side fix for the drafter KV cache. The engine allocated every drafter layer as
// a ring of `sliding_window` slots and passed that window to the attention kernel on every layer, so the full
// attention layer (z-lab: no window, block-bidirectional) silently lost every context key older than the window.
// plan_draft_kv() sizes each layer by its type: an SWA layer keeps the W-slot ring, a full layer gets a power-of-two
// ring covering the DFlash2 context limit, and the kernel window argument for a full layer is that ring size (the
// kernel uses one value as both window and ring mask, so this is exact up to the limit). Above the limit the
// source policy falls back to the blk.48 MTP head with the FR-Spec shortlist.
//
// Pure host code: no GPU, no file I/O, no environment reads.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vegajet::spec_replay {

// ---- drafter KV plan -------------------------------------------------------------------------------------------
enum class LayerKind : uint8_t { kSwa = 0, kFull = 1 };
const char* layer_kind_name(LayerKind k);

struct DraftKvConfig {
    uint32_t n_layer = 0;
    std::vector<uint8_t> swa_pattern;  // per layer: 1 = sliding-window layer, 0 = full; empty = unknown
    uint32_t swa_window = 4096;        // GGUF attention.sliding_window (power of two)
    uint32_t block_len = 8;            // B = K + 1
    uint32_t max_ctx = 32768;          // engine --max-ctx
    uint32_t full_ctx_cap = 32768;     // DFlash2 context limit; above it the source policy falls back
    uint32_t n_kv = 8, head_dim = 128; // FP16 K and V rows
};

struct DraftKvLayer {
    LayerKind kind = LayerKind::kSwa;
    uint32_t slots = 0;        // ring slots (power of two)
    uint32_t kernel_win = 0;   // value passed as `swa_win` to vj_dflash2_attn_kernel
    uint32_t ring_mask = 0;    // value passed as `win_mask` to vj_dflash2_kv_inject_kernel (= slots - 1)
    uint64_t bytes = 0;        // K + V bytes for this layer
};

struct DraftKvPlan {
    bool ok = false;
    std::string error;
    bool pattern_known = false;     // false: no per-layer pattern, every layer treated as SWA (legacy behaviour)
    std::vector<DraftKvLayer> layers;
    uint64_t total_bytes = 0;
    // Largest anchor position at which every layer still sees exactly its reference key set for all B queries:
    // full layers need s_pos + B - 1 < slots. UINT32_MAX when there is no full layer.
    uint32_t max_anchor_pos = 0;
    std::string summary;
};

// The fixed plan (per-layer sizing as described above).
DraftKvPlan plan_draft_kv(const DraftKvConfig& cfg);
// What d51c19a allocates: every layer a `swa_window`-slot ring with the window applied (kept as the failing fixture).
DraftKvPlan legacy_draft_kv(const DraftKvConfig& cfg);

// ---- source policy (> 32K context fallback) ---------------------------------------------------------------------
enum class Source : uint8_t { kDFlash2 = 0, kMtpFrSpec, kMtp, kSuffixOnly, kRaw };
const char* source_name(Source s);

struct SourceConfig {
    bool has_dflash2 = false;
    bool has_mtp = false;           // blk.48 nextn head present
    uint32_t fr_spec_rows = 0;      // 0 = full-vocabulary MTP LM head
    uint32_t dflash_max_anchor = 0; // DraftKvPlan::max_anchor_pos
    uint32_t long_ctx = 32768;      // DFlash2 is used while the anchor stays at or below this position
};

struct SourcePlan {
    Source initial = Source::kRaw;
    Source after_switch = Source::kRaw;
    uint32_t switch_at_pos = UINT32_MAX;  // first anchor position that uses after_switch (UINT32_MAX: never)
    bool mtp_kv_from_prefill = false;     // the MTP KV cache must be fed from prefill on (switch may happen)
};
// Per request: prompt_len tokens, at most max_new generated.
SourcePlan plan_sources(const SourceConfig& cfg, uint32_t prompt_len, uint32_t max_new);
Source source_at(const SourcePlan& p, uint32_t anchor_pos);

// ---- FR-Spec shortlist (the engine's construction, as a host function) -------------------------------------------
struct Shortlist {
    uint32_t rows = 0, n_ranks = 2, rows_per_rank = 0;
    std::vector<std::vector<int32_t>> tok_map;  // [rank][local row] -> vocabulary id (striped: global i -> rank i % n)
    std::vector<int32_t> rank_of, row_of;       // [vocab] -> rank / local row, -1 when not in the shortlist
    bool contains(int32_t tok) const {
        return tok >= 0 && static_cast<size_t>(tok) < rank_of.size() && rank_of[static_cast<size_t>(tok)] >= 0;
    }
};
// scores: [vocab] (higher = more frequent). The last `pinned_tail` ids are always included first.
bool build_shortlist(const std::vector<double>& scores, uint32_t rows, uint32_t n_ranks, uint32_t pinned_tail,
                     Shortlist* out, std::string* err);
// Checks: rows_per_rank is a multiple of 256, ids unique and in range, maps round-trip, tail pinned.
bool check_shortlist(const Shortlist& s, uint32_t vocab, uint32_t pinned_tail, std::string* err);

// ---- replay --------------------------------------------------------------------------------------------------------
// Scripted target: the greedy continuation is `truth` (prompt followed by the greedy output). The target's
// prediction after a prefix equals truth only when the prefix is a prefix of truth; otherwise it is a
// deterministic off-truth token, so a verify batch built on a wrong prefix is visible in the trace.
struct Script {
    std::vector<int32_t> truth;
    uint32_t prompt_len = 0;
    uint32_t vocab = 1u << 17;
    // accept_plan[c] = how many leading drafts of neural cycle c the drafter gets right when its view is correct
    // (cycled; values >= K mean "all"). A drafter whose cache view differs from the reference loses everything
    // from the first query slot whose view differs.
    std::vector<uint32_t> accept_plan;
};

struct ReplayConfig {
    uint32_t k_draft = 7;               // K (B = K + 1)
    uint32_t n_gen = 128;               // tokens to generate (the bench counts the first, prefill-predicted one)
    uint32_t prefill_chunk = 512;
    uint32_t ring_slots = 16;           // kGdnRingSlots
    uint32_t suffix_max_draft = 14;
    bool hybrid_suffix = false;         // CascadeHybrid with min_match = min_chain = 4
    bool spec_gdn = false;              // --spec-gdn (two state slots) instead of the GDN ring
    uint32_t pmin_stop_at = 0;          // > 0: the p_min gate cuts every neural draft to this many tokens
    DraftKvPlan kv;                     // drafter cache layout (fixed or legacy)
    std::vector<uint8_t> causal_layer;  // per layer: block mask causal (1) or bidirectional (0) as the engine runs it
    SourcePlan sources;                 // DFlash2 / MTP+FR-Spec switching
    const Shortlist* shortlist = nullptr;  // drafted tokens must be in it (nullptr: full vocabulary)
};

struct LayerView {     // what drafter layer l saw for one query slot
    uint32_t layer = 0, slot = 0;
    uint32_t keys_expected = 0, keys_seen = 0;
    int64_t first_missing = -1;  // first reference key position the engine did not see (or saw with a wrong token)
};

struct CycleRow {
    uint32_t cycle = 0;
    Source source = Source::kRaw;
    bool suffix = false;
    uint32_t anchor_pos = 0;
    int32_t anchor_tok = -1;
    uint32_t k = 0;                    // drafts proposed
    std::vector<int32_t> drafts, preds;
    uint32_t accepted = 0;
    int32_t bonus = -1;
    uint32_t n_commit = 0;
    std::vector<uint32_t> inject_pos;  // positions written into the drafter cache after this cycle
    std::vector<int32_t> inject_tok;   // token whose target taps were written there
    uint32_t gdn_state_pos = 0;        // tokens absorbed by the committed GDN state after the cycle
    std::vector<int32_t> shortlist_rows; // (rank << 24 | row) of each neural draft, -1 when not drafted via a row
    std::vector<LayerView> view_diffs; // drafter layers / query slots whose key set differed from the reference
    std::vector<uint32_t> mask_diff_layers;  // layers whose block mask differs from z-lab
};

struct Trace {
    std::vector<CycleRow> rows;
    std::vector<int32_t> output;   // generated tokens (first = prefill prediction)
    uint64_t drafted = 0, accepted = 0, neural_cycles = 0, suffix_cycles = 0;
    double mean_accepted() const { return neural_cycles ? static_cast<double>(neural_accepted) / neural_cycles : 0.0; }
    uint64_t neural_accepted = 0;
    std::string error;             // bookkeeping invariant violated (empty when consistent)
};

// The engine's loop with the given layout / masks.
Trace replay_engine(const Script& s, const ReplayConfig& cfg);
// z-lab reference: exact key sets (full layers see 0..p-1, SWA layers p+i-W+1..p-1), causal SWA / bidirectional
// full block masks, no cache capacity limit.
Trace replay_reference(const Script& s, const ReplayConfig& cfg);

enum class Field : uint8_t { kNone = 0, kTokenId, kPosition, kShortlist, kCache, kMask, kLength };
const char* field_name(Field f);

struct Divergence {
    Field field = Field::kNone;
    int cycle = -1;          // index into the engine trace rows
    int slot = -1;           // draft slot / layer view index
    std::string detail;
};
Divergence first_divergence(const Trace& engine, const Trace& reference);

// Deterministic token streams for the code / prose cases (no tokenizer; ids in [0, vocab)).
std::vector<int32_t> synth_code_stream(uint32_t n, uint32_t vocab, uint64_t seed);
std::vector<int32_t> synth_prose_stream(uint32_t n, uint32_t vocab, uint64_t seed);

}  // namespace vegajet::spec_replay

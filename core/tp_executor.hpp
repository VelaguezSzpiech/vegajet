// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
// Generic N-rank executor layout (v1.3 M4): pure host data, no device I/O and no policy.
//
// The layout binds, for an owner-ordered device list:
//   * the per-rank row/head/vocabulary maps the decode path addresses (from tp_sharder),
//   * every row-parallel reduction site with its epoch/bank/slice schedule (from tp_collective),
//   * the per-rank buffer sizes implied by those maps.
// It describes the generic per-op path, which serves N != 2 and N == 2 when the owner
// disables the fused fast path. The TP=2 super-kernel path is a separate execution mode
// and is never implied by this layout.
//
// Nothing here decides strategy: row counts, ranks and the peer matrix come from the owner.
#include "core/tp_collective.hpp"
#include "core/tp_sharder.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace vegajet::tp {

// Execution features whose placement exists only for the TP=2 fast path.
struct FeatureFlags {
    bool super_kernels = false;  // fused M=1 super-kernels (VEGAJET_ENABLE_SUPER)
    bool asym_experts = false;   // D1 asymmetric expert layout (VEGAJET_V12_ASYM_EXPERTS)
    bool mtp_draft = false;      // blk.48 MTP head / draft GGUF
    bool speculative = false;    // suffix or DFlash2 draft-verify cycles
    bool int8_wire = false;      // block-32 INT8 all-reduce wire
    bool fused_verify = false;   // fused verify+norm kernel
    bool prefill_dot8 = false;   // W4A4 prefill
    bool ring_state = false;     // multi-slot ReplaySSM ring (two-write spec state)
};
struct FeatureUse {
    const char* feature = "";  // human name, quoted in the failure message
    const char* flag = "";     // the switch that selects it
    bool enabled = false;
    uint32_t min_ranks = 2, max_ranks = 2;  // supported rank range for this feature
};
std::vector<FeatureUse> feature_uses(const FeatureFlags& features);
// Empty when every requested feature supports `ranks`; otherwise one clause per
// offending feature naming it, its switch and the ranks it supports.
std::string admit_features(uint32_t ranks, const FeatureFlags& features);

struct ExecutorRequest {
    uint32_t m = 1;          // tokens per step (decode 1, verify <= 16, prefill chunk)
    uint32_t d_model = 0;
    uint32_t layers = 0;
    uint32_t ar_sites_per_layer = 2;  // mixer output + expert/dense FF output
    bool vocab_merge = true;          // final vocabulary-partial reduction on the owner
    // One ABI frame bound. A row is 2 B/element on the wire; a slice holds as many rows as fit.
    uint32_t max_frame_bytes = 2u << 20;
};
struct ArSlice {
    uint32_t offset = 0, rows = 0;  // rows of this step carried by the slice
    uint32_t epoch = 0, bank = 0;
    bool drain = false;             // quiesce before the frame is reused
};
struct ArSite {
    uint32_t index = 0;        // step-local site number, 0-based
    int32_t layer = -1;        // -1 for a model-level site
    uint32_t elements = 0;     // values reduced per row
    uint64_t wire_bytes = 0;   // align64(2 * elements) per push
    std::vector<ArSlice> slices;
};
struct RankLayout {
    uint32_t rank = 0;
    std::string device_key;
    uint64_t usable_vram_bytes = 0, weight_bytes = 0;
    uint64_t hidden_bytes = 0;   // m * d_model * 4
    uint64_t partial_bytes = 0;  // one slice of this rank's contribution
    uint64_t logits_bytes = 0;   // this rank's vocabulary rows (FP32)
    uint64_t vocab_gather_bytes = 0;  // owner only: the merged full-vocabulary vector
    uint64_t rx_payload_bytes = 0, rx_flag_bytes = 0;  // from the ABI-1 collective plan
    shard::Range vocab, dense, expert;  // logical rows owned
    uint32_t q_heads = 0, kv_heads = 0, k_groups = 0, v_heads = 0;
    uint32_t pushes_per_epoch = 0;  // non-local destinations of this rank's partial
    uint64_t total_bytes() const;
    uint64_t headroom_bytes() const;
};
struct ExecutorLayout {
    uint32_t ranks = 0, m = 1, d_model = 0, layers = 0;
    uint32_t ar_sites_per_layer = 2, max_frame_bytes = 2u << 20;
    bool vocab_merge = true;
    bool ar_bypass = false;
    CollectivePlan collective;
    std::vector<RankLayout> rank;
    std::vector<ArSite> sites;
    uint32_t epochs_per_step = 0;
    uint32_t frames_per_rank = 0;  // source frames a destination polls per bank
    uint64_t rx_bytes_per_rank() const { return collective.rx_payload_bytes_per_rank +
                                                collective.rx_flag_bytes_per_rank; }
};
// Slices of one site. `next_epoch` advances past the last epoch used; the schedule fails
// closed instead of wrapping. `elements` above the frame bound is an error, not a split.
std::vector<ArSlice> plan_ar_slices(uint32_t m, uint32_t elements, uint32_t max_frame_bytes,
                                    uint32_t first_epoch, uint32_t* next_epoch);
// Throws invalid_argument/overflow_error/runtime_error on an inconsistent request or capacity.
// `peers` is the directed capability matrix in rank order; empty means unknown links, which
// plans host-bounce transports exactly as tp::plan_collective does for an absent profile.
ExecutorLayout plan_executor(const ModelSpec& model, const std::vector<DeviceProfile>& devices,
                             const PeerMatrix& peers, const ShardOptions& options,
                             const ExecutorRequest& request);
// Full-coverage audit of a layout against the model it was planned for. False + message on
// any gap: missing rank, uncovered vocabulary row, uncovered head, non-monotonic epoch,
// a slice that exceeds its frame, rx bytes that disagree with the collective plan, or a
// rank whose allocation exceeds its usable VRAM.
bool check_executor_layout(const ExecutorLayout& layout, const ModelSpec& model, std::string* err);
// One line per rank plus the site/slice summary; identities come from the device keys.
std::string format_executor_layout(const ExecutorLayout& layout);

} // namespace vegajet::tp

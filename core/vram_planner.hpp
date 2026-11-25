// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Measured-bandwidth VRAM promotion planner (namespace vram; independent of quant::VramBudgetPlanner in HEAD,
// which scores promotions with a VALU roofline model).
//
// A promotion old -> new format raises bytes per weight by bpw_new / bpw_old. It pays off only if the decode
// kernel for the new format streams proportionally faster:
//
//     t_new < t_old   <=>   bytes_new / bw_new < bytes_old / bw_old   <=>   bw_new / bw_old > bpw_new / bpw_old
//
// where bw is the effective weight-streaming GB/s of that format's kernel at the step's batch size M (from
// `vegajet bench`, not the datasheet). The planner:
//   * enforces that rule strictly, plus explicit bans (IQ2_S -> VJ_Q4_A8 at M=1, a measured regression);
//   * orders accepted candidates by policy class first (routed IQ3_XXS MoE Down before anything else), then by
//     ROI = ms saved per token / extra GiB;
//   * promotes per tensor (per layer), so a budget can be filled partially;
//   * keeps >= 2.0 GiB per GPU free (activations, scratch, allocator slack), on top of KV, state and caches.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vegajet::vram {

enum class Fmt : uint8_t {
    IQ2_S = 0,          // packed sub-4-bit (disk bpw)
    IQ3_XXS,
    Q4_K,               // disk-format reference
    Q6_K,
    VJ_Q4_A8,           // asymmetric u4 + FP16 (d, m) per 32
    VJ_Q4_DOT8,         // s4 + FP16 d per 32
    VJ_DOT8_EXACT,      // exact 16-level repack of IQ3_XXS (lossless)
    VJ_Q8_DP4A,
    F16,
    Count,
};
const char* fmt_name(Fmt f);
double fmt_bpw(Fmt f);

enum class Role : uint8_t { Dense = 0, Router, SharedExpert, RoutedGateUp, RoutedDown, LmHead, Embed };
const char* role_name(Role r);

// Effective decode streaming GB/s of each format's kernel, per batch bucket (M <= m_max).
struct BwEntry { Fmt fmt; uint32_t m_max; double gbs; };
class BandwidthTable {
public:
    BandwidthTable() = default;
    explicit BandwidthTable(std::vector<BwEntry> e) : e_(std::move(e)) {}
    // Planning defaults (gfx906, 60 CU) until replaced by `vegajet bench` numbers; see vram_planner.cpp.
    static BandwidthTable gfx906_defaults();
    // Smallest bucket with m <= m_max; 0 if the format has no entry.
    double gbs(Fmt f, uint32_t m) const;
    void set(Fmt f, uint32_t m_max, double gbs);
private:
    std::vector<BwEntry> e_;
};

struct TensorSpec {
    std::string name;
    Role role = Role::Dense;
    int32_t layer = -1;
    Fmt base = Fmt::VJ_Q4_A8;          // format it is uploaded in without promotion
    std::vector<Fmt> candidates;       // possible promotions (lossless or accepted-quality only)
    uint64_t weights = 0;              // total weights (all experts)
    double read_fraction = 1.0;        // fraction read per decode token (routed: top_k / n_experts)
    std::vector<double> gpu_share;     // fraction of the tensor stored on each GPU (sums to 1, or replicated: all 1)
};

struct GpuBudget {
    uint64_t vram_bytes = 0;
    double bw_scale = 1.0;             // this GPU's streaming ceiling relative to the table's reference GPU
    uint64_t kv_bytes = 0, state_bytes = 0, cache_bytes = 0, other_bytes = 0;
};

struct Ban { Fmt from, to; uint32_t m_max; const char* why; };

struct PlannerConfig {
    uint32_t m = 1;                          // decode batch the plan is optimized for
    uint64_t reserve_bytes = 2ull << 30;     // per GPU; values below 2 GiB are raised to 2 GiB
    std::vector<Ban> bans = default_bans();
    bool iq3_down_first = true;              // policy class 0: routed IQ3_XXS MoE Down, ahead of any ROI order
    static std::vector<Ban> default_bans();
};

enum class Verdict : uint8_t { Promoted, KeptBase, RejectedRatio, RejectedBan, RejectedBudget };
const char* verdict_name(Verdict v);

struct Decision {
    std::string name;
    Role role = Role::Dense;
    int32_t layer = -1;
    Fmt from = Fmt::VJ_Q4_A8, to = Fmt::VJ_Q4_A8;
    Verdict verdict = Verdict::KeptBase;
    double bw_ratio = 0.0, bpw_ratio = 0.0;  // of the best candidate considered
    double ms_saved = 0.0;                   // per decode token (max over GPUs), if promoted
    double extra_gib = 0.0;                  // max over GPUs
    std::string why;
};

struct GpuPlan {
    uint64_t vram = 0, reserve = 0, kv = 0, state = 0, cache = 0, other = 0;
    uint64_t weights_base = 0, weights_final = 0;
    int64_t headroom() const {  // bytes left above the reserve
        return static_cast<int64_t>(vram) - static_cast<int64_t>(reserve + kv + state + cache + other + weights_final);
    }
};

struct PlanResult {
    bool ok = false;           // baseline fits on every GPU
    std::string error;
    std::vector<Decision> decisions;   // one per tensor, input order
    std::vector<uint32_t> promotion_order;  // indices into decisions, in acceptance order
    std::vector<GpuPlan> gpus;
    double ms_per_token_base = 0.0, ms_per_token_final = 0.0;  // weight streaming time, slowest GPU
};

uint64_t tensor_bytes(Fmt f, uint64_t weights);
// Decode streaming time of one tensor on one GPU, ms per token.
double stream_ms(Fmt f, uint64_t weights, double share, double read_fraction, double gbs);

PlanResult plan(const std::vector<TensorSpec>& tensors, const std::vector<GpuBudget>& gpus,
                const BandwidthTable& bw, const PlannerConfig& cfg = {});

// `vegajet plan` dry-run report (plain text, deterministic).
std::string format_plan(const PlanResult& r, const PlannerConfig& cfg);

} // namespace vegajet::vram

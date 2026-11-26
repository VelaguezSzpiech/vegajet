// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Normal-engine caller/state machine for the sealed GDN tree-WY verify/commit pair.
//
// This is the missing production-side caller, not another kernel implementation. It consumes the
// sealed descriptors in core/gdn_treewy_launch.hpp (VerifyArgs/CommitArgs built by the same
// build_verify_args()/build_commit_args() helpers) and drives the already-registered
// `vj_gdn_treewy` module, built with -fno-unroll-loops. The only arithmetic used for selection is
// the host-side select_acceptance() walk; no recurrence, WY solve, convolution, normalization or
// quantizer is reimplemented here.
//
// One Session is one speculative verify cycle. A cycle:
//   1. configure(): validate the sealed width (5/8/12), the parent tree and the dispatch budget;
//   2. bind_layer(): validate one GDN layer's state slots, records, output, wire planes, conv-ring
//      slots and the post-verify normalization/quantization references;
//   3. enqueue_verify()/enqueue_norm() per layer: the sealed verify writes records + raw output
//      only and leaves the live state untouched; the existing normal-engine gated RMSNorm + INT8
//      quantizer turns each node row into the q8/(d,sum) pair consumed by the GDN out projection;
//   4. resolve(): walk the target tokens, choose the committed node, enqueue the sealed commit
//      into the inactive state slot, and only after every layer's commit is enqueued and finished
//      flip the active slot. A rejected or partial walk therefore rolls the S0 slot and the conv
//      ring cursor back to the last accepted node; a failed commit never flips anything.
//
// Chain trees (parent[i] == i-1) are what the current normal engine can submit: the already emitted
// context token is node 0 (draft[0] == verified[0] == context token), so the sealed width carries
// width-1 new drafts and resolve() drops the root token from the emitted stream. Branching trees
// are supported by the same API; the engine simply has no branching draft source yet.
//
// Independent opt-in: an unsupported width, an absent sealed-wire producer, a missing module, a
// descriptor alias, an incomplete layer set or a budget overrun returns SequentialFallback before
// any device work, leaving the live state and both cursors byte-identical to the sequential path.
// A device launch failure after verification is fatal for the request (the caller aborts); it must
// not silently continue with a half-verified state.
//
// CPU-only: no HIP include, no device open. The Backend interface is implemented by the normal
// engine (models/ornith/ornith_treewy.cpp) and by deterministic fixtures in the CPU test.
#pragma once

#include "core/gdn_treewy_launch.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vegajet::gdn::treewy_engine {

constexpr uint32_t kMaxLayers = 128;         // one request binds both shards' copies of each GDN layer
constexpr size_t kTraceCapacity = 128;       // bounded observable history for a benchmark run

// ---- post-verify quantization ABI --------------------------------------------------------
// Byte-identical to the existing `vj_spec_gdn_norm_quant_kernel` kernarg image in
// models/ornith/ornith_gpu.cpp (five pointers, four u32, one float). Normalization is reused, not
// re-implemented; the caller's backend only enqueues the existing kernel.
struct NormQuantArgs {
    uint64_t raw_output = 0;   // [m][nv][128] FP32 raw recurrent output from the sealed verify
    uint64_t gate_source = 0;  // [m][stride_in] projection buffer, z gate at z_off + hv*128
    uint64_t norm_weight = 0;  // [128] FP32 gated RMSNorm weight
    uint64_t out_q8 = 0;       // [m][nv][128] INT8
    uint64_t out_ds = 0;       // [m][nv][4] float2 (d, block sum)
    uint32_t m = 0;
    uint32_t nv = 32;
    uint32_t stride_in = 0;
    uint32_t z_off = 0;
    float eps = 0.0f;
};
static_assert(sizeof(NormQuantArgs) == 64, "normal-engine norm/quant kernarg image is 64 bytes");
static_assert(offsetof(NormQuantArgs, out_ds) == 32 && offsetof(NormQuantArgs, m) == 40 &&
                  offsetof(NormQuantArgs, eps) == 56,
              "normal-engine norm/quant kernarg offsets");

// ---- one layer's device references ------------------------------------------------------
struct WirePlanes {
    uint64_t query = 0;  // [m][8][128] packed FP32
    uint64_t key = 0;    // [m][8][128]
    uint64_t value = 0;  // [m][32][128]
    uint64_t decay = 0;  // [m][32]
    uint64_t beta = 0;   // [m][32]
    uint64_t parent = 0; // [m] int32
    bool complete() const { return query && key && value && decay && beta && parent; }
};

struct NormRef {
    uint64_t raw_output = 0;
    uint64_t gate_source = 0;
    uint64_t norm_weight = 0;
    uint64_t out_q8 = 0;
    uint64_t out_ds = 0;
    uint32_t nv = 32;
    uint32_t stride_in = 0;
    uint32_t z_off = 0;
    float eps = 0.0f;
    bool enabled = false;
    bool complete() const {
        return enabled && raw_output && gate_source && norm_weight && out_q8 && out_ds;
    }
};

struct LayerBinding {
    uint32_t device = 0;             // shard/GPU ordinal handed back to the backend
    uint64_t state_slot[2] = {0, 0}; // two full [32][128][128] FP32 slots
    uint32_t active_slot = 0;        // S0 lives in state_slot[active_slot]
    uint64_t records = 0;            // one sealed 128-record block, durable until commit
    uint64_t output = 0;             // [m][32][128] FP32 raw output
    uint64_t parent = 0;             // [m] int32 device copy of the same tree
    WirePlanes wire;                 // packed q/k/v/decay/beta (sealed ABI)
    NormRef norm;                    // post-verify gated RMSNorm + INT8 quantizer
    // Producer sources for the sealed-wire stage (ignored when the wire is pre-supplied).
    uint64_t conv_source = 0;        // interleaved [m][6144] conv+q/k/v plane
    uint64_t decay_source = 0;       // [m][32]
    uint64_t beta_source = 0;        // [m][32]
    uint64_t parent_source = 0;      // [m] int32 staging; the producer copies it to `wire.parent`
    uint32_t conv_stride = 6144;
    uint32_t nk = 8;
    uint32_t nv = 32;
    // Post-node conv-ring slots; when omitted the chain mapping (snapshot + i + 1) % slots is used.
    bool has_conv_slots = false;
    uint32_t conv_slot[vegajet::treewy::max_m] = {};
};

// Engine request: one verify cycle over one tree.
struct Request {
    uint32_t width = 0;   // sealed widths only: 5, 8 or 12
    uint32_t layers = 0;  // GDN layers bound for this cycle
    bool context_is_root = true;  // node 0 is the already emitted context token
    bool produce_wire = true;     // ask the backend to fill the sealed planes from conv sources
    bool require_norm = true;     // every layer must expose the reused norm/quant stage
    uint64_t budget_bytes = 0;    // 0 disables the check
    uint32_t conv_head = 0;       // ring cursor before this cycle
    uint32_t conv_slots = 17;     // == kGdnRingSlots in the normal engine
    std::vector<int32_t> parent;  // [width], host copy for validation and selection
    std::vector<int32_t> draft;   // [width], node tokens (node 0 == context token)
    std::vector<int32_t> verified;  // [width], target tokens; empty until resolve()
};

enum class AcceptanceKind : uint8_t {
    Natural = 0,  // use Request::verified as delivered by the target LM head
    Prefix = 1,   // deterministic fixture: accept the first `prefix` chain nodes
    Reject = 2,   // deterministic fixture: accept no node
};

struct Acceptance {
    AcceptanceKind kind = AcceptanceKind::Natural;
    uint32_t prefix = 0;
};

// Pre-dispatch decision. SequentialFallback means the caller must keep its existing sequential
// verifier; no device work, state write or cursor change has happened.
struct Decision {
    vegajet::gdn::treewy_launch::Dispatch dispatch =
        vegajet::gdn::treewy_launch::Dispatch::Unsupported;
    std::string reason;
    bool treewy() const { return dispatch == vegajet::gdn::treewy_launch::Dispatch::TreeWy; }
};

enum class FallbackReason : uint8_t {
    None = 0,
    Disabled,
    WidthNotSealed,
    WidthExceedsSequential,
    EmptyLayers,
    TooManyLayers,
    InvalidParent,
    InvalidDraft,
    InvalidVerified,
    ContextRootRejected,
    BudgetExceeded,
    BackendUnavailable,
    DescriptorInvalid,
    WireUnavailable,
    WireProduceFailed,
    IncompleteLayers,
    VerifyFailed,
    NormFailed,
    CommitFailed,
    ConvSlotMismatch,
    StateSlotAlias,
};
const char* fallback_reason_name(FallbackReason r);

// ---- backend interface ------------------------------------------------------------------
struct WireRequest {
    uint32_t layer = 0;
    uint32_t device = 0;
    uint32_t width = 0;
    uint32_t nk = 8;
    uint32_t nv = 32;
    uint32_t conv_stride = 6144;
    uint64_t conv_source = 0;
    uint64_t decay_source = 0;
    uint64_t beta_source = 0;
    uint64_t parent_source = 0;  // [m] int32 staging written by the caller each cycle
    WirePlanes out;  // producer writes these packed planes
};

struct VerifyCall {
    uint32_t layer = 0;
    uint32_t device = 0;
    vegajet::gdn::treewy_launch::LaunchPlan launch;
    vegajet::gdn::treewy_launch::VerifyArgs args{};
};

struct NormCall {
    uint32_t layer = 0;
    uint32_t device = 0;
    NormQuantArgs args{};
};

struct CommitCall {
    uint32_t layer = 0;
    uint32_t device = 0;
    vegajet::gdn::treewy_launch::LaunchPlan launch;
    vegajet::gdn::treewy_launch::CommitArgs args{};
};

class Backend {
public:
    virtual ~Backend() = default;
    // False when the registered module or the sealed-wire producer is not present in this build.
    virtual bool available() const { return true; }
    virtual bool wire_available() const { return true; }
    // Fills LayerBinding::wire from the interleaved conv/decay/beta source buffers.
    virtual bool produce_wire(const WireRequest&, std::string& err) {
        (void)err;
        err = "sealed-wire producer is not installed";
        return false;
    }
    // Copies the host parent chain into one shard's device parent buffer (once per cycle).
    virtual bool upload_parent(uint32_t, uint64_t, const int32_t*, uint32_t, std::string& err) {
        (void)err;
        return true;  // pre-supplied wire: the caller already owns the device copy
    }
    // Enqueues only; the caller's normal submit/wait remains in force for verify/norm.
    virtual bool verify(const VerifyCall&, std::string& err) = 0;
    virtual bool normalize(const NormCall&, std::string& err) = 0;
    // Enqueues one sealed commit (into the inactive slot); finish() submits and waits.
    virtual bool commit(const CommitCall&, std::string& err) = 0;
    // Submits/waits every shard after the last commit is enqueued. False is fatal for the cycle.
    virtual bool finish(std::string&) { return true; }
};

// ---- observable state -------------------------------------------------------------------
enum class Op : uint8_t { Verify = 1, Norm = 2, Commit = 3, Restore = 4, Fallback = 5 };
struct TraceEvent {
    Op op = Op::Verify;
    uint32_t layer = 0;
    uint32_t node = vegajet::gdn::treewy_launch::kNoNode;
    bool ok = false;
    uint64_t bytes = 0;
};

struct Stats {
    uint64_t configured = 0;
    uint64_t treewy_selected = 0;
    uint64_t fallbacks = 0;
    uint64_t verify_launches = 0;
    uint64_t norm_launches = 0;
    uint64_t commit_launches = 0;
    uint64_t restore_launches = 0;
    uint64_t wire_produced = 0;
    uint64_t wire_supplied = 0;
    uint64_t full_accepts = 0;
    uint64_t partial_accepts = 0;
    uint64_t rejects = 0;
    uint64_t root_only = 0;  // context-root chain with no accepted draft (engine rollback case)
    uint64_t accepted_nodes = 0;
    uint64_t committed_nodes = 0;
    uint64_t emitted_tokens = 0;
    uint64_t bonus_pending = 0;
    uint64_t cursor_advances = 0;
    uint64_t cursor_rollbacks = 0;
    uint64_t state_flips = 0;
    uint64_t descriptor_errors = 0;
    uint64_t backend_errors = 0;
    uint64_t bytes_records = 0;
    uint64_t bytes_state = 0;
    uint64_t bytes_wire = 0;
    uint64_t bytes_output = 0;
    std::vector<TraceEvent> trace;

    void reset();
    void event(Op op, uint32_t layer, uint32_t node, bool ok, uint64_t bytes);
    std::string summary() const;
};

// ---- result of one resolved cycle -------------------------------------------------------
struct PostVerify {
    uint32_t rows = 0;          // rows the norm stage produced (== width)
    uint32_t first_new_row = 0; // 1 on a context-root chain: row 0 is the emitted context
    // The only q8/(d,sum) rows the out-projection / residual consumer may use. Rows outside this
    // set are speculative: the engine may compute them, but the next cycle re-embeds/re-projects
    // those token positions. On a chain this is row 0..selected_node; on a tree it is the path.
    std::vector<uint32_t> accepted_rows;
    std::vector<int32_t> emitted;         // new tokens this cycle, root excluded, bonus excluded
    bool bonus_pending = false;           // full accept: the caller emits one ordinary target token
};

struct Resolution {
    bool used = false;   // a tree-WY cycle was active
    bool ok = false;     // commit + flip completed (a clean no-commit walk also sets this)
    bool fatal = false;  // device failure after verification: the caller must abort the request
    FallbackReason reason = FallbackReason::None;
    std::string message;
    vegajet::gdn::treewy_launch::Selection selection;
    uint32_t path_nodes = 0;  // committed nodes along the accepted root path (includes node 0)
    uint32_t accepted_node = vegajet::gdn::treewy_launch::kNoNode;
    int32_t commit_accepted = 0;  // sealed CommitArgs::accepted (node id + 1)
    uint32_t state_advance = 0;   // tokens whose state is committed (root included)
    uint32_t new_tokens = 0;      // emitted now, bonus excluded
    uint32_t conv_head = 0;       // cursor after this cycle
    std::vector<uint32_t> active_slot;  // per bound layer after the flip
    bool bonus_pending = false;
    PostVerify post;
};

// ---- session ----------------------------------------------------------------------------
class Session {
public:
    explicit Session(uint32_t conv_slots = 17) : conv_slots_(conv_slots) {}

    // Validates the width/tree/width budget and selects TreeWy or SequentialFallback.
    Decision configure(const Request& req);

    // Attaches one layer's device references. False leaves the session inactive and records the
    // precise fallback reason; nothing was dispatched.
    bool bind_layer(uint32_t layer, const LayerBinding& binding, std::string& err);

    // Enqueues the sealed verify (and producer when requested) for one layer.
    bool enqueue_verify(Backend& backend, uint32_t layer, std::string& err);
    // Enqueues the existing normal-engine norm/quant stage over the sealed raw output rows.
    bool enqueue_norm(Backend& backend, uint32_t layer, std::string& err);

    // Chooses the committed node, enqueues commit(s) into the inactive slots and flips only after
    // the backend reports success. A clean no-accept walk launches no commit and restores cursors.
    Resolution resolve(Backend& backend, const Acceptance& acceptance, std::string& err);

    // Drops an uncommitted cycle: no state write, conv cursor restored.
    void abort();

    bool active() const { return active_; }
    bool blocked() const { return !blocked_reason_.empty(); }
    const std::string& blocked_reason() const { return blocked_reason_; }
    const Request& request() const { return req_; }
    const LayerBinding& binding(uint32_t layer) const { return layers_[layer].binding; }
    bool layer_bound(uint32_t layer) const { return layer < layers_.size() && layers_[layer].bound; }
    bool layer_verified(uint32_t layer) const {
        return layer < layers_.size() && layers_[layer].verified;
    }
    bool layer_normalized(uint32_t layer) const {
        return layer < layers_.size() && layers_[layer].normalized;
    }
    uint32_t conv_slots() const { return conv_slots_; }
    uint32_t snapshot_conv_head() const { return snapshot_conv_head_; }
    const Stats& stats() const { return stats_; }
    Stats& stats() { return stats_; }
    FallbackReason last_reason() const { return last_reason_; }
    std::string last_message() const { return last_message_; }

private:
    struct LayerState {
        LayerBinding binding;
        bool bound = false;
        bool verified = false;
        bool normalized = false;
        bool commit_enqueued = false;
    };

    bool fail(FallbackReason reason, const std::string& message);
    bool validate_conv_slots(uint32_t layer, std::string& err) const;
    bool validate_buffers(uint32_t layer, std::string& err) const;
    void restore_cursors();
    bool build_acceptance(AcceptanceKind kind, uint32_t prefix, std::vector<int32_t>& verified,
                          std::string& err) const;

    uint32_t conv_slots_ = 17;
    Request req_;
    std::vector<LayerState> layers_;
    bool active_ = false;
    bool verified_all_ = false;
    bool normalized_all_ = false;
    uint32_t snapshot_conv_head_ = 0;
    uint32_t selected_conv_head_ = 0;
    FallbackReason last_reason_ = FallbackReason::None;
    std::string last_message_;
    std::string blocked_reason_;
    Stats stats_;
};

}  // namespace vegajet::gdn::treewy_engine

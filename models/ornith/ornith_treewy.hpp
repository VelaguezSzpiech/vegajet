// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Normal-engine (non-super) adapter for the sealed GDN tree-WY caller.
//
// The engine's existing `--spec-gdn` path verifies one chronological batch with
// vj_spec_gdn_verify_kernel and either flips its two state slots or recomputes on reject. This
// adapter keeps that contract but routes the same cycle through the sealed
// `vj_gdn_treewy_verify_kernel` / `vj_gdn_treewy_commit_kernel` pair:
//
//   * the emitted context token is node 0 (draft[0] == verified[0] == context); the sealed width
//     therefore carries width-1 new drafts;
//   * the conv kernel still writes q/k/v/decay/beta for the chain; the independent producer
//     (ops/gdn_wire_pack.hpp + ops/kernels/vj_gdn_wire_pack.hip) packs those into the token-major
//     planes the sealed ABI requires. The producer is installed by the engine through
//     install_gdn_wire_producer(); without it the cycle falls back to the sequential verifier
//     before any device work;
//   * the post-verify gated RMSNorm + INT8 quantizer is the engine's existing
//     vj_spec_gdn_norm_quant_kernel, pointed at the sealed raw output (no new norm kernel);
//   * the S0 slot pair is the engine's existing spec_gdn_s_slot_va[2]; commit writes the inactive
//     slot and only the caller flips after every layer's commit, exactly like the sequential
//     full-accept pointer flip. A partial/reject walk commits node accepted+1 (root included),
//     which is the same prefix the sequential recompute path would produce;
//   * the conv-ring cursor is advanced by the normal engine exactly as today (ring_head/conv_head
//     += accepted+1). The adapter snapshots it, validates the per-node slot order, and rolls the
//     snapshot back on a rejected/no-commit walk; a mismatch is a pre-dispatch fallback.
//
// Opt-in: VEGAJET_TREEWY=1 (paired with --spec-gdn). Deterministic driver hooks:
//   VEGAJET_TREEWY_FORCE_WIDTH=5|8|12       pad/truncate the draft chain to that sealed width
//   VEGAJET_TREEWY_FORCE_ACCEPT=natural|reject|prefix:N
//                                          clamp the target's own accepted prefix; it can only
//                                          shorten it (reject -> root-only, prefix:N -> min(natural,
//                                          N)). It never rewrites a target prediction, so a
//                                          mismatched draft can never be forced into the state.
// They exist only to make the engine path reproducible; they never invent a kernel result.
//
// CPU-only: this file and its .cpp compile without /dev/kfd. RuntimeBackend is the production
// backend; the CPU test supplies a deterministic Backend and drives the exact same Hook.
#pragma once

#include "core/gdn_treewy_engine.hpp"
#include "ops/gdn_wire_pack.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace vegajet::kfd {
class Runtime;
struct LoadedKernel;
}  // namespace vegajet::kfd

namespace vegajet::ornith::treewy {

struct Options {
    bool enabled = false;
    uint32_t force_width = 0;  // 0 = off, otherwise 5, 8 or 12
    enum class ForceAccept : uint8_t { Natural = 0, Full = 1, Reject = 2, Prefix = 3 };
    ForceAccept force_accept = ForceAccept::Natural;
    uint32_t force_prefix = 0;  // number of accepted drafts for ForceAccept::Prefix
    bool verbose = false;

    static Options from_env();
};

std::string describe(const Options& opt);

// Target-authority clamp used by the engine's deterministic driver hooks. It may only SHORTEN the
// prefix the target verifier already produced: Natural/Full return the natural count unchanged,
// Reject returns 0 (root-only rollback) and Prefix returns min(prefix, natural). It never rewrites
// a target prediction and can never increase acceptance.
uint32_t clamp_accepted(uint32_t natural, uint32_t drafts, Options::ForceAccept mode,
                        uint32_t prefix);

// One candidate verify cycle as the engine sees it before step_batch.
struct CycleInput {
    std::vector<int32_t> tokens;  // [context, draft...]; context first
    uint32_t conv_head = 0;
    uint32_t active_slot = 0;  // engine spec_gdn_active before this cycle
};

// Per-GDN-layer device references supplied from inside the engine's layer loop.
struct LayerInputs {
    uint32_t gdn_index = 0;
    uint32_t device = 0;
    uint32_t active_slot = 0;  // engine spec_gdn_active snapshot for this cycle
    uint64_t state_slot[2] = {0, 0};
    uint64_t records = 0;
    uint64_t output = 0;
    uint64_t parent = 0;
    uint64_t conv_source = 0;
    uint64_t decay_source = 0;
    uint64_t beta_source = 0;
    uint64_t gate_source = 0;
    uint64_t norm_weight = 0;
    uint64_t out_q8 = 0;
    uint64_t out_ds = 0;
    uint32_t stride_in = 10304;
    uint32_t z_off = 6144;
    uint32_t nv = 32;
    float eps = 0.0f;
};

struct CycleResult {
    bool used = false;   // adapter owned this cycle
    bool ok = false;     // commit path completed
    bool fatal = false;  // device failure after verification: abort the request
    uint32_t path_nodes = 0;
    uint32_t new_tokens = 0;
    uint32_t accepted_node = vegajet::gdn::treewy_launch::kNoNode;
    uint32_t active_slot = 0;  // engine spec_gdn_active after this cycle
    uint32_t conv_head = 0;
    bool bonus_pending = false;
    std::vector<int32_t> emitted;  // new tokens, context root and bonus excluded
    std::string reason;
};

// The sealed-wire producer contract. It runs on the engine's device stream immediately after the
// conv kernel for that layer and must fill request.out from request.conv_source/decay_source/
// beta_source/parent_source without changing the value order. Enqueue only: do not submit or wait,
// because the sealed verify and the reused norm kernel follow on the same stream. Installing
// nullptr disables the tree-WY path.
using SealedWireProducer =
    std::function<bool(kfd::Runtime&, size_t gpu, const gdn::treewy_engine::WireRequest&, std::string&)>;
void install_sealed_wire_producer(SealedWireProducer producer);
bool sealed_wire_installed();

// Builds the validated device pack call for one layer/shard. Shared by the installed producer and
// by the CPU tests, so the descriptor the tests exercise is the descriptor production enqueues.
bool build_gdn_wire_call(const gdn::treewy_engine::WireRequest& req,
                         vegajet::gdn::wire_pack::Args& args, uint32_t& grid_x, std::string& err);

// Installs the real GDN q/k/v/decay/beta/parent producer backed by the registered
// `vj_gdn_wire_pack` module. Both shard kernels must be present; on failure this clears any
// previously installed producer and returns false (the engine keeps the sequential verifier).
bool install_gdn_wire_producer(const kfd::LoadedKernel* shard0, const kfd::LoadedKernel* shard1);

// Production backend: enqueues the registered sealed module and the existing norm/quant kernel.
class RuntimeBackend final : public gdn::treewy_engine::Backend {
public:
    RuntimeBackend(kfd::Runtime& rt, const kfd::LoadedKernel* verify[2],
                   const kfd::LoadedKernel* commit[2], const kfd::LoadedKernel* norm[2]);
    bool available() const override;
    bool wire_available() const override;
    bool produce_wire(const gdn::treewy_engine::WireRequest&, std::string& err) override;
    bool upload_parent(uint32_t device, uint64_t dst, const int32_t* host, uint32_t width,
                       std::string& err) override;
    bool verify(const gdn::treewy_engine::VerifyCall&, std::string& err) override;
    bool normalize(const gdn::treewy_engine::NormCall&, std::string& err) override;
    bool commit(const gdn::treewy_engine::CommitCall&, std::string& err) override;
    bool finish(std::string& err) override;

private:
    kfd::Runtime& rt_;
    const kfd::LoadedKernel* verify_[2];
    const kfd::LoadedKernel* commit_[2];
    const kfd::LoadedKernel* norm_[2];
};

// One engine cycle. begin() validates and snapshots; enqueue_layer() is called for every GDN
// layer in execution order; commit() resolves the acceptance walk and commits. All methods are
// synchronous host-side calls that only enqueue device work through the backend.
class Hook {
public:
    Hook(Options opt, uint32_t gdn_layers, uint32_t conv_slots, const uint64_t arena_va[2],
         const uint64_t arena_bytes[2], gdn::treewy_engine::Backend& backend);
    ~Hook();

    bool begin(const CycleInput& in);
    bool active() const { return session_.active(); }
    bool enqueue_layer(const LayerInputs& in, std::string& err);
    CycleResult commit(uint32_t accepted_drafts, uint32_t draft_count);
    void abort();
    const gdn::treewy_engine::Stats& stats() const { return session_.stats(); }
    const gdn::treewy_engine::Session& session() const { return session_; }
    const gdn::treewy_launch::ScratchPlan& plan() const { return plan_; }
    uint64_t output_va(uint32_t device) const {
        return device < 2u ? arena_va_[device] + plan_.output_offset : 0;
    }
    void print_stats(const char* tag) const;

private:
    Options opt_;
    uint32_t gdn_layers_ = 0;
    uint32_t conv_slots_ = 17;
    uint64_t arena_va_[2] = {0, 0};
    uint64_t arena_bytes_[2] = {0, 0};
    gdn::treewy_engine::Backend& backend_;
    gdn::treewy_engine::Session session_;
    gdn::treewy_launch::ScratchPlan plan_;
    uint32_t width_ = 0;
    uint32_t active_in_ = 0;
    uint32_t conv_head_in_ = 0;
    std::vector<int32_t> tokens_;
};

}  // namespace vegajet::ornith::treewy

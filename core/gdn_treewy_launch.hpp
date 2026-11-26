// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Host-side launch, scratch and selection layer for the sealed GDN tree-WY verifier
// (ops/kernels/vj_gdn_treewy_verify.hiph). CPU-only: it never opens a device, never claims
// device arithmetic parity, and never authorizes a commit by itself.
//
// Sealed device contract, one layer per launch:
//   verify: initial,q,k,v,decay,beta,parent -> output,records   128 workgroups x 256 threads
//   commit: records,M,accepted -> state                        128 workgroups x 256 threads
//   accepted==0 restores S0; accepted==N>0 selects node N-1 and its ancestors.
// The two kernels are reused unchanged. This file supplies only what the seal leaves to the
// host: validated kernarg images, record/output/parent scratch sizing from a byte budget,
// the tree acceptance walk, and the resulting state action.
//
// Build/loader: the pair is a registered code object (CMake `vj_gdn_treewy`, compile-only
// `-fno-unroll-loops`: verify 26 VGPR, commit 18 VGPR, 0 B private on TheRock 7.12) and both
// symbols are in ops/kernels/hsaco_gates.txt with no waiver. Device parity is NOT established
// by this file or by the CPU reference; one recurrent layer still needs the owner A/B before
// any production dispatch may use the pair. The sequential spec-GDN path stays in use until then.
//
// Rejection never mutates the live state: a selection with no accepted node returns
// StateAction::None, so the caller launches no commit. A commit reads only the selected
// node's ancestor closure. Acceptance itself is the target verifier's decision: this layer
// compares the target's per-node tokens with the draft tokens and never invents a token.
//
// Node convention (differs from the production chain kernel, which feeds the already emitted
// context token as its first slot): every tree node is a token whose state follows S0.
//   1. build_verify_args() -> verify launch: records + per-node outputs, live state untouched;
//   2. the target's LM head turns each node output into verified_token[i];
//   3. select_acceptance() walks parent links from S0 and accepts the lowest-index child whose
//      draft token equals the target token;
//   4. a rejected walk (accepted_nodes == 0) emits the target token of the first root child and
//      commits nothing - S0 stays the live state;
//   5. an accepted walk commits exactly node accepted_node_id (kernarg accepted = id + 1) and
//      emits the accepted draft tokens followed by the rejected frontier's target token, or, when
//      the path ends at a leaf, sets bonus_pending so the caller runs one ordinary target step.
// Integration choice: production's S0 is the state *before* the already emitted context token.
// To reproduce its commit rule, make that context token the root node (draft[0] == verified[0] ==
// context token) and drop emitted[0] when emitting; the sealed width then carries width-1 new
// draft tokens. A tree that omits the context node must start from a state that already includes
// it. Never pass the accepted-node count as CommitArgs::accepted: only CommitNode uses the id.
#pragma once

#include "ops/kernels/vj_gdn_treewy_verify.hiph"

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

namespace vegajet::gdn::treewy_launch {

// ---- sealed geometry -------------------------------------------------------------------
constexpr uint32_t kValueHeads = 32;
constexpr uint32_t kQueryHeads = 8;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kSplitV = 4;  // value columns handled per workgroup
constexpr uint32_t kWorkgroups = kValueHeads * kSplitV;
constexpr uint32_t kBlockThreads = 256;
constexpr uint32_t kMaxNodes = static_cast<uint32_t>(vegajet::treewy::max_m);
// The production chain kernel takes at most 16 verified slots (kMaxSpecBatch in
// models/ornith/ornith_gpu.cpp); above that neither path can run the request.
constexpr uint32_t kSequentialMaxWidth = 16;
constexpr uint64_t kArenaAlignment = 256;  // matches core/workspace_packer arena alignment
constexpr uint64_t kStateBytesPerLayer = static_cast<uint64_t>(kValueHeads) * kHeadDim * kHeadDim * sizeof(float);
constexpr uint64_t kRecordBytes = sizeof(vegajet::treewy::Record);
constexpr uint64_t kRecordBlockBytes = static_cast<uint64_t>(kWorkgroups) * kRecordBytes;

static_assert(kMaxNodes == 12, "sealed verifier is defined for M in {5,8,12} up to max_m == 12");
static_assert(kStateBytesPerLayer == 2097152, "state ABI is [32 heads][128 rows][128 cols] FP32");
static_assert(kRecordBytes == 6172 * sizeof(float), "sealed Record layout changed; re-derive the sizing");
static_assert(kRecordBlockBytes % kArenaAlignment == 0, "one layer of records also satisfies the arena alignment");
static_assert(sizeof(int) == 4, "the sealed kernels take a 32-bit int M/accepted");

constexpr bool is_sealed_width(uint32_t width) { return width == 5 || width == 8 || width == 12; }

namespace detail {

constexpr uint64_t align_up(uint64_t x, uint64_t a) { return ((x + a - 1) / a) * a; }
inline bool checked_add(uint64_t a, uint64_t b, uint64_t& out) {
    if (b > UINT64_MAX - a) return false;
    out = a + b;
    return true;
}
inline bool checked_mul(uint64_t a, uint64_t b, uint64_t& out) {
    if (a != 0 && b > UINT64_MAX / a) return false;
    out = a * b;
    return true;
}
inline bool checked_align(uint64_t x, uint64_t a, uint64_t& out) {
    uint64_t bumped = 0;
    if (!checked_add(x, a - 1, bumped)) return false;
    out = (bumped / a) * a;
    return true;
}

}  // namespace detail

// ---- buffer sizes required by the sealed ABI -------------------------------------------
constexpr uint64_t qk_plane_bytes(uint32_t m) { return static_cast<uint64_t>(m) * kQueryHeads * kHeadDim * sizeof(float); }
constexpr uint64_t value_plane_bytes(uint32_t m) { return static_cast<uint64_t>(m) * kValueHeads * kHeadDim * sizeof(float); }
constexpr uint64_t gate_plane_bytes(uint32_t m) { return static_cast<uint64_t>(m) * kValueHeads * sizeof(float); }
constexpr uint64_t parent_plane_bytes(uint32_t m) { return static_cast<uint64_t>(m) * sizeof(int32_t); }
// Staged kernel-layout inputs (query, key, value, decay, beta), each block 256-aligned.
constexpr uint64_t staged_inputs_plane_bytes(uint32_t m) {
    return detail::align_up(qk_plane_bytes(m), kArenaAlignment) * 2u +
           detail::align_up(value_plane_bytes(m), kArenaAlignment) +
           detail::align_up(gate_plane_bytes(m), kArenaAlignment) * 2u;
}

// ---- scratch sizing --------------------------------------------------------------------
struct ScratchPlan {
    uint32_t width = 0;
    uint32_t layers = 0;
    bool stage_inputs = false;
    uint64_t records_offset = 0, records_bytes = 0;   // one 128-record block per layer
    uint64_t output_offset = 0, output_bytes = 0;     // [m][32][128] FP32
    uint64_t parent_store_offset = 0, parent_store_bytes = 0;  // [m] int32, padded to the arena alignment
    uint64_t parent_source_offset = 0, parent_source_bytes = 0;  // [m] int32 staging for the producer
    uint64_t inputs_offset = 0, inputs_bytes = 0;     // optional staged kernel-layout inputs
    uint64_t staged_query_offset = 0, staged_key_offset = 0, staged_value_offset = 0;
    uint64_t staged_decay_offset = 0, staged_beta_offset = 0;
    uint64_t total_bytes = 0;
    std::string error;
    bool valid() const { return error.empty(); }
};

// Sizes the arena for one verify pass over `layers` GDN layers. Records stay durable until
// each layer is committed; output and staged inputs are reused per layer by the ordered queue.
inline ScratchPlan size_scratch(uint32_t width, uint32_t layers, bool stage_inputs) {
    ScratchPlan p;
    p.width = width;
    p.layers = layers;
    p.stage_inputs = stage_inputs;
    if (!is_sealed_width(width)) {
        p.error = "tree-WY width must be 5, 8 or 12";
        return p;
    }
    if (layers == 0) {
        p.error = "at least one layer is required";
        return p;
    }
    uint64_t at = 0;
    auto place = [&at](uint64_t bytes, uint64_t& offset, uint64_t& stored) -> bool {
        offset = at;
        stored = bytes;
        uint64_t next = 0;
        if (!detail::checked_add(at, bytes, next) || !detail::checked_align(next, kArenaAlignment, next)) return false;
        at = next;
        return true;
    };
    if (!detail::checked_mul(kRecordBlockBytes, layers, p.records_bytes)) {
        p.error = "record bytes overflow";
        return p;
    }
    if (!place(p.records_bytes, p.records_offset, p.records_bytes)) {
        p.error = "records do not fit the arena";
        return p;
    }
    if (!place(value_plane_bytes(width), p.output_offset, p.output_bytes)) {
        p.error = "output does not fit the arena";
        return p;
    }
    const uint64_t parent_store = detail::align_up(parent_plane_bytes(width), kArenaAlignment);
    if (!place(parent_store, p.parent_store_offset, p.parent_store_bytes)) {
        p.error = "parent store does not fit the arena";
        return p;
    }
    // The producer needs its own device staging for the parent chain: the pack kernel reads one
    // range and writes the sealed parent plane, so source and destination can never alias.
    if (!place(parent_store, p.parent_source_offset, p.parent_source_bytes)) {
        p.error = "parent staging does not fit the arena";
        return p;
    }
    if (stage_inputs) {
        p.inputs_offset = at;
        uint64_t ignored = 0;
        if (!place(qk_plane_bytes(width), p.staged_query_offset, ignored)) {
            p.error = "staged query does not fit the arena";
            return p;
        }
        if (!place(qk_plane_bytes(width), p.staged_key_offset, ignored)) {
            p.error = "staged key does not fit the arena";
            return p;
        }
        if (!place(value_plane_bytes(width), p.staged_value_offset, ignored)) {
            p.error = "staged value does not fit the arena";
            return p;
        }
        if (!place(gate_plane_bytes(width), p.staged_decay_offset, ignored)) {
            p.error = "staged decay does not fit the arena";
            return p;
        }
        if (!place(gate_plane_bytes(width), p.staged_beta_offset, ignored)) {
            p.error = "staged beta does not fit the arena";
            return p;
        }
        p.inputs_bytes = at - p.inputs_offset;
    }
    p.total_bytes = at;
    return p;
}

// Maps a scratch arena produced by size_scratch() to the sealed input planes. The parent store is
// always mapped because it is always sized; q/k/v/decay/beta are mapped only when staging was
// reserved. The caller copies token-major q/k [m][8][128], v [m][32][128], decay/beta [m][32] and
// parent [m] into these buffers; this layer never reads or rewrites model data.
struct StagedInputs {
    float* query = nullptr;
    float* key = nullptr;
    float* value = nullptr;
    float* decay = nullptr;
    float* beta = nullptr;
    int32_t* parent = nullptr;
    int32_t* parent_source = nullptr;
    bool ok() const { return query && key && value && decay && beta && parent && parent_source; }
};

inline StagedInputs staged_inputs(uint8_t* arena_base, const ScratchPlan& plan) {
    StagedInputs s;
    if (!arena_base || !plan.valid()) return s;
    s.parent = reinterpret_cast<int32_t*>(arena_base + plan.parent_store_offset);
    s.parent_source = reinterpret_cast<int32_t*>(arena_base + plan.parent_source_offset);
    if (!plan.stage_inputs) return s;
    s.query = reinterpret_cast<float*>(arena_base + plan.staged_query_offset);
    s.key = reinterpret_cast<float*>(arena_base + plan.staged_key_offset);
    s.value = reinterpret_cast<float*>(arena_base + plan.staged_value_offset);
    s.decay = reinterpret_cast<float*>(arena_base + plan.staged_decay_offset);
    s.beta = reinterpret_cast<float*>(arena_base + plan.staged_beta_offset);
    return s;
}

// ---- launch planning and dispatch selection ---------------------------------------------
enum class Dispatch : uint8_t { Unsupported = 0, TreeWy = 1, SequentialFallback = 2 };

struct PlanRequest {
    uint32_t width = 0;         // tree width the caller will submit; sealed widths are 5, 8 and 12
    uint32_t layers = 1;        // recurrent layers verified one launch at a time
    uint64_t budget_bytes = 0;  // scratch budget; 0 disables the check
    bool stage_inputs = false;  // reserve kernel-layout staging for q/k/v/decay/beta
};

struct LaunchPlan {
    Dispatch dispatch = Dispatch::Unsupported;
    uint32_t width = 0;
    uint32_t layers = 0;
    // Same geometry for the verify and commit entry points. grid_x is work-items, which is
    // what KfdRuntime::enqueue_kernel expects (it derives block_count_x); 128 * 256 = 32768.
    uint32_t workgroups = kWorkgroups;
    uint32_t grid_x = kWorkgroups * kBlockThreads;
    uint32_t grid_y = 1, grid_z = 1;
    uint32_t block_x = kBlockThreads, block_y = 1, block_z = 1;
    uint32_t dynamic_lds_bytes = 0;  // the sealed kernels use static LDS only
    ScratchPlan scratch;
    std::string reason;
    bool ok() const { return dispatch == Dispatch::TreeWy; }
};

inline LaunchPlan plan_launch(const PlanRequest& req) {
    LaunchPlan p;
    p.width = req.width;
    p.layers = req.layers;
    if (req.width == 0) {
        p.dispatch = Dispatch::Unsupported;
        p.reason = "tree-WY verify needs a nonzero width";
        return p;
    }
    if (!is_sealed_width(req.width)) {
        if (req.width > kSequentialMaxWidth) {
            p.dispatch = Dispatch::Unsupported;
            p.reason = "width " + std::to_string(req.width) + " exceeds the 16-slot sequential verify ring";
            return p;
        }
        p.dispatch = Dispatch::SequentialFallback;
        p.reason = "width " + std::to_string(req.width) +
                   " is not a sealed tree-WY width (5/8/12); keep the sequential spec-GDN path";
        return p;
    }
    p.scratch = size_scratch(req.width, req.layers, req.stage_inputs);
    if (!p.scratch.valid()) {
        p.dispatch = Dispatch::Unsupported;
        p.reason = p.scratch.error;
        return p;
    }
    if (req.budget_bytes != 0 && p.scratch.total_bytes > req.budget_bytes) {
        p.dispatch = Dispatch::SequentialFallback;
        p.reason = "tree-WY scratch " + std::to_string(p.scratch.total_bytes) + " B exceeds the " +
                   std::to_string(req.budget_bytes) + " B budget; keep the sequential spec-GDN path";
        return p;
    }
    p.dispatch = Dispatch::TreeWy;
    p.reason = "sealed tree-WY verify width " + std::to_string(req.width) + " over " +
               std::to_string(req.layers) + " layer(s)";
    return p;
}

// Chooses the largest sealed width that does not exceed `max_width` and fits the budget. Use this
// when the draft tree is built from scratch; a caller that already fixed a width must call
// plan_launch() instead, because shrinking a width would change the submitted tree.
struct WidthChoice {
    uint32_t width = 0;
    ScratchPlan scratch;
    bool fits = false;
    std::string error;
};

inline WidthChoice select_width(uint32_t max_width, uint32_t layers, uint64_t budget_bytes, bool stage_inputs) {
    WidthChoice choice;
    for (uint32_t width : {12u, 8u, 5u}) {
        if (width > max_width) continue;
        ScratchPlan candidate = size_scratch(width, layers, stage_inputs);
        if (!candidate.valid()) {
            choice.error = candidate.error;
            continue;
        }
        choice.scratch = candidate;
        if (budget_bytes != 0 && candidate.total_bytes > budget_bytes) continue;
        choice.width = width;
        choice.fits = true;
        choice.error.clear();
        return choice;
    }
    if (choice.error.empty()) choice.error = "no sealed tree-WY width fits the requested maximum and budget";
    return choice;
}

// ---- kernarg images --------------------------------------------------------------------
// Byte-exact HIP kernarg layouts. The explicit block ends at offset 76 for the verify kernel,
// so `reserved` is an explicit 8-byte pad that keeps the implicit kernarg block at offset 80,
// matching align_up(explicit bytes, 8) used by the loader.
struct VerifyArgs {
    uint64_t initial, query, key, value, decay, beta, parent, output, records;
    int32_t m;
    int32_t reserved;
};
static_assert(std::is_standard_layout<VerifyArgs>::value, "verify kernarg must be standard layout");
static_assert(sizeof(VerifyArgs) == 80, "verify kernarg image is 80 bytes");
static_assert(offsetof(VerifyArgs, records) == 64 && offsetof(VerifyArgs, m) == 72, "verify kernarg offsets");

struct CommitArgs {
    uint64_t records;
    uint64_t state;
    int32_t m;
    int32_t accepted;
};
static_assert(std::is_standard_layout<CommitArgs>::value, "commit kernarg must be standard layout");
static_assert(sizeof(CommitArgs) == 24, "commit kernarg image is 24 bytes");
static_assert(offsetof(CommitArgs, state) == 8 && offsetof(CommitArgs, accepted) == 20, "commit kernarg offsets");

struct BufferExtent {
    uint64_t ptr = 0;
    uint64_t bytes = 0;
};

struct VerifyBuffers {
    BufferExtent initial, query, key, value, decay, beta, parent, output, records;
};

struct CommitBuffers {
    BufferExtent records, state;
};

namespace detail {

inline bool check_extent(const char* name, const BufferExtent& b, uint64_t need, std::string& err) {
    if (b.ptr == 0) {
        err = std::string(name) + " pointer is null";
        return false;
    }
    if (b.bytes < need) {
        err = std::string(name) + " is smaller than the sealed ABI needs";
        return false;
    }
    if ((b.ptr & 3u) != 0) {
        err = std::string(name) + " is not 4-byte aligned";
        return false;
    }
    if (b.ptr > UINT64_MAX - b.bytes) {
        err = std::string(name) + " range overflows the address space";
        return false;
    }
    return true;
}

struct Span {
    const char* name;
    uint64_t begin;
    uint64_t end;
};

inline bool check_disjoint(const Span* spans, size_t count, std::string& err) {
    for (size_t i = 0; i < count; ++i) {
        for (size_t j = i + 1; j < count; ++j) {
            if (spans[i].begin < spans[j].end && spans[j].begin < spans[i].end) {
                err = std::string("buffers ") + spans[i].name + " and " + spans[j].name +
                      " overlap; the sealed kernels require disjoint ranges";
                return false;
            }
        }
    }
    return true;
}

}  // namespace detail

inline bool build_verify_args(const VerifyBuffers& b, uint32_t m, VerifyArgs& out, std::string& err) {
    err.clear();
    if (!is_sealed_width(m)) {
        err = "verify width must be 5, 8 or 12";
        return false;
    }
    struct Req {
        const char* name;
        const BufferExtent* extent;
        uint64_t need;
    };
    const Req req[] = {
        {"initial", &b.initial, kStateBytesPerLayer},
        {"query", &b.query, qk_plane_bytes(m)},
        {"key", &b.key, qk_plane_bytes(m)},
        {"value", &b.value, value_plane_bytes(m)},
        {"decay", &b.decay, gate_plane_bytes(m)},
        {"beta", &b.beta, gate_plane_bytes(m)},
        {"parent", &b.parent, parent_plane_bytes(m)},
        {"output", &b.output, value_plane_bytes(m)},
        {"records", &b.records, kRecordBlockBytes},
    };
    for (const Req& r : req)
        if (!detail::check_extent(r.name, *r.extent, r.need, err)) return false;
    detail::Span spans[sizeof(req) / sizeof(req[0])];
    for (size_t i = 0; i < sizeof(req) / sizeof(req[0]); ++i)
        spans[i] = detail::Span{req[i].name, req[i].extent->ptr, req[i].extent->ptr + req[i].need};
    if (!detail::check_disjoint(spans, sizeof(spans) / sizeof(spans[0]), err)) return false;

    out.initial = b.initial.ptr;
    out.query = b.query.ptr;
    out.key = b.key.ptr;
    out.value = b.value.ptr;
    out.decay = b.decay.ptr;
    out.beta = b.beta.ptr;
    out.parent = b.parent.ptr;
    out.output = b.output.ptr;
    out.records = b.records.ptr;
    out.m = static_cast<int32_t>(m);
    out.reserved = 0;
    return true;
}

inline bool build_commit_args(const CommitBuffers& b, uint32_t m, int32_t accepted, CommitArgs& out,
                              std::string& err) {
    err.clear();
    if (!is_sealed_width(m)) {
        err = "commit width must be 5, 8 or 12";
        return false;
    }
    if (accepted < 0 || accepted > static_cast<int32_t>(m)) {
        err = "accepted must be in [0, M]; 0 restores S0, N selects node N-1";
        return false;
    }
    if (!detail::check_extent("records", b.records, kRecordBlockBytes, err)) return false;
    if (!detail::check_extent("state", b.state, kStateBytesPerLayer, err)) return false;
    const detail::Span spans[2] = {{"records", b.records.ptr, b.records.ptr + kRecordBlockBytes},
                                   {"state", b.state.ptr, b.state.ptr + kStateBytesPerLayer}};
    if (!detail::check_disjoint(spans, 2, err)) return false;
    out.records = b.records.ptr;
    out.state = b.state.ptr;
    out.m = static_cast<int32_t>(m);
    out.accepted = accepted;
    return true;
}

// Layer L's durable record block inside a records arena sized by size_scratch().
inline bool records_layer_ptr(uint64_t records_base, uint32_t layer, uint32_t layers, uint64_t& out) {
    if (layer >= layers) return false;
    uint64_t offset = 0;
    if (!detail::checked_mul(kRecordBlockBytes, layer, offset)) return false;
    return detail::checked_add(records_base, offset, out);
}

// ---- tree acceptance and state action ---------------------------------------------------
// The sealed verify kernel writes records and output only; it never writes the live state.
// Therefore every accepted node - including the last node of a fully accepted chain - needs a
// commit launch. There is deliberately no slot-swap fast path here: the inactive state slot
// does not hold the accepted state after this verify.
enum class StateAction : uint8_t {
    None = 0,       // no accepted node: launch nothing, the live state is already the rollback state
    CommitNode = 1, // launch commit with Selection::commit_accepted
};

constexpr uint32_t kNoNode = 0xFFFFFFFFu;

struct Selection {
    StateAction action = StateAction::None;
    uint32_t accepted_nodes = 0;    // length of the accepted path
    uint32_t accepted_node_id = kNoNode;  // last accepted node index; kNoNode when nothing was accepted
    // CommitArgs::accepted. This is the selected node's id + 1, NOT the accepted-node count:
    // in a branching tree those differ, and the kernel selects a node, not a depth.
    int32_t commit_accepted = 0;
    int32_t correction_token = -1;  // target token at the rejected frontier, -1 when there is none
    bool bonus_pending = false;     // accepted a leaf: the caller must run one ordinary target step
    std::vector<uint32_t> path;     // accepted node ids, root first
    std::vector<int32_t> emitted;   // accepted draft tokens in order, then the correction token
    std::string error;
    bool ok() const { return error.empty(); }
};

inline bool validate_tree(const int32_t* parent, uint32_t width, std::string& err) {
    err.clear();
    if (!is_sealed_width(width)) {
        err = "tree-WY width must be 5, 8 or 12";
        return false;
    }
    if (!parent) {
        err = "parent array is null";
        return false;
    }
    for (uint32_t i = 0; i < width; ++i) {
        if (parent[i] < -1 || parent[i] >= static_cast<int32_t>(i)) {
            err = "parent[i] must be -1 (continues S0) or in [0, i)";
            return false;
        }
    }
    return true;
}

inline bool is_chain(const int32_t* parent, uint32_t width) {
    for (uint32_t i = 1; i < width; ++i)
        if (parent[i] != static_cast<int32_t>(i - 1)) return false;
    return true;
}

inline bool commit_required(const Selection& s) {
    return s.ok() && s.action == StateAction::CommitNode;
}

// Lowest-index child wins when two siblings propose the same verified token. This is a host
// policy, not a target-model decision; it must stay deterministic across replays.
inline Selection select_acceptance(const int32_t* parent, const int32_t* draft, const int32_t* verified,
                                   uint32_t width) {
    Selection s;
    std::string err;
    if (!validate_tree(parent, width, err)) {
        s.error = err;
        return s;
    }
    if (!draft || !verified) {
        s.error = "draft/verified token array is null";
        return s;
    }
    int32_t at = -1;
    for (;;) {
        int32_t pick = -1;
        for (uint32_t i = 0; i < width; ++i) {
            if (parent[i] != at || draft[i] != verified[i]) continue;
            pick = static_cast<int32_t>(i);
            break;
        }
        if (pick < 0) break;
        s.path.push_back(static_cast<uint32_t>(pick));
        s.emitted.push_back(draft[pick]);
        at = pick;
    }
    s.accepted_nodes = static_cast<uint32_t>(s.path.size());
    if (s.accepted_nodes != 0) {
        s.accepted_node_id = s.path.back();
        s.commit_accepted = static_cast<int32_t>(s.accepted_node_id) + 1;
    }
    int32_t frontier = -1;
    for (uint32_t i = 0; i < width; ++i) {
        if (parent[i] == at) {
            frontier = static_cast<int32_t>(i);
            break;
        }
    }
    if (frontier >= 0) {
        s.correction_token = verified[frontier];
        s.emitted.push_back(s.correction_token);
    } else {
        s.bonus_pending = true;
    }
    if (s.accepted_nodes == 0) {
        s.action = StateAction::None;
    } else {
        s.action = StateAction::CommitNode;
    }
    return s;
}

// Convenience wrapper: refuses to build a kernarg image for a selection that needs no commit.
inline bool make_commit_args(const Selection& s, const CommitBuffers& b, uint32_t m, CommitArgs& out,
                             std::string& err) {
    err.clear();
    if (!s.ok()) {
        err = s.error;
        return false;
    }
    if (s.action != StateAction::CommitNode) {
        err = "this selection does not launch a commit";
        return false;
    }
    return build_commit_args(b, m, s.commit_accepted, out, err);
}

}  // namespace vegajet::gdn::treewy_launch

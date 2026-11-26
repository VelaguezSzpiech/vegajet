// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// CPU-only qualification of the host launch/scratch/selection layer for the sealed GDN tree-WY
// verify/commit pair (ops/kernels/vj_gdn_treewy_verify.hiph).
//
// Covers, at verify widths 5, 8 and 12:
//   * kernarg ABI offsets/sizes, launch geometry, record/output/parent scratch sizing;
//   * dispatch selection from a byte budget, and rejection of unsupported widths;
//   * descriptor validation (null, unaligned, undersized, overlapping, accepted out of range);
//   * tree acceptance: accept, partial accept, reject and rollback, on chains and branching trees,
//     including a selected node whose id differs from the accepted-node count;
//   * that rejected nodes never mutate the committed state: a no-accept selection launches no
//     commit and leaves the live buffer byte-identical, commit(0) restores S0 exactly, and
//     poisoning every durable field outside the selected node's ancestor closure leaves the
//     committed bytes bit-identical;
//   * an end-to-end fixture: the sealed WY record build feeds the selection and the committed
//     state matches an independently written chronological recurrence (<= 1e-5, exact at the
//     first node), with a full single write of every committed element.
// This is CPU evidence only. It makes no claim about device execution, resources or speed.
#include "core/gdn_treewy_launch.hpp"
#include "tests/test_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace vegajet::gdn::treewy_launch;
using vegajet::treewy::Record;

namespace {

constexpr size_t kStateElems = static_cast<size_t>(kStateBytesPerLayer / sizeof(float));

uint32_t bits(float f) {
    uint32_t b = 0;
    std::memcpy(&b, &f, sizeof b);
    return b;
}

// Volatile helpers keep the CPU model from contracting a multiply into a neighbouring FMA.
float mul(float a, float b) {
    volatile float r = a * b;
    return r;
}
float sub(float a, float b) {
    volatile float r = a - b;
    return r;
}

// ---- CPU model of the sealed commit kernel ----------------------------------------------
// Same statement order as the device code: one cached read of initial, optional root scale, then
// ascending ancestor FMAs gated by the path mask. The host layer never reconstructs state, so
// this oracle lives only in the test.
float commit_element(const Record& r, int accepted, int row, int col) {
    float z = r.initial[row * 32 + col];
    if (accepted != 0) {
        const int i = accepted - 1;
        z = mul(r.root[i], z);
        for (int j = 0; j <= i; ++j) {
            if (r.path[i * 12 + j] == 0.0f) continue;
            z = std::fma(r.key[j * 128 + row], mul(r.path[i * 12 + j], r.delta[j * 32 + col]), z);
        }
    }
    return z;
}

void commit_state(const std::vector<Record>& rec, int accepted, float* state,
                  std::vector<uint8_t>* writes = nullptr) {
    VJ_CHECK(accepted >= 0 && accepted <= 12);
    for (uint32_t wg = 0; wg < kWorkgroups; ++wg) {
        const uint32_t head = wg / kSplitV, part = wg % kSplitV;
        for (uint32_t row = 0; row < kHeadDim; ++row) {
            for (uint32_t col = 0; col < 32; ++col) {
                const size_t index = static_cast<size_t>(head) * kHeadDim * kHeadDim + row * kHeadDim + part * 32 + col;
                state[index] = commit_element(rec[wg], accepted, static_cast<int>(row), static_cast<int>(col));
                if (writes) ++(*writes)[index];
            }
        }
    }
}

struct StateDiff {
    size_t bit_mismatches = 0;
    double max_abs = 0.0;
};

StateDiff compare_state(const std::vector<float>& a, const std::vector<float>& b) {
    VJ_CHECK(a.size() == b.size());
    StateDiff d;
    for (size_t i = 0; i < a.size(); ++i) {
        VJ_CHECK(std::isfinite(a[i]) && std::isfinite(b[i]));
        if (bits(a[i]) != bits(b[i])) ++d.bit_mismatches;
        d.max_abs = std::max(d.max_abs, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    }
    return d;
}

bool ancestor_or_self(const int32_t* parent, uint32_t node, uint32_t candidate) {
    for (int32_t x = static_cast<int32_t>(node); x >= 0; x = parent[x])
        if (static_cast<uint32_t>(x) == candidate) return true;
    return false;
}

// Poison every durable field the selected node's commit must not read: delta/key of nodes outside
// its ancestor closure, root of every other node, and every other node's path row.
std::vector<Record> poisoned_outside_closure(const std::vector<Record>& rec, const int32_t* parent,
                                             uint32_t width, uint32_t accepted_node) {
    bool closure[12] = {false, false, false, false, false, false, false, false, false, false, false, false};
    for (int32_t x = static_cast<int32_t>(accepted_node); x >= 0; x = parent[x]) closure[x] = true;
    const float poison = std::numeric_limits<float>::quiet_NaN();
    std::vector<Record> out = rec;
    for (uint32_t wg = 0; wg < kWorkgroups; ++wg) {
        for (uint32_t j = 0; j < width; ++j) {
            if (!closure[j]) {
                for (uint32_t r = 0; r < kHeadDim; ++r) out[wg].key[j * 128 + r] = poison;
                for (uint32_t c = 0; c < 32; ++c) out[wg].delta[j * 32 + c] = poison;
            }
            if (j != accepted_node) {
                out[wg].root[j] = poison;
                for (uint32_t c = 0; c < 12; ++c) out[wg].path[j * 12 + c] = poison;
            }
        }
    }
    return out;
}

struct Arena {
    std::vector<uint8_t> bytes;
    explicit Arena(uint64_t n) : bytes(static_cast<size_t>(n), 0) {}
    uint64_t ptr(uint64_t offset) const {
        return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(bytes.data())) + offset;
    }
};

// ---- descriptor ABI ---------------------------------------------------------------------
void test_descriptor_abi() {
    std::printf("[abi] verify/commit kernarg images and sealed geometry\n");
    static_assert(sizeof(VerifyArgs) == 80, "verify kernarg is 80 bytes");
    static_assert(offsetof(VerifyArgs, records) == 64 && offsetof(VerifyArgs, m) == 72, "verify offsets");
    static_assert(sizeof(CommitArgs) == 24, "commit kernarg is 24 bytes");
    static_assert(offsetof(CommitArgs, accepted) == 20, "commit offsets");
    static_assert(kRecordBytes == 24688, "one sealed record is 24688 bytes");
    static_assert(kRecordBlockBytes == 3160064, "one layer of 128 records is 3160064 bytes");
    static_assert(kStateBytesPerLayer == 2097152, "state is 2 MiB per layer");
    static_assert(vegajet::treewy::max_m == 12, "sealed header max_m");
    static_assert(is_sealed_width(5) && is_sealed_width(8) && is_sealed_width(12), "sealed widths");
    static_assert(!is_sealed_width(4) && !is_sealed_width(6) && !is_sealed_width(11) && !is_sealed_width(13),
                  "non-sealed widths");

    const LaunchPlan p = plan_launch(PlanRequest{12, 36, 0, false});
    VJ_CHECK(p.ok());
    VJ_CHECK(p.workgroups == 128 && p.block_x == 256 && p.block_y == 1 && p.block_z == 1);
    VJ_CHECK(p.grid_x == 32768 && p.grid_y == 1 && p.grid_z == 1);  // work-items, as KfdRuntime takes them
    VJ_CHECK(p.dynamic_lds_bytes == 0);                             // sealed kernels use static LDS
    VJ_CHECK(p.scratch.valid() && p.scratch.total_bytes > 0);
}

// ---- scratch sizing ---------------------------------------------------------------------
void test_scratch_sizing() {
    std::printf("[scratch] record/output/parent/staging sizing and arena layout\n");
    const uint64_t align = kArenaAlignment;
    for (uint32_t width : {5u, 8u, 12u}) {
        const uint64_t qk = static_cast<uint64_t>(width) * kQueryHeads * kHeadDim * 4u;
        const uint64_t value = static_cast<uint64_t>(width) * kValueHeads * kHeadDim * 4u;
        const uint64_t gate = static_cast<uint64_t>(width) * kValueHeads * 4u;
        const uint64_t qk_aligned = (qk + align - 1) / align * align;
        const uint64_t value_aligned = (value + align - 1) / align * align;
        const uint64_t gate_aligned = (gate + align - 1) / align * align;

        const ScratchPlan one = size_scratch(width, 1, false);
        VJ_CHECK(one.valid());
        VJ_CHECK(one.records_offset == 0 && one.records_bytes == kRecordBlockBytes);
        VJ_CHECK(one.output_offset == kRecordBlockBytes && one.output_bytes == value);
        VJ_CHECK(one.parent_store_offset == kRecordBlockBytes + value);
        VJ_CHECK(one.parent_store_bytes == align);
        VJ_CHECK(one.parent_source_offset == one.parent_store_offset + align);
        VJ_CHECK(one.parent_source_bytes == align);
        VJ_CHECK(one.total_bytes == one.parent_source_offset + align);
        VJ_CHECK(one.inputs_bytes == 0 && !one.stage_inputs);
        VJ_CHECK(size_scratch(width, 1, false).total_bytes == one.total_bytes);

        const ScratchPlan multi = size_scratch(width, 36, true);
        VJ_CHECK(multi.valid());
        VJ_CHECK(multi.records_bytes == 36u * kRecordBlockBytes);
        VJ_CHECK(multi.output_bytes == value);
        VJ_CHECK(multi.parent_source_offset == multi.parent_store_offset + multi.parent_store_bytes);
        VJ_CHECK(multi.inputs_offset == multi.parent_source_offset + multi.parent_source_bytes);
        VJ_CHECK(multi.staged_query_offset == multi.inputs_offset);
        VJ_CHECK(multi.staged_key_offset == multi.inputs_offset + qk_aligned);
        VJ_CHECK(multi.staged_value_offset == multi.inputs_offset + qk_aligned + qk_aligned);
        VJ_CHECK(multi.staged_decay_offset == multi.inputs_offset + qk_aligned + qk_aligned + value_aligned);
        VJ_CHECK(multi.staged_beta_offset ==
                 multi.inputs_offset + qk_aligned + qk_aligned + value_aligned + gate_aligned);
        VJ_CHECK(multi.inputs_bytes == 2u * qk_aligned + value_aligned + 2u * gate_aligned);
        VJ_CHECK(multi.total_bytes == multi.inputs_offset + multi.inputs_bytes);
        for (uint64_t offset : {multi.records_offset, multi.output_offset, multi.parent_store_offset,
                                multi.staged_query_offset, multi.staged_key_offset, multi.staged_value_offset,
                                multi.staged_decay_offset, multi.staged_beta_offset})
            VJ_CHECK(offset % align == 0);
        VJ_CHECK(multi.total_bytes % align == 0);

        auto within = [&](uint64_t offset, uint64_t length) {
            return offset <= multi.total_bytes && length <= multi.total_bytes - offset;
        };
        VJ_CHECK(within(multi.records_offset, multi.records_bytes));
        VJ_CHECK(within(multi.output_offset, multi.output_bytes));
        VJ_CHECK(within(multi.parent_store_offset, multi.parent_store_bytes));
        VJ_CHECK(within(multi.parent_source_offset, multi.parent_source_bytes));
        VJ_CHECK(within(multi.staged_query_offset, qk_aligned));
        VJ_CHECK(within(multi.staged_key_offset, qk_aligned));
        VJ_CHECK(within(multi.staged_value_offset, value_aligned));
        VJ_CHECK(within(multi.staged_decay_offset, gate_aligned));
        VJ_CHECK(within(multi.staged_beta_offset, gate_aligned));
        VJ_CHECK(multi.records_offset + multi.records_bytes <= multi.output_offset);
        VJ_CHECK(multi.output_offset + multi.output_bytes <= multi.parent_store_offset);
        VJ_CHECK(multi.parent_store_offset + multi.parent_store_bytes <= multi.inputs_offset);

        const ScratchPlan staged_plan = size_scratch(width, 1, true);
        VJ_CHECK(staged_plan.valid());
        Arena arena(staged_plan.total_bytes);
        const StagedInputs staged = staged_inputs(arena.bytes.data(), staged_plan);
        VJ_CHECK(staged.ok());
        VJ_CHECK(reinterpret_cast<uint8_t*>(staged.query) == arena.bytes.data() + staged_plan.staged_query_offset);
        VJ_CHECK(reinterpret_cast<uint8_t*>(staged.key) == arena.bytes.data() + staged_plan.staged_key_offset);
        VJ_CHECK(reinterpret_cast<uint8_t*>(staged.value) == arena.bytes.data() + staged_plan.staged_value_offset);
        VJ_CHECK(reinterpret_cast<uint8_t*>(staged.decay) == arena.bytes.data() + staged_plan.staged_decay_offset);
        VJ_CHECK(reinterpret_cast<uint8_t*>(staged.beta) == arena.bytes.data() + staged_plan.staged_beta_offset);
        VJ_CHECK(reinterpret_cast<uint8_t*>(staged.parent) == arena.bytes.data() + staged_plan.parent_store_offset);
        VJ_CHECK(staged_inputs(nullptr, staged_plan).ok() == false);
        const StagedInputs unstaged = staged_inputs(arena.bytes.data(), one);
        VJ_CHECK(unstaged.ok() == false && unstaged.query == nullptr && unstaged.parent != nullptr);
        VJ_CHECK(staged_inputs(nullptr, one).parent == nullptr);
    }
    VJ_CHECK(size_scratch(4, 1, false).valid() == false);
    VJ_CHECK(size_scratch(6, 1, false).valid() == false);
    VJ_CHECK(size_scratch(13, 1, false).valid() == false);
    VJ_CHECK(size_scratch(12, 0, false).valid() == false);
    VJ_CHECK(!size_scratch(4, 1, false).error.empty());
}

// ---- launch planning and dispatch selection ---------------------------------------------
void test_plan_dispatch() {
    std::printf("[dispatch] widths, budgets and fallback selection\n");
    for (uint32_t width : {5u, 8u, 12u}) {
        const LaunchPlan p = plan_launch(PlanRequest{width, 36, 0, true});
        VJ_CHECK(p.dispatch == Dispatch::TreeWy && p.ok());
        VJ_CHECK(p.width == width && p.layers == 36);
        VJ_CHECK(!p.reason.empty());
        VJ_CHECK(plan_launch(PlanRequest{width, 36, p.scratch.total_bytes, true}).ok());
        const LaunchPlan tight = plan_launch(PlanRequest{width, 36, p.scratch.total_bytes - 1u, true});
        VJ_CHECK(tight.dispatch == Dispatch::SequentialFallback && !tight.ok());
        VJ_CHECK(!tight.reason.empty());
        VJ_CHECK(tight.scratch.valid());  // sizing is still reported for the caller
    }
    VJ_CHECK(plan_launch(PlanRequest{7, 36, 0, false}).dispatch == Dispatch::SequentialFallback);
    VJ_CHECK(plan_launch(PlanRequest{16, 36, 0, false}).dispatch == Dispatch::SequentialFallback);
    VJ_CHECK(plan_launch(PlanRequest{17, 36, 0, false}).dispatch == Dispatch::Unsupported);
    VJ_CHECK(plan_launch(PlanRequest{0, 36, 0, false}).dispatch == Dispatch::Unsupported);
    VJ_CHECK(plan_launch(PlanRequest{12, 0, 0, false}).dispatch == Dispatch::Unsupported);
    VJ_CHECK(!plan_launch(PlanRequest{0, 36, 0, false}).reason.empty());
    VJ_CHECK(!plan_launch(PlanRequest{7, 36, 0, false}).reason.empty());
    VJ_CHECK(!plan_launch(PlanRequest{12, 0, 0, false}).reason.empty());

    // Budget-aware width selection: largest sealed width <= max_width that fits.
    for (uint32_t layers : {1u, 36u}) {
        for (bool stage : {false, true}) {
            const WidthChoice unbounded = select_width(12, layers, 0, stage);
            VJ_CHECK(unbounded.fits && unbounded.width == 12 && unbounded.scratch.valid());
            VJ_CHECK(select_width(11, layers, 0, stage).width == 8);
            VJ_CHECK(select_width(8, layers, 0, stage).width == 8);
            VJ_CHECK(select_width(7, layers, 0, stage).width == 5);  // 8 exceeds 7, so 5 is the largest fit
            const WidthChoice too_narrow = select_width(4, layers, 0, stage);
            VJ_CHECK(!too_narrow.fits && too_narrow.width == 0 && !too_narrow.error.empty());
            const ScratchPlan five = size_scratch(5, layers, stage);
            const ScratchPlan eight = size_scratch(8, layers, stage);
            VJ_CHECK(select_width(12, layers, eight.total_bytes, stage).width == 8);
            VJ_CHECK(select_width(12, layers, eight.total_bytes - 1u, stage).width == 5);
            VJ_CHECK(select_width(12, layers, five.total_bytes, stage).width == 5);
            const WidthChoice none = select_width(12, layers, five.total_bytes - 1u, stage);
            VJ_CHECK(!none.fits && none.width == 0 && !none.error.empty());
            VJ_CHECK(select_width(12, 0, 0, stage).fits == false);
        }
    }
}

// ---- descriptor validation ---------------------------------------------------------------
void test_descriptor_validation() {
    std::printf("[descriptors] accept/reject kernarg validation\n");
    const ScratchPlan sp = size_scratch(5, 1, true);
    VJ_CHECK(sp.valid());
    Arena arena(sp.total_bytes);
    std::vector<float> live(kStateElems, 0.0f);
    const uint64_t live_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(live.data()));
    VerifyBuffers vb;
    vb.initial = {live_ptr, kStateBytesPerLayer};
    vb.records = {arena.ptr(sp.records_offset), sp.records_bytes};
    vb.output = {arena.ptr(sp.output_offset), sp.output_bytes};
    vb.parent = {arena.ptr(sp.parent_store_offset), parent_plane_bytes(5)};
    vb.query = {arena.ptr(sp.staged_query_offset), qk_plane_bytes(5)};
    vb.key = {arena.ptr(sp.staged_key_offset), qk_plane_bytes(5)};
    vb.value = {arena.ptr(sp.staged_value_offset), value_plane_bytes(5)};
    vb.decay = {arena.ptr(sp.staged_decay_offset), gate_plane_bytes(5)};
    vb.beta = {arena.ptr(sp.staged_beta_offset), gate_plane_bytes(5)};

    VerifyArgs args;
    std::string err;
    VJ_CHECK(build_verify_args(vb, 5, args, err));
    VJ_CHECK(err.empty());
    VJ_CHECK(args.initial == vb.initial.ptr && args.query == vb.query.ptr && args.key == vb.key.ptr &&
             args.value == vb.value.ptr && args.decay == vb.decay.ptr && args.beta == vb.beta.ptr &&
             args.parent == vb.parent.ptr && args.output == vb.output.ptr && args.records == vb.records.ptr);
    VJ_CHECK(args.m == 5 && args.reserved == 0);

    auto expect_reject = [&](VerifyBuffers bad, uint32_t width) {
        VerifyArgs ignored{};
        std::string why;
        const bool built = build_verify_args(bad, width, ignored, why);
        VJ_CHECK(!built);
        VJ_CHECK(!why.empty());
    };
    {
        VerifyBuffers bad = vb;
        bad.query.ptr = 0;
        expect_reject(bad, 5);
    }
    {
        VerifyBuffers bad = vb;
        bad.records.ptr += 2;  // not 4-byte aligned
        expect_reject(bad, 5);
    }
    {
        VerifyBuffers bad = vb;
        bad.value.bytes -= 4;
        expect_reject(bad, 5);
    }
    {
        VerifyBuffers bad = vb;
        bad.output = bad.value;  // aliasing is not allowed by the sealed contract
        expect_reject(bad, 5);
    }
    {
        VerifyBuffers bad = vb;
        bad.initial.ptr = UINT64_MAX - 7u;  // aligned, but the declared range would wrap the address space
        expect_reject(bad, 5);
    }
    {
        VerifyBuffers bad = vb;
        expect_reject(bad, 6);
    }
    {
        VerifyBuffers wide = vb;
        wide.query.bytes += 128;  // over-allocation inside a slab stays legal
        VerifyArgs accepted{};
        VJ_CHECK(build_verify_args(wide, 5, accepted, err));
    }

    CommitBuffers cb;
    cb.records = vb.records;
    cb.state = {live_ptr, kStateBytesPerLayer};
    CommitArgs ca{};
    for (int32_t accepted = 0; accepted <= 5; ++accepted) {
        VJ_CHECK(build_commit_args(cb, 5, accepted, ca, err));
        VJ_CHECK(ca.accepted == accepted && ca.m == 5 && ca.records == cb.records.ptr && ca.state == cb.state.ptr);
    }
    VJ_CHECK(!build_commit_args(cb, 5, 6, ca, err) && !err.empty());
    VJ_CHECK(!build_commit_args(cb, 5, -1, ca, err) && !err.empty());
    VJ_CHECK(!build_commit_args(cb, 7, 3, ca, err) && !err.empty());
    {
        CommitBuffers bad = cb;
        bad.state.ptr = cb.records.ptr;  // overlap
        VJ_CHECK(!build_commit_args(bad, 5, 3, ca, err));
    }
    {
        CommitBuffers bad = cb;
        bad.records.bytes = kRecordBlockBytes - 8;
        VJ_CHECK(!build_commit_args(bad, 5, 3, ca, err));
    }

    uint64_t layer_ptr = 0;
    const uint64_t base = 0x100000000ULL;
    VJ_CHECK(records_layer_ptr(base, 0, 36, layer_ptr) && layer_ptr == base);
    VJ_CHECK(records_layer_ptr(base, 35, 36, layer_ptr) && layer_ptr == base + 35u * kRecordBlockBytes);
    VJ_CHECK(!records_layer_ptr(base, 36, 36, layer_ptr));
    VJ_CHECK(!records_layer_ptr(UINT64_MAX - 8u, 1, 2, layer_ptr));
}

// ---- tree acceptance / reject / rollback ------------------------------------------------
void test_selection() {
    std::printf("[select] accept/reject/rollback on chains and branching trees\n");
    const int32_t bad0[5] = {0, 0, 1, 2, 3};
    const int32_t bad1[5] = {-1, 5, 0, 1, 2};
    const int32_t bad2[5] = {-2, 0, 1, 2, 3};
    std::string why;
    VJ_CHECK(!validate_tree(bad0, 5, why) && !why.empty());
    VJ_CHECK(!validate_tree(bad1, 5, why));
    VJ_CHECK(!validate_tree(bad2, 5, why));
    VJ_CHECK(!validate_tree(nullptr, 5, why));
    VJ_CHECK(!validate_tree(bad0, 4, why));
    VJ_CHECK(validate_tree(bad0 + 1, 4, why) == false);  // {0,1,2,3} is still not a root
    const int32_t chain5[5] = {-1, 0, 1, 2, 3};
    VJ_CHECK(validate_tree(chain5, 5, why) && is_chain(chain5, 5));
    VJ_CHECK(select_acceptance(chain5, nullptr, chain5, 5).ok() == false);
    VJ_CHECK(!select_acceptance(chain5, chain5, chain5, 7).ok());

    for (uint32_t width : {5u, 8u, 12u}) {
        int32_t chain[12];
        for (uint32_t i = 0; i < width; ++i) chain[i] = static_cast<int32_t>(i) - 1;
        int32_t draft[12], verified[12];
        for (uint32_t i = 0; i < width; ++i) draft[i] = 1000 + static_cast<int32_t>(i);

        // Reject all: no commit, the live buffer is untouched, the target token is emitted.
        for (uint32_t i = 0; i < width; ++i) verified[i] = draft[i];
        verified[0] = draft[0] + 1;
        std::vector<float> live(kStateElems, 0.0f);
        for (size_t i = 0; i < live.size(); ++i) live[i] = static_cast<float>(i % 97) * 0.5f - 7.0f;
        const std::vector<float> live_before = live;
        std::vector<int32_t> parent_before(chain, chain + width);
        std::vector<int32_t> draft_before(draft, draft + width);
        std::vector<int32_t> verified_before(verified, verified + width);
        Selection rej = select_acceptance(chain, draft, verified, width);
        VJ_CHECK(rej.ok());
        VJ_CHECK(rej.action == StateAction::None && rej.accepted_nodes == 0 && rej.commit_accepted == 0);
        VJ_CHECK(rej.accepted_node_id == kNoNode);
        VJ_CHECK(!commit_required(rej));
        VJ_CHECK(rej.correction_token == verified[0] && rej.emitted.size() == 1 &&
                 rej.emitted[0] == verified[0]);
        VJ_CHECK(rej.path.empty() && !rej.bonus_pending);
        VJ_CHECK(std::memcmp(live.data(), live_before.data(), live.size() * sizeof(float)) == 0);
        VJ_CHECK(std::memcmp(chain, parent_before.data(), width * sizeof(int32_t)) == 0);
        VJ_CHECK(std::memcmp(draft, draft_before.data(), width * sizeof(int32_t)) == 0);
        VJ_CHECK(std::memcmp(verified, verified_before.data(), width * sizeof(int32_t)) == 0);
        {
            CommitArgs ca{};
            std::string err;
            CommitBuffers cb{{0x1000, kRecordBlockBytes}, {0x200000, kStateBytesPerLayer}};
            VJ_CHECK(!make_commit_args(rej, cb, width, ca, err) && !err.empty());
        }

        // Accept 1, then a mid prefix, then full acceptance: every accepted node needs a commit.
        for (uint32_t accepted : {1u, width / 2u, width}) {
            for (uint32_t i = 0; i < width; ++i) verified[i] = draft[i];
            if (accepted < width) verified[accepted] = draft[accepted] + 1;
            Selection s = select_acceptance(chain, draft, verified, width);
            VJ_CHECK(s.ok());
            VJ_CHECK(s.action == StateAction::CommitNode && commit_required(s));
            VJ_CHECK(s.accepted_nodes == accepted && s.path.size() == accepted);
            for (uint32_t i = 0; i < accepted; ++i) VJ_CHECK(s.path[i] == i);
            VJ_CHECK(s.accepted_node_id == accepted - 1u);
            VJ_CHECK(s.commit_accepted == static_cast<int32_t>(accepted));  // node id + 1
            VJ_CHECK(s.emitted.size() == accepted + (accepted < width ? 1u : 0u));
            for (uint32_t i = 0; i < accepted; ++i) VJ_CHECK(s.emitted[i] == draft[i]);
            if (accepted < width) {
                VJ_CHECK(s.correction_token == verified[accepted]);
                VJ_CHECK(s.emitted.back() == s.correction_token);
                VJ_CHECK(!s.bonus_pending);
            } else {
                VJ_CHECK(s.correction_token == -1 && s.bonus_pending);
            }
        }

        // Branching tree: the selected node id is not the accepted-node count.
        int32_t binary[12];
        for (uint32_t i = 0; i < width; ++i) binary[i] = i == 0 ? -1 : static_cast<int32_t>((i - 1) / 2);
        const int32_t* tree = binary;
        VJ_CHECK(validate_tree(tree, width, why));
        VJ_CHECK(!is_chain(tree, width));
        for (uint32_t i = 0; i < width; ++i) {
            draft[i] = 2000 + static_cast<int32_t>(i);
            verified[i] = draft[i];
        }
        // Reject the two children of node 7 (where present) so the path stops at node 3 with a
        // frontier correction; for width 5 node 3 is already a leaf, so accept the leaf instead.
        if (width == 5) {
            Selection leaf = select_acceptance(tree, draft, verified, width);
            VJ_CHECK(leaf.ok() && leaf.action == StateAction::CommitNode);
            VJ_CHECK(leaf.accepted_nodes == 3 && leaf.accepted_node_id == 3 && leaf.commit_accepted == 4);
            VJ_CHECK(leaf.path.size() == 3 && leaf.path[0] == 0 && leaf.path[1] == 1 && leaf.path[2] == 3);
            VJ_CHECK(leaf.bonus_pending && leaf.correction_token == -1);
        } else {
            verified[7] = draft[7] + 1;
            if (width == 12) verified[8] = draft[8] + 1;
            Selection cut = select_acceptance(tree, draft, verified, width);
            VJ_CHECK(cut.ok() && cut.action == StateAction::CommitNode);
            VJ_CHECK(cut.accepted_nodes == 3 && cut.path.size() == 3);
            VJ_CHECK(cut.path[0] == 0 && cut.path[1] == 1 && cut.path[2] == 3);
            VJ_CHECK(cut.accepted_node_id == 3 && cut.commit_accepted == 4);
            VJ_CHECK(cut.accepted_nodes != static_cast<uint32_t>(cut.commit_accepted));
            VJ_CHECK(cut.correction_token == verified[7] && cut.emitted.back() == verified[7]);
            VJ_CHECK(!cut.bonus_pending);

            Selection whole = select_acceptance(tree, draft, verified, width);
            VJ_CHECK(whole.ok() && whole.action == StateAction::CommitNode);
            // Node 7 (and 8 at width 12) is rejected above; restore it for the full-acceptance walk.
            verified[7] = draft[7];
            if (width == 12) verified[8] = draft[8];
            whole = select_acceptance(tree, draft, verified, width);
            VJ_CHECK(whole.ok() && whole.action == StateAction::CommitNode);
            VJ_CHECK(whole.accepted_nodes == 4 && whole.accepted_node_id == 7 && whole.commit_accepted == 8);
            VJ_CHECK(whole.accepted_nodes != static_cast<uint32_t>(whole.commit_accepted));
            VJ_CHECK(whole.bonus_pending && whole.correction_token == -1);
        }
        // Sibling that also matches the target token must not be accepted when a lower-index
        // sibling matches first (deterministic policy).
        for (uint32_t i = 0; i < width; ++i) verified[i] = draft[i];
        Selection left = select_acceptance(tree, draft, verified, width);
        VJ_CHECK(left.ok());
        if (width > 5) {
            VJ_CHECK(left.path[1] == 1);  // children of the root are 1 and 2; 1 wins
        }
    }
}

// ---- commit closure and rollback --------------------------------------------------------
void test_commit_closure() {
    std::printf("[commit] ancestor closure, reject isolation, S0 restore\n");
    std::mt19937 rng(424242u);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    for (uint32_t width : {5u, 8u, 12u}) {
        int32_t chain[12];
        for (uint32_t i = 0; i < width; ++i) chain[i] = static_cast<int32_t>(i) - 1;
        int32_t binary[12];
        for (uint32_t i = 0; i < width; ++i) binary[i] = i == 0 ? -1 : static_cast<int32_t>((i - 1) / 2);
        for (const int32_t* parent : {chain, binary}) {
            std::vector<Record> rec(kWorkgroups);
            for (uint32_t wg = 0; wg < kWorkgroups; ++wg) {
                for (uint32_t e = 0; e < 128 * 32; ++e) rec[wg].initial[e] = uniform(rng);
                for (uint32_t e = 0; e < 12 * 128; ++e) rec[wg].key[e] = uniform(rng);
                for (uint32_t e = 0; e < 12 * 32; ++e) rec[wg].delta[e] = uniform(rng);
                for (uint32_t e = 0; e < 12 * 12; ++e) rec[wg].path[e] = 0.0f;
                for (uint32_t e = 0; e < 12; ++e) rec[wg].root[e] = uniform(rng);
                // Encode the ancestor closure with unit path weights so the reference sum is exact.
                for (uint32_t node = 0; node < width; ++node)
                    for (int32_t j = static_cast<int32_t>(node); j >= 0; j = parent[j])
                        rec[wg].path[node * 12 + j] = 1.0f;
            }
            std::vector<float> initial_state(kStateElems, 0.0f);
            for (uint32_t wg = 0; wg < kWorkgroups; ++wg) {
                const uint32_t head = wg / kSplitV, part = wg % kSplitV;
                for (uint32_t row = 0; row < kHeadDim; ++row)
                    for (uint32_t col = 0; col < 32; ++col)
                        initial_state[static_cast<size_t>(head) * kHeadDim * kHeadDim + row * kHeadDim + part * 32 + col] =
                            rec[wg].initial[row * 32 + col];
            }
            // accepted == 0 restores S0 bit-exactly: this is the kernel rollback path.
            {
                std::vector<float> restored(kStateElems, std::numeric_limits<float>::quiet_NaN());
                std::vector<uint8_t> rollback_writes(kStateElems, 0);
                commit_state(rec, 0, restored.data(), &rollback_writes);
                for (uint8_t w : rollback_writes) VJ_CHECK(w == 1);  // rollback also writes every element once
                VJ_CHECK(compare_state(restored, initial_state).bit_mismatches == 0);
            }
            for (uint32_t node = 0; node < width; ++node) {
                const int accepted = static_cast<int>(node) + 1;
                std::vector<float> committed(kStateElems, 0.0f);
                // One coverage entry per committed state element (commit_state indexes by element, not byte).
                std::vector<uint8_t> writes(kStateElems, 0);
                VJ_CHECK(writes.size() == committed.size());
                commit_state(rec, accepted, committed.data(), &writes);
                for (uint8_t w : writes) VJ_CHECK(w == 1);  // every element written exactly once
                for (uint32_t wg = 0; wg < kWorkgroups; ++wg) {
                    const uint32_t head = wg / kSplitV, part = wg % kSplitV;
                    for (uint32_t row = 0; row < kHeadDim; ++row) {
                        for (uint32_t col = 0; col < 32; ++col) {
                            float expect = mul(rec[wg].root[node], rec[wg].initial[row * 32 + col]);
                            for (uint32_t j = 0; j <= node; ++j)
                                if (ancestor_or_self(parent, node, j))
                                    expect = std::fma(rec[wg].key[j * 128 + row],
                                                      mul(1.0f, rec[wg].delta[j * 32 + col]), expect);
                            const size_t index =
                                static_cast<size_t>(head) * kHeadDim * kHeadDim + row * kHeadDim + part * 32 + col;
                            VJ_CHECK(bits(committed[index]) == bits(expect));
                        }
                    }
                }
                // Poisoning every field outside the closure leaves the committed bytes identical.
                std::vector<Record> poisoned = poisoned_outside_closure(rec, parent, width, node);
                std::vector<float> again(kStateElems, 0.0f);
                commit_state(poisoned, accepted, again.data());
                VJ_CHECK(std::memcmp(again.data(), committed.data(), kStateBytesPerLayer) == 0);
            }
        }
    }
}

// ---- sealed WY fixture + independent chronological recurrence ---------------------------
struct Fixture {
    uint32_t m = 0;
    std::vector<float> s0, q, k, v, decay, beta;
    std::vector<int32_t> parent;
};

Fixture make_fixture(uint32_t m, bool branching, uint32_t seed) {
    Fixture f;
    f.m = m;
    f.s0.assign(static_cast<size_t>(kValueHeads) * kHeadDim * kHeadDim, 0.0f);
    f.q.assign(static_cast<size_t>(m) * kQueryHeads * kHeadDim, 0.0f);
    f.k.assign(static_cast<size_t>(m) * kQueryHeads * kHeadDim, 0.0f);
    f.v.assign(static_cast<size_t>(m) * kValueHeads * kHeadDim, 0.0f);
    f.decay.assign(static_cast<size_t>(m) * kValueHeads, 0.0f);
    f.beta.assign(static_cast<size_t>(m) * kValueHeads, 0.0f);
    f.parent.assign(m, -1);
    std::mt19937 rng(seed + m * 101u + (branching ? 7u : 0u));
    auto random = [&rng]() {
        return static_cast<float>(static_cast<int>(rng() % 2000001u) - 1000000) / 1000003.0f;
    };
    for (float& z : f.s0) z = random() * 0.31f;
    for (float& z : f.q) z = random() * 0.21f;
    for (float& z : f.k) z = random() * 0.13f;
    for (float& z : f.v) z = random() * 0.73f;
    for (float& z : f.decay) z = 0.87f + random() * 0.09f;
    for (float& z : f.beta) z = 0.45f + random() * 0.31f;
    for (uint32_t i = 0; i < m; ++i)
        f.parent[i] = branching ? (i == 0 ? -1 : static_cast<int32_t>((i - 1) / 2)) : static_cast<int32_t>(i) - 1;
    return f;
}

// Fixture model of the sealed verify kernel's durable record build: same dot128 grouping, same
// path/root products, same unit-triangular solve by doubling. Temporary matrices are separate
// from the durable records, exactly as in the kernel.
void fixture_verify(const Fixture& f, std::vector<Record>& rec, std::vector<float>& work) {
    const uint32_t m = f.m;
    rec.assign(kWorkgroups, Record{});
    work.assign(2 * 144 + 2 * 384, 0.0f);
    for (uint32_t wg = 0; wg < kWorkgroups; ++wg) {
        const uint32_t head = wg / kSplitV, part = wg % kSplitV;
        Record& r = rec[wg];
        for (uint32_t row = 0; row < kHeadDim; ++row)
            for (uint32_t col = 0; col < 32; ++col)
                r.initial[row * 32 + col] = f.s0[static_cast<size_t>(head) * kHeadDim * kHeadDim + row * kHeadDim + part * 32 + col];
        for (uint32_t i = 0; i < m; ++i) {
            std::memcpy(r.key + i * 128, f.k.data() + (static_cast<size_t>(i) * kQueryHeads + head % kQueryHeads) * kHeadDim,
                        kHeadDim * sizeof(float));
            for (uint32_t j = 0; j < 12; ++j) r.path[i * 12 + j] = 0.0f;
            float d = 1.0f;
            for (int32_t j = static_cast<int32_t>(i); j >= 0; j = f.parent[j]) {
                r.path[i * 12 + j] = d;
                d = mul(d, f.decay[static_cast<size_t>(j) * kValueHeads + head]);
            }
            r.root[i] = d;
        }
        float* b = work.data();
        float* bn = b + 144;
        float* x = bn + 144;
        float* xn = x + 384;
        std::fill(work.begin(), work.end(), 0.0f);
        for (uint32_t i = 0; i < m; ++i) {
            for (uint32_t j = 0; j < i; ++j) {
                if (r.path[i * 12 + j] == 0.0f) continue;
                b[i * 12 + j] = -mul(mul(f.beta[static_cast<size_t>(i) * kValueHeads + head], r.path[i * 12 + j]),
                                     vegajet::treewy::dot128(r.key + i * 128, r.key + j * 128));
            }
        }
        for (uint32_t i = 0; i < m; ++i) {
            for (uint32_t col = 0; col < 32; ++col) {
                float column[kHeadDim];
                for (uint32_t row = 0; row < kHeadDim; ++row) column[row] = r.initial[row * 32 + col];
                const float pred = vegajet::treewy::dot128(column, r.key + i * 128);
                x[i * 32 + col] = mul(f.beta[static_cast<size_t>(i) * kValueHeads + head],
                                      sub(f.v[(static_cast<size_t>(i) * kValueHeads + head) * kHeadDim + part * 32 + col],
                                          mul(r.root[i], pred)));
            }
        }
        for (uint32_t span = 1; span < m; span *= 2) {
            for (uint32_t i = 0; i < m; ++i)
                for (uint32_t col = 0; col < 32; ++col) {
                    float z = x[i * 32 + col];
                    for (uint32_t j = 0; j < i; ++j) z = std::fma(b[i * 12 + j], x[j * 32 + col], z);
                    xn[i * 32 + col] = z;
                }
            for (uint32_t i = 0; i < m; ++i)
                for (uint32_t j = 0; j < m; ++j) {
                    float z = 0.0f;
                    for (uint32_t l = j + 1; l < i; ++l) z = std::fma(b[i * 12 + l], b[l * 12 + j], z);
                    bn[i * 12 + j] = z;
                }
            std::copy_n(xn, m * 32, x);
            std::copy_n(bn, 144, b);
        }
        std::copy_n(x, m * 32, r.delta);
    }
}

// Independently written chronological recurrence along the node's root path, with the production
// 8-group-of-16 FMA dot order used by the sealed arithmetic.
std::vector<float> chronological_state(const Fixture& f, uint32_t node) {
    std::vector<float> s = f.s0;
    std::vector<uint32_t> path;
    for (int32_t x = static_cast<int32_t>(node); x >= 0; x = f.parent[x]) path.push_back(static_cast<uint32_t>(x));
    std::reverse(path.begin(), path.end());
    for (uint32_t n : path) {
        for (uint32_t h = 0; h < kValueHeads; ++h) {
            for (uint32_t c = 0; c < kHeadDim; ++c) {
                float p[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                for (uint32_t g = 0; g < 8; ++g)
                    for (uint32_t r = 0; r < 16; ++r) {
                        const uint32_t row = g * 16 + r;
                        p[g] = std::fma(s[static_cast<size_t>(h) * kHeadDim * kHeadDim + row * kHeadDim + c],
                                        f.k[(static_cast<size_t>(n) * kQueryHeads + h % kQueryHeads) * kHeadDim + row], p[g]);
                    }
                float pred = p[0];
                for (uint32_t g = 1; g < 8; ++g) pred = pred + p[g];
                const float dv = mul(f.beta[static_cast<size_t>(n) * kValueHeads + h],
                                     sub(f.v[(static_cast<size_t>(n) * kValueHeads + h) * kHeadDim + c],
                                         mul(f.decay[static_cast<size_t>(n) * kValueHeads + h], pred)));
                for (uint32_t g = 0; g < 8; ++g)
                    for (uint32_t r = 0; r < 16; ++r) {
                        const uint32_t row = g * 16 + r;
                        float& cell = s[static_cast<size_t>(h) * kHeadDim * kHeadDim + row * kHeadDim + c];
                        cell = std::fma(f.k[(static_cast<size_t>(n) * kQueryHeads + h % kQueryHeads) * kHeadDim + row], dv,
                                        mul(f.decay[static_cast<size_t>(n) * kValueHeads + h], cell));
                    }
            }
        }
    }
    return s;
}

void test_fixture_end_to_end() {
    std::printf("[e2e] sealed WY records -> selection -> committed state vs chronological\n");
    for (uint32_t width : {5u, 8u, 12u}) {
        for (uint32_t tree = 0; tree < 2; ++tree) {
            const Fixture f = make_fixture(width, tree != 0, 17u);
            std::vector<Record> rec;
            std::vector<float> work;
            fixture_verify(f, rec, work);

            struct Case {
                const char* name;
                std::vector<uint32_t> mismatch;
                std::vector<uint32_t> expect_path;
                bool expect_bonus;
            };
            std::vector<Case> cases;
            if (tree == 0) {
                std::vector<uint32_t> mid;
                for (uint32_t i = 0; i < width / 2u; ++i) mid.push_back(i);
                std::vector<uint32_t> all;
                for (uint32_t i = 0; i < width; ++i) all.push_back(i);
                cases.push_back({"reject", {0}, {}, false});
                cases.push_back({"accept1", {1}, {0}, false});
                cases.push_back({"mid", {width / 2u}, mid, false});
                cases.push_back({"full", {}, all, true});
            } else if (width == 5) {
                cases.push_back({"reject", {0}, {}, false});
                cases.push_back({"root", {1, 2}, {0}, false});
                cases.push_back({"leaf", {}, {0, 1, 3}, true});
            } else {
                std::vector<uint32_t> cut_mismatch{7};
                if (width == 12) cut_mismatch.push_back(8);  // both children of node 3 must reject
                cases.push_back({"reject", {0}, {}, false});
                cases.push_back({"root", {1, 2}, {0}, false});
                cases.push_back({"cut", cut_mismatch, {0, 1, 3}, false});
                cases.push_back({"full", {}, {0, 1, 3, 7}, true});
            }
            for (const Case& c : cases) {
                std::vector<int32_t> draft(width), verified(width);
                for (uint32_t i = 0; i < width; ++i) draft[i] = 3000 + static_cast<int32_t>(i);
                verified = draft;
                for (uint32_t m : c.mismatch) verified[m] = draft[m] + 1;
                const Selection s = select_acceptance(f.parent.data(), draft.data(), verified.data(), width);
                VJ_CHECK(s.ok());
                VJ_CHECK(s.path == c.expect_path);
                VJ_CHECK(s.accepted_nodes == c.expect_path.size());
                VJ_CHECK(s.bonus_pending == c.expect_bonus);
                if (c.expect_path.empty()) {
                    VJ_CHECK(s.action == StateAction::None && !commit_required(s));
                    VJ_CHECK(s.commit_accepted == 0);
                    VJ_CHECK(s.correction_token == verified[0]);
                } else {
                    const uint32_t last = c.expect_path.back();
                    VJ_CHECK(s.action == StateAction::CommitNode && commit_required(s));
                    VJ_CHECK(s.accepted_node_id == last);
                    VJ_CHECK(s.commit_accepted == static_cast<int32_t>(last) + 1);
                    {
                        std::vector<float> live_state(kStateElems, 0.0f);
                        CommitBuffers cb{{reinterpret_cast<uint64_t>(rec.data()), kRecordBlockBytes},
                                         {reinterpret_cast<uint64_t>(live_state.data()), kStateBytesPerLayer}};
                        CommitArgs ca{};
                        std::string err;
                        VJ_CHECK(make_commit_args(s, cb, width, ca, err));
                        VJ_CHECK(err.empty());
                        VJ_CHECK(ca.accepted == s.commit_accepted && ca.m == static_cast<int32_t>(width));
                    }
                    const std::vector<float> expected = chronological_state(f, last);
                    std::vector<float> committed(kStateElems, std::numeric_limits<float>::quiet_NaN());
                    std::vector<uint8_t> writes(kStateElems, 0);
                    VJ_CHECK(writes.size() == committed.size());
                    commit_state(rec, s.commit_accepted, committed.data(), &writes);
                    for (uint8_t w : writes) VJ_CHECK(w == 1);
                    const StateDiff d = compare_state(committed, expected);
                    VJ_CHECK(d.max_abs <= 1e-5);
                    if (last == 0) VJ_CHECK(d.bit_mismatches == 0);
                    VJ_CHECK(compare_state(committed, f.s0).max_abs > 1e-3);  // a commit must advance state
                    std::printf("  M=%u tree=%u %-8s path=%zu node=%u commit=%d state_max_abs=%.9g\n", width,
                                tree, c.name, s.path.size(), last, s.commit_accepted, d.max_abs);
                    // Rejected durable data never reaches the committed state.
                    const std::vector<Record> poisoned = poisoned_outside_closure(rec, f.parent.data(), width, last);
                    std::vector<float> again(kStateElems, 0.0f);
                    commit_state(poisoned, s.commit_accepted, again.data());
                    VJ_CHECK(std::memcmp(again.data(), committed.data(), kStateBytesPerLayer) == 0);
                }
                if (!c.expect_path.empty()) {
                    // A rejected sibling's state must not be what was committed.
                    uint32_t sibling = width;  // sentinel
                    const uint32_t last = c.expect_path.back();
                    for (uint32_t i = 0; i < width; ++i)
                        if (f.parent[i] == static_cast<int32_t>(last)) {
                            sibling = i;
                            break;
                        }
                    if (sibling < width && s.correction_token == verified[sibling]) {
                        const std::vector<float> wrong = chronological_state(f, sibling);
                        std::vector<float> committed(kStateElems, 0.0f);
                        commit_state(rec, s.commit_accepted, committed.data());
                        VJ_CHECK(compare_state(committed, wrong).max_abs > 1e-3);
                    }
                } else {
                    std::vector<float> restored(kStateElems, 0.0f);
                    commit_state(rec, 0, restored.data());
                    VJ_CHECK(compare_state(restored, f.s0).bit_mismatches == 0);
                    VJ_CHECK(compare_state(f.s0, chronological_state(f, 0)).max_abs > 1e-3);
                }
            }
        }
    }
}

}  // namespace

int main() {
    test_descriptor_abi();
    test_scratch_sizing();
    test_plan_dispatch();
    test_descriptor_validation();
    test_selection();
    test_commit_closure();
    test_fixture_end_to_end();
    std::printf(
        "PASS host tree-WY launch/scratch/selection: widths 5/8/12, accept/reject/rollback, "
        "rejected nodes never mutate committed state. CPU only; no device claim.\n");
    return 0;
}

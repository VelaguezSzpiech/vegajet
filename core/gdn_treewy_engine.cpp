// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/gdn_treewy_engine.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>

namespace vegajet::gdn::treewy_engine {

namespace tl = vegajet::gdn::treewy_launch;

const char* fallback_reason_name(FallbackReason r) {
    switch (r) {
        case FallbackReason::None: return "none";
        case FallbackReason::Disabled: return "disabled";
        case FallbackReason::WidthNotSealed: return "width-not-sealed";
        case FallbackReason::WidthExceedsSequential: return "width-exceeds-sequential";
        case FallbackReason::EmptyLayers: return "empty-layers";
        case FallbackReason::TooManyLayers: return "too-many-layers";
        case FallbackReason::InvalidParent: return "invalid-parent";
        case FallbackReason::InvalidDraft: return "invalid-draft";
        case FallbackReason::InvalidVerified: return "invalid-verified";
        case FallbackReason::ContextRootRejected: return "context-root-rejected";
        case FallbackReason::BudgetExceeded: return "budget-exceeded";
        case FallbackReason::BackendUnavailable: return "backend-unavailable";
        case FallbackReason::DescriptorInvalid: return "descriptor-invalid";
        case FallbackReason::WireUnavailable: return "wire-unavailable";
        case FallbackReason::WireProduceFailed: return "wire-produce-failed";
        case FallbackReason::IncompleteLayers: return "incomplete-layers";
        case FallbackReason::VerifyFailed: return "verify-failed";
        case FallbackReason::NormFailed: return "norm-failed";
        case FallbackReason::CommitFailed: return "commit-failed";
        case FallbackReason::ConvSlotMismatch: return "conv-slot-mismatch";
        case FallbackReason::StateSlotAlias: return "state-slot-alias";
    }
    return "unknown";
}

void Stats::reset() {
    *this = Stats{};
}

void Stats::event(Op op, uint32_t layer, uint32_t node, bool ok, uint64_t bytes) {
    if (trace.size() >= kTraceCapacity) trace.erase(trace.begin());
    trace.push_back(TraceEvent{op, layer, node, ok, bytes});
}

std::string Stats::summary() const {
    char buf[2048];
    std::snprintf(buf, sizeof buf,
                  "treewy=%llu fallback=%llu verify=%llu norm=%llu commit=%llu restore=%llu "
                  "full=%llu partial=%llu reject=%llu root_only=%llu accepted_nodes=%llu committed_nodes=%llu "
                  "emitted=%llu bonus=%llu cursor_advance=%llu cursor_rollback=%llu flips=%llu "
                  "descriptor_err=%llu backend_err=%llu records_bytes=%llu state_bytes=%llu "
                  "wire_bytes=%llu output_bytes=%llu",
                  static_cast<unsigned long long>(treewy_selected),
                  static_cast<unsigned long long>(fallbacks),
                  static_cast<unsigned long long>(verify_launches),
                  static_cast<unsigned long long>(norm_launches),
                  static_cast<unsigned long long>(commit_launches),
                  static_cast<unsigned long long>(restore_launches),
                  static_cast<unsigned long long>(full_accepts),
                  static_cast<unsigned long long>(partial_accepts),
                  static_cast<unsigned long long>(rejects),
                  static_cast<unsigned long long>(root_only),
                  static_cast<unsigned long long>(accepted_nodes),
                  static_cast<unsigned long long>(committed_nodes),
                  static_cast<unsigned long long>(emitted_tokens),
                  static_cast<unsigned long long>(bonus_pending),
                  static_cast<unsigned long long>(cursor_advances),
                  static_cast<unsigned long long>(cursor_rollbacks),
                  static_cast<unsigned long long>(state_flips),
                  static_cast<unsigned long long>(descriptor_errors),
                  static_cast<unsigned long long>(backend_errors),
                  static_cast<unsigned long long>(bytes_records),
                  static_cast<unsigned long long>(bytes_state),
                  static_cast<unsigned long long>(bytes_wire),
                  static_cast<unsigned long long>(bytes_output));
    return std::string(buf);
}

namespace {

bool overlap(uint64_t a0, uint64_t a1, uint64_t b0, uint64_t b1) {
    return a0 < b1 && b0 < a1;
}

bool range_ok(uint64_t ptr, uint64_t bytes) {
    return ptr != 0 && bytes != 0 && ptr <= UINT64_MAX - bytes;
}

}  // namespace

bool Session::fail(FallbackReason reason, const std::string& message) {
    last_reason_ = reason;
    last_message_ = message;
    blocked_reason_ = message;
    ++stats_.fallbacks;
    stats_.event(Op::Fallback, 0, tl::kNoNode, false, 0);
    active_ = false;
    verified_all_ = false;
    normalized_all_ = false;
    return false;
}

Decision Session::configure(const Request& req) {
    if (active_) abort();
    req_ = req;
    layers_.clear();
    blocked_reason_.clear();
    last_reason_ = FallbackReason::None;
    last_message_.clear();
    verified_all_ = false;
    normalized_all_ = false;
    ++stats_.configured;

    Decision d;
    const auto fallback = [&](FallbackReason reason, const std::string& msg) {
        d.dispatch = tl::Dispatch::SequentialFallback;
        d.reason = msg;
        last_reason_ = reason;
        last_message_ = msg;
        blocked_reason_ = msg;
        ++stats_.fallbacks;
        stats_.event(Op::Fallback, 0, tl::kNoNode, false, 0);
        return d;
    };

    if (req.width == 0) return fallback(FallbackReason::WidthNotSealed, "tree-WY needs a nonzero width");
    if (req.width > tl::kSequentialMaxWidth) {
        const std::string msg =
            "width " + std::to_string(req.width) + " exceeds the sequential verify ring";
        last_reason_ = FallbackReason::WidthExceedsSequential;
        last_message_ = msg;
        blocked_reason_ = msg;
        ++stats_.fallbacks;
        stats_.event(Op::Fallback, 0, tl::kNoNode, false, 0);
        d.dispatch = tl::Dispatch::Unsupported;
        d.reason = msg;
        return d;
    }
    if (!tl::is_sealed_width(req.width)) {
        return fallback(FallbackReason::WidthNotSealed,
                        "width " + std::to_string(req.width) + " is not a sealed tree-WY width (5/8/12)");
    }
    if (req.layers == 0) return fallback(FallbackReason::EmptyLayers, "at least one GDN layer is required");
    if (req.layers > kMaxLayers) {
        return fallback(FallbackReason::TooManyLayers,
                        "layer count " + std::to_string(req.layers) + " exceeds the caller limit");
    }
    if (req.parent.size() != req.width) {
        return fallback(FallbackReason::InvalidParent, "parent array does not match the width");
    }
    if (req.draft.size() != req.width) {
        return fallback(FallbackReason::InvalidDraft, "draft token array does not match the width");
    }
    std::string tree_err;
    if (!tl::validate_tree(req.parent.data(), req.width, tree_err)) {
        return fallback(FallbackReason::InvalidParent, tree_err);
    }
    if (req.context_is_root && req.parent[0] != -1) {
        return fallback(FallbackReason::InvalidParent, "node 0 must continue S0 (parent -1)");
    }
    for (uint32_t i = 0; i < req.width; ++i) {
        if (req.draft[i] == std::numeric_limits<int32_t>::min()) {
            return fallback(FallbackReason::InvalidDraft, "draft token is not a valid token id");
        }
    }
    if (!req.verified.empty()) {
        if (req.verified.size() != req.width) {
            return fallback(FallbackReason::InvalidVerified, "verified token array does not match the width");
        }
        for (uint32_t i = 0; i < req.width; ++i) {
            if (req.verified[i] == std::numeric_limits<int32_t>::min()) {
                return fallback(FallbackReason::InvalidVerified, "verified token is not a valid token id");
            }
        }
    }
    if (req.conv_slots == 0 || conv_slots_ == 0) {
        return fallback(FallbackReason::ConvSlotMismatch, "the conv ring must have at least one slot");
    }
    if (req.conv_slots != conv_slots_) {
        return fallback(FallbackReason::ConvSlotMismatch,
                        "the request conv-ring slot count differs from this session");
    }

    tl::PlanRequest plan_req;
    plan_req.width = req.width;
    plan_req.layers = req.layers;
    plan_req.budget_bytes = req.budget_bytes;
    plan_req.stage_inputs = req.produce_wire;
    const tl::LaunchPlan plan = tl::plan_launch(plan_req);
    if (plan.dispatch != tl::Dispatch::TreeWy) {
        FallbackReason reason = FallbackReason::DescriptorInvalid;
        if (!plan.scratch.valid()) reason = FallbackReason::DescriptorInvalid;
        if (req.budget_bytes != 0 && plan.scratch.valid() &&
            plan.scratch.total_bytes > req.budget_bytes) {
            reason = FallbackReason::BudgetExceeded;
        }
        if (plan.dispatch == tl::Dispatch::SequentialFallback && !tl::is_sealed_width(req.width))
            reason = FallbackReason::WidthNotSealed;
        return fallback(reason, plan.reason);
    }

    snapshot_conv_head_ = req.conv_head % req.conv_slots;
    selected_conv_head_ = snapshot_conv_head_;
    layers_.resize(req.layers);
    active_ = true;
    ++stats_.treewy_selected;
    d.dispatch = tl::Dispatch::TreeWy;
    d.reason = plan.reason;
    return d;
}

bool Session::validate_conv_slots(uint32_t layer, std::string& err) const {
    const LayerBinding& b = layers_[layer].binding;
    if (req_.conv_slots == 0) {
        err = "conv ring has no slots";
        return false;
    }
    for (uint32_t i = 0; i < req_.width; ++i) {
        const uint32_t expect = (snapshot_conv_head_ + i + 1u) % req_.conv_slots;
        if (b.has_conv_slots) {
            if (b.conv_slot[i] != expect) {
                err = "layer " + std::to_string(layer) + " node " + std::to_string(i) +
                      " conv slot " + std::to_string(b.conv_slot[i]) +
                      " does not match the sealed-wire order " + std::to_string(expect);
                return false;
            }
        } else if (b.conv_slot[i] != 0u) {
            err = "layer " + std::to_string(layer) + " has conv slots but did not set has_conv_slots";
            return false;
        }
    }
    return true;
}

bool Session::validate_buffers(uint32_t layer, std::string& err) const {
    const LayerBinding& b = layers_[layer].binding;
    if (!tl::is_sealed_width(req_.width)) {
        err = "width is not sealed";
        return false;
    }
    if (!b.wire.complete()) {
        err = "layer " + std::to_string(layer) + " sealed wire planes are incomplete";
        return false;
    }
    if (!b.parent) {
        err = "layer " + std::to_string(layer) + " parent buffer is null";
        return false;
    }
    if (b.records == 0 || b.output == 0) {
        err = "layer " + std::to_string(layer) + " records/output buffer is null";
        return false;
    }
    if (!range_ok(b.state_slot[0], tl::kStateBytesPerLayer) ||
        !range_ok(b.state_slot[1], tl::kStateBytesPerLayer)) {
        err = "layer " + std::to_string(layer) + " state slot range overflow";
        return false;
    }
    if (b.state_slot[0] == b.state_slot[1]) {
        err = "layer " + std::to_string(layer) + " active and inactive state slots alias";
        return false;
    }
    if (b.active_slot > 1u) {
        err = "layer " + std::to_string(layer) + " active slot index is not 0 or 1";
        return false;
    }
    if (req_.require_norm && !b.norm.complete()) {
        err = "layer " + std::to_string(layer) + " post-verify norm/quant references are incomplete";
        return false;
    }
    if (b.norm.enabled && b.norm.raw_output != b.output) {
        err = "layer " + std::to_string(layer) +
              " norm stage is not pointed at this layer's sealed raw output";
        return false;
    }

    tl::VerifyBuffers vb;
    vb.initial = {b.state_slot[b.active_slot], tl::kStateBytesPerLayer};
    vb.query = {b.wire.query, tl::qk_plane_bytes(req_.width)};
    vb.key = {b.wire.key, tl::qk_plane_bytes(req_.width)};
    vb.value = {b.wire.value, tl::value_plane_bytes(req_.width)};
    vb.decay = {b.wire.decay, tl::gate_plane_bytes(req_.width)};
    vb.beta = {b.wire.beta, tl::gate_plane_bytes(req_.width)};
    vb.parent = {b.wire.parent, tl::parent_plane_bytes(req_.width)};
    vb.output = {b.output, tl::value_plane_bytes(req_.width)};
    vb.records = {b.records, tl::kRecordBlockBytes};
    tl::VerifyArgs va{};
    if (!tl::build_verify_args(vb, req_.width, va, err)) return false;

    tl::CommitBuffers cb;
    cb.records = {b.records, tl::kRecordBlockBytes};
    cb.state = {b.state_slot[b.active_slot ^ 1u], tl::kStateBytesPerLayer};
    tl::CommitArgs ca{};
    if (!tl::build_commit_args(cb, req_.width, 0, ca, err)) return false;

    const uint64_t active0 = b.state_slot[b.active_slot];
    const uint64_t active1 = active0 + tl::kStateBytesPerLayer;
    const uint64_t inactive0 = b.state_slot[b.active_slot ^ 1u];
    const uint64_t inactive1 = inactive0 + tl::kStateBytesPerLayer;
    const uint64_t rec0 = b.records, rec1 = rec0 + tl::kRecordBlockBytes;
    const uint64_t out0 = b.output, out1 = out0 + tl::value_plane_bytes(req_.width);
    const uint64_t q0 = b.wire.query, q1 = q0 + tl::qk_plane_bytes(req_.width);
    const uint64_t k0 = b.wire.key, k1 = k0 + tl::qk_plane_bytes(req_.width);
    const uint64_t v0 = b.wire.value, v1 = v0 + tl::value_plane_bytes(req_.width);
    const uint64_t d0 = b.wire.decay, d1 = d0 + tl::gate_plane_bytes(req_.width);
    const uint64_t be0 = b.wire.beta, be1 = be0 + tl::gate_plane_bytes(req_.width);
    const uint64_t p0 = b.wire.parent, p1 = p0 + tl::parent_plane_bytes(req_.width);
    const uint64_t ps0 = b.parent_source, ps1 = ps0 + tl::parent_plane_bytes(req_.width);
    if (req_.produce_wire) {
        if (!range_ok(ps0, tl::parent_plane_bytes(req_.width))) {
            err = "layer " + std::to_string(layer) + " parent staging is null";
            return false;
        }
        if (b.conv_source == 0 || b.decay_source == 0 || b.beta_source == 0) {
            err = "layer " + std::to_string(layer) + " producer source buffers are incomplete";
            return false;
        }
    }
    if (overlap(active0, active1, inactive0, inactive1)) {
        err = "layer " + std::to_string(layer) + " state slots overlap";
        return false;
    }
    const uint64_t guards[][2] = {{rec0, rec1}, {out0, out1}, {q0, q1}, {k0, k1}, {v0, v1},
                                  {d0, d1}, {be0, be1}, {p0, p1}, {ps0, ps1}};
    for (const auto& gr : guards) {
        if (overlap(inactive0, inactive1, gr[0], gr[1])) {
            err = "layer " + std::to_string(layer) + " inactive state slot overlaps verify scratch";
            return false;
        }
        if (overlap(rec0, rec1, gr[0], gr[1]) && gr[0] != rec0) {
            err = "layer " + std::to_string(layer) + " records overlap verify scratch";
            return false;
        }
    }
    if (req_.produce_wire) {
        // The producer reads conv/decay/beta/parent staging and writes the sealed planes. A source
        // aliasing a destination, a durable record block or the committed slot would corrupt the
        // cycle (or a previous width's staged rows), so it is a pre-dispatch failure.
        const uint64_t conv_bytes =
            static_cast<uint64_t>(req_.width) * b.conv_stride * sizeof(float);
        if (!range_ok(b.conv_source, conv_bytes)) {
            err = "layer " + std::to_string(layer) + " conv source range overflow";
            return false;
        }
        const uint64_t src_spans[][2] = {
            {b.conv_source, b.conv_source + conv_bytes},
            {b.decay_source, b.decay_source + tl::gate_plane_bytes(req_.width)},
            {b.beta_source, b.beta_source + tl::gate_plane_bytes(req_.width)},
            {ps0, ps1},
        };
        const uint64_t dst_spans[][2] = {{out0, out1}, {q0, q1}, {k0, k1}, {v0, v1},
                                         {d0, d1}, {be0, be1}, {p0, p1}, {rec0, rec1},
                                         {inactive0, inactive1}, {active0, active1}};
        for (size_t si = 0; si < sizeof(src_spans) / sizeof(src_spans[0]); ++si) {
            for (size_t di = 0; di < sizeof(dst_spans) / sizeof(dst_spans[0]); ++di) {
                if (overlap(src_spans[si][0], src_spans[si][1], dst_spans[di][0], dst_spans[di][1])) {
                    err = "layer " + std::to_string(layer) +
                          " producer source overlaps verify scratch or state";
                    return false;
                }
            }
            for (size_t sj = si + 1; sj < sizeof(src_spans) / sizeof(src_spans[0]); ++sj) {
                if (overlap(src_spans[si][0], src_spans[si][1], src_spans[sj][0], src_spans[sj][1])) {
                    err = "layer " + std::to_string(layer) + " producer sources overlap";
                    return false;
                }
            }
        }
    }
    if (b.norm.enabled) {
        const uint64_t q80 = b.norm.out_q8;
        const uint64_t q81 = q80 + static_cast<uint64_t>(req_.width) * b.norm.nv * 128u;
        const uint64_t ds0 = b.norm.out_ds;
        const uint64_t ds1 = ds0 + static_cast<uint64_t>(req_.width) * b.norm.nv * 4u * 8u;
        if (!range_ok(q80, q81 - q80) || !range_ok(ds0, ds1 - ds0)) {
            err = "layer " + std::to_string(layer) + " norm output range overflow";
            return false;
        }
        if (overlap(q80, q81, ds0, ds1)) {
            err = "layer " + std::to_string(layer) + " norm q8 and scale outputs overlap";
            return false;
        }
        if (overlap(q80, q81, out0, out1) || overlap(ds0, ds1, out0, out1)) {
            err = "layer " + std::to_string(layer) + " norm output overlaps the raw output";
            return false;
        }
        if (overlap(q80, q81, rec0, rec1) || overlap(ds0, ds1, rec0, rec1)) {
            err = "layer " + std::to_string(layer) + " norm output overlaps durable records";
            return false;
        }
        if (overlap(q80, q81, inactive0, inactive1) || overlap(ds0, ds1, inactive0, inactive1) ||
            overlap(q80, q81, active0, active1) || overlap(ds0, ds1, active0, active1)) {
            err = "layer " + std::to_string(layer) + " norm output overlaps a state slot";
            return false;
        }
    }
    return true;
}

bool Session::bind_layer(uint32_t layer, const LayerBinding& binding, std::string& err) {
    err.clear();
    if (!active_) {
        err = blocked_reason_.empty() ? "tree-WY session is not active" : blocked_reason_;
        return false;
    }
    if (layer >= layers_.size()) {
        err = "layer index out of range";
        return false;
    }
    if (layers_[layer].bound) {
        err = "layer already bound";
        return false;
    }
    layers_[layer].binding = binding;
    layers_[layer].bound = true;
    layers_[layer].verified = false;
    layers_[layer].normalized = false;
    layers_[layer].commit_enqueued = false;

    std::string why;
    if (!validate_conv_slots(layer, why)) {
        ++stats_.descriptor_errors;
        fail(FallbackReason::ConvSlotMismatch, why);
        err = why;
        return false;
    }
    if (!validate_buffers(layer, why)) {
        ++stats_.descriptor_errors;
        fail(FallbackReason::DescriptorInvalid, why);
        err = why;
        return false;
    }
    stats_.bytes_records += tl::kRecordBlockBytes;
    stats_.bytes_state += 2u * tl::kStateBytesPerLayer;
    stats_.bytes_wire += tl::qk_plane_bytes(req_.width) * 2u + tl::value_plane_bytes(req_.width) +
                         tl::gate_plane_bytes(req_.width) * 2u + tl::parent_plane_bytes(req_.width);
    stats_.bytes_output += tl::value_plane_bytes(req_.width);
    return true;
}

bool Session::enqueue_verify(Backend& backend, uint32_t layer, std::string& err) {
    err.clear();
    if (!active_) {
        err = blocked_reason_.empty() ? "tree-WY session is not active" : blocked_reason_;
        return false;
    }
    if (layer >= layers_.size() || !layers_[layer].bound) {
        err = "layer is not bound";
        return false;
    }
    LayerState& ls = layers_[layer];
    if (ls.verified) {
        err = "layer verify already enqueued";
        return false;
    }
    const LayerBinding& b = ls.binding;
    if (req_.produce_wire) {
        WireRequest wr;
        wr.layer = layer;
        wr.device = b.device;
        wr.width = req_.width;
        wr.nk = b.nk;
        wr.nv = b.nv;
        wr.conv_stride = b.conv_stride;
        wr.conv_source = b.conv_source;
        wr.decay_source = b.decay_source;
        wr.beta_source = b.beta_source;
        wr.parent_source = b.parent_source;
        wr.out = b.wire;
        if (!backend.produce_wire(wr, err)) {
            ++stats_.backend_errors;
            fail(FallbackReason::WireProduceFailed,
                 "sealed-wire producer failed for layer " + std::to_string(layer) + ": " + err);
            return false;
        }
        ++stats_.wire_produced;
    } else {
        ++stats_.wire_supplied;
    }

    tl::VerifyBuffers vb;
    vb.initial = {b.state_slot[b.active_slot], tl::kStateBytesPerLayer};
    vb.query = {b.wire.query, tl::qk_plane_bytes(req_.width)};
    vb.key = {b.wire.key, tl::qk_plane_bytes(req_.width)};
    vb.value = {b.wire.value, tl::value_plane_bytes(req_.width)};
    vb.decay = {b.wire.decay, tl::gate_plane_bytes(req_.width)};
    vb.beta = {b.wire.beta, tl::gate_plane_bytes(req_.width)};
    vb.parent = {b.wire.parent, tl::parent_plane_bytes(req_.width)};
    vb.output = {b.output, tl::value_plane_bytes(req_.width)};
    vb.records = {b.records, tl::kRecordBlockBytes};
    VerifyCall call;
    call.layer = layer;
    call.device = b.device;
    call.launch = tl::plan_launch(tl::PlanRequest{req_.width, req_.layers, 0, req_.produce_wire});
    if (!call.launch.ok()) {
        ++stats_.descriptor_errors;
        fail(FallbackReason::DescriptorInvalid, call.launch.reason);
        err = call.launch.reason;
        return false;
    }
    if (!tl::build_verify_args(vb, req_.width, call.args, err)) {
        ++stats_.descriptor_errors;
        fail(FallbackReason::DescriptorInvalid, err);
        return false;
    }
    if (!backend.verify(call, err)) {
        ++stats_.backend_errors;
        fail(FallbackReason::VerifyFailed,
             "sealed verify launch failed for layer " + std::to_string(layer) + ": " + err);
        return false;
    }
    ls.verified = true;
    ++stats_.verify_launches;
    stats_.event(Op::Verify, layer, tl::kNoNode, true, tl::kRecordBlockBytes);
    verified_all_ = true;
    for (const LayerState& s : layers_) {
        if (!s.verified) {
            verified_all_ = false;
            break;
        }
    }
    return true;
}

bool Session::enqueue_norm(Backend& backend, uint32_t layer, std::string& err) {
    err.clear();
    if (!active_) {
        err = blocked_reason_.empty() ? "tree-WY session is not active" : blocked_reason_;
        return false;
    }
    if (layer >= layers_.size() || !layers_[layer].verified) {
        err = "layer verify has not been enqueued";
        return false;
    }
    LayerState& ls = layers_[layer];
    if (ls.normalized) {
        err = "layer norm already enqueued";
        return false;
    }
    if (!ls.binding.norm.enabled) {
        if (req_.require_norm) {
            ++stats_.descriptor_errors;
            fail(FallbackReason::DescriptorInvalid,
                 "layer " + std::to_string(layer) + " has no post-verify norm stage");
            err = last_message_;
            return false;
        }
        ls.normalized = true;
        return true;
    }
    NormCall call;
    call.layer = layer;
    call.device = ls.binding.device;
    call.args.raw_output = ls.binding.norm.raw_output;
    call.args.gate_source = ls.binding.norm.gate_source;
    call.args.norm_weight = ls.binding.norm.norm_weight;
    call.args.out_q8 = ls.binding.norm.out_q8;
    call.args.out_ds = ls.binding.norm.out_ds;
    call.args.m = req_.width;
    call.args.nv = ls.binding.norm.nv;
    call.args.stride_in = ls.binding.norm.stride_in;
    call.args.z_off = ls.binding.norm.z_off;
    call.args.eps = ls.binding.norm.eps;
    if (!backend.normalize(call, err)) {
        ++stats_.backend_errors;
        fail(FallbackReason::NormFailed,
             "norm/quant launch failed for layer " + std::to_string(layer) + ": " + err);
        return false;
    }
    ls.normalized = true;
    ++stats_.norm_launches;
    stats_.event(Op::Norm, layer, tl::kNoNode, true,
                 static_cast<uint64_t>(req_.width) * ls.binding.norm.nv * 128u);
    normalized_all_ = true;
    for (const LayerState& s : layers_) {
        if (!s.normalized) {
            normalized_all_ = false;
            break;
        }
    }
    return true;
}

bool Session::build_acceptance(AcceptanceKind kind, uint32_t prefix, std::vector<int32_t>& verified,
                               std::string& err) const {
    verified.clear();
    if (kind == AcceptanceKind::Natural) {
        if (req_.verified.size() != req_.width) {
            err = "natural acceptance needs a verified token for every node";
            return false;
        }
        verified = req_.verified;
        return true;
    }
    if (!tl::is_chain(req_.parent.data(), req_.width)) {
        err = "deterministic acceptance fixtures require a chain tree";
        return false;
    }
    if (kind == AcceptanceKind::Reject) prefix = 0;
    if (prefix > req_.width) {
        err = "acceptance prefix exceeds the sealed width";
        return false;
    }
    verified = req_.draft;
    for (uint32_t i = prefix; i < req_.width; ++i) {
        // A deterministic mismatch; selection stops at the first rejected child.
        int32_t v = req_.draft[i];
        v = (v == std::numeric_limits<int32_t>::max()) ? v - 1 : v + 1;
        verified[i] = v;
    }
    return true;
}

void Session::restore_cursors() {
    selected_conv_head_ = snapshot_conv_head_;
}

Resolution Session::resolve(Backend& backend, const Acceptance& acceptance, std::string& err) {
    err.clear();
    Resolution r;
    if (!active_) {
        r.reason = last_reason_;
        r.message = blocked_reason_.empty() ? last_message_ : blocked_reason_;
        err = r.message;
        return r;
    }
    r.used = true;
    if (!verified_all_ || !normalized_all_) {
        fail(FallbackReason::IncompleteLayers,
             "every bound GDN layer must be verified and normalized before resolve");
        r.reason = last_reason_;
        r.message = last_message_;
        err = r.message;
        return r;
    }

    std::vector<int32_t> verified;
    std::string why;
    if (!build_acceptance(acceptance.kind, acceptance.prefix, verified, why)) {
        fail(FallbackReason::InvalidVerified, why);
        r.reason = last_reason_;
        r.message = last_message_;
        err = why;
        return r;
    }
    tl::Selection sel = tl::select_acceptance(req_.parent.data(), req_.draft.data(), verified.data(),
                                             req_.width);
    if (!sel.ok()) {
        fail(FallbackReason::DescriptorInvalid, sel.error);
        r.reason = last_reason_;
        r.message = last_message_;
        err = sel.error;
        return r;
    }
    r.selection = sel;
    r.path_nodes = sel.accepted_nodes;
    r.accepted_node = sel.accepted_node_id;
    r.commit_accepted = sel.commit_accepted;
    r.bonus_pending = sel.bonus_pending;
    r.post.rows = req_.width;
    r.post.first_new_row = req_.context_is_root ? 1u : 0u;
    r.post.accepted_rows = sel.path;
    r.post.emitted = sel.emitted;
    if (req_.context_is_root && !r.post.emitted.empty()) {
        r.post.emitted.erase(r.post.emitted.begin());
    }
    r.post.bonus_pending = sel.bonus_pending;
    r.new_tokens = static_cast<uint32_t>(r.post.emitted.size());
    r.state_advance = sel.accepted_nodes;

    if (sel.accepted_nodes == req_.width) {
        ++stats_.full_accepts;
    } else if (sel.accepted_nodes == 0) {
        ++stats_.rejects;
    } else {
        ++stats_.partial_accepts;
    }
    stats_.accepted_nodes += sel.accepted_nodes;

    if (req_.context_is_root && sel.accepted_nodes == 1u) ++stats_.root_only;

    if (req_.context_is_root && sel.accepted_nodes == 0) {
        fail(FallbackReason::ContextRootRejected,
             "the emitted context node must stay accepted; use a non-root tree for a full rollback");
        r.ok = false;
        r.reason = last_reason_;
        r.message = last_message_;
        err = r.message;
        return r;
    }

    if (sel.accepted_nodes == 0 || sel.action == tl::StateAction::None) {
        // No node selected: no commit, S0 stays live, the conv cursor is unchanged.
        restore_cursors();
        ++stats_.cursor_rollbacks;
        r.ok = true;
        r.reason = FallbackReason::None;
        r.conv_head = selected_conv_head_;
        r.active_slot.resize(layers_.size());
        for (uint32_t i = 0; i < layers_.size(); ++i) r.active_slot[i] = layers_[i].binding.active_slot;
        stats_.event(Op::Fallback, 0, tl::kNoNode, true, 0);
        active_ = false;
        verified_all_ = false;
        normalized_all_ = false;
        return r;
    }

    // Commit exactly the selected node (id + 1), never the accepted-node count.
    const uint32_t selected = sel.accepted_node_id;
    // Conv-slot ownership is per logical layer, not per bound shard; every binding carries the
    // same node order. Never index the layer vector by node id (nodes can outnumber layers).
    const LayerBinding& slot_ref = layers_.front().binding;
    if (slot_ref.has_conv_slots) {
        // The producer wrote node i's post-token conv state to slot (snapshot + i + 1); a
        // branching path may select a node whose id is larger than its depth, so use that
        // node's own slot rather than reconstructing the cursor from the path length.
        selected_conv_head_ = slot_ref.conv_slot[selected] % req_.conv_slots;
    } else {
        selected_conv_head_ =
            (snapshot_conv_head_ + static_cast<uint32_t>(sel.path.size())) % req_.conv_slots;
    }
    const tl::LaunchPlan plan =
        tl::plan_launch(tl::PlanRequest{req_.width, req_.layers, 0, req_.produce_wire});
    bool commit_failed = false;
    for (uint32_t i = 0; i < layers_.size(); ++i) {
        LayerState& ls = layers_[i];
        if (!ls.bound) {
            commit_failed = true;
            why = "layer " + std::to_string(i) + " was never bound";
            break;
        }
        tl::CommitBuffers cb;
        cb.records = {ls.binding.records, tl::kRecordBlockBytes};
        cb.state = {ls.binding.state_slot[ls.binding.active_slot ^ 1u], tl::kStateBytesPerLayer};
        CommitCall call;
        call.layer = i;
        call.device = ls.binding.device;
        call.launch = plan;
        if (!tl::build_commit_args(cb, req_.width, sel.commit_accepted, call.args, why)) {
            commit_failed = true;
            break;
        }
        if (!backend.commit(call, why)) {
            ++stats_.backend_errors;
            commit_failed = true;
            break;
        }
        ls.commit_enqueued = true;
        ++stats_.commit_launches;
        stats_.event(Op::Commit, i, sel.accepted_node_id, true, tl::kStateBytesPerLayer);
    }

    if (!commit_failed) {
        if (!backend.finish(why)) {
            ++stats_.backend_errors;
            commit_failed = true;
        }
    }

    if (commit_failed) {
        // The inactive slots may hold partial commits; restoring S0 there and leaving active_slot
        // alone keeps the live state byte-identical. A device failure is still fatal: the caller
        // must not continue as if the cycle had succeeded.
        for (uint32_t i = 0; i < layers_.size(); ++i) {
            LayerState& ls = layers_[i];
            if (!ls.commit_enqueued) continue;
            tl::CommitBuffers cb;
            cb.records = {ls.binding.records, tl::kRecordBlockBytes};
            cb.state = {ls.binding.state_slot[ls.binding.active_slot ^ 1u], tl::kStateBytesPerLayer};
            CommitCall restore;
            restore.layer = i;
            restore.device = ls.binding.device;
            restore.launch = plan;
            std::string ignored;
            if (tl::build_commit_args(cb, req_.width, 0, restore.args, ignored)) {
                if (backend.commit(restore, ignored)) {
                    ++stats_.restore_launches;
                    stats_.event(Op::Restore, i, tl::kNoNode, true, tl::kStateBytesPerLayer);
                }
            }
        }
        std::string finish_err;
        backend.finish(finish_err);
        restore_cursors();
        ++stats_.cursor_rollbacks;
        fail(FallbackReason::CommitFailed,
             "sealed tree-WY commit failed: " + why + "; live state and cursors were left unchanged");
        r.used = true;
        r.ok = false;
        r.fatal = true;
        r.reason = last_reason_;
        r.message = last_message_;
        r.conv_head = selected_conv_head_;
        r.active_slot.resize(layers_.size());
        for (uint32_t i = 0; i < layers_.size(); ++i) r.active_slot[i] = layers_[i].binding.active_slot;
        err = r.message;
        return r;
    }

    for (uint32_t i = 0; i < layers_.size(); ++i) {
        layers_[i].binding.active_slot ^= 1u;
        ++stats_.state_flips;
    }
    ++stats_.cursor_advances;
    ++stats_.committed_nodes;
    stats_.emitted_tokens += r.new_tokens;
    if (r.bonus_pending) ++stats_.bonus_pending;
    r.ok = true;
    r.reason = FallbackReason::None;
    r.conv_head = selected_conv_head_;
    r.active_slot.resize(layers_.size());
    for (uint32_t i = 0; i < layers_.size(); ++i) r.active_slot[i] = layers_[i].binding.active_slot;
    active_ = false;
    verified_all_ = false;
    normalized_all_ = false;
    return r;
}

void Session::abort() {
    if (!active_) return;
    restore_cursors();
    ++stats_.cursor_rollbacks;
    stats_.event(Op::Fallback, 0, tl::kNoNode, false, 0);
    active_ = false;
    verified_all_ = false;
    normalized_all_ = false;
}

}  // namespace vegajet::gdn::treewy_engine

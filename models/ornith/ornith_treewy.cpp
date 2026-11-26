// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "models/ornith/ornith_treewy.hpp"

#include "core/kfd_runtime.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <utility>

namespace vegajet::ornith::treewy {

namespace te = vegajet::gdn::treewy_engine;
namespace tl = vegajet::gdn::treewy_launch;
namespace wp = vegajet::gdn::wire_pack;

namespace {

std::mutex& producer_mutex() {
    static std::mutex m;
    return m;
}

SealedWireProducer& producer_slot() {
    static SealedWireProducer producer;
    return producer;
}

bool env_on(const char* v) {
    if (!v || !*v) return false;
    return std::strcmp(v, "0") != 0 && std::strcmp(v, "off") != 0 &&
           std::strcmp(v, "false") != 0 && std::strcmp(v, "no") != 0;
}

uint32_t env_u32(const char* v, uint32_t fallback) {
    if (!v || !*v) return fallback;
    const long parsed = std::strtol(v, nullptr, 10);
    if (parsed < 0) return fallback;
    return static_cast<uint32_t>(parsed);
}

}  // namespace

void install_sealed_wire_producer(SealedWireProducer producer) {
    std::lock_guard<std::mutex> lock(producer_mutex());
    producer_slot() = std::move(producer);
}

bool sealed_wire_installed() {
    std::lock_guard<std::mutex> lock(producer_mutex());
    return static_cast<bool>(producer_slot());
}

namespace {

SealedWireProducer copy_producer() {
    std::lock_guard<std::mutex> lock(producer_mutex());
    return producer_slot();
}

}  // namespace

bool build_gdn_wire_call(const te::WireRequest& req, wp::Args& args, uint32_t& grid_x,
                         std::string& err) {
    args = wp::Args{};
    args.conv_source = req.conv_source;
    args.decay_source = req.decay_source;
    args.beta_source = req.beta_source;
    args.query = req.out.query;
    args.key = req.out.key;
    args.value = req.out.value;
    args.decay = req.out.decay;
    args.beta = req.out.beta;
    args.parent_source = req.parent_source;
    args.parent = req.out.parent;
    args.m = req.width;
    args.nk = req.nk;
    args.nv = req.nv;
    args.conv_stride = req.conv_stride;
    if (!wp::validate(args, err)) return false;
    grid_x = wp::grid_x(req.width);
    return true;
}

bool install_gdn_wire_producer(const kfd::LoadedKernel* shard0, const kfd::LoadedKernel* shard1) {
    if (!shard0 || !shard1) {
        install_sealed_wire_producer(nullptr);
        return false;
    }
    // The engine owns the runtime for the whole run and clears the producer before returning, so
    // capturing the two shard kernels here cannot outlive their module.
    install_sealed_wire_producer(
        [shard0, shard1](kfd::Runtime& rt, size_t gpu, const te::WireRequest& req,
                         std::string& err) -> bool {
            if (gpu >= 2u) {
                err = "sealed-wire producer device index is out of range";
                return false;
            }
            wp::Args args;
            uint32_t grid = 0;
            if (!build_gdn_wire_call(req, args, grid, err)) return false;
            const kfd::LoadedKernel* kernel = (gpu == 0u) ? shard0 : shard1;
            if (!kernel) {
                err = "sealed-wire producer module is not loaded on this shard";
                return false;
            }
            // Barrier-ordered on this shard's queue: conv -> pack -> sealed verify -> norm.
            if (!rt.enqueue_kernel(gpu, *kernel, grid, 1, 1, wp::kThreads, 1, 1, &args,
                                   sizeof(args), true)) {
                err = "enqueue_kernel(vj_gdn_wire_pack_kernel) failed";
                return false;
            }
            return true;
        });
    return true;
}

Options Options::from_env() {
    Options o;
    o.enabled = env_on(std::getenv("VEGAJET_TREEWY"));
    const uint32_t fw = env_u32(std::getenv("VEGAJET_TREEWY_FORCE_WIDTH"), 0u);
    if (fw == 5u || fw == 8u || fw == 12u) o.force_width = fw;
    o.verbose = env_on(std::getenv("VEGAJET_TREEWY_VERBOSE"));
    const char* fa = std::getenv("VEGAJET_TREEWY_FORCE_ACCEPT");
    if (fa && *fa) {
        if (std::strcmp(fa, "full") == 0) {
            o.force_accept = ForceAccept::Full;
        } else if (std::strcmp(fa, "reject") == 0) {
            o.force_accept = ForceAccept::Reject;
        } else if (std::strncmp(fa, "prefix:", 7) == 0) {
            o.force_accept = ForceAccept::Prefix;
            const long n = std::strtol(fa + 7, nullptr, 10);
            o.force_prefix = (n > 0) ? static_cast<uint32_t>(n) : 0u;
        }
    }
    return o;
}

uint32_t clamp_accepted(uint32_t natural, uint32_t drafts, Options::ForceAccept mode,
                        uint32_t prefix) {
    if (natural > drafts) natural = drafts;  // the target cannot accept more drafts than submitted
    switch (mode) {
        case Options::ForceAccept::Reject:
            return 0u;
        case Options::ForceAccept::Prefix:
            return prefix < natural ? prefix : natural;
        case Options::ForceAccept::Natural:
        case Options::ForceAccept::Full:
        default:
            // Full is not a force mode: full acceptance is only possible naturally.
            return natural;
    }
}

std::string describe(const Options& opt) {
    const char* fa = "natural";
    if (opt.force_accept == Options::ForceAccept::Full) fa = "full";
    else if (opt.force_accept == Options::ForceAccept::Reject) fa = "reject";
    else if (opt.force_accept == Options::ForceAccept::Prefix) fa = "prefix";
    char buf[192];
    std::snprintf(buf, sizeof buf, "enabled=%d force_width=%u force_accept=%s prefix=%u verbose=%d",
                  opt.enabled ? 1 : 0, opt.force_width, fa, opt.force_prefix,
                  opt.verbose ? 1 : 0);
    return std::string(buf);
}

// ---- production backend -----------------------------------------------------------------
RuntimeBackend::RuntimeBackend(kfd::Runtime& rt, const kfd::LoadedKernel* verify[2],
                               const kfd::LoadedKernel* commit[2],
                               const kfd::LoadedKernel* norm[2])
    : rt_(rt) {
    for (size_t g = 0; g < 2; ++g) {
        verify_[g] = verify ? verify[g] : nullptr;
        commit_[g] = commit ? commit[g] : nullptr;
        norm_[g] = norm ? norm[g] : nullptr;
    }
}

bool RuntimeBackend::available() const {
    return verify_[0] && verify_[1] && commit_[0] && commit_[1] && norm_[0] && norm_[1] &&
           rt_.num_gpus() >= 2u;
}

bool RuntimeBackend::wire_available() const { return sealed_wire_installed(); }

bool RuntimeBackend::produce_wire(const te::WireRequest& req, std::string& err) {
    err.clear();
    if (!req.out.complete()) {
        err = "sealed-wire destination planes are incomplete";
        return false;
    }
    SealedWireProducer producer = copy_producer();
    if (!producer) {
        err = "sealed-wire producer is not installed";
        return false;
    }
    if (req.device >= rt_.num_gpus()) {
        err = "sealed-wire producer device index is out of range";
        return false;
    }
    return producer(rt_, req.device, req, err);
}

bool RuntimeBackend::upload_parent(uint32_t device, uint64_t dst, const int32_t* host,
                                   uint32_t width, std::string& err) {
    err.clear();
    if (device >= rt_.num_gpus() || dst == 0 || !host || width == 0) {
        err = "invalid parent upload";
        return false;
    }
    if (!rt_.upload_to_vram(device, dst, host, static_cast<size_t>(width) * sizeof(int32_t))) {
        err = "parent upload failed";
        return false;
    }
    return true;
}

bool RuntimeBackend::verify(const te::VerifyCall& call, std::string& err) {
    err.clear();
    if (call.device >= 2u || !verify_[call.device]) {
        err = "sealed tree-WY verify module is not loaded";
        return false;
    }
    if (!call.launch.ok()) {
        err = call.launch.reason.empty() ? "invalid verify launch plan" : call.launch.reason;
        return false;
    }
    if (!rt_.enqueue_kernel(call.device, *verify_[call.device], call.launch.grid_x,
                            call.launch.grid_y, call.launch.grid_z, call.launch.block_x,
                            call.launch.block_y, call.launch.block_z, &call.args,
                            sizeof(call.args), true, call.launch.dynamic_lds_bytes)) {
        err = "enqueue_kernel(vj_gdn_treewy_verify_kernel) failed";
        return false;
    }
    return true;
}

bool RuntimeBackend::normalize(const te::NormCall& call, std::string& err) {
    err.clear();
    if (call.device >= 2u || !norm_[call.device]) {
        err = "normal-engine norm/quant kernel is not loaded";
        return false;
    }
    if (call.args.m == 0 || call.args.nv == 0) {
        err = "invalid norm/quant dimensions";
        return false;
    }
    const uint32_t grid_x = ((call.args.m * call.args.nv + 3u) / 4u) * 256u;
    if (!rt_.enqueue_kernel(call.device, *norm_[call.device], grid_x, 1, 1, 256, 1, 1,
                            &call.args, sizeof(call.args), true)) {
        err = "enqueue_kernel(vj_spec_gdn_norm_quant_kernel) failed";
        return false;
    }
    return true;
}

bool RuntimeBackend::commit(const te::CommitCall& call, std::string& err) {
    err.clear();
    if (call.device >= 2u || !commit_[call.device]) {
        err = "sealed tree-WY commit module is not loaded";
        return false;
    }
    if (!call.launch.ok()) {
        err = call.launch.reason.empty() ? "invalid commit launch plan" : call.launch.reason;
        return false;
    }
    if (!rt_.enqueue_kernel(call.device, *commit_[call.device], call.launch.grid_x,
                            call.launch.grid_y, call.launch.grid_z, call.launch.block_x,
                            call.launch.block_y, call.launch.block_z, &call.args,
                            sizeof(call.args), true, call.launch.dynamic_lds_bytes)) {
        err = "enqueue_kernel(vj_gdn_treewy_commit_kernel) failed";
        return false;
    }
    return true;
}

bool RuntimeBackend::finish(std::string& err) {
    err.clear();
    if (!rt_.submit_batch(0) || !rt_.submit_batch(1) || !rt_.wait_idle(0) || !rt_.wait_idle(1)) {
        err = "tree-WY submit/wait failed";
        return false;
    }
    return true;
}

// ---- engine hook ------------------------------------------------------------------------
Hook::Hook(Options opt, uint32_t gdn_layers, uint32_t conv_slots, const uint64_t arena_va[2],
           const uint64_t arena_bytes[2], te::Backend& backend)
    : opt_(opt),
      gdn_layers_(gdn_layers),
      conv_slots_(conv_slots),
      backend_(backend),
      session_(conv_slots) {
    for (size_t g = 0; g < 2; ++g) {
        arena_va_[g] = arena_va ? arena_va[g] : 0;
        arena_bytes_[g] = arena_bytes ? arena_bytes[g] : 0;
    }
}

Hook::~Hook() = default;

bool Hook::begin(const CycleInput& in) {
    if (!opt_.enabled) return false;
    if (session_.active()) session_.abort();
    width_ = static_cast<uint32_t>(in.tokens.size());
    active_in_ = in.active_slot;
    conv_head_in_ = in.conv_head;
    tokens_ = in.tokens;
    if (!backend_.available() || !backend_.wire_available()) {
        if (opt_.verbose) std::printf("[treewy] fallback backend/module/producer unavailable\n");
        return false;
    }
    if (!tl::is_sealed_width(width_)) {
        if (opt_.verbose)
            std::printf("[treewy] fallback width=%u (sealed widths are 5/8/12)\n", width_);
        return false;
    }
    if (gdn_layers_ == 0 || gdn_layers_ * 2u > te::kMaxLayers) {
        if (opt_.verbose) std::printf("[treewy] fallback gdn_layers=%u\n", gdn_layers_);
        return false;
    }
    plan_ = tl::size_scratch(width_, gdn_layers_, true);
    if (!plan_.valid()) {
        if (opt_.verbose) std::printf("[treewy] fallback scratch: %s\n", plan_.error.c_str());
        return false;
    }
    for (size_t g = 0; g < 2; ++g) {
        if (arena_va_[g] == 0 || arena_bytes_[g] < plan_.total_bytes) {
            if (opt_.verbose)
                std::printf("[treewy] fallback arena GPU%zu bytes=%llu need=%llu\n", g,
                            static_cast<unsigned long long>(arena_bytes_[g]),
                            static_cast<unsigned long long>(plan_.total_bytes));
            return false;
        }
    }

    te::Request req;
    req.width = width_;
    // One binding per (GDN layer, shard): both GPUs hold their own records/output/wire planes
    // and each sealed commit writes that shard's inactive state slot.
    req.layers = gdn_layers_ * 2u;
    req.context_is_root = true;
    req.produce_wire = true;
    req.require_norm = true;
    req.budget_bytes = 0;  // the per-device arena was sized by size_scratch(width, gdn_layers)
    req.conv_head = in.conv_head;
    req.conv_slots = conv_slots_;
    req.parent.resize(width_);
    for (uint32_t i = 0; i < width_; ++i) req.parent[i] = (i == 0u) ? -1 : static_cast<int32_t>(i - 1u);
    req.draft = in.tokens;
    std::string err;
    // The producer copies this staging chain into the sealed parent plane for every layer, so the
    // upload is repeated each cycle: a reused width with a changed chain can never leave stale
    // parent ids behind, and the copy is at most 48 bytes. It runs before configure() so a failed
    // upload cannot count a selected cycle, and before any conv/verify work on the device stream.
    for (size_t g = 0; g < 2; ++g) {
        const uint64_t dst = arena_va_[g] + plan_.parent_source_offset;
        if (!backend_.upload_parent(static_cast<uint32_t>(g), dst, req.parent.data(), width_,
                                    err)) {
            if (opt_.verbose)
                std::printf("[treewy] fallback parent upload GPU%zu: %s\n", g, err.c_str());
            return false;
        }
    }
    const te::Decision d = session_.configure(req);
    if (!d.treewy()) {
        // configure() already counted and recorded the precise fallback reason.
        if (opt_.verbose) std::printf("[treewy] fallback: %s (%s)\n", d.reason.c_str(),
                                      te::fallback_reason_name(session_.last_reason()));
        return false;
    }
    if (opt_.verbose) {
        std::printf("[treewy] begin width=%u layers=%u conv_head=%u active=%u\n", width_,
                    gdn_layers_, in.conv_head, in.active_slot);
    }
    return true;
}

bool Hook::enqueue_layer(const LayerInputs& in, std::string& err) {
    err.clear();
    if (!session_.active()) {
        err = "tree-WY cycle is not active";
        return false;
    }
    if (in.gdn_index >= gdn_layers_) {
        err = "GDN layer index out of range";
        return false;
    }
    if (in.device >= 2u) {
        err = "device index out of range";
        return false;
    }
    if (in.active_slot != active_in_) {
        err = "layer active state slot disagrees with the cycle snapshot";
        return false;
    }
    const uint32_t binding_index = in.gdn_index * 2u + in.device;
    const uint64_t base = arena_va_[in.device];
    te::LayerBinding b;
    b.device = in.device;
    b.state_slot[0] = in.state_slot[0];
    b.state_slot[1] = in.state_slot[1];
    b.active_slot = active_in_;
    b.records = base + plan_.records_offset +
                static_cast<uint64_t>(in.gdn_index) * tl::kRecordBlockBytes;
    b.output = base + plan_.output_offset;
    b.parent = base + plan_.parent_store_offset;
    b.wire.query = base + plan_.staged_query_offset;
    b.wire.key = base + plan_.staged_key_offset;
    b.wire.value = base + plan_.staged_value_offset;
    b.wire.decay = base + plan_.staged_decay_offset;
    b.wire.beta = base + plan_.staged_beta_offset;
    b.wire.parent = b.parent;
    b.conv_source = in.conv_source;
    b.decay_source = in.decay_source;
    b.beta_source = in.beta_source;
    b.parent_source = base + plan_.parent_source_offset;
    b.conv_stride = 6144;
    b.nk = 8u;
    b.nv = in.nv;
    b.norm.raw_output = b.output;
    b.norm.gate_source = in.gate_source;
    b.norm.norm_weight = in.norm_weight;
    b.norm.out_q8 = in.out_q8;
    b.norm.out_ds = in.out_ds;
    b.norm.nv = in.nv;
    b.norm.stride_in = in.stride_in;
    b.norm.z_off = in.z_off;
    b.norm.eps = in.eps;
    b.norm.enabled = in.gate_source && in.norm_weight && in.out_q8 && in.out_ds;
    b.has_conv_slots = true;
    for (uint32_t i = 0; i < width_; ++i)
        b.conv_slot[i] = (session_.snapshot_conv_head() + i + 1u) % conv_slots_;

    if (!session_.bind_layer(binding_index, b, err)) {
        session_.abort();
        return false;
    }
    if (!session_.enqueue_verify(backend_, binding_index, err)) {
        session_.abort();
        return false;
    }
    if (!session_.enqueue_norm(backend_, binding_index, err)) {
        session_.abort();
        return false;
    }
    return true;
}

CycleResult Hook::commit(uint32_t accepted_drafts, uint32_t draft_count) {
    CycleResult out;
    if (!session_.active()) {
        out.reason = "no active tree-WY cycle";
        return out;
    }
    out.used = true;
    if (draft_count + 1u != width_) {
        out.reason = "draft_count does not match the sealed width";
        out.fatal = true;
        session_.abort();
        return out;
    }
    if (accepted_drafts > draft_count) {
        out.reason = "accepted drafts exceed the submitted drafts";
        out.fatal = true;
        session_.abort();
        return out;
    }
    te::Acceptance acc;
    acc.kind = te::AcceptanceKind::Prefix;
    acc.prefix = accepted_drafts + 1u;
    std::string err;
    const te::Resolution r = session_.resolve(backend_, acc, err);
    out.path_nodes = r.path_nodes;
    out.new_tokens = r.new_tokens;
    out.accepted_node = r.accepted_node;
    out.bonus_pending = r.bonus_pending;
    out.conv_head = r.conv_head;
    out.emitted = r.post.emitted;
    out.ok = r.ok;
    out.fatal = r.fatal;
    if (!r.ok) {
        out.reason = std::string(te::fallback_reason_name(r.reason)) + ": " + r.message;
        return out;
    }
    out.active_slot = r.active_slot.empty() ? active_in_ : r.active_slot[0];
    for (size_t i = 1; i < r.active_slot.size(); ++i) {
        if (r.active_slot[i] != out.active_slot) {
            out.ok = false;
            out.fatal = true;
            out.reason = "layers disagree on the post-commit active state slot";
            return out;
        }
    }
    if (r.path_nodes != accepted_drafts + 1u || r.state_advance != accepted_drafts + 1u) {
        out.ok = false;
        out.fatal = true;
        out.reason = "sealed path length differs from the engine's accepted prefix";
        return out;
    }
    if (out.conv_head != (conv_head_in_ % conv_slots_ + r.path_nodes) % conv_slots_) {
        out.ok = false;
        out.fatal = true;
        out.reason = "conv-ring cursor does not match the accepted prefix";
        return out;
    }
    if (opt_.verbose) {
        std::printf("[treewy] cycle width=%u acc=%u path=%u node=%u new=%u bonus=%u "
                    "conv_head=%u active=%u\n",
                    width_, accepted_drafts, r.path_nodes, r.accepted_node, r.new_tokens,
                    r.bonus_pending ? 1u : 0u, out.conv_head, out.active_slot);
    }
    return out;
}

void Hook::abort() { session_.abort(); }

void Hook::print_stats(const char* tag) const {
    std::printf("[treewy] %s %s\n", tag ? tag : "stats", session_.stats().summary().c_str());
}

}  // namespace vegajet::ornith::treewy

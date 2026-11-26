// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// CPU qualification of the normal-engine tree-WY caller (core/gdn_treewy_engine.*) and its
// Ornith adapter (models/ornith/ornith_treewy.*).
//
// The sealed kernels are NOT re-implemented here. A deterministic fake backend stands in for the
// registered `vj_gdn_treewy` module and applies a documented integer state fold per node; this
// test exercises the caller's descriptors, selection, commit-node rule, S0 slot policy, conv-ring
// cursor, reused norm/quant stage, emitted-token view and fallbacks.
//
// Covered at widths 5, 8 and 12:
//   * full, partial and root-only rollback against a sequential verifier model (the engine's
//     existing full-accept flip vs recompute-on-reject prefix semantics), on chains and trees;
//   * a branching path whose selected node id differs from the accepted-node count;
//   * rejected siblings never reach the committed slot;
//   * multi-shard/multi-layer binding, all-or-nothing flips and commit-failure restoration;
//   * conv-ring slot order, wrap, advance and rollback; descriptor aliasing and budget fallbacks;
//   * post-verify norm/quant row coverage and the emitted-token view (root dropped, bonus flagged);
//   * an exact driver of models/ornith/ornith_treewy.cpp Hook over 36 GDN layers x 2 shards with
//     the normal engine's acceptance accounting (acc drafts -> acc+1 committed nodes), deterministic
//     acceptance modes and observable nonzero dispatch counters.
//
// This is orchestration evidence, not device parity. Numeric/device evidence lives in
// tests/test_gdn_treewy_launch.cpp, tests/test_gdn_treewy_verify.cpp and the owner A/B procedure.
#include "core/gdn_treewy_engine.hpp"
#include "core/kfd_runtime.hpp"
#include "models/ornith/ornith_treewy.hpp"
#include "tests/test_common.hpp"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace te = vegajet::gdn::treewy_engine;
namespace tl = vegajet::gdn::treewy_launch;
namespace wp = vegajet::gdn::wire_pack;
using vegajet::ornith::treewy::CycleInput;
using vegajet::ornith::treewy::Hook;
using vegajet::ornith::treewy::LayerInputs;
using vegajet::ornith::treewy::Options;

namespace {

constexpr uint32_t kGdnLayersEngine = 36;  // Ornith-122B: 36 GDN + 12 GQA layers
constexpr uint32_t kConvSlots = 17;        // == kGdnRingSlots in the normal engine
constexpr size_t kState = static_cast<size_t>(tl::kStateBytesPerLayer);
constexpr uint32_t kProbe = 4096;          // first/last probe window of a fake state slot

uint64_t g_cases = 0;
uint64_t g_treewy = 0;
uint64_t g_fallback = 0;
uint64_t g_forced = 0;
uint64_t g_driver_cycles = 0;
uint64_t g_driver_tokens = 0;

struct Digest {
    uint64_t w[4] = {0, 0, 0, 0};
};

bool operator==(const Digest& a, const Digest& b) {
    return std::memcmp(a.w, b.w, sizeof a.w) == 0;
}
bool operator!=(const Digest& a, const Digest& b) { return !(a == b); }

uint64_t mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

Digest fold(Digest d, int32_t token) {
    const uint64_t t = static_cast<uint32_t>(token);
    for (uint32_t i = 0; i < 4; ++i)
        d.w[i] = mix64(d.w[i] ^ (t + 0x100000001b3ULL * (i + 1u)));
    return d;
}

Digest initial_digest(uint32_t layer, uint64_t seed) {
    Digest d;
    for (uint32_t i = 0; i < 4; ++i)
        d.w[i] = mix64(seed ^ (static_cast<uint64_t>(layer) << 32) ^
                       (0xcbf29ce484222325ULL * (i + 1u)));
    return d;
}

Digest poison_digest(uint32_t layer, uint64_t seed) {
    return initial_digest(layer, seed ^ 0xdeadbeefcafef00dULL);
}

void write_probe(uint8_t* p, Digest d) {
    std::memcpy(p, d.w, sizeof d.w);
    for (uint32_t off = 32; off < kProbe; off += 8) {
        const uint64_t v = mix64(d.w[0] + off + 0x51ed270bULL);
        std::memcpy(p + off, &v, sizeof v);
    }
    uint8_t* tail = p + kState - kProbe;
    std::memcpy(tail, d.w, sizeof d.w);
    for (uint32_t off = 32; off < kProbe; off += 8) {
        const uint64_t v = mix64(d.w[1] + off + 0x9e3779b9ULL);
        std::memcpy(tail + off, &v, sizeof v);
    }
}

Digest read_digest(const uint8_t* p) {
    Digest d;
    std::memcpy(d.w, p, sizeof d.w);
    return d;
}

uint64_t probe_hash(const uint8_t* p) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (uint32_t i = 0; i < kProbe; i += 8) {
        uint64_t v = 0;
        std::memcpy(&v, p + i, sizeof v);
        h = mix64(h ^ v);
    }
    const uint8_t* tail = p + kState - kProbe;
    for (uint32_t i = 0; i < kProbe; i += 8) {
        uint64_t v = 0;
        std::memcpy(&v, tail + i, sizeof v);
        h = mix64(h ^ v);
    }
    return h;
}

uint64_t va_of(const std::vector<uint64_t>& v) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(v.data()));
}

// ---- fixture: device-visible buffers for one caller cycle --------------------------------
struct Fixture {
    uint32_t layers = 0;   // GDN layers per shard
    uint32_t width = 0;    // current sealed width
    uint32_t devices = 1;  // 1 = direct Session, 2 = engine Hook (both shards)
    uint32_t conv_head = 0;
    uint32_t active = 0;
    uint64_t seed = 0x123456789abcdef0ULL;
    std::vector<int32_t> parent;
    std::vector<int32_t> tokens;
    tl::ScratchPlan plan;

    std::vector<uint64_t> state_words[2];
    std::vector<uint64_t> arena_words[2];
    std::vector<uint64_t> q8_words[2];
    std::vector<uint64_t> ds_words[2];
    std::vector<uint64_t> gate_words[2];
    std::vector<uint64_t> nw_words[2];
    std::vector<uint64_t> conv_words[2];  // production interleaved conv rows [12][6144]
    std::vector<uint64_t> dec_words[2];   // [12][32]
    std::vector<uint64_t> beta_words[2];  // [12][32]

    Fixture(uint32_t layer_count, uint32_t sealed_width, uint32_t device_count,
            std::vector<int32_t> parents, std::vector<int32_t> toks, uint32_t conv_head_in,
            uint32_t active_in)
        : layers(layer_count),
          width(sealed_width),
          devices(device_count),
          conv_head(conv_head_in),
          active(active_in),
          parent(std::move(parents)),
          tokens(std::move(toks)) {
        VJ_CHECK(parent.size() == width && tokens.size() == width);
        plan = tl::size_scratch(12u, layers, true);  // widest arena; cycles may shrink
        VJ_CHECK(plan.valid());
        for (uint32_t g = 0; g < devices; ++g) {
            state_words[g].assign((static_cast<uint64_t>(layers) * 2u * kState + 7u) / 8u + 8u, 0);
            arena_words[g].assign((plan.total_bytes + 7u) / 8u + 8u, 0);
            q8_words[g].assign((static_cast<uint64_t>(12u) * 32u * 128u + 7u) / 8u + 8u, 0);
            ds_words[g].assign((static_cast<uint64_t>(12u) * 32u * 4u * 8u + 7u) / 8u + 8u, 0);
            gate_words[g].assign(4096u / 8u, 0);
            nw_words[g].assign(128u * 4u / 8u, 0);
            conv_words[g].assign((static_cast<uint64_t>(12u) * 6144u * 4u + 7u) / 8u + 8u, 0);
            dec_words[g].assign((12u * 32u * 4u + 7u) / 8u + 8u, 0);
            beta_words[g].assign((12u * 32u * 4u + 7u) / 8u + 8u, 0);
        }
    }

    uint8_t* state_ptr(uint32_t g, uint32_t layer, uint32_t slot) {
        VJ_CHECK(g < devices && layer < layers && slot < 2);
        const uint64_t off = (static_cast<uint64_t>(layer) * 2u + slot) * kState;
        return reinterpret_cast<uint8_t*>(state_words[g].data()) + off;
    }
    uint64_t state_va(uint32_t g, uint32_t layer, uint32_t slot) {
        return va_of(state_words[g]) + (static_cast<uint64_t>(layer) * 2u + slot) * kState;
    }
    uint64_t arena_va(uint32_t g) { return va_of(arena_words[g]); }
    uint64_t output_va(uint32_t g) { return arena_va(g) + plan.output_offset; }
    uint64_t parent_va(uint32_t g) { return arena_va(g) + plan.parent_store_offset; }
    uint64_t q8_va(uint32_t g) { return va_of(q8_words[g]); }
    uint64_t ds_va(uint32_t g) { return va_of(ds_words[g]); }
    uint64_t gate_va(uint32_t g) { return va_of(gate_words[g]); }
    uint64_t nw_va(uint32_t g) { return va_of(nw_words[g]); }
    uint64_t conv_va(uint32_t g) { return va_of(conv_words[g]); }
    uint64_t dec_va(uint32_t g) { return va_of(dec_words[g]); }
    uint64_t beta_va(uint32_t g) { return va_of(beta_words[g]); }
    uint64_t parent_source_va(uint32_t g) { return arena_va(g) + plan.parent_source_offset; }
    uint64_t records_va(uint32_t g, uint32_t gdn_layer) {
        return arena_va(g) + plan.records_offset +
               static_cast<uint64_t>(gdn_layer) * tl::kRecordBlockBytes;
    }

    Digest s0(uint32_t layer) const { return initial_digest(layer, seed); }

    void reset() {
        for (uint32_t g = 0; g < devices; ++g) {
            for (uint32_t l = 0; l < layers; ++l) {
                for (uint32_t slot = 0; slot < 2; ++slot) {
                    write_probe(state_ptr(g, l, slot),
                                slot == active ? s0(l) : poison_digest(l, seed));
                }
            }
        }
    }
};

te::LayerBinding make_binding(Fixture& fx, uint32_t device, uint32_t gdn_layer,
                              uint32_t active_slot) {
    te::LayerBinding b;
    b.device = device;
    b.state_slot[0] = fx.state_va(device, gdn_layer, 0);
    b.state_slot[1] = fx.state_va(device, gdn_layer, 1);
    b.active_slot = active_slot;
    b.records = fx.records_va(device, gdn_layer);
    b.output = fx.output_va(device);
    b.parent = fx.parent_va(device);
    b.wire.query = fx.arena_va(device) + fx.plan.staged_query_offset;
    b.wire.key = fx.arena_va(device) + fx.plan.staged_key_offset;
    b.wire.value = fx.arena_va(device) + fx.plan.staged_value_offset;
    b.wire.decay = fx.arena_va(device) + fx.plan.staged_decay_offset;
    b.wire.beta = fx.arena_va(device) + fx.plan.staged_beta_offset;
    b.wire.parent = b.parent;
    b.conv_source = fx.conv_va(device);
    b.decay_source = fx.dec_va(device);
    b.beta_source = fx.beta_va(device);
    b.parent_source = fx.parent_source_va(device);
    b.conv_stride = 6144;
    b.nk = 8;
    b.nv = 32;
    b.norm.raw_output = b.output;
    b.norm.gate_source = fx.gate_va(device);
    b.norm.norm_weight = fx.nw_va(device);
    b.norm.out_q8 = fx.q8_va(device);
    b.norm.out_ds = fx.ds_va(device);
    b.norm.nv = 32;
    b.norm.stride_in = 10304;
    b.norm.z_off = 6144;
    b.norm.eps = 1e-6f;
    b.norm.enabled = true;
    return b;
}

// ---- deterministic backend --------------------------------------------------------------
struct FakeBackend : te::Backend {
    struct Slot {
        uint64_t active_va = 0;
        uint64_t inactive_va = 0;
        uint8_t* active = nullptr;
        uint8_t* inactive = nullptr;
        Digest s0;
    };
    uint32_t width = 0;
    uint32_t layers = 0;
    std::vector<int32_t> parent;
    std::vector<int32_t> tokens;
    std::vector<Slot> slots;
    std::vector<std::vector<Digest>> nodes;
    uint64_t output_va[2] = {0, 0};
    bool available_ = true;
    bool wire_ok = true;
    uint32_t fail_verify_layer = UINT32_MAX;
    uint32_t fail_commit_layer = UINT32_MAX;
    bool fail_finish = false;
    uint64_t verify_calls = 0;
    uint64_t norm_calls = 0;
    uint64_t commit_calls = 0;
    uint64_t restore_calls = 0;
    uint64_t produce_calls = 0;
    uint64_t parent_uploads = 0;
    uint64_t written_states = 0;
    std::vector<uint32_t> norm_m;

    void configure(uint32_t layer_count, uint32_t w, std::vector<int32_t> parents,
                   std::vector<int32_t> toks, uint64_t out0, uint64_t out1) {
        layers = layer_count;
        width = w;
        parent = std::move(parents);
        tokens = std::move(toks);
        output_va[0] = out0;
        output_va[1] = out1;
        slots.assign(layers, Slot{});
        nodes.assign(layers, std::vector<Digest>(width));
        norm_m.assign(layers, 0);
        verify_calls = norm_calls = commit_calls = restore_calls = 0;
        produce_calls = parent_uploads = written_states = 0;
        fail_verify_layer = fail_commit_layer = UINT32_MAX;
        fail_finish = false;
        available_ = true;
        wire_ok = true;
    }

    void set_slot(uint32_t layer, uint64_t active_va, uint64_t inactive_va, uint8_t* active,
                  uint8_t* inactive, Digest s0) {
        VJ_CHECK(layer < slots.size());
        slots[layer] = Slot{active_va, inactive_va, active, inactive, s0};
    }

    bool available() const override { return available_; }
    bool wire_available() const override { return wire_ok; }

    bool produce_wire(const te::WireRequest& req, std::string& err) override {
        if (!wire_ok) {
            err = "fake sealed-wire producer disabled";
            return false;
        }
        if (!req.out.complete()) {
            err = "fake producer received incomplete planes";
            return false;
        }
        ++produce_calls;
        return true;
    }

    bool upload_parent(uint32_t device, uint64_t dst, const int32_t* host, uint32_t w,
                       std::string& err) override {
        if (device >= 2u || dst == 0 || !host || w != width) {
            err = "fake parent upload rejected";
            return false;
        }
        ++parent_uploads;
        return true;
    }

    bool verify(const te::VerifyCall& call, std::string& err) override {
        if (call.layer >= layers) {
            err = "verify layer out of range";
            return false;
        }
        if (call.layer == fail_verify_layer) {
            err = "injected verify failure";
            return false;
        }
        if (call.args.initial != slots[call.layer].active_va) {
            err = "sealed verify did not read the live active slot";
            return false;
        }
        const uint64_t before = probe_hash(slots[call.layer].active);
        for (uint32_t i = 0; i < width; ++i) {
            const Digest d = (parent[i] < 0)
                                 ? slots[call.layer].s0
                                 : nodes[call.layer][static_cast<uint32_t>(parent[i])];
            nodes[call.layer][i] = fold(d, tokens[i]);
        }
        const uint64_t after = probe_hash(slots[call.layer].active);
        if (before != after) {
            err = "sealed verify mutated the live state";
            return false;
        }
        ++verify_calls;
        return true;
    }

    bool normalize(const te::NormCall& call, std::string& err) override {
        if (call.layer >= layers || call.args.m != width) {
            err = "norm/quant call does not cover the sealed width";
            return false;
        }
        if (call.device >= 2u || call.args.raw_output != output_va[call.device]) {
            err = "norm/quant stage is not pointed at the sealed raw output";
            return false;
        }
        if (!call.args.out_q8 || !call.args.out_ds || call.args.nv != 32) {
            err = "norm/quant outputs are invalid";
            return false;
        }
        norm_m[call.layer] = call.args.m;
        ++norm_calls;
        return true;
    }

    bool commit(const te::CommitCall& call, std::string& err) override {
        if (call.layer >= layers) {
            err = "commit layer out of range";
            return false;
        }
        if (call.layer == fail_commit_layer) {
            err = "injected commit failure";
            return false;
        }
        const int32_t accepted = call.args.accepted;
        if (accepted < 0 || accepted > static_cast<int32_t>(width)) {
            err = "commit accepted is out of range";
            return false;
        }
        Digest d;
        if (accepted == 0) {
            d = slots[call.layer].s0;
            ++restore_calls;
        } else {
            d = nodes[call.layer][static_cast<uint32_t>(accepted - 1)];
        }
        write_probe(slots[call.layer].inactive, d);
        ++commit_calls;
        ++written_states;
        return true;
    }

    bool finish(std::string& err) override {
        if (fail_finish) {
            err = "injected submit/wait failure";
            return false;
        }
        return true;
    }
};

std::vector<int32_t> chain_parent(uint32_t width) {
    std::vector<int32_t> p(width);
    for (uint32_t i = 0; i < width; ++i) p[i] = (i == 0u) ? -1 : static_cast<int32_t>(i - 1u);
    return p;
}

std::vector<int32_t> chain_tokens(uint32_t width, int32_t base) {
    std::vector<int32_t> t(width);
    for (uint32_t i = 0; i < width; ++i) t[i] = base + static_cast<int32_t>(i);
    return t;
}

Digest path_digest(const Fixture& fx, uint32_t layer, const std::vector<int32_t>& tokens,
                   const std::vector<uint32_t>& path) {
    Digest d = fx.s0(layer);
    for (uint32_t node : path) d = fold(d, tokens[node]);
    return d;
}

Digest prefix_digest(const Fixture& fx, uint32_t layer, const std::vector<int32_t>& tokens,
                     uint32_t count) {
    Digest d = fx.s0(layer);
    for (uint32_t i = 0; i < count; ++i) d = fold(d, tokens[i]);
    return d;
}

std::string hex_digest(const Digest& d) {
    char buf[80];
    std::snprintf(buf, sizeof buf, "%016llx%016llx%016llx%016llx",
                  static_cast<unsigned long long>(d.w[0]),
                  static_cast<unsigned long long>(d.w[1]),
                  static_cast<unsigned long long>(d.w[2]),
                  static_cast<unsigned long long>(d.w[3]));
    return std::string(buf);
}

void check_state(Fixture& fx, uint32_t device, uint32_t layer, uint32_t slot,
                 const Digest& expect) {
    const Digest got = read_digest(fx.state_ptr(device, layer, slot));
    if (got != expect) {
        std::fprintf(stderr, "state mismatch device=%u layer=%u slot=%u got=%s expect=%s\n", device,
                     layer, slot, hex_digest(got).c_str(), hex_digest(expect).c_str());
    }
    VJ_CHECK(got == expect);
}

void fill_slots(FakeBackend& fake, Fixture& fx, uint32_t session_layers, uint32_t device_count) {
    for (uint32_t g = 0; g < device_count; ++g) {
        for (uint32_t l = 0; l < fx.layers; ++l) {
            const uint32_t idx = (device_count == 1u) ? l : l * 2u + g;
            VJ_CHECK(idx < session_layers);
            fake.set_slot(idx, fx.state_va(g, l, fx.active), fx.state_va(g, l, fx.active ^ 1u),
                          fx.state_ptr(g, l, fx.active), fx.state_ptr(g, l, fx.active ^ 1u),
                          fx.s0(l));
        }
    }
}

te::Resolution run_direct(Fixture& fx, FakeBackend& fake, const te::Acceptance& acc,
                          bool bind_all, bool* out_bound) {
    ++g_cases;
    fake.configure(fx.layers, fx.width, fx.parent, fx.tokens, fx.output_va(0), fx.output_va(0));
    fill_slots(fake, fx, fx.layers, 1u);
    te::Request req;
    req.width = fx.width;
    req.layers = fx.layers;
    req.context_is_root = true;
    req.produce_wire = true;
    req.require_norm = true;
    req.conv_head = fx.conv_head;
    req.conv_slots = kConvSlots;
    req.parent = fx.parent;
    req.draft = fx.tokens;
    te::Session session(kConvSlots);
    const te::Decision d = session.configure(req);
    if (!d.treewy()) {
        ++g_fallback;
        if (out_bound) *out_bound = false;
        te::Resolution r;
        r.reason = session.last_reason();
        r.message = session.last_message();
        return r;
    }
    ++g_treewy;
    const uint32_t binds = bind_all ? fx.layers : (fx.layers > 1u ? fx.layers - 1u : 0u);
    for (uint32_t l = 0; l < binds; ++l) {
        std::string err;
        VJ_CHECK(session.bind_layer(l, make_binding(fx, 0, l, fx.active), err));
        VJ_CHECK(session.enqueue_verify(fake, l, err));
        VJ_CHECK(session.enqueue_norm(fake, l, err));
    }
    if (out_bound) *out_bound = true;
    std::string err;
    return session.resolve(fake, acc, err);
}

// ---- chain full/partial/root-only vs the sequential verifier model ----------------------
void test_chain_full_partial_reject() {
    std::printf("[chain] widths 5/8/12 full/partial/root-only vs sequential verifier model\n");
    for (uint32_t width : {5u, 8u, 12u}) {
        const std::vector<int32_t> parent = chain_parent(width);
        const std::vector<int32_t> tokens = chain_tokens(width, 900);
        for (uint32_t mode = 0; mode < 3; ++mode) {
            Fixture fx(3u, width, 1u, parent, tokens, 15u, 0u);
            fx.reset();
            FakeBackend fake;
            te::Acceptance acc;
            acc.kind = te::AcceptanceKind::Prefix;
            acc.prefix = (mode == 0u) ? width : ((mode == 1u) ? (width / 2u) : 1u);
            bool bound = false;
            const te::Resolution r = run_direct(fx, fake, acc, true, &bound);
            VJ_CHECK(bound && r.used && r.ok && !r.fatal);
            const uint32_t path = acc.prefix;
            VJ_CHECK(r.path_nodes == path);
            VJ_CHECK(r.state_advance == path);
            VJ_CHECK(r.selection.accepted_nodes == path);
            VJ_CHECK(r.selection.commit_accepted == static_cast<int32_t>(path));
            VJ_CHECK(r.selection.accepted_node_id == path - 1u);
            const bool full = (path == width);
            const uint32_t expected_new = full ? width - 1u : path;
            VJ_CHECK(r.new_tokens == expected_new);
            VJ_CHECK(r.bonus_pending == full);
            VJ_CHECK(r.post.emitted.size() == expected_new);
            if (full) {
                for (uint32_t i = 0; i < expected_new; ++i) VJ_CHECK(r.post.emitted[i] == tokens[i + 1u]);
            } else {
                for (uint32_t i = 0; i + 1u < path; ++i) VJ_CHECK(r.post.emitted[i] == tokens[i + 1u]);
                VJ_CHECK(r.post.emitted.back() == tokens[path] + 1);
            }
            VJ_CHECK(r.post.first_new_row == 1u);
            VJ_CHECK(r.post.accepted_rows.size() == path);
            VJ_CHECK(r.conv_head == (15u + path) % kConvSlots);
            for (uint32_t l = 0; l < fx.layers; ++l) {
                VJ_CHECK(r.active_slot[l] == 1u);  // commit flips the pair
                check_state(fx, 0, l, r.active_slot[l], prefix_digest(fx, l, tokens, path));
            }
            VJ_CHECK(fake.verify_calls == fx.layers);
            VJ_CHECK(fake.norm_calls == fx.layers);
            VJ_CHECK(fake.commit_calls == fx.layers);
            VJ_CHECK(fake.produce_calls == fx.layers);
            // Sequential verifier model: full accept flips the pair; partial/root-only recompute
            // in place on the active slot. The logical state value must match the sealed commit
            // even though the sealed path always writes the inactive slot and then flips.
            for (uint32_t l = 0; l < fx.layers; ++l) {
                struct SeqPair {
                    Digest slot[2];
                    uint32_t active = 0;
                } seq;
                seq.slot[0] = fx.s0(l);
                seq.slot[1] = poison_digest(l, fx.seed);
                const Digest committed = prefix_digest(fx, l, tokens, path);
                if (full) {
                    seq.active = 1u;
                    seq.slot[1] = committed;
                } else {
                    seq.slot[0] = committed;  // recompute-on-reject, in place
                }
                VJ_CHECK(read_digest(fx.state_ptr(0, l, r.active_slot[l])) == seq.slot[seq.active]);
                if (!full) VJ_CHECK(r.active_slot[l] == 1u && seq.active == 0u);
            }
            std::printf("  M=%u mode=%s path=%u commit=%d new=%zu conv_head=%u digest=%s\n", width,
                        mode == 0u ? "full" : (mode == 1u ? "partial" : "root-only"), path,
                        r.commit_accepted, r.post.emitted.size(), r.conv_head,
                        hex_digest(read_digest(fx.state_ptr(0, 0, r.active_slot[0]))).c_str());
        }
    }
}

// ---- branching tree: selected node id differs from the path length ----------------------
void test_branching_tree() {
    std::printf("[tree] branching path, node id != count, rejected sibling closure\n");
    const uint32_t width = 8;
    const std::vector<int32_t> parent = {-1, 0, 0, 1, 1, 3, 3, 5};
    const std::vector<int32_t> tokens = chain_tokens(width, 1200);
    Fixture fx(2u, width, 1u, parent, tokens, 4u, 0u);
    fx.reset();
    FakeBackend fake;
    fake.configure(fx.layers, width, parent, tokens, fx.output_va(0), fx.output_va(0));
    fill_slots(fake, fx, fx.layers, 1u);
    te::Request req;
    req.width = width;
    req.layers = fx.layers;
    req.context_is_root = true;
    req.produce_wire = true;
    req.require_norm = true;
    req.conv_head = 4u;
    req.conv_slots = kConvSlots;
    req.parent = parent;
    req.draft = tokens;
    req.verified = tokens;
    // Path 0 -> 1 -> 3 -> 5; cut nodes 2 (child of 0), 4 (child of 1), 6 (child of 3) and 7.
    req.verified[2] = tokens[2] + 1;
    req.verified[4] = tokens[4] + 1;
    req.verified[6] = tokens[6] + 1;
    req.verified[7] = tokens[7] + 1;
    te::Session session(kConvSlots);
    VJ_CHECK(session.configure(req).treewy());
    for (uint32_t l = 0; l < fx.layers; ++l) {
        std::string err;
        VJ_CHECK(session.bind_layer(l, make_binding(fx, 0, l, 0), err));
        VJ_CHECK(session.enqueue_verify(fake, l, err));
        VJ_CHECK(session.enqueue_norm(fake, l, err));
    }
    std::string err;
    const te::Resolution r = session.resolve(fake, te::Acceptance{}, err);
    VJ_CHECK(r.used && r.ok && !r.fatal);
    VJ_CHECK(r.path_nodes == 4u);
    VJ_CHECK(r.selection.accepted_node_id == 5u);
    VJ_CHECK(r.commit_accepted == 6);  // node id + 1, not the accepted-node count
    VJ_CHECK(r.selection.path == (std::vector<uint32_t>{0u, 1u, 3u, 5u}));
    VJ_CHECK(r.new_tokens == 4u);
    VJ_CHECK(r.post.emitted ==
             (std::vector<int32_t>{tokens[1], tokens[3], tokens[5], tokens[7] + 1}));
    VJ_CHECK(r.conv_head == 8u);
    const std::vector<uint32_t> path = {0u, 1u, 3u, 5u};
    for (uint32_t l = 0; l < fx.layers; ++l)
        check_state(fx, 0, l, r.active_slot[l], path_digest(fx, l, tokens, path));
    // The rejected sibling (node 4, also a child of 1) must not be the committed state.
    VJ_CHECK(read_digest(fx.state_ptr(0, 0, r.active_slot[0])) != fake.nodes[0][4]);
    std::printf("  tree path=0,1,3,5 selected_node=5 commit=%d new=4 conv_head=%u\n",
                r.commit_accepted, r.conv_head);
}

// ---- multi-shard/layer atomic flip and commit-failure restoration -----------------------
void test_multilayer_atomicity() {
    std::printf("[layers] all-or-nothing flip and commit-failure restoration\n");
    const uint32_t width = 8;
    const std::vector<int32_t> parent = chain_parent(width);
    const std::vector<int32_t> tokens = chain_tokens(width, 1400);
    {
        Fixture fx(4u, width, 1u, parent, tokens, 2u, 0u);
        fx.reset();
        FakeBackend fake;
        bool bound = false;
        const te::Resolution r =
            run_direct(fx, fake, te::Acceptance{te::AcceptanceKind::Prefix, 3}, true, &bound);
        VJ_CHECK(bound && r.ok && !r.fatal);
        VJ_CHECK(fake.commit_calls == fx.layers && fake.restore_calls == 0);
        for (uint32_t l = 0; l < fx.layers; ++l) {
            VJ_CHECK(r.active_slot[l] == 1u);
            check_state(fx, 0, l, 1u, prefix_digest(fx, l, tokens, 3));
        }
        VJ_CHECK(fake.written_states == fx.layers);
    }
    {   // commit failure on layer 2: earlier inactive slots are restored, nothing flips
        Fixture fx(4u, width, 1u, parent, tokens, 2u, 0u);
        fx.reset();
        FakeBackend fake;
        fake.configure(fx.layers, width, parent, tokens, fx.output_va(0), fx.output_va(0));
        fake.fail_commit_layer = 2u;
        fill_slots(fake, fx, fx.layers, 1u);
        te::Request req;
        req.width = width;
        req.layers = fx.layers;
        req.context_is_root = true;
        req.produce_wire = true;
        req.require_norm = true;
        req.conv_head = 2u;
        req.conv_slots = kConvSlots;
        req.parent = parent;
        req.draft = tokens;
        te::Session session(kConvSlots);
        VJ_CHECK(session.configure(req).treewy());
        for (uint32_t l = 0; l < fx.layers; ++l) {
            std::string err;
            VJ_CHECK(session.bind_layer(l, make_binding(fx, 0, l, 0), err));
            VJ_CHECK(session.enqueue_verify(fake, l, err));
            VJ_CHECK(session.enqueue_norm(fake, l, err));
        }
        std::string err;
        const te::Resolution r =
            session.resolve(fake, te::Acceptance{te::AcceptanceKind::Prefix, 3}, err);
        VJ_CHECK(r.used && !r.ok && r.fatal && r.reason == te::FallbackReason::CommitFailed);
        VJ_CHECK(fake.restore_calls == 2u);  // layers 0 and 1 only
        for (uint32_t l = 0; l < fx.layers; ++l) {
            VJ_CHECK(read_digest(fx.state_ptr(0, l, 0)) == fx.s0(l));  // active unchanged
            if (l < 2u) VJ_CHECK(read_digest(fx.state_ptr(0, l, 1)) == fx.s0(l));  // restored
        }
        VJ_CHECK(r.conv_head == 2u);
        std::printf("  commit failure layer=2 fatal=1 restore=%llu active=unchanged\n",
                    static_cast<unsigned long long>(fake.restore_calls));
    }
    {   // submit/wait failure after all commits: restore inactive slots, active slot 1 stays S0
        Fixture fx(2u, width, 1u, parent, tokens, 2u, 1u);
        fx.reset();
        FakeBackend fake;
        fake.configure(fx.layers, width, parent, tokens, fx.output_va(0), fx.output_va(0));
        fake.fail_finish = true;
        fill_slots(fake, fx, fx.layers, 1u);
        te::Request req;
        req.width = width;
        req.layers = fx.layers;
        req.context_is_root = true;
        req.produce_wire = true;
        req.require_norm = true;
        req.conv_head = 2u;
        req.conv_slots = kConvSlots;
        req.parent = parent;
        req.draft = tokens;
        te::Session session(kConvSlots);
        VJ_CHECK(session.configure(req).treewy());
        for (uint32_t l = 0; l < fx.layers; ++l) {
            std::string err;
            VJ_CHECK(session.bind_layer(l, make_binding(fx, 0, l, 1), err));
            VJ_CHECK(session.enqueue_verify(fake, l, err));
            VJ_CHECK(session.enqueue_norm(fake, l, err));
        }
        std::string err;
        const te::Resolution r =
            session.resolve(fake, te::Acceptance{te::AcceptanceKind::Prefix, 2}, err);
        VJ_CHECK(r.used && !r.ok && r.fatal && r.reason == te::FallbackReason::CommitFailed);
        for (uint32_t l = 0; l < fx.layers; ++l) {
            VJ_CHECK(read_digest(fx.state_ptr(0, l, 1)) == fx.s0(l));
            VJ_CHECK(read_digest(fx.state_ptr(0, l, 0)) == fx.s0(l));  // restored
        }
    }
}

// ---- conv-ring cursor, descriptor order and post-verify view ----------------------------
void test_conv_ring_and_norm_view() {
    std::printf("[cursor] conv-ring order, wrap, descriptors and post-verify view\n");
    const uint32_t width = 12;
    const std::vector<int32_t> parent = chain_parent(width);
    const std::vector<int32_t> tokens = chain_tokens(width, 1600);
    for (uint32_t head : {0u, 15u}) {
        Fixture fx(2u, width, 1u, parent, tokens, head, 0u);
        fx.reset();
        FakeBackend fake;
        bool bound = false;
        const te::Resolution r =
            run_direct(fx, fake, te::Acceptance{te::AcceptanceKind::Prefix, 7}, true, &bound);
        VJ_CHECK(bound && r.ok);
        VJ_CHECK(r.conv_head == (head + 7u) % kConvSlots);
        VJ_CHECK(r.post.rows == width);
        VJ_CHECK(r.post.first_new_row == 1u);
        VJ_CHECK(r.post.accepted_rows.size() == 7u);
        VJ_CHECK(r.post.emitted.size() == 7u);
        VJ_CHECK(r.post.emitted.back() == tokens[7] + 1);
        for (uint32_t l = 0; l < fx.layers; ++l) {
            VJ_CHECK(fake.norm_m[l] == width);
            for (uint32_t node = 0; node < 7u; ++node)
                VJ_CHECK(std::find(r.post.accepted_rows.begin(), r.post.accepted_rows.end(), node) !=
                         r.post.accepted_rows.end());
            VJ_CHECK(std::find(r.post.accepted_rows.begin(), r.post.accepted_rows.end(), 7u) ==
                     r.post.accepted_rows.end());
        }
        std::printf("  head=%u width=%u prefix=7 -> conv_head=%u rows=%u emitted=%zu\n", head, width,
                    r.conv_head, r.post.rows, r.post.emitted.size());
    }
    {   // A producer that reports a different slot order falls back before any device call.
        Fixture fx(1u, 5u, 1u, chain_parent(5u), chain_tokens(5u, 1700), 0u, 0u);
        fx.reset();
        FakeBackend fake;
        fake.configure(fx.layers, fx.width, fx.parent, fx.tokens, fx.output_va(0), fx.output_va(0));
        fill_slots(fake, fx, fx.layers, 1u);
        te::Request req;
        req.width = 5u;
        req.layers = 1u;
        req.context_is_root = true;
        req.produce_wire = true;
        req.require_norm = true;
        req.conv_head = 0u;
        req.conv_slots = kConvSlots;
        req.parent = fx.parent;
        req.draft = fx.tokens;
        te::Session session(kConvSlots);
        VJ_CHECK(session.configure(req).treewy());
        te::LayerBinding b = make_binding(fx, 0, 0, 0);
        b.has_conv_slots = true;
        for (uint32_t i = 0; i < 5u; ++i) b.conv_slot[i] = (i + 2u) % kConvSlots;
        std::string err;
        VJ_CHECK(!session.bind_layer(0, b, err));
        VJ_CHECK(session.last_reason() == te::FallbackReason::ConvSlotMismatch);
        VJ_CHECK(fake.verify_calls == 0u && fake.commit_calls == 0u);
    }
}

// ---- fallbacks --------------------------------------------------------------------------

void test_branch_conv_and_no_accept() {
    std::printf("[tree/cursor] selected-node conv slot and non-root no-accept rollback\n");
    {   // Branching producer order: node 5 owns conv slot (start + 5 + 1), not (start + path).
        const uint32_t width = 8;
        const std::vector<int32_t> parent = {-1, 0, 0, 1, 1, 3, 3, 5};
        const std::vector<int32_t> tokens = chain_tokens(width, 2300);
        Fixture fx(1u, width, 1u, parent, tokens, 2u, 0u);
        fx.reset();
        FakeBackend fake;
        fake.configure(fx.layers, width, parent, tokens, fx.output_va(0), fx.output_va(0));
        fill_slots(fake, fx, fx.layers, 1u);
        te::Request req;
        req.width = width;
        req.layers = 1u;
        req.context_is_root = true;
        req.produce_wire = true;
        req.require_norm = true;
        req.conv_head = 2u;
        req.conv_slots = kConvSlots;
        req.parent = parent;
        req.draft = tokens;
        req.verified = tokens;
        for (uint32_t node : {2u, 4u, 6u, 7u}) req.verified[node] = tokens[node] + 1;
        te::Session session(kConvSlots);
        VJ_CHECK(session.configure(req).treewy());
        te::LayerBinding b = make_binding(fx, 0, 0, 0);
        b.has_conv_slots = true;
        for (uint32_t i = 0; i < width; ++i) b.conv_slot[i] = (2u + i + 1u) % kConvSlots;
        std::string err;
        VJ_CHECK(session.bind_layer(0, b, err));
        VJ_CHECK(session.enqueue_verify(fake, 0, err));
        VJ_CHECK(session.enqueue_norm(fake, 0, err));
        const te::Resolution r = session.resolve(fake, te::Acceptance{}, err);
        VJ_CHECK(r.ok && r.path_nodes == 4u && r.selection.accepted_node_id == 5u);
        VJ_CHECK(r.commit_accepted == 6);
        VJ_CHECK(r.conv_head == 8u);  // slot after node 5, not snapshot + path length
    }
    {   // A non-root tree with no accepted node launches no commit and rolls the cursor back.
        const uint32_t width = 5;
        const std::vector<int32_t> parent = chain_parent(width);
        const std::vector<int32_t> tokens = chain_tokens(width, 2400);
        Fixture fx(1u, width, 1u, parent, tokens, 7u, 0u);
        fx.reset();
        FakeBackend fake;
        fake.configure(fx.layers, width, parent, tokens, fx.output_va(0), fx.output_va(0));
        fill_slots(fake, fx, fx.layers, 1u);
        te::Request req;
        req.width = width;
        req.layers = 1u;
        req.context_is_root = false;
        req.produce_wire = true;
        req.require_norm = true;
        req.conv_head = 7u;
        req.conv_slots = kConvSlots;
        req.parent = parent;
        req.draft = tokens;
        req.verified = tokens;
        req.verified[0] = tokens[0] + 1;  // no root child matches
        te::Session session(kConvSlots);
        VJ_CHECK(session.configure(req).treewy());
        std::string err;
        std::string why;
        VJ_CHECK(session.bind_layer(0, make_binding(fx, 0, 0, 0), err));
        VJ_CHECK(session.enqueue_verify(fake, 0, err));
        VJ_CHECK(session.enqueue_norm(fake, 0, err));
        const te::Resolution r = session.resolve(fake, te::Acceptance{}, why);
        VJ_CHECK(r.used && r.ok && !r.fatal && r.reason == te::FallbackReason::None);
        VJ_CHECK(r.path_nodes == 0u && r.state_advance == 0u);
        VJ_CHECK(r.selection.action == tl::StateAction::None);
        VJ_CHECK(r.conv_head == 7u);
        VJ_CHECK(r.new_tokens == 1u);  // target token of the first root child
        VJ_CHECK(fake.commit_calls == 0u);
        VJ_CHECK(read_digest(fx.state_ptr(0, 0, 0)) == fx.s0(0));
    }
}

void test_fallbacks() {
    std::printf("[fallback] widths, budget, descriptors, incomplete layers, producer\n");
    const std::vector<int32_t> parent = chain_parent(5u);
    const std::vector<int32_t> tokens = chain_tokens(5u, 1800);
    auto configure = [&](uint32_t width, uint64_t budget, bool bad_parent, te::Session& s,
                         te::Request& req) {
        req.width = width;
        req.layers = 1u;
        req.context_is_root = true;
        req.produce_wire = true;
        req.require_norm = true;
        req.budget_bytes = budget;
        req.conv_slots = kConvSlots;
        req.parent = chain_parent(width);
        req.draft = chain_tokens(width, 1800);
        if (bad_parent) req.parent[1] = 5;
        return s.configure(req);
    };
    {
        te::Session s(kConvSlots);
        te::Request req;
        VJ_CHECK(!configure(6u, 0u, false, s, req).treewy());
        VJ_CHECK(s.last_reason() == te::FallbackReason::WidthNotSealed);
        ++g_fallback;
    }
    {
        te::Session s(kConvSlots);
        te::Request req;
        VJ_CHECK(!configure(20u, 0u, false, s, req).treewy());
        VJ_CHECK(s.last_reason() == te::FallbackReason::WidthExceedsSequential);
        ++g_fallback;
    }
    {
        te::Session s(kConvSlots);
        te::Request req;
        VJ_CHECK(!configure(5u, 1024u, false, s, req).treewy());
        VJ_CHECK(s.last_reason() == te::FallbackReason::BudgetExceeded);
        ++g_fallback;
    }
    {
        te::Session s(kConvSlots);
        te::Request req;
        VJ_CHECK(!configure(5u, 0u, true, s, req).treewy());
        VJ_CHECK(s.last_reason() == te::FallbackReason::InvalidParent);
        ++g_fallback;
    }
    {   // aliasing state slots is rejected at bind time, before any dispatch
        Fixture fx(1u, 5u, 1u, parent, tokens, 0u, 0u);
        fx.reset();
        te::Request req;
        req.width = 5u;
        req.layers = 1u;
        req.context_is_root = true;
        req.produce_wire = true;
        req.require_norm = true;
        req.conv_slots = kConvSlots;
        req.conv_head = 0u;
        req.parent = parent;
        req.draft = tokens;
        te::Session s(kConvSlots);
        VJ_CHECK(s.configure(req).treewy());
        te::LayerBinding b = make_binding(fx, 0, 0, 0);
        b.state_slot[1] = b.state_slot[0];
        std::string err;
        VJ_CHECK(!s.bind_layer(0, b, err));
        VJ_CHECK(s.last_reason() == te::FallbackReason::DescriptorInvalid);
        ++g_fallback;
    }
    {   // resolve before every layer is bound/normalized falls back cleanly
        Fixture fx(2u, 5u, 1u, parent, tokens, 0u, 0u);
        fx.reset();
        FakeBackend fake;
        bool bound = false;
        const te::Resolution r =
            run_direct(fx, fake, te::Acceptance{te::AcceptanceKind::Prefix, 5u}, false, &bound);
        VJ_CHECK(bound && !r.ok && r.reason == te::FallbackReason::IncompleteLayers);
        ++g_fallback;
        VJ_CHECK(fake.commit_calls == 0u);
    }
    {   // producer failure is a pre-resolve fallback and leaves state untouched
        Fixture fx(1u, 5u, 1u, parent, tokens, 0u, 0u);
        fx.reset();
        FakeBackend fake;
        fake.configure(fx.layers, fx.width, fx.parent, fx.tokens, fx.output_va(0), fx.output_va(0));
        fake.wire_ok = false;
        fill_slots(fake, fx, fx.layers, 1u);
        te::Request req;
        req.width = 5u;
        req.layers = 1u;
        req.context_is_root = true;
        req.produce_wire = true;
        req.require_norm = true;
        req.conv_slots = kConvSlots;
        req.conv_head = 0u;
        req.parent = parent;
        req.draft = tokens;
        te::Session s(kConvSlots);
        VJ_CHECK(s.configure(req).treewy());
        std::string err;
        VJ_CHECK(s.bind_layer(0, make_binding(fx, 0, 0, 0), err));
        VJ_CHECK(!s.enqueue_verify(fake, 0, err));
        VJ_CHECK(s.last_reason() == te::FallbackReason::WireProduceFailed);
        ++g_fallback;
        VJ_CHECK(read_digest(fx.state_ptr(0, 0, 0)) == fx.s0(0));
    }
    {   // Hook refuses before dispatch when the module/producer is unavailable
        Options o;
        o.enabled = true;
        FakeBackend fake;
        fake.configure(1u, 5u, parent, tokens, 0, 0);
        fake.available_ = false;
        uint64_t ava[2] = {1, 1};
        uint64_t bytes[2] = {1, 1};
        Hook hook(o, 1u, kConvSlots, ava, bytes, fake);
        CycleInput in;
        in.tokens = tokens;
        VJ_CHECK(!hook.begin(in));
        ++g_fallback;
        VJ_CHECK(hook.stats().verify_launches == 0u);
    }
}

void test_acceptance_forcing() {
    std::printf("[force] deterministic acceptance prefixes at 5/8/12\n");
    for (uint32_t width : {5u, 8u, 12u}) {
        Fixture fx(1u, width, 1u, chain_parent(width), chain_tokens(width, 2000), 0u, 0u);
        for (uint32_t prefix = 1u; prefix <= width; ++prefix) {
            fx.reset();
            FakeBackend fake;
            bool bound = false;
            const te::Resolution r =
                run_direct(fx, fake, te::Acceptance{te::AcceptanceKind::Prefix, prefix}, true,
                           &bound);
            VJ_CHECK(bound && r.ok);
            VJ_CHECK(r.path_nodes == prefix);
            VJ_CHECK(r.selection.commit_accepted == static_cast<int32_t>(prefix));
            VJ_CHECK(r.bonus_pending == (prefix == width));
            VJ_CHECK(r.new_tokens == ((prefix == width) ? width - 1u : prefix));
            ++g_forced;
        }
    }
}

// ---- target authority -------------------------------------------------------------------
// The only acceptance helper the engine calls. It may shorten the target's own prefix, never
// extend or rewrite it, so no mismatched draft can be forced into the committed state.
void test_target_authority() {
    using Focus = Options::ForceAccept;
    uint64_t cases = 0;
    for (uint32_t drafts = 0; drafts <= 12u; ++drafts) {
        for (uint32_t natural = 0; natural <= drafts; ++natural) {
            for (Focus mode : {Focus::Natural, Focus::Full, Focus::Reject, Focus::Prefix}) {
                for (uint32_t prefix = 0; prefix <= 12u; ++prefix) {
                    const uint32_t got = vegajet::ornith::treewy::clamp_accepted(natural, drafts,
                                                                                mode, prefix);
                    VJ_CHECK(got <= natural);  // never accepts more than the target did
                    if (mode == Focus::Natural || mode == Focus::Full) {
                        VJ_CHECK(got == natural);  // "full" cannot be forced
                    } else if (mode == Focus::Reject) {
                        VJ_CHECK(got == 0u);
                    } else {
                        VJ_CHECK(got == std::min(prefix, natural));
                    }
                    ++cases;
                }
            }
        }
    }
    VJ_CHECK(vegajet::ornith::treewy::clamp_accepted(12u, 5u, Focus::Natural, 0u) == 5u);
    std::printf("TREEWY_CLAMP cases=%llu max_increase=0 force_full=disabled\n",
                static_cast<unsigned long long>(cases));
}

// ---- installed production producer over the real Hook -----------------------------------
// The engine never supplies the wire itself: the installed producer packs the conv layout into
// the sealed planes. This test installs a producer through the public seam, then drives the exact
// Ornith Hook with a backend that applies the same build_gdn_wire_call descriptor the device path
// enqueues. The host mirror (ops/gdn_wire_pack.hpp) writes the planes, so the bytes the sealed
// verifier would read are checked here.
struct HostProducerBackend final : FakeBackend {
    uint32_t last_grid = 0;

    bool wire_available() const override { return vegajet::ornith::treewy::sealed_wire_installed(); }

    bool upload_parent(uint32_t device, uint64_t dst, const int32_t* host, uint32_t w,
                       std::string& err) override {
        if (device >= 2u || dst == 0 || !host || w == 0) {
            err = "host producer parent upload rejected";
            return false;
        }
        std::memcpy(reinterpret_cast<void*>(static_cast<uintptr_t>(dst)), host,
                    static_cast<size_t>(w) * sizeof(int32_t));
        ++parent_uploads;
        return true;
    }

    bool produce_wire(const te::WireRequest& req, std::string& err) override {
        if (!vegajet::ornith::treewy::sealed_wire_installed()) {
            err = "sealed-wire producer is not installed";
            return false;
        }
        wp::Args a;
        uint32_t grid = 0;
        if (!vegajet::ornith::treewy::build_gdn_wire_call(req, a, grid, err)) return false;
        if (grid != wp::grid_x(req.width)) {
            err = "wire pack grid does not cover the sealed width";
            return false;
        }
        wp::HostBuffers hb;
        hb.conv = reinterpret_cast<const float*>(static_cast<uintptr_t>(a.conv_source));
        hb.decay = reinterpret_cast<const float*>(static_cast<uintptr_t>(a.decay_source));
        hb.beta = reinterpret_cast<const float*>(static_cast<uintptr_t>(a.beta_source));
        hb.query = reinterpret_cast<float*>(static_cast<uintptr_t>(a.query));
        hb.key = reinterpret_cast<float*>(static_cast<uintptr_t>(a.key));
        hb.value = reinterpret_cast<float*>(static_cast<uintptr_t>(a.value));
        hb.decay_out = reinterpret_cast<float*>(static_cast<uintptr_t>(a.decay));
        hb.beta_out = reinterpret_cast<float*>(static_cast<uintptr_t>(a.beta));
        hb.parent = reinterpret_cast<const int32_t*>(static_cast<uintptr_t>(a.parent_source));
        hb.parent_out = reinterpret_cast<int32_t*>(static_cast<uintptr_t>(a.parent));
        if (!wp::pack_host(a, hb, err)) return false;
        last_grid = grid;
        ++produce_calls;
        return true;
    }
};

void fill_producer_sources(Fixture& fx) {
    for (uint32_t g = 0; g < fx.devices; ++g) {
        float* conv = reinterpret_cast<float*>(fx.conv_words[g].data());
        float* dec = reinterpret_cast<float*>(fx.dec_words[g].data());
        float* bet = reinterpret_cast<float*>(fx.beta_words[g].data());
        for (uint32_t t = 0; t < 12u; ++t) {
            for (uint32_t i = 0; i < 6144u; ++i)
                conv[static_cast<uint64_t>(t) * 6144u + i] =
                    static_cast<float>((g + 1u) * 1.0e-3) +
                    static_cast<float>(t) * 1.0e-2f + static_cast<float>(i) * 1.0e-6f;
            for (uint32_t i = 0; i < 32u; ++i) {
                dec[static_cast<uint64_t>(t) * 32u + i] = 0.90f + 0.001f * static_cast<float>(i);
                bet[static_cast<uint64_t>(t) * 32u + i] = 0.10f + 0.001f * static_cast<float>(i);
            }
        }
    }
}

void poison_staged_planes(Fixture& fx, const tl::ScratchPlan& plan) {
    const uint32_t width = plan.width;
    const uint32_t qk = width * wp::kQueryHeads * wp::kHeadDim;
    const uint32_t vdim = width * wp::kValueHeads * wp::kHeadDim;
    const uint32_t gate = width * wp::kValueHeads;
    for (uint32_t g = 0; g < fx.devices; ++g) {
        uint8_t* base = reinterpret_cast<uint8_t*>(fx.arena_words[g].data());
        tl::StagedInputs si = tl::staged_inputs(base, plan);
        VJ_CHECK(si.ok());
        for (uint32_t i = 0; i < qk; ++i) {
            si.query[i] = std::numeric_limits<float>::quiet_NaN();
            si.key[i] = std::numeric_limits<float>::quiet_NaN();
        }
        for (uint32_t i = 0; i < vdim; ++i)
            si.value[i] = std::numeric_limits<float>::quiet_NaN();
        for (uint32_t i = 0; i < gate; ++i) {
            si.decay[i] = std::numeric_limits<float>::quiet_NaN();
            si.beta[i] = std::numeric_limits<float>::quiet_NaN();
        }
        for (uint32_t i = 0; i < width; ++i) {
            si.parent[i] = std::numeric_limits<int32_t>::min();
            si.parent_source[i] = std::numeric_limits<int32_t>::min();
        }
    }
}

void check_producer_planes(Fixture& fx, const tl::ScratchPlan& plan) {
    const uint32_t width = fx.width;
    for (uint32_t g = 0; g < fx.devices; ++g) {
        uint8_t* base = reinterpret_cast<uint8_t*>(fx.arena_words[g].data());
        tl::StagedInputs si = tl::staged_inputs(base, plan);
        VJ_CHECK(si.ok());
        const float* conv = reinterpret_cast<const float*>(fx.conv_words[g].data());
        const float* dec = reinterpret_cast<const float*>(fx.dec_words[g].data());
        const float* bet = reinterpret_cast<const float*>(fx.beta_words[g].data());
        const uint32_t qk = wp::kQueryHeads * wp::kHeadDim;
        const uint32_t vdim = wp::kValueHeads * wp::kHeadDim;
        for (uint32_t t = 0; t < width; ++t) {
            const float* src = conv + static_cast<uint64_t>(t) * wp::kConvStride;
            for (uint32_t i = 0; i < qk; ++i) {
                VJ_CHECK(si.query[static_cast<uint64_t>(t) * qk + i] == src[i]);
                VJ_CHECK(si.key[static_cast<uint64_t>(t) * qk + i] == src[qk + i]);
            }
            for (uint32_t i = 0; i < vdim; ++i)
                VJ_CHECK(si.value[static_cast<uint64_t>(t) * vdim + i] == src[2u * qk + i]);
            for (uint32_t i = 0; i < wp::kValueHeads; ++i) {
                VJ_CHECK(si.decay[static_cast<uint64_t>(t) * 32u + i] ==
                         dec[static_cast<uint64_t>(t) * 32u + i]);
                VJ_CHECK(si.beta[static_cast<uint64_t>(t) * 32u + i] ==
                         bet[static_cast<uint64_t>(t) * 32u + i]);
            }
            VJ_CHECK(si.parent[t] == fx.parent[t]);
        }
        // Every element the sealed verifier reads was poisoned before the cycle and is now
        // source-exact: no element of the consumed prefix can be stale.
        for (uint32_t i = 0; i < width * wp::kQueryHeads * wp::kHeadDim; ++i) {
            VJ_CHECK(!std::isnan(si.query[i]) && !std::isnan(si.key[i]));
        }
        for (uint32_t i = 0; i < width * wp::kValueHeads * wp::kHeadDim; ++i)
            VJ_CHECK(!std::isnan(si.value[i]));
    }
}

void test_installed_producer() {
    std::printf("[producer] installed GDN pack over the normal-engine Hook: widths 5/8/12\n");
    // The kernel-backed factory refuses incomplete shards and clears the slot.
    VJ_CHECK(!vegajet::ornith::treewy::install_gdn_wire_producer(nullptr, nullptr));
    VJ_CHECK(!vegajet::ornith::treewy::sealed_wire_installed());
    uint64_t installed_cycles = 0;
    for (uint32_t width : {5u, 8u, 12u}) {
        for (uint32_t mode = 0; mode < 3; ++mode) {
            const std::vector<int32_t> parent = chain_parent(width);
            const std::vector<int32_t> tokens = chain_tokens(width, 700u + width);
            Fixture fx(2u, width, 2u, parent, tokens, 5u, 0u);
            fill_producer_sources(fx);
            const tl::ScratchPlan plan = tl::size_scratch(width, 2u, true);
            VJ_CHECK(plan.valid() && plan.parent_source_bytes >= wp::parent_bytes(width));
            poison_staged_planes(fx, plan);

            vegajet::ornith::treewy::install_sealed_wire_producer(
                [](vegajet::kfd::Runtime&, size_t, const te::WireRequest&, std::string&) { return true; });
            VJ_CHECK(vegajet::ornith::treewy::sealed_wire_installed());

            HostProducerBackend fake;
            fake.configure(2u * 2u, width, parent, tokens, fx.arena_va(0) + plan.output_offset,
                           fx.arena_va(1) + plan.output_offset);
            for (uint32_t g = 0; g < 2u; ++g) {
                for (uint32_t l = 0; l < 2u; ++l) {
                    const uint32_t idx = l * 2u + g;
                    fake.set_slot(idx, fx.state_va(g, l, 0), fx.state_va(g, l, 1),
                                  fx.state_ptr(g, l, 0), fx.state_ptr(g, l, 1), fx.s0(l));
                }
            }
            Options opt;
            opt.enabled = true;
            uint64_t ava[2] = {fx.arena_va(0), fx.arena_va(1)};
            uint64_t bytes[2] = {fx.plan.total_bytes, fx.plan.total_bytes};
            Hook hook(opt, 2u, kConvSlots, ava, bytes, fake);
            CycleInput in;
            in.tokens = tokens;
            in.conv_head = 5u;
            in.active_slot = 0u;
            VJ_CHECK(hook.begin(in));
            for (uint32_t l = 0; l < 2u; ++l) {
                for (uint32_t g = 0; g < 2u; ++g) {
                    LayerInputs li;
                    li.gdn_index = l;
                    li.device = g;
                    li.state_slot[0] = fx.state_va(g, l, 0);
                    li.state_slot[1] = fx.state_va(g, l, 1);
                    li.records = fx.records_va(g, l);
                    li.output = hook.output_va(g);
                    li.parent = fx.parent_va(g);
                    li.conv_source = fx.conv_va(g);
                    li.decay_source = fx.dec_va(g);
                    li.beta_source = fx.beta_va(g);
                    li.gate_source = fx.gate_va(g);
                    li.norm_weight = fx.nw_va(g);
                    li.out_q8 = fx.q8_va(g);
                    li.out_ds = fx.ds_va(g);
                    li.nv = 32u;
                    li.stride_in = 10304u;
                    li.z_off = 6144u;
                    li.eps = 1e-6f;
                    std::string err;
                    VJ_CHECK(hook.enqueue_layer(li, err));
                    VJ_CHECK(err.empty());
                }
            }
            const uint32_t acc = (mode == 0u) ? (width - 1u) : ((mode == 1u) ? (width / 2u) : 0u);
            const vegajet::ornith::treewy::CycleResult r = hook.commit(acc, width - 1u);
            VJ_CHECK(r.used && r.ok && !r.fatal);
            VJ_CHECK(hook.stats().treewy_selected == 1u);
            VJ_CHECK(hook.stats().fallbacks == 0u);
            VJ_CHECK(hook.stats().verify_launches == 4u);
            VJ_CHECK(hook.stats().norm_launches == 4u);
            VJ_CHECK(hook.stats().commit_launches == 4u);
            VJ_CHECK(hook.stats().wire_produced == 4u);
            VJ_CHECK(fake.produce_calls == 4u);
            VJ_CHECK(fake.parent_uploads == 2u);
            check_producer_planes(fx, hook.plan());
            ++installed_cycles;
            ++g_cases;
            ++g_treewy;
        }
    }
    vegajet::ornith::treewy::install_sealed_wire_producer(nullptr);
    VJ_CHECK(!vegajet::ornith::treewy::sealed_wire_installed());
    std::printf("TREEWY_WIRE selected=%llu fallback=0 widths=5/8/12 producer=installed\n",
                static_cast<unsigned long long>(installed_cycles));
}

// ---- exact normal-engine driver ---------------------------------------------------------
void test_normal_engine_driver() {
    std::printf("[driver] Ornith Hook: %u GDN layers x 2 shards, widths 5/8/12\n",
                kGdnLayersEngine);
    Fixture fx(kGdnLayersEngine, 12u, 2u, chain_parent(12u), chain_tokens(12u, 3000), 13u, 0u);
    Options opt;
    opt.enabled = true;
    opt.verbose = false;
    uint64_t ava[2] = {fx.arena_va(0), fx.arena_va(1)};
    uint64_t bytes[2] = {fx.plan.total_bytes, fx.plan.total_bytes};
    for (uint32_t width : {5u, 8u, 12u}) {
        const std::vector<int32_t> parent = chain_parent(width);
        const std::vector<int32_t> tokens = chain_tokens(width, 3000 + static_cast<int32_t>(width));
        for (uint32_t mode = 0; mode < 3; ++mode) {
            const uint32_t acc = (mode == 0u) ? (width - 1u) : ((mode == 1u) ? (width / 2u) : 0u);
            fx.width = width;
            fx.parent = parent;
            fx.tokens = tokens;
            fx.active = 0u;
            fx.reset();
            FakeBackend fake;
            const tl::ScratchPlan width_plan = tl::size_scratch(width, kGdnLayersEngine, true);
            VJ_CHECK(width_plan.valid());
            fake.configure(kGdnLayersEngine * 2u, width, parent, tokens,
                           fx.arena_va(0) + width_plan.output_offset,
                           fx.arena_va(1) + width_plan.output_offset);
            Hook hook(opt, kGdnLayersEngine, kConvSlots, ava, bytes, fake);
            CycleInput in;
            in.tokens = tokens;
            in.conv_head = 13u;
            in.active_slot = 0u;
            VJ_CHECK(hook.begin(in));
            VJ_CHECK(hook.output_va(0) == fx.arena_va(0) + width_plan.output_offset);
            for (uint32_t g = 0; g < 2u; ++g) {
                for (uint32_t l = 0; l < kGdnLayersEngine; ++l) {
                    const uint32_t idx = l * 2u + g;
                    fake.set_slot(idx, fx.state_va(g, l, 0), fx.state_va(g, l, 1),
                                  fx.state_ptr(g, l, 0), fx.state_ptr(g, l, 1), fx.s0(l));
                }
            }
            for (uint32_t l = 0; l < kGdnLayersEngine; ++l) {
                for (uint32_t g = 0; g < 2u; ++g) {
                    LayerInputs li;
                    li.gdn_index = l;
                    li.device = g;
                    li.state_slot[0] = fx.state_va(g, l, 0);
                    li.state_slot[1] = fx.state_va(g, l, 1);
                    li.records = fx.records_va(g, l);
                    li.output = hook.output_va(g);
                    li.parent = fx.parent_va(g);
                    li.conv_source = fx.conv_va(g);
                    li.decay_source = fx.dec_va(g);
                    li.beta_source = fx.beta_va(g);
                    li.gate_source = fx.gate_va(g);
                    li.norm_weight = fx.nw_va(g);
                    li.out_q8 = fx.q8_va(g);
                    li.out_ds = fx.ds_va(g);
                    li.nv = 32;
                    li.stride_in = 10304;
                    li.z_off = 6144;
                    li.eps = 1e-6f;
                    std::string err;
                    if (!hook.enqueue_layer(li, err)) {
                        std::fprintf(stderr, "driver enqueue failed layer=%u gpu=%u: %s\n", l, g,
                                     err.c_str());
                        VJ_CHECK(false);
                    }
                    VJ_CHECK(err.empty());
                }
            }
            const auto r = hook.commit(acc, width - 1u);
            VJ_CHECK(r.used && r.ok && !r.fatal);
            const uint32_t path = acc + 1u;
            VJ_CHECK(r.path_nodes == path);
            VJ_CHECK(r.active_slot == 1u);
            VJ_CHECK(r.conv_head == (13u + path) % kConvSlots);
            VJ_CHECK(r.bonus_pending == (acc == width - 1u));
            const uint32_t expected_new = (acc == width - 1u) ? width - 1u : path;
            VJ_CHECK(r.new_tokens == expected_new);
            VJ_CHECK(r.emitted.size() == expected_new);
            if (acc == width - 1u) {
                for (uint32_t i = 0; i < expected_new; ++i)
                    VJ_CHECK(r.emitted[i] == tokens[i + 1u]);
            } else {
                for (uint32_t i = 0; i + 1u < path; ++i) VJ_CHECK(r.emitted[i] == tokens[i + 1u]);
                VJ_CHECK(r.emitted.back() == tokens[path] + 1);
            }
            for (uint32_t g = 0; g < 2u; ++g) {
                for (uint32_t l = 0; l < kGdnLayersEngine; ++l) {
                    VJ_CHECK(read_digest(fx.state_ptr(g, l, 1)) ==
                             prefix_digest(fx, l, tokens, path));
                    VJ_CHECK(read_digest(fx.state_ptr(g, l, 0)) == fx.s0(l));
                }
            }
            VJ_CHECK(fake.verify_calls == kGdnLayersEngine * 2u);
            VJ_CHECK(fake.norm_calls == kGdnLayersEngine * 2u);
            VJ_CHECK(fake.commit_calls == kGdnLayersEngine * 2u);
            VJ_CHECK(fake.produce_calls == kGdnLayersEngine * 2u);
            VJ_CHECK(fake.parent_uploads == 2u);
            VJ_CHECK(hook.stats().verify_launches == kGdnLayersEngine * 2u);
            VJ_CHECK(hook.stats().norm_launches == kGdnLayersEngine * 2u);
            VJ_CHECK(hook.stats().commit_launches == kGdnLayersEngine * 2u);
            VJ_CHECK(hook.stats().treewy_selected == 1u);
            ++g_driver_cycles;
            ++g_treewy;
            ++g_cases;
            g_driver_tokens += expected_new;
            std::printf("  driver M=%u mode=%s path=%u active=%u conv_head=%u new=%u\n", width,
                        mode == 0u ? "full" : (mode == 1u ? "partial" : "root-only"), r.path_nodes,
                        r.active_slot, r.conv_head, r.new_tokens);
        }
    }
    std::printf("TREEWY_DRIVER cycles=%llu emitted=%llu layers=%u shards=2\n",
                static_cast<unsigned long long>(g_driver_cycles),
                static_cast<unsigned long long>(g_driver_tokens), kGdnLayersEngine);
}

}  // namespace

int main() {
    test_target_authority();
    test_chain_full_partial_reject();
    test_branching_tree();
    test_multilayer_atomicity();
    test_conv_ring_and_norm_view();
    test_branch_conv_and_no_accept();
    test_fallbacks();
    test_acceptance_forcing();
    test_normal_engine_driver();
    test_installed_producer();
    std::printf("TREEWY_ENGINE cases=%llu treewy=%llu fallback=%llu forced=%llu selected_nonzero=%d\n",
                static_cast<unsigned long long>(g_cases),
                static_cast<unsigned long long>(g_treewy),
                static_cast<unsigned long long>(g_fallback),
                static_cast<unsigned long long>(g_forced), g_treewy > 0u ? 1 : 0);
    std::printf("TREEWY_DISPATCH treewy=%llu fallback=%llu\n",
                static_cast<unsigned long long>(g_treewy),
                static_cast<unsigned long long>(g_fallback));
    VJ_CHECK(g_treewy > 0u);
    std::printf("PASS tree-WY normal-engine caller: 5/8/12, full/partial/rollback, multi-layer, "
                "conv cursor, norm view, fallbacks. CPU only; no device claim.\n");
    return 0;
}

// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// v1.3 M4: host-side mappings of the generic N-rank executor for N = 1, 2, 3, 4 and 8.
// Covers rank row ranges, head maps, the all-reduce plan (epochs/banks/slices), buffer sizes
// and rank-loop coverage, plus fail-closed admission for the TP=2-only fast-path features.
#include "core/gpu_select.hpp"
#include "core/tp_executor.hpp"
#include "tests/test_common.hpp"

#include <cstdio>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
#include <string>
#include <vector>

using namespace vegajet;
using namespace vegajet::tp;
namespace {

template <class F>
void fails_with(F f, const std::string& needle) {
    bool threw = false;
    std::string what;
    try {
        f();
    } catch (const std::exception& e) {
        threw = true;
        what = e.what();
    }
    VJ_CHECK(threw);
    if (what.find(needle) == std::string::npos)
        std::cerr << "expected \"" << needle << "\" in: " << what << "\n";
    VJ_CHECK(what.find(needle) != std::string::npos);
}

std::vector<DeviceProfile> devices(size_t n) {
    std::vector<DeviceProfile> d(n);
    for (size_t r = 0; r < n; ++r) {
        d[r].unique_id = "exec-" + std::to_string(r);
        d[r].bdf = gpusel::bdf(0, static_cast<uint32_t>((r + 1) << 8));
        d[r].bw_hbm_gbs = 800.0 + 40.0 * static_cast<double>(r % 3);
        d[r].mclk = 1000;
        d[r].n_cu = 60;
        d[r].pcie_link_gbs = 16;
        d[r].numa_node = 0;
        d[r].usable_vram_bytes = 16ULL << 30;
    }
    return d;
}

// Every directed edge is P2P, so the collective plans peer-push routes.
PeerMatrix full_mesh(size_t n) {
    PeerMatrix p(n, std::vector<PeerLink>(n));
    for (auto& row : p)
        for (auto& e : row) e = {true, 25.0};
    return p;
}

// Ornith-shaped model: 3072 wide, 48 trunk layers, 64/8 GQA heads, 8 GDN groups.
ModelSpec model_spec() {
    ModelSpec m;
    m.replicated_bytes = 3ULL << 30;  // norms, KV/state reserve per rank
    m.rows = {{"vocab", RowKind::Vocabulary, 151936, 3072},
              {"expert_ff", RowKind::MoeIntermediate, 256 * 512, 1968},
              {"dense_proj", RowKind::Dense, 48 * 3072, 3072},
              {"shared_ff", RowKind::Dense, 48 * 512, 3072}};
    m.attention = {{"attn", 64, 8, 48ULL * 3072 * 4, 48ULL * 2 * 3072 * 4},
                   {"gdn", 32, 8, 40ULL * 3072 * 4, 40ULL * 3072 * 4}};
    return m;
}

ExecutorRequest request(uint32_t m, uint32_t frame = 2u << 20) {
    ExecutorRequest r;
    r.m = m;
    r.d_model = 3072;
    r.layers = 48;
    r.max_frame_bytes = frame;
    return r;
}

void coverage_and_maps(uint32_t n) {
    const auto devs = devices(n);
    const auto spec = model_spec();
    for (uint32_t m : {1u, 5u, 16u}) {
        const auto layout = plan_executor(spec, devs, full_mesh(n), ShardOptions{}, request(m));
        std::string err;
        VJ_CHECK(check_executor_layout(layout, spec, &err));
        VJ_CHECK(layout.ranks == n);
        VJ_CHECK(layout.rank.size() == n);
        VJ_CHECK(layout.m == m);
        // Rank-loop coverage: every rank has an identity and an allocation, in owner order.
        for (uint32_t r = 0; r < n; ++r) {
            VJ_CHECK(layout.rank[r].rank == r);
            VJ_CHECK(layout.rank[r].device_key == devs[r].key());
            VJ_CHECK(layout.rank[r].hidden_bytes == uint64_t(m) * 3072 * 4);
            VJ_CHECK(layout.rank[r].logits_bytes == layout.rank[r].vocab.size() * 4);
            VJ_CHECK(layout.rank[r].total_bytes() <= layout.rank[r].usable_vram_bytes);
        }
        // Vocabulary partition: contiguous, complete, no rank over the tensor.
        uint64_t pos = 0;
        for (uint32_t r = 0; r < n; ++r) {
            VJ_CHECK(layout.rank[r].vocab.begin == pos);
            pos = layout.rank[r].vocab.end;
        }
        VJ_CHECK(pos == 151936);
        // Head maps: every Q head and GDN group is owned exactly once.
        uint32_t q = 0, kv = 0, groups = 0, v = 0;
        for (const auto& r : layout.rank) {
            q += r.q_heads;
            kv += r.kv_heads;
            groups += r.k_groups;
            v += r.v_heads;
        }
        VJ_CHECK(q == 64);
        VJ_CHECK(groups == 8);
        VJ_CHECK(kv > 0 && v > 0);
        VJ_CHECK(kv <= 8 * n);  // a straddling GQA group may be held by two ranks
    }
}

void ar_plan(uint32_t n) {
    const auto devs = devices(n);
    const auto spec = model_spec();
    const auto layout = plan_executor(spec, devs, full_mesh(n), ShardOptions{}, request(1));
    VJ_CHECK(layout.sites.size() == 48u * 2u + 1u);
    VJ_CHECK(layout.sites.back().layer == -1);
    if (n == 1) {
        VJ_CHECK(layout.ar_bypass);
        VJ_CHECK(layout.epochs_per_step == 0);
        VJ_CHECK(layout.frames_per_rank == 0);
        for (const auto& r : layout.rank) {
            VJ_CHECK(r.rx_payload_bytes == 0 && r.rx_flag_bytes == 0 && r.partial_bytes == 0);
            VJ_CHECK(r.pushes_per_epoch == 0);
        }
        for (const auto& s : layout.sites) VJ_CHECK(s.slices.empty());
        return;
    }
    VJ_CHECK(!layout.ar_bypass);
    VJ_CHECK(layout.frames_per_rank == n);
    const auto reference = plan_collective(3072, full_mesh(n));
    VJ_CHECK(layout.collective.rx_payload_bytes_per_rank == reference.rx_payload_bytes_per_rank);
    VJ_CHECK(layout.collective.rx_flag_bytes_per_rank == reference.rx_flag_bytes_per_rank);
    for (const auto& r : layout.rank) {
        VJ_CHECK(r.pushes_per_epoch == n - 1);
        VJ_CHECK(r.partial_bytes == 1u * 3072 * 4);
    }
    uint32_t epoch = 0;
    for (const auto& s : layout.sites) {
        VJ_CHECK(!s.slices.empty());
        for (const auto& sl : s.slices) {
            VJ_CHECK(sl.epoch == epoch + 1);
            epoch = sl.epoch;
            VJ_CHECK(sl.bank == (sl.epoch & 1u));
            VJ_CHECK(sl.rows == 1 && sl.offset == 0);
            VJ_CHECK(!sl.drain);
        }
    }
    VJ_CHECK(layout.epochs_per_step == layout.sites.size());
}

void slices_and_epochs() {
    // 1024-wide row = 2048 B on the wire; a 4096 B frame holds two rows.
    uint32_t next = 1;
    const auto slices = plan_ar_slices(5, 1024, 4096, 1, &next);
    VJ_CHECK(slices.size() == 3);
    VJ_CHECK(slices[0].offset == 0 && slices[0].rows == 2 && slices[0].drain);
    VJ_CHECK(slices[1].offset == 2 && slices[1].rows == 2 && slices[1].drain);
    VJ_CHECK(slices[2].offset == 4 && slices[2].rows == 1 && !slices[2].drain);
    VJ_CHECK(slices[0].epoch == 1 && slices[0].bank == 1);
    VJ_CHECK(slices[2].epoch == 3 && slices[2].bank == 1);
    VJ_CHECK(next == 4);
    // A frame smaller than one row cannot be split further.
    fails_with([] { uint32_t n = 1; plan_ar_slices(1, 4096, 4096, 1, &n); }, "frame bound");
    // Epoch exhaustion fails closed instead of wrapping.
    fails_with([] { uint32_t n = 1; plan_ar_slices(4, 1024, 4096, 0xFFFFFFFEu, &n); }, "epoch budget");
    // Drain is a function of reuse, not of row count.
    uint32_t n2 = 1;
    const auto single = plan_ar_slices(1, 3072, 2u << 20, 7, &n2);
    VJ_CHECK(single.size() == 1 && !single[0].drain && single[0].epoch == 7 && n2 == 8);
}

void layout_geometry(uint32_t n) {
    // A prefill-sized step still fits one slice per site and keeps the schedule monotonic.
    const auto spec = model_spec();
    const auto layout = plan_executor(spec, devices(n), full_mesh(n), ShardOptions{}, request(16, 2u << 20));
    VJ_CHECK(check_executor_layout(layout, spec, nullptr));
    VJ_CHECK(layout.sites.size() == 97);
    if (n > 1) {
        VJ_CHECK(layout.rank[0].partial_bytes == 16u * 3072 * 4);
        VJ_CHECK(layout.epochs_per_step == 97);
    }
    // A two-row frame splits every site into eight slices with a drain between them.
    const auto tight = plan_executor(spec, devices(n), full_mesh(n), ShardOptions{}, request(16, 12288));
    VJ_CHECK(check_executor_layout(tight, spec, nullptr));
    if (n > 1) {
        VJ_CHECK(tight.sites[0].slices.size() == 8);
        VJ_CHECK(tight.sites[0].slices.front().drain);
        VJ_CHECK(!tight.sites[0].slices.back().drain);
        VJ_CHECK(tight.epochs_per_step == 97u * 8u);
    }
}

void owner_overrides() {
    // Owner row counts are binding and must still cover the vocabulary exactly.
    const auto spec = model_spec();
    ShardOptions o;
    o.tensors["vocab"].rows = {151936 / 2, 151936 / 2};
    const auto layout = plan_executor(spec, devices(2), full_mesh(2), o, request(1));
    VJ_CHECK(layout.rank[0].vocab.size() == 151936 / 2);
    VJ_CHECK(layout.rank[1].vocab.size() == 151936 / 2);
    // Owner fractions that do not cover the tensor fail inside the sharder.
    ShardOptions bad;
    bad.tensors["vocab"].rows = {151936 / 2, 151936 / 2 - 32};
    fails_with([&] { plan_executor(spec, devices(2), full_mesh(2), bad, request(1)); }, "cover");
    // Unknown tensor names are rejected, never ignored.
    ShardOptions unknown;
    unknown.tensors["nope"].fractions = {0.5, 0.5};
    fails_with([&] { plan_executor(spec, devices(2), full_mesh(2), unknown, request(1)); }, "unknown tensor");
}

void capacity_and_arguments() {
    auto tiny = devices(3);
    for (auto& d : tiny) d.usable_vram_bytes = 2ULL << 30;  // replicated floor is 3 GiB
    fails_with([&] { plan_executor(model_spec(), tiny, full_mesh(3), ShardOptions{}, request(1)); }, "usable VRAM");
    fails_with([] { plan_executor(model_spec(), devices(2), full_mesh(2), ShardOptions{}, request(0)); }, "tokens");
    fails_with([] { plan_executor(model_spec(), devices(2), full_mesh(3), ShardOptions{}, request(1)); }, "peer matrix");
    ExecutorRequest no_layers;
    no_layers.m = 1;
    no_layers.d_model = 3072;
    fails_with([&] { plan_executor(model_spec(), devices(2), full_mesh(2), ShardOptions{}, no_layers); }, "layer");
    // Unknown links plan host-bounce, never an optimistic peer read.
    const auto unknown = plan_executor(model_spec(), devices(3), PeerMatrix{}, ShardOptions{}, request(1));
    for (uint32_t s = 0; s < 3; ++s)
        for (uint32_t d = 0; d < 3; ++d)
            if (s != d) VJ_CHECK(unknown.collective.routes[s][d] == Transport::HostBounce);
}

void admission() {
    FeatureFlags none;
    VJ_CHECK(admit_features(1, none).empty());
    VJ_CHECK(admit_features(2, none).empty());
    VJ_CHECK(admit_features(8, none).empty());
    struct Case {
        bool FeatureFlags::*member;
        const char* needle;
        const char* flag;
    } cases[] = {
        {&FeatureFlags::super_kernels, "super-kernels", "VEGAJET_ENABLE_SUPER"},
        {&FeatureFlags::asym_experts, "D1 asymmetric expert layout", "VEGAJET_V12_ASYM_EXPERTS"},
        {&FeatureFlags::mtp_draft, "MTP draft head", "draft/MTP"},
        {&FeatureFlags::speculative, "speculative draft-verify", "suffix/DFlash2"},
        {&FeatureFlags::int8_wire, "INT8 all-reduce wire", "VEGAJET_V12_AR_INT8"},
        {&FeatureFlags::fused_verify, "fused verify+norm", "VEGAJET_V12_VERIFY_FUSED_NQ"},
        {&FeatureFlags::prefill_dot8, "W4A4 prefill", "--prefill-dot8"},
        {&FeatureFlags::ring_state, "multi-slot ReplaySSM ring", "spec GDN"},
    };
    for (const auto& c : cases) {
        FeatureFlags f;
        f.*c.member = true;
        for (uint32_t n : {3u, 4u, 8u}) {
            const std::string why = admit_features(n, f);
            VJ_CHECK(why.find(c.needle) != std::string::npos);
            VJ_CHECK(why.find(c.flag) != std::string::npos);
            VJ_CHECK(why.find("TP=" + std::to_string(n)) != std::string::npos);
            VJ_CHECK(why.find("TP=2") != std::string::npos);
        }
        // TP=1 cannot run a two-rank schedule either.
        VJ_CHECK(!admit_features(1, f).empty());
        // TP=2 is the fast path: unchanged, admitted.
        VJ_CHECK(admit_features(2, f).empty());
    }
    // Several unsupported features are all named, not just the first.
    FeatureFlags many;
    many.super_kernels = many.asym_experts = many.mtp_draft = true;
    const std::string why = admit_features(3, many);
    VJ_CHECK(why.find("super-kernels") != std::string::npos);
    VJ_CHECK(why.find("D1 asymmetric") != std::string::npos);
    VJ_CHECK(why.find("MTP draft head") != std::string::npos);
    VJ_CHECK(std::count(why.begin(), why.end(), ';') == 3);  // three features plus the trailer
}

void determinism_and_format() {
    const auto spec = model_spec();
    const auto a = plan_executor(spec, devices(3), full_mesh(3), ShardOptions{}, request(1));
    const auto b = plan_executor(spec, devices(3), full_mesh(3), ShardOptions{}, request(1));
    VJ_CHECK(a.rank.size() == b.rank.size());
    for (size_t r = 0; r < a.rank.size(); ++r) {
        VJ_CHECK(a.rank[r].vocab.begin == b.rank[r].vocab.begin);
        VJ_CHECK(a.rank[r].vocab.end == b.rank[r].vocab.end);
        VJ_CHECK(a.rank[r].total_bytes() == b.rank[r].total_bytes());
    }
    const std::string text = format_executor_layout(a);
    VJ_CHECK(text.find("TP=3") != std::string::npos);
    for (uint32_t r = 0; r < 3; ++r) VJ_CHECK(text.find("rank " + std::to_string(r)) != std::string::npos);
    VJ_CHECK(text.find("sites, ") != std::string::npos);
}

void detection() {
    // A layout that lost a rank, a row or a slice is rejected instead of being run.
    const auto spec = model_spec();
    auto layout = plan_executor(spec, devices(3), full_mesh(3), ShardOptions{}, request(1));
    std::string err;
    auto broken = layout;
    broken.rank.pop_back();
    VJ_CHECK(!check_executor_layout(broken, spec, &err) && err.find("rank table") != std::string::npos);
    broken = layout;
    broken.rank[1].vocab.begin += 32;
    VJ_CHECK(!check_executor_layout(broken, spec, &err) && err.find("contiguous") != std::string::npos);
    broken = layout;
    broken.sites[3].slices[0].epoch = broken.sites[2].slices[0].epoch;
    VJ_CHECK(!check_executor_layout(broken, spec, &err) && err.find("strictly increasing") != std::string::npos);
    broken = layout;
    broken.rank[0].rx_payload_bytes += 64;
    VJ_CHECK(!check_executor_layout(broken, spec, &err) && err.find("RX payload") != std::string::npos);
    broken = layout;
    broken.rank[2].hidden_bytes -= 4;
    VJ_CHECK(!check_executor_layout(broken, spec, &err) && err.find("hidden buffer") != std::string::npos);
    broken = layout;
    broken.sites.pop_back();
    VJ_CHECK(!check_executor_layout(broken, spec, &err) && err.find("site count") != std::string::npos);
}

} // namespace

int main() {
    coverage_and_maps(1);
    coverage_and_maps(2);
    coverage_and_maps(3);
    coverage_and_maps(4);
    coverage_and_maps(8);
    ar_plan(1);
    ar_plan(2);
    ar_plan(3);
    ar_plan(4);
    ar_plan(8);
    slices_and_epochs();
    for (uint32_t n : {1u, 2u, 3u, 4u, 8u}) layout_geometry(n);
    owner_overrides();
    capacity_and_arguments();
    admission();
    determinism_and_format();
    detection();
    std::printf("test_tp_executor: PASS (N=1,2,3,4,8)\n");
    return 0;
}

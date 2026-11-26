// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/tp_collective.hpp"
#include "core/hw_profile.hpp"
#include "ops/kernels/vj_ar_tp3_alltoall.hiph"
#include "tests/test_common.hpp"
#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>

using namespace vegajet;
using namespace vegajet::tp;
namespace {
template<class F> void fails(F f) {
    bool threw = false; try { f(); } catch (const std::exception&) { threw = true; } VJ_CHECK(threw);
}
PeerMatrix matrix(uint32_t n, bool direct = true) {
    PeerMatrix m(n, std::vector<PeerLink>(n));
    for (uint32_t s = 0; s < n; ++s) for (uint32_t d = 0; d < n; ++d)
        if (s != d) m[s][d] = {direct, direct ? (s % 3 == 0 ? 16.0 : s % 3 == 1 ? 8.0 : 4.0) : 0};
    return m;
}
std::vector<float> partial(uint32_t rank, uint32_t epoch, size_t len) {
    std::vector<float> values(len);
    for (size_t i = 0; i < len; ++i) values[i] = static_cast<float>((rank + 1) * 3 + epoch % 23) * 0.00101f + static_cast<float>(i % 11) * -0.12345f;
    return values;
}
std::vector<float> oracle(uint32_t n, uint32_t epoch, size_t len) {
    if (n == 1) return partial(0, epoch, len);
    std::vector<float> result(len, 0);
    for (uint32_t r = 0; r < n; ++r) {
        const auto input = partial(r, epoch, len);
        for (size_t i = 0; i < len; ++i) result[i] += vj_tp3::from_half(vj_tp3::to_half(input[i]));
    }
    return result;
}
void numerics() {
    for (uint32_t n : {1, 2, 3, 4, 6, 8, 12, 16, 20, 23}) for (size_t len : {1, 31, 32, 33, 3073}) {
        auto m = matrix(n);
        if (n > 2) m[0][n - 1] = {}; // exercise directed host bounce alongside P2P
        const auto p = plan_collective(len, m);
        VJ_CHECK(p.bypass == (n == 1));
        VJ_CHECK(p.payload_stride % 64 == 0 && p.rx_flag_bytes_per_rank % 64 == 0);
        CpuCollective cpu(p);
        VJ_CHECK(cpu.mailbox_frames() == (n == 1 ? 0 : 2ULL * n * n));
        if (n == 1) VJ_CHECK(p.routes.empty() && !p.banks && !p.rx_payload_bytes_per_rank && !p.rx_flag_bytes_per_rank);
        std::mt19937 rng(n * static_cast<uint32_t>(len));
        for (uint32_t epoch = 1; epoch <= 6; ++epoch) {
            for (uint32_t r = 0; r < n; ++r) cpu.begin(r, epoch, partial(r, epoch, len));
            if (n != 1) {
                std::vector<std::pair<uint32_t, uint32_t>> pushes;
                for (uint32_t s = 0; s < n; ++s) for (uint32_t d = 0; d < n; ++d) pushes.emplace_back(s, d);
                std::shuffle(pushes.begin(), pushes.end(), rng);
                VJ_CHECK(!cpu.try_consume(0, epoch));
                for (const auto& e : pushes) cpu.push(e.first, e.second);
            }
            if (n > 1) { VJ_CHECK(cpu.try_consume(0, epoch)); fails([&] { cpu.result(0, epoch); }); }
            for (uint32_t r = n > 1 ? 1 : 0; r < n; ++r) VJ_CHECK(cpu.try_consume(r, epoch));
            for (uint32_t r = 0; r < n; ++r) VJ_CHECK(cpu.result(r, epoch) == oracle(n, epoch, len));
        }
    }
}
void asynchronous() {
    // No epoch barriers: random source/destination delivery and delayed ranks across 19 bank reuses.
    for (uint32_t n : {2, 3, 4, 6, 8, 12, 16, 20}) for (uint32_t seed : {1, 17, 91}) {
        const size_t len = 65;
        CpuCollective cpu(plan_collective(len, matrix(n)));
        std::mt19937 rng(seed);
        std::vector<uint32_t> epoch(n, 1), phase(n, 0);
        std::vector<std::vector<uint32_t>> destinations(n);
        size_t completed = 0, steps = 0;
        while (completed < n) {
            VJ_CHECK(++steps < 2000000);
            uint32_t r = rng() % n;
            if (epoch[r] > 20) continue;
            if (phase[r] == 0) {
                cpu.begin(r, epoch[r], partial(r, epoch[r], len));
                destinations[r].resize(n); std::iota(destinations[r].begin(), destinations[r].end(), 0);
                std::shuffle(destinations[r].begin(), destinations[r].end(), rng);
                phase[r] = 1;
            } else if (phase[r] == 1) {
                cpu.push(r, destinations[r].back()); destinations[r].pop_back();
                if (destinations[r].empty()) phase[r] = 2;
            } else if (cpu.try_consume(r, epoch[r])) {
                if (++epoch[r] == 21) ++completed;
                phase[r] = 0;
            }
        }
        for (uint32_t r = 0; r < n; ++r) VJ_CHECK(cpu.result(r, 20) == oracle(n, 20, len));
        VJ_CHECK(!cpu.failed());
    }
}
void reuse_faults_wrap() {
    auto p = plan_collective(2, matrix(3));
    CpuCollective cpu(p);
    for (uint32_t r = 0; r < 3; ++r) cpu.publish(r, 1, {1.0001f, 2.0001f});
    VJ_CHECK(cpu.try_consume(0, 1)); VJ_CHECK(cpu.try_consume(1, 1));
    cpu.publish(0, 2, {3, 4}); cpu.publish(1, 2, {3, 4});
    VJ_CHECK(!cpu.try_consume(0, 2)); // slow rank still holds epoch 1, other parity must remain intact
    VJ_CHECK(cpu.try_consume(2, 1));
    cpu.publish(2, 2, {3, 4});
    for (uint32_t r = 0; r < 3; ++r) VJ_CHECK(cpu.try_consume(r, 2));
    VJ_CHECK(cpu.result(2, 2) == std::vector<float>({9, 12}));
    cpu.publish(0, 3, {1, 2});
    fails([&] { cpu.publish(0, 4, {5, 6}); });
    VJ_CHECK(cpu.failed()); fails([&] { cpu.result(2, 2); });
    fails([&] { cpu.reset(false); }); cpu.reset(true);
    for (uint32_t r = 0; r < 3; ++r) cpu.publish(r, 1, {1, 2});
    VJ_CHECK(cpu.try_consume(0, 1));
    fails([&] { cpu.timeout(2); });
    fails([&] { cpu.try_consume(1, 1); });
    cpu.reset(true);
    cpu.begin(0, 1, {1, 2}); cpu.push(0, 0);
    fails([&] { cpu.push(0, 0); });
    for (uint32_t n : {1, 2, 3, 20}) {
        CpuCollective near_wrap(plan_collective(1, matrix(n)), UINT32_MAX - 1);
        for (uint32_t e : {UINT32_MAX - 1, UINT32_MAX}) {
            for (uint32_t r = 0; r < n; ++r) near_wrap.publish(r, e, {1.0001f});
            for (uint32_t r = 0; r < n; ++r) VJ_CHECK(near_wrap.try_consume(r, e));
            VJ_CHECK(near_wrap.result(0, e)[0] == (n == 1 ? 1.0001f : static_cast<float>(n)));
        }
        fails([&] { near_wrap.publish(0, 0, {1}); }); VJ_CHECK(near_wrap.failed());
        near_wrap.reset(true); near_wrap.publish(0, 1, {1});
    }
    fails([&] { plan_collective(UINT64_MAX, matrix(2)); });
    fails([&] { plan_collective(0, matrix(1)); });
    fails([&] { plan_collective(1, {}); });
    p.banks = 1; fails([&] { CpuCollective bad(p); });
    p = plan_collective(2, matrix(3)); p.routes[0][1] = Transport::Local;
    fails([&] { CpuCollective bad(p); });
    p = plan_collective(2, matrix(1)); p.epoch_flag_stride = 64;
    fails([&] { CpuCollective bad(p); });
}
void topology() {
    auto m = matrix(6, false);
    for (uint32_t begin : {0, 3}) for (uint32_t a = begin; a < begin + 3; ++a)
        for (uint32_t b = begin; b < begin + 3; ++b) if (a != b) m[a][b] = {true, a % 2 ? 8.0 : 16.0};
    auto t = plan_topology(m);
    VJ_CHECK(t.islands == std::vector<std::vector<uint32_t>>({{0, 1, 2}, {3, 4, 5}}));
    VJ_CHECK(t.stages == t.islands && t.stage_transports[0] == Transport::HostBounce);
    auto p = plan_collective(3072, m);
    VJ_CHECK(p.routes[0][1] == Transport::PeerPush && p.routes[0][3] == Transport::HostBounce);
    CollectiveOptions o; o.max_tp = 2;
    VJ_CHECK(plan_topology(m, o).stages.size() == 4);
    o = {}; o.min_p2p_gbs = 12;
    VJ_CHECK(plan_topology(m, o).islands.size() > 2);
    VJ_CHECK(plan_collective(1, m, o).routes[0][1] == Transport::PeerPush); // grouping threshold does not revoke valid directed P2P
    o = {}; o.max_tp = 2;
    const auto flat = plan_collective(1, matrix(4), o);
    const auto proposed = plan_topology(matrix(4), o);
    VJ_CHECK(flat.ranks == 4 && proposed.stages.size() == 2);
    VJ_CHECK(format_collective_plan(flat, proposed).find("whole-matrix benchmark collective") != std::string::npos);
    // Connected A-B-C is not a pairwise-P2P island; asymmetric capability isn't mutual P2P.
    m = matrix(3, false); m[0][1] = m[1][0] = m[1][2] = m[2][1] = {true, 8};
    t = plan_topology(m);
    VJ_CHECK(t.islands == std::vector<std::vector<uint32_t>>({{0, 1}, {2}}));
    m[0][1] = {}; VJ_CHECK(plan_topology(m).islands[0] == std::vector<uint32_t>({0}));
    o = collective_options_from_json(R"({"schema":1,"allow_host_tp":true,"stages":[[2,0],[1]],"host_bounce":[[1,2]]})");
    t = plan_topology(m, o);
    VJ_CHECK(t.stages[0] == std::vector<uint32_t>({2, 0}));
    VJ_CHECK(plan_collective(1, m, o).routes[1][2] == Transport::HostBounce);
    o.allow_host_tp = false; fails([&] { plan_topology(m, o); });
    o.allow_host_tp = true; o.stages = {{0}, {1}}; fails([&] { plan_topology(m, o); });
    o.stages = {{0, 1}, {1, 2}}; fails([&] { plan_topology(m, o); });
    o = {}; o.host_bounce = {{0, 7}}; fails([&] { plan_collective(1, m, o); });
    for (const auto& text : {R"({"schema":1,"stages":[[]]})", R"({"schema":1,"stage":[[0]]})",
                           R"({"schema":1,"schema":1})", R"({"schema":1,"max_tp":-1})"})
        fails([&] { collective_options_from_json(text); });
    hw::Profile legacy; legacy.gpus.resize(3);
    for (uint32_t r = 0; r < 3; ++r) {
        auto& g = legacy.gpus[r]; g.index = r; g.p2p.resize(3);
        for (uint32_t d = 0; d < 3; ++d) if (r != d) g.p2p[d] = {hw::LinkKind::Pcie, 8, 1};
    }
    VJ_CHECK(plan_topology(peer_matrix_from_json(hw::to_json(legacy))).islands.size() == 1);
    const auto printed = format_collective_plan(p, plan_topology(matrix(6)));
    VJ_CHECK(printed.find("FP16") != std::string::npos && printed.find("host-bounce") != std::string::npos);
    for (uint32_t n : {1, 2, 3, 4, 6, 8, 12, 16, 20, 23}) {
        const auto peers = matrix(n);
        VJ_CHECK(plan_topology(peers).islands.size() == 1);
        VJ_CHECK(plan_topology(peers).stages[0].size() == n);
        VJ_CHECK(plan_topology(peers).islands == plan_topology(peers).islands);
        VJ_CHECK(plan_topology(matrix(n, false)).islands.size() == n);
    }
}
} // namespace
int main() {
    numerics(); asynchronous(); reuse_faults_wrap(); topology();
    std::cout << "tp_collective: FP16 N-rank reference, asynchronous parity reuse, epoch exhaustion, fail-stop, topology overrides PASS\n";
}

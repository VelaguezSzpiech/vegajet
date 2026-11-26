// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/tp_runtime_topology.hpp"
#include "core/tp_collective_abi.hpp"
#include "tests/test_common.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>
using namespace vegajet::tp;
namespace {
struct Fixture {
    std::filesystem::path root = std::filesystem::temp_directory_path() / ("vj-tp-topology-" + std::to_string(getpid()));
    ~Fixture() { std::filesystem::remove_all(root); }
    void write(const std::string& path, const std::string& text) {
        const auto target = root / path;
        std::filesystem::create_directories(target.parent_path());
        std::ofstream(target) << text;
    }
    void edge(uint32_t s, uint32_t d, unsigned flags = 0, unsigned type = 2, unsigned mbps = 8000) {
        write("sys/class/kfd/kfd/topology/nodes/" + std::to_string(s) + "/io_links/" + std::to_string(d) + "/properties",
              "node_from " + std::to_string(s) + "\nnode_to " + std::to_string(d) + "\ntype " + std::to_string(type) +
              "\nflags " + std::to_string(flags) + "\nmin_bandwidth " + std::to_string(mbps) + "\n");
    }
};
}
int main() {
    for (uint32_t n : {1, 2, 3, 4, 8}) {
        Fixture f;
        std::vector<RuntimeDevice> devices;
        for (uint32_t r = 0; r < n; ++r) {
            // Source literals are split: fixtures must not leak real device identifiers.
            devices.push_back({r + 1, 128 + r, "0000" ":0" + std::to_string(r) + ":00.0"});
            f.write("sys/class/drm/renderD" + std::to_string(128 + r) + "/device/unique_id", r == 0 ? "0x0000ABCD\n" : "0\n");
        }
        for (uint32_t s = 1; s <= n; ++s) for (uint32_t d = 1; d <= n; ++d) if (s != d) f.edge(s, d, 2); // NON_COHERENT is allowed
        auto topology = read_runtime_topology(f.root.string(), devices);
        VJ_CHECK(topology.identities[0] == "unique_id:abcd");
        for (uint32_t r = 1; r < n; ++r) VJ_CHECK(topology.identities[r] == "bdf:" + devices[r].bdf);
        for (uint32_t s = 0; s < n; ++s) for (uint32_t d = 0; d < n; ++d) {
            VJ_CHECK(topology.peers[s][d].p2p == (s != d));
            VJ_CHECK(topology.peers[s][d].gbs == (s == d ? 0 : 8));
        }
        auto plan = plan_collective(33, topology.peers);
        VJ_CHECK(plan.ranks == n && plan.bypass == (n == 1));
        VJ_CHECK(plan.rx_payload_bytes_per_rank == (n == 1 ? 0 : uint64_t(2) * n * 128));
        VJ_CHECK(plan.rx_flag_bytes_per_rank == (n == 1 ? 0 : uint64_t(2) * n * 64));
        std::vector<MailboxAddresses> boxes;
        if (n != 1) for (uint32_t r = 0; r < n; ++r) {
            const uint64_t base = uint64_t(r + 1) * 65536;
            boxes.push_back({base, base + 4096, base + 8192, base + 12288});
        }
        auto routes = collective_route_table(plan, boxes);
        VJ_CHECK(routes.size() == (n == 1 ? 0 : n * n));
        for (uint32_t s = 0; n != 1 && s < n; ++s) for (uint32_t d = 0; d < n; ++d) {
            VJ_CHECK(routes[s * n + d].push_payload == boxes[d].payload);
            VJ_CHECK(routes[s * n + d].read_payload == boxes[s].payload);
            VJ_CHECK(routes[s * n + d].push_flags == boxes[d].flags);
            VJ_CHECK(routes[s * n + d].read_flags == boxes[s].flags);
        }
        if (n > 1) {
            f.edge(1, n, 16); // NO_PEER_TO_PEER_DMA overrides bandwidth
            topology = read_runtime_topology(f.root.string(), devices);
            VJ_CHECK(!topology.peers[0][n - 1].p2p && topology.peers[n - 1][0].p2p);
            plan = plan_collective(3073, topology.peers);
            VJ_CHECK(plan.routes[0][n - 1] == Transport::HostBounce);
            VJ_CHECK(plan.routes[n - 1][0] == Transport::PeerPush);
            routes = collective_route_table(plan, boxes);
            VJ_CHECK(routes[n - 1].push_payload == boxes[n - 1].host_payload);
            VJ_CHECK(routes[(n - 1) * n].read_payload == boxes[n - 1].host_payload);
            VJ_CHECK(routes[(n - 1) * n].push_payload == boxes[0].payload);
            auto missing = boxes; missing[n - 1].host_flags = 0;
            bool missing_threw = false;
            try { (void)collective_route_table(plan, missing); } catch (const std::invalid_argument&) { missing_threw = true; }
            VJ_CHECK(missing_threw);
            f.edge(n, 1, 0, 0); // CPU/unknown is not P2P
            VJ_CHECK(!read_runtime_topology(f.root.string(), devices).peers[n - 1][0].p2p);
            auto duplicate = devices;
            duplicate[1] = duplicate[0];
            bool threw = false;
            try { (void)read_runtime_topology(f.root.string(), duplicate); } catch (const std::invalid_argument&) { threw = true; }
            VJ_CHECK(threw);
        }
        // Selected rank ordering, not KFD node ordinal, controls the matrix.
        std::reverse(devices.begin(), devices.end());
        VJ_CHECK(read_runtime_topology(f.root.string(), devices).identities.back() == "unique_id:abcd");
    }
    std::cout << "runtime topology/identity and mailbox sizes PASS N=1,2,3,4,8\n";
}

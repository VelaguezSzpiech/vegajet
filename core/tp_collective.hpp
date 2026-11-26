// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace vegajet::tp {
struct PeerLink { bool p2p = false; double gbs = 0; };
using PeerMatrix = std::vector<std::vector<PeerLink>>;
// Reads schema-1 hw_profile.json's gpus[].p2p entries. Missing edges stay unknown/host.
PeerMatrix peer_matrix_from_json(const std::string& text);
struct CollectiveOptions {
    double min_p2p_gbs = 0;
    uint32_t max_tp = 0; // zero: entire pairwise-P2P island; owner can cap group size
    bool allow_host_tp = false;
    std::vector<std::vector<uint32_t>> stages;
    std::vector<std::pair<uint32_t, uint32_t>> host_bounce;
};
CollectiveOptions collective_options_from_json(const std::string& text);
enum class Transport { Local, PeerPush, HostBounce };
struct TopologyPlan {
    std::vector<std::vector<uint32_t>> islands; // disjoint cliques, not connected components
    std::vector<std::vector<uint32_t>> stages; // one TP group per ordered pipeline stage
    std::vector<Transport> stage_transports; // stage[i].front() -> stage[i+1].front()
};
TopologyPlan plan_topology(const PeerMatrix& peers, const CollectiveOptions& options = {});
struct CollectivePlan {
    uint32_t abi = 1, ranks = 0;
    uint64_t elements = 0;
    bool bypass = false;
    uint32_t banks = 0, epoch_flag_stride = 0;
    uint64_t payload_stride = 0, rx_payload_bytes_per_rank = 0, rx_flag_bytes_per_rank = 0;
    std::vector<std::vector<Transport>> routes; // [source][destination]; local RX frame included
};
CollectivePlan plan_collective(uint64_t elements, const PeerMatrix& peers,
                               const CollectiveOptions& options = {});
std::string format_collective_plan(const CollectivePlan& plan, const TopologyPlan& topology);

// A deterministic interleaving model, not proof of GPU fences or BAR mappings.
// Each rank completes all outgoing pushes before consuming, and consumes e before
// beginning e+1. This is the TP3 header/RT-B1 two-bank reuse rule generalized to N.
class CpuCollective {
public:
    explicit CpuCollective(const CollectivePlan& plan, uint32_t first_epoch = 1);
    void begin(uint32_t source, uint32_t epoch, const std::vector<float>& partial);
    void push(uint32_t source, uint32_t destination);
    void publish(uint32_t source, uint32_t epoch, const std::vector<float>& partial);
    bool try_consume(uint32_t rank, uint32_t epoch);
    // Tentative outputs are exposed only after EVERY rank consumed this epoch.
    std::vector<float> result(uint32_t rank, uint32_t epoch) const;
    void timeout(uint32_t rank);
    bool failed() const { return failed_; }
    const std::string& error() const { return error_; }
    uint64_t mailbox_frames() const;
    // Simulation acknowledgement only. A GPU implementation needs actual queue quiescence.
    void reset(bool all_queues_quiescent);
private:
    struct alignas(64) EpochFlag { uint32_t epoch = 0; };
    static_assert(sizeof(EpochFlag) == 64, "epoch flags occupy isolated cache lines");
    struct Frame { std::vector<uint16_t> wire; EpochFlag ready; };
    struct RankState {
        uint32_t published = 0, consumed = 0, pending_epoch = 0;
        std::vector<uint16_t> pending;
        std::vector<bool> sent;
        std::vector<float> output;
    };
    CollectivePlan plan_;
    std::vector<std::vector<Frame>> rx_; // exclusively polled by destination owner
    std::vector<RankState> states_;
    bool failed_ = false;
    std::string error_;
    void check_rank(uint32_t rank) const;
    void healthy() const;
    [[noreturn]] void fault(const std::string& message);
};
} // namespace vegajet::tp

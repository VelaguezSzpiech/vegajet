// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/tp_collective.hpp"
#include "core/tp_json.hpp"
#include "ops/kernels/vj_ar_tp3_alltoall.hiph"
#include <numeric>
#include <sstream>

namespace vegajet::tp {
using namespace detail;
namespace {
void validate(const PeerMatrix& p) {
    require(!p.empty() && p.size() <= UINT32_MAX, "collective needs 1..UINT32_MAX ranks");
    for (const auto& row : p) {
        require(row.size() == p.size(), "peer matrix must be square");
        for (const auto& l : row)
            require(std::isfinite(l.gbs) && l.gbs >= 0 && (!l.p2p || l.gbs > 0), "invalid P2P bandwidth");
    }
}
void validate_options(const CollectiveOptions& o, size_t n) {
    require(std::isfinite(o.min_p2p_gbs) && o.min_p2p_gbs >= 0, "invalid minimum P2P bandwidth");
    std::set<std::pair<uint32_t, uint32_t>> seen;
    for (const auto& edge : o.host_bounce)
        require(edge.first < n && edge.second < n && edge.first != edge.second && seen.insert(edge).second,
                "invalid/duplicate host-bounce edge");
}
Transport transport(const PeerMatrix& p, size_t s, size_t d, const CollectiveOptions& o) {
    if (s == d) return Transport::Local;
    if (std::find(o.host_bounce.begin(), o.host_bounce.end(), std::make_pair(static_cast<uint32_t>(s), static_cast<uint32_t>(d))) != o.host_bounce.end())
        return Transport::HostBounce;
    return p[s][d].p2p ? Transport::PeerPush : Transport::HostBounce;
}
bool island_pair(const PeerMatrix& p, size_t a, size_t b, const CollectiveOptions& o) {
    return transport(p, a, b, o) == Transport::PeerPush && transport(p, b, a, o) == Transport::PeerPush &&
           std::min(p[a][b].gbs, p[b][a].gbs) >= o.min_p2p_gbs;
}
const char* name(Transport t) {
    return t == Transport::Local ? "local" : (t == Transport::PeerPush ? "peer-push" : "host-bounce");
}
uint32_t rank_number(const json::Value& v) {
    const auto n = integer(v); require(n <= UINT32_MAX, "rank out of range"); return static_cast<uint32_t>(n);
}
uint64_t align64(uint64_t n) { return mul(add(n, 63) / 64, 64); }
} // namespace
PeerMatrix peer_matrix_from_json(const std::string& text) {
    const auto root = detail::parse(text);
    keys(root, {}, false);
    require(integer(field(root, "schema")) == 1, "unsupported peer profile schema");
    const auto& g = field(root, "gpus");
    require(g.is_array() && !g.arr.empty(), "profile requires gpus array");
    PeerMatrix peers(g.arr.size(), std::vector<PeerLink>(g.arr.size()));
    for (size_t s = 0; s < g.arr.size(); ++s) {
        keys(g.arr[s], {}, false);
        auto row = g.arr[s].get("p2p");
        if (!row) continue;
        require(row->is_array() && row->arr.size() == peers.size(), "P2P row size mismatch");
        for (size_t d = 0; d < peers.size(); ++d) {
            const auto& edge = row->arr[d];
            keys(edge, {"kind", "gbs", "latency_us"});
            const auto kind = string(field(edge, "kind"));
            require(kind == "unknown" || kind == "none" || kind == "pcie" || kind == "xgmi", "unknown link kind");
            peers[s][d].p2p = s != d && (kind == "pcie" || kind == "xgmi");
            peers[s][d].gbs = number(field(edge, "gbs"));
        }
    }
    validate(peers);
    return peers;
}
CollectiveOptions collective_options_from_json(const std::string& text) {
    const auto root = detail::parse(text);
    keys(root, {"schema", "min_p2p_gbs", "max_tp", "allow_host_tp", "stages", "host_bounce"});
    require(integer(field(root, "schema")) == 1, "unsupported collective options schema");
    CollectiveOptions o;
    if (auto p = root.get("min_p2p_gbs")) o.min_p2p_gbs = number(*p);
    if (auto p = root.get("max_tp")) o.max_tp = rank_number(*p);
    if (auto p = root.get("allow_host_tp")) o.allow_host_tp = boolean(*p);
    auto groups = [&](const json::Value& values) {
        require(values.is_array() && !values.arr.empty(), "rank groups must be nonempty array");
        std::vector<std::vector<uint32_t>> result;
        for (const auto& group : values.arr) {
            require(group.is_array() && !group.arr.empty(), "rank group must be nonempty array");
            std::vector<uint32_t> ranks;
            for (const auto& rank : group.arr) ranks.push_back(rank_number(rank));
            result.push_back(std::move(ranks));
        }
        return result;
    };
    if (auto p = root.get("stages")) o.stages = groups(*p);
    if (auto p = root.get("host_bounce")) {
        require(p->is_array(), "host_bounce must be array");
        if (!p->arr.empty()) for (const auto& edge : groups(*p)) {
            require(edge.size() == 2, "host-bounce edge needs source,destination");
            o.host_bounce.emplace_back(edge[0], edge[1]);
        }
    }
    return o;
}
TopologyPlan plan_topology(const PeerMatrix& peers, const CollectiveOptions& options) {
    validate(peers); validate_options(options, peers.size());
    TopologyPlan plan;
    std::vector<bool> used(peers.size(), false);
    // Deterministic greedy clique cover: seed by rank, append strongest compatible peer.
    for (uint32_t seed = 0; seed < peers.size(); ++seed) {
        if (used[seed]) continue;
        std::vector<uint32_t> island{seed}; used[seed] = true;
        while (true) {
            uint32_t best = static_cast<uint32_t>(peers.size()); double strongest = -1;
            for (uint32_t r = 0; r < peers.size(); ++r) if (!used[r]) {
                double weakest = std::numeric_limits<double>::infinity(); bool fits = true;
                for (uint32_t member : island) {
                    if (!island_pair(peers, r, member, options)) { fits = false; break; }
                    weakest = std::min(weakest, std::min(peers[r][member].gbs, peers[member][r].gbs));
                }
                if (fits && weakest > strongest) { strongest = weakest; best = r; }
            }
            if (best == peers.size()) break;
            used[best] = true; island.push_back(best);
        }
        std::sort(island.begin(), island.end());
        plan.islands.push_back(island);
        const size_t cap = options.max_tp ? options.max_tp : island.size();
        for (size_t begin = 0; begin < island.size(); begin += cap)
            plan.stages.emplace_back(island.begin() + begin, island.begin() + std::min(begin + cap, island.size()));
    }
    if (!options.stages.empty()) {
        std::vector<bool> covered(peers.size(), false);
        for (const auto& stage : options.stages) {
            require(!stage.empty() && (!options.max_tp || stage.size() <= options.max_tp), "invalid TP stage size");
            for (uint32_t r : stage) {
                require(r < peers.size() && !covered[r], "owner stages must cover ranks exactly once");
                covered[r] = true;
            }
            if (!options.allow_host_tp) for (uint32_t a : stage) for (uint32_t b : stage)
                require(a == b || island_pair(peers, a, b, options), "owner TP stage crosses non-P2P edges; set allow_host_tp explicitly");
        }
        require(std::all_of(covered.begin(), covered.end(), [](bool b) { return b; }), "owner stages omit ranks");
        plan.stages = options.stages;
    }
    for (size_t i = 1; i < plan.stages.size(); ++i)
        plan.stage_transports.push_back(transport(peers, plan.stages[i - 1][0], plan.stages[i][0], options));
    return plan;
}
CollectivePlan plan_collective(uint64_t elements, const PeerMatrix& peers, const CollectiveOptions& options) {
    validate(peers); validate_options(options, peers.size());
    require(elements > 0, "collective needs nonzero elements");
    CollectivePlan p;
    p.ranks = static_cast<uint32_t>(peers.size()); p.elements = elements; p.bypass = p.ranks == 1;
    if (p.bypass) return p; // no routes, frames, flags, RX buffers or FP16 rounding
    p.banks = 2; p.epoch_flag_stride = 64;
    p.payload_stride = align64(mul(elements, 2));
    p.rx_payload_bytes_per_rank = mul(mul(p.banks, p.ranks), p.payload_stride);
    p.rx_flag_bytes_per_rank = mul(mul(p.banks, p.ranks), p.epoch_flag_stride);
    p.routes.resize(p.ranks, std::vector<Transport>(p.ranks));
    for (uint32_t s = 0; s < p.ranks; ++s) for (uint32_t d = 0; d < p.ranks; ++d)
        p.routes[s][d] = transport(peers, s, d, options);
    return p;
}
std::string format_collective_plan(const CollectivePlan& p, const TopologyPlan& t) {
    std::ostringstream o;
    o << "Scope: whole-matrix benchmark collective; pipeline stages below are separate topology recommendations.\n";
    o << "TP=" << p.ranks << " collective=" << (p.bypass ? "bypass" : "direct-push-all-to-all")
      << " wire=" << (p.bypass ? "none" : "FP16") << " elements=" << p.elements << " ABI=" << p.abi << '\n';
    o << "per-rank isolated RX: payload=" << p.rx_payload_bytes_per_rank << " flags=" << p.rx_flag_bytes_per_rank
      << " banks=" << p.banks << " flag-stride=" << p.epoch_flag_stride << '\n';
    auto groups = [&](const char* label, const auto& gs) {
        o << label << ':';
        for (const auto& g : gs) { o << " ["; for (size_t i = 0; i < g.size(); ++i) o << (i ? "," : "") << g[i]; o << ']'; }
        o << '\n';
    };
    groups("P2P islands", t.islands); groups("pipeline TP stages", t.stages);
    for (size_t i = 0; i < t.stage_transports.size(); ++i)
        o << "stage " << i << " -> " << i + 1 << ": " << name(t.stage_transports[i]) << '\n';
    for (size_t s = 0; s < p.routes.size(); ++s) for (size_t d = 0; d < p.routes.size(); ++d)
        if (s != d) o << "rank " << s << " -> " << d << ": " << name(p.routes[s][d]) << '\n';
    o << "epochs: 1..UINT32_MAX, no wrap; fault => discard outputs, quiesce all queues, reset\n";
    return o.str();
}
CpuCollective::CpuCollective(const CollectivePlan& plan, uint32_t first_epoch) : plan_(plan) {
    require(plan.abi == 1 && plan.ranks && plan.elements && first_epoch, "invalid CPU collective plan/epoch");
    require(plan.bypass == (plan.ranks == 1), "invalid bypass plan");
    if (!plan.bypass) {
        const uint64_t frames = mul(plan.ranks, 2);
        require(plan.banks == 2 && plan.epoch_flag_stride == 64 &&
                plan.payload_stride == align64(mul(plan.elements, 2)) &&
                plan.rx_payload_bytes_per_rank == mul(frames, plan.payload_stride) &&
                plan.rx_flag_bytes_per_rank == mul(frames, 64) && plan.routes.size() == plan.ranks,
                "invalid collective ABI layout");
        for (size_t s = 0; s < plan.routes.size(); ++s) {
            require(plan.routes[s].size() == plan.ranks, "invalid route matrix");
            for (size_t d = 0; d < plan.routes[s].size(); ++d) {
                const auto route = plan.routes[s][d];
                require((s == d && route == Transport::Local) ||
                        (s != d && (route == Transport::PeerPush || route == Transport::HostBounce)),
                        "invalid collective route");
            }
        }
        require(frames <= SIZE_MAX && plan.elements <= SIZE_MAX, "CPU simulation size overflow");
        rx_.resize(plan.ranks);
        for (auto& inbox : rx_) {
            inbox.resize(static_cast<size_t>(frames));
            for (auto& frame : inbox) frame.wire.resize(static_cast<size_t>(plan.elements));
        }
    } else require(!plan.banks && !plan.epoch_flag_stride && !plan.payload_stride &&
                   !plan.rx_payload_bytes_per_rank && !plan.rx_flag_bytes_per_rank && plan.routes.empty(),
                   "bypass must allocate no P2P buffers");
    states_.resize(plan.ranks);
    for (auto& s : states_) s.published = s.consumed = first_epoch - 1;
}
void CpuCollective::healthy() const {
    if (failed_) throw std::runtime_error("collective fail-stop: " + error_);
}
void CpuCollective::check_rank(uint32_t rank) const {
    require(rank < plan_.ranks, "collective rank out of range");
}
[[noreturn]] void CpuCollective::fault(const std::string& message) {
    failed_ = true; error_ = message;
    for (auto& s : states_) s.output.clear();
    throw std::runtime_error("collective fail-stop: " + message);
}
void CpuCollective::begin(uint32_t source, uint32_t epoch, const std::vector<float>& partial) {
    healthy(); check_rank(source);
    auto& s = states_[source];
    if (!epoch || s.published == UINT32_MAX) fault("epoch exhausted; wraparound forbidden");
    if (s.pending_epoch || s.consumed != s.published || epoch != s.published + 1) fault("rank must consume e before publishing e+1");
    if (partial.size() != plan_.elements) fault("partial size mismatch");
    if (plan_.bypass) { s.output = partial; s.published = epoch; return; }
    s.pending.resize(partial.size());
    for (size_t i = 0; i < partial.size(); ++i) s.pending[i] = vj_tp3::to_half(partial[i]);
    s.sent.assign(plan_.ranks, false); s.pending_epoch = epoch;
}
void CpuCollective::push(uint32_t source, uint32_t destination) {
    healthy(); check_rank(source); check_rank(destination);
    auto& s = states_[source];
    if (plan_.bypass || !s.pending_epoch || s.sent[destination]) fault("duplicate or unordered push");
    const size_t frame_index = static_cast<size_t>(s.pending_epoch & 1u) * plan_.ranks + source;
    auto& frame = rx_[destination][frame_index];
    // Local polling only. Host-bounce and peer pushes have identical publication ordering:
    // payload completion precedes the destination-local epoch flag, never the reverse.
    if (plan_.routes[source][destination] == Transport::HostBounce) {
        // Model two completed copies via host staging, not an inaccessible peer read.
        const std::vector<uint16_t> host_staging = s.pending;
        frame.wire = host_staging;
    } else frame.wire = s.pending;
    frame.ready.epoch = s.pending_epoch;
    s.sent[destination] = true;
    if (std::all_of(s.sent.begin(), s.sent.end(), [](bool b) { return b; })) {
        s.published = s.pending_epoch; s.pending_epoch = 0;
    }
}
void CpuCollective::publish(uint32_t source, uint32_t epoch, const std::vector<float>& partial) {
    begin(source, epoch, partial);
    if (!plan_.bypass) for (uint32_t d = 0; d < plan_.ranks; ++d) push(source, d);
}
bool CpuCollective::try_consume(uint32_t rank, uint32_t epoch) {
    healthy(); check_rank(rank);
    auto& s = states_[rank];
    if (!epoch || s.consumed == UINT32_MAX || epoch != s.consumed + 1) fault("consume epoch out of order");
    if (s.published != epoch) return false; // finish every outgoing push first
    if (!plan_.bypass) {
        const size_t bank = static_cast<size_t>(epoch & 1u) * plan_.ranks;
        for (uint32_t source = 0; source < plan_.ranks; ++source) {
            const auto observed = rx_[rank][bank + source].ready.epoch;
            if (observed > epoch) fault("future epoch overwrote an unconsumed mailbox");
            if (observed != epoch) return false;
        }
        s.output.assign(static_cast<size_t>(plan_.elements), 0);
        for (uint32_t source = 0; source < plan_.ranks; ++source)
            for (size_t i = 0; i < s.output.size(); ++i)
                s.output[i] += vj_tp3::from_half(rx_[rank][bank + source].wire[i]);
    }
    s.consumed = epoch;
    return true;
}
std::vector<float> CpuCollective::result(uint32_t rank, uint32_t epoch) const {
    healthy(); check_rank(rank);
    require(epoch && states_[rank].consumed == epoch && states_[rank].published == epoch && !states_[rank].pending_epoch,
            "result epoch is not the rank's last retired epoch");
    for (const auto& s : states_) require(s.consumed >= epoch, "result is tentative until every rank retires");
    return states_[rank].output;
}
void CpuCollective::timeout(uint32_t rank) { healthy(); check_rank(rank); fault("rank timeout: " + std::to_string(rank)); }
uint64_t CpuCollective::mailbox_frames() const { return plan_.bypass ? 0 : mul(mul(plan_.ranks, plan_.ranks), 2); }
void CpuCollective::reset(bool quiescent) {
    if (!quiescent) fault("reset requires all queues quiescent");
    // Reset is an out-of-band acknowledgement after stopping the entire group.
    for (auto& inbox : rx_) for (auto& frame : inbox) { std::fill(frame.wire.begin(), frame.wire.end(), 0); frame.ready.epoch = 0; }
    for (auto& s : states_) s = RankState{};
    failed_ = false; error_.clear();
}
} // namespace vegajet::tp

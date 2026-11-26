// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
#include <cstdint>
#if !defined(__HIPCC__)
#include "core/tp_collective.hpp"
#include <stdexcept>
#include <vector>
#endif
namespace vegajet::tp {
// Per-source descriptor table: outgoing destinations followed via push_*;
// incoming sources via read_*. read_* are owner-local VRAM or host bounce.
struct CollectiveRoute {
    uint64_t push_payload = 0, push_flags = 0, read_payload = 0, read_flags = 0;
};
static_assert(sizeof(CollectiveRoute) == 32, "collective route ABI");
struct CollectiveArgs {
    uint64_t partial, output, routes, status;
    uint64_t elements, payload_stride;
    uint32_t ranks, rank, epoch, max_poll_iters;
};
static_assert(sizeof(CollectiveArgs) == 64, "collective kernel ABI");
#if !defined(__HIPCC__)
struct MailboxAddresses { uint64_t payload, flags, host_payload, host_flags; };
inline std::vector<CollectiveRoute> collective_route_table(const CollectivePlan& plan, const std::vector<MailboxAddresses>& boxes) {
    const size_t n = plan.ranks;
    if (plan.bypass) {
        if (n != 1 || !boxes.empty()) throw std::invalid_argument("bypass has no mailboxes");
        return {};
    }
    if (!n || boxes.size() != n || plan.routes.size() != n) throw std::invalid_argument("collective route shape");
    for (const auto& row : plan.routes) if (row.size() != n) throw std::invalid_argument("collective route row");
    std::vector<CollectiveRoute> result(n * n);
    for (size_t r = 0; r < n; ++r) for (size_t peer = 0; peer < n; ++peer) {
        const auto& out = boxes[peer]; const auto& in = boxes[r];
        const bool ph = plan.routes[r][peer] == Transport::HostBounce;
        const bool rh = plan.routes[peer][r] == Transport::HostBounce;
        auto& route = result[r * n + peer];
        route = {ph ? out.host_payload : out.payload, ph ? out.host_flags : out.flags,
                 rh ? in.host_payload : in.payload, rh ? in.host_flags : in.flags};
        if (!route.push_payload || !route.push_flags || !route.read_payload || !route.read_flags ||
            (route.push_payload & 63) || (route.push_flags & 63) || (route.read_payload & 63) || (route.read_flags & 63))
            throw std::invalid_argument("collective missing/misaligned mailbox");
    }
    return result;
}
#endif
}

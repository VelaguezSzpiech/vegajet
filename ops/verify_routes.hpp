// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
#include <stdint.h>
#ifdef __HIP__
#define VJ_VERIFY_HD __host__ __device__
#else
#define VJ_VERIFY_HD
#endif
namespace vegajet { namespace verify {
// Leader for each consecutive pair of occurrences of an expert. An unpaired
// final occurrence points to itself. All routes have exactly one writer.
VJ_VERIFY_HD inline uint32_t partner(const int32_t* ids, uint32_t n, uint32_t route) {
    uint32_t ordinal = 0u;
    for (uint32_t r = 0; r < route; ++r) ordinal += ids[r] == ids[route];
    if (ordinal & 1u) return n;
    for (uint32_t r = route + 1u; r < n; ++r) if (ids[r] == ids[route]) return r;
    return route;
}
}}
#undef VJ_VERIFY_HD

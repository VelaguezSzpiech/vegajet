// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
#include <cstdint>
#include <cstring>
#if defined(__HIPCC__)
#define VJ_AM_HD __host__ __device__
#else
#define VJ_AM_HD
#endif
namespace vegajet::acceptmoe {
// Warm-step policy: plain is the historical dispatch AND routing policy. Commit
// only changes standalone verify batches; ring-super batches retain exact routing.
inline bool commit_mode(const char* mode) {
    return mode && std::strcmp(mode, "commit") == 0;
}
inline bool plain_equivalent(float acceptance, float cold_penalty, uint32_t window) {
    return acceptance == 1.0f && (cold_penalty == 1.0f || window == 0u);
}
// Commitment is the probability of reaching this draft position. At zero
// commitment routing is unbiased; at one it uses the full union preference.
VJ_AM_HD inline float multiplier(bool in_union, bool cold, float commitment,
                                 float penalty, float cold_penalty) {
    return (in_union ? 1.0f : commitment == 1.0f ? penalty : 1.0f - commitment * (1.0f - penalty)) *
           (cold ? cold_penalty : 1.0f);
}
VJ_AM_HD inline bool cold(uint32_t last, uint32_t tick, uint32_t window) {
    return window != 0u && (last == 0u || tick - last > window);
}
}
#undef VJ_AM_HD

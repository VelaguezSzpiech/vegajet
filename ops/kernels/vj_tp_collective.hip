// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "core/tp_collective_abi.hpp"
namespace {
__device__ __forceinline__ void push(uint32_t* p, uint32_t v) {
    asm volatile("flat_store_dword %0, %1 glc slc" :: "v"(p), "v"(v) : "memory");
}
__device__ __forceinline__ uint32_t read(const uint32_t* p) {
    uint32_t v;
    asm volatile("flat_load_dword %0, %1 glc slc\n\ts_waitcnt vmcnt(0)" : "=v"(v) : "v"(p) : "memory");
    return v;
}
__device__ __forceinline__ void drain() {
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __threadfence_system();
}
}
// One resident wave/rank, canonical rank-order FP32 sum, FP16 RNE on wire.
// All peer operations are writes. Each rank polls exclusively its own inbox.
extern "C" __global__ __launch_bounds__(64)
void vj_tp_allreduce_fp16(vegajet::tp::CollectiveArgs a) {
    const unsigned lane = threadIdx.x;
    auto* status = reinterpret_cast<uint32_t*>(a.status);
    const auto* routes = reinterpret_cast<const vegajet::tp::CollectiveRoute*>(a.routes);
    const auto* input = reinterpret_cast<const float*>(a.partial);
    auto* output = reinterpret_cast<float*>(a.output);
    if (!a.epoch || a.rank >= a.ranks || a.ranks < 2 || !a.elements || blockDim.x != 64 || gridDim.x != 1) {
        if (!lane) push(status, 1);
        return;
    }
    const uint64_t frame = uint64_t(a.epoch & 1u) * a.ranks + a.rank;
    unsigned bad = 0;
    #pragma nounroll
    for (uint64_t pair = lane; pair < (a.elements + 1) / 2; pair += 64) {
        float x = input[pair * 2], y = pair * 2 + 1 < a.elements ? input[pair * 2 + 1] : 0.0f;
        bad |= !(fabsf(x) <= 65504.0f && fabsf(y) <= 65504.0f);
        const uint32_t packed = uint32_t(__half_as_ushort(__float2half_rn(x))) |
                                (uint32_t(__half_as_ushort(__float2half_rn(y))) << 16);
        #pragma nounroll
        for (unsigned d = 0; d < a.ranks; ++d)
            push(reinterpret_cast<uint32_t*>(routes[d].push_payload + frame * a.payload_stride) + pair, packed);
    }
    if (__ballot(bad)) {
        if (!lane) push(status, 2);
        return; // never publish non-finite/overflowed payloads
    }
    // RT-B1: EVERY writer lane drains posted payload before the publishing lane
    // can issue any epoch. A lane-0-only fence is not sufficient.
    drain();
    __syncthreads();
    unsigned ready = 1;
    if (!lane) {
        #pragma nounroll
        for (unsigned d = 0; d < a.ranks; ++d)
            push(reinterpret_cast<uint32_t*>(routes[d].push_flags + frame * 64), a.epoch);
        drain();
        #pragma nounroll
        for (unsigned s = 0; s < a.ranks; ++s) {
            const uint64_t source_frame = uint64_t(a.epoch & 1u) * a.ranks + s;
            auto* flag = reinterpret_cast<const uint32_t*>(routes[s].read_flags + source_frame * 64);
            bool found = false;
            #pragma nounroll
            for (unsigned poll = 0; poll < a.max_poll_iters; ++poll) {
                const unsigned seen = read(flag);
                if (seen == a.epoch) { found = true; break; }
                if (seen > a.epoch) break;
                asm volatile("s_sleep 1" ::: "memory");
            }
            if (!found) { ready = 0; push(status, 4); break; }
        }
    }
    ready = __shfl(ready, 0);
    __syncthreads();
    if (!ready) return;
    #pragma nounroll
    for (uint64_t i = lane; i < a.elements; i += 64) {
        float sum = 0;
        #pragma nounroll
        for (unsigned s = 0; s < a.ranks; ++s) {
            const uint64_t source_frame = uint64_t(a.epoch & 1u) * a.ranks + s;
            const uint32_t word = read(reinterpret_cast<const uint32_t*>(routes[s].read_payload + source_frame * a.payload_stride) + i / 2);
            const float value = __half2float(__ushort_as_half(static_cast<unsigned short>(word >> ((i & 1) * 16))));
            sum = s == 0 ? value : __fadd_rn(sum, value);
        }
        output[i] = sum;
    }
}

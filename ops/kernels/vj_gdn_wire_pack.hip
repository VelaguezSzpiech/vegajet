// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Independent production-input packer for the sealed GDN tree-WY verifier.
//
// This is a value-preserving strided copy, not arithmetic:
//   conv[t][0 .. nk*128)          -> query[t]
//   conv[t][nk*128 .. 2*nk*128)   -> key[t]
//   conv[t][2*nk*128 .. +nv*128)  -> value[t]
//   decay_source[t][nv]           -> decay[t]
//   beta_source[t][nv]            -> beta[t]
//   parent_source[0 .. m)         -> parent[0 .. m)
//
// One workgroup per token; every destination element the sealed verifier reads is rewritten, so
// no stale row from an earlier cycle or an earlier (wider) width survives. The kernel is enqueued
// with an AQL barrier on the engine's device stream immediately after that layer's conv kernel and
// before the sealed verify, so the ordering is stream order and needs no host wait.
//
// Grid: m workgroups x 256 threads (host passes grid_x = m*256 work-items).
#include "ops/gdn_wire_pack.hpp"

#include <hip/hip_runtime.h>
#include <stdint.h>

using vegajet::gdn::wire_pack::Args;
static_assert(sizeof(Args) == 96, "vj_gdn_wire_pack_kernel kernarg layout changed");
static_assert(offsetof(Args, m) == 80 && offsetof(Args, conv_stride) == 92,
              "vj_gdn_wire_pack_kernel kernarg offsets changed");

extern "C" __global__ __launch_bounds__(256, 4)
void vj_gdn_wire_pack_kernel(
    const float* __restrict__ conv_source,
    const float* __restrict__ decay_source,
    const float* __restrict__ beta_source,
    float* __restrict__ query,
    float* __restrict__ key,
    float* __restrict__ value,
    float* __restrict__ decay,
    float* __restrict__ beta,
    const int* __restrict__ parent_source,
    int* __restrict__ parent,
    uint32_t m,
    uint32_t nk,
    uint32_t nv,
    uint32_t conv_stride) {
    using namespace vegajet::gdn::wire_pack;
    const uint32_t token = blockIdx.x;
    if (token >= m || nk != kQueryHeads || nv != kValueHeads || conv_stride < kConvStride ||
        blockDim.x != kThreads) {
        return;
    }
    const uint32_t tid = threadIdx.x;
    const uint32_t qk = nk * kHeadDim;
    const uint32_t vdim = nv * kHeadDim;
    const float* src = conv_source + static_cast<uint64_t>(token) * conv_stride;
    float* q = query + static_cast<uint64_t>(token) * qk;
    float* k = key + static_cast<uint64_t>(token) * qk;
    float* v = value + static_cast<uint64_t>(token) * vdim;
    for (uint32_t i = tid; i < qk; i += kThreads) {
        q[i] = src[i];
        k[i] = src[qk + i];
    }
    for (uint32_t i = tid; i < vdim; i += kThreads) v[i] = src[2u * qk + i];
    if (tid < nv) {
        const uint64_t g = static_cast<uint64_t>(token) * nv + tid;
        decay[g] = decay_source[g];
        beta[g] = beta_source[g];
    }
    if (blockIdx.x == 0u && tid < m) parent[tid] = parent_source[tid];
}

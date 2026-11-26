// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
#include "core/kfd_runtime.hpp"
#include "ops/kvq_launch.hpp"
#include "ops/flash_decode_splitkv_launch.hpp"
#include <algorithm>
#include <stdexcept>
#include <vector>
namespace vegajet::ornith::l4 {
// Intersect global GQA groups with the rank's contiguous Q shard. Empty groups are legal.
inline std::vector<kvq::HeadRange> ranges(uint32_t nq, uint32_t nkv, uint32_t qo,
                                         uint32_t ko, uint32_t group) {
    if (!group) throw std::invalid_argument("zero GQA group");
    std::vector<kvq::HeadRange> r;
    for (uint32_t k=0;k<nkv;++k) {
        const uint64_t lo=uint64_t(ko+k)*group, hi=lo+group;
        auto local=[&](uint64_t x) { return uint32_t(std::min<uint64_t>(nq,x>qo?x-qo:0)); };
        r.push_back({local(lo),local(hi)});
    }
    if (!kvq::valid_shape(256,nq,nkv,1,1,kvq::DeviceFormat::Q8,r.data()))
        throw std::invalid_argument("unsupported rank GQA ranges (maximum 16 Q heads per KV head)");
    return r;
}
// Per-head dispatches use M=1, including verify/prefill/batch rows. Four waves/WG.
constexpr uint32_t splits(uint32_t tokens, uint32_t kv_heads, bool chronological) {
    return chronological || tokens<1920 ? 1 : kvq::split_count(kv_heads);
}
constexpr uint64_t cache_bytes(uint32_t capacity, uint32_t heads, kvq::DeviceFormat f) {
    return uint64_t(capacity)*heads*kvq::vector_bytes(f,256);
}
constexpr uint64_t wire_offset(uint32_t capacity, uint32_t head, uint32_t sequence,
                               uint32_t sequence_capacity) {
    return (uint64_t(head)*capacity+uint64_t(sequence)*sequence_capacity)*384;
}
constexpr uint64_t native_offset(uint32_t sequence, uint32_t sequence_capacity,
                                 uint32_t heads, kvq::DeviceFormat f) {
    return uint64_t(sequence)*sequence_capacity*heads*kvq::vector_bytes(f,256);
}
struct PrepArgs {
    uint64_t qkv,qw,kw,q,gate,k,v;
    uint32_t rows,pos,nq,nkv,rope;
    float base,eps;
};
struct WireKvArgs { uint64_t k,v,kw,vw; uint32_t heads,head,token; };
struct WireQArgs { uint64_t q,wire; uint32_t begin,end; };
struct EpilogueArgs { uint64_t attn,gate,out,ds; uint32_t begin,end; };
static_assert(sizeof(PrepArgs)==88 && offsetof(PrepArgs,base)==76);
static_assert(sizeof(WireKvArgs)==48 && offsetof(WireKvArgs,token)==40);
static_assert(sizeof(WireQArgs)==24 && sizeof(EpilogueArgs)==40);
// Only opt-in buffers are added to the normal executor state.
struct State {
    uint64_t prepared_k=0,prepared_v=0,ranges=0,signs=0,partial=0,lse=0,out=0;
    uint64_t wire_q=0,wire_scratch=0;
    std::vector<uint64_t> wire_k,wire_v;
};
struct Kernels {
    const kfd::LoadedKernel *prep=nullptr,*append=nullptr,*partial=nullptr,*merge=nullptr;
    const kfd::LoadedKernel *wire_kv=nullptr,*wire_q=nullptr,*partition=nullptr,*wire_merge=nullptr,*epilogue=nullptr;
};
} // namespace vegajet::ornith::l4

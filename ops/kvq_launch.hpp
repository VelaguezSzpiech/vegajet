// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
#include <cstdint>
#include <cstddef>
#include <string_view>
namespace vegajet::kvq {
inline constexpr const char* kEnvVar = "VEGAJET_KV_QUANT";
enum class DeviceFormat : uint32_t { Fp16=0, Q8=1, Q4=2, Turbo3=3 };
constexpr uint32_t vector_bytes(DeviceFormat f, uint32_t d) {
    return f==DeviceFormat::Fp16 ? 2*d : (f==DeviceFormat::Q8 ? d+2*(d/32) :
        (f==DeviceFormat::Q4 ? 18*(d/32) : 14*(d/32)));
}
// Local Q-head indices belonging to this KV head. Supports TP straddling.
struct HeadRange { uint32_t begin, end; };
struct AppendArgs {
    uint64_t input, cache, signs;
    uint32_t dim, kv_heads, token, format;
};
struct PartialArgs {
    uint64_t q, k, v, ranges, signs, partial, lse;
    uint32_t dim, q_heads, kv_heads, tokens, splits, format;
    float attention_scale;
};
struct MergeArgs {
    uint64_t partial, lse, output;
    uint32_t q_heads, splits, dim;
};
// 256 threads = four waves per tile; at least 60 tiles => 240 launched waves.
constexpr uint32_t split_count(uint32_t kv_heads, bool chronological=false) {
    return kv_heads==0 ? 0 : chronological ? 1 : 1+59/kv_heads;
}
// No getenv here: the main session owns process policy and cache lifetime.
inline bool parse_format(std::string_view value, DeviceFormat& out) {
    if(value=="fp16") out=DeviceFormat::Fp16;
    else if(value=="q8_0") out=DeviceFormat::Q8;
    else if(value=="q4_0") out=DeviceFormat::Q4;
    else if(value=="turbo3") out=DeviceFormat::Turbo3;
    else return false;
    return true;
}
// Ranges must partition [0,q_heads) in KV-head order; empty KV ranges are allowed.
// Byte offsets in the kernel are uint32: enforce this limit before allocation/dispatch.
inline bool valid_shape(uint32_t dim, uint32_t q_heads, uint32_t kv_heads,
                        uint32_t tokens, uint32_t splits, DeviceFormat f,
                        const HeadRange* ranges) {
    if((dim!=128 && dim!=256) || !q_heads || !kv_heads || !splits || splits>UINT32_MAX/256 || !ranges ||
       uint32_t(f)>3 || uint64_t(tokens)*kv_heads>UINT32_MAX/vector_bytes(f,dim) ||
       uint64_t(q_heads)*splits>SIZE_MAX/(dim*4)) return false;
    uint32_t next=0;
    for(uint32_t k=0;k<kv_heads;++k) {
        if(ranges[k].begin!=next || ranges[k].end<next || ranges[k].end-next>16 || ranges[k].end>q_heads) return false;
        next=ranges[k].end;
    }
    return next==q_heads;
}
constexpr uint32_t kAppendKernargBytes=40, kPartialKernargBytes=84, kMergeKernargBytes=36;
static_assert(offsetof(PartialArgs,attention_scale)==80 && offsetof(MergeArgs,dim)==32);
static_assert(sizeof(AppendArgs)==40 && sizeof(PartialArgs)==88 && sizeof(MergeArgs)==40);
}

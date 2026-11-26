// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "ops/sub4_launch.hpp"
namespace {
__device__ __forceinline__ uint64_t uniform_u64(uint64_t v) {
    uint32_t lo,hi;
    asm("v_readfirstlane_b32 %0, %2\nv_readfirstlane_b32 %1, %3"
        : "=s"(lo), "=s"(hi) : "v"(uint32_t(v)), "v"(uint32_t(v>>32)));
    return uint64_t(lo)|(uint64_t(hi)<<32);
}
__device__ __forceinline__ uint32_t compact(uint32_t a,uint32_t b) {
    a &= 0x0f0f0f0f; b &= 0x0f0f0f0f;
    a=(a|(a>>4))&0x00ff00ff; b=(b|(b>>4))&0x00ff00ff;
    return ((a|(a>>8))&65535) | ((b|(b>>8))<<16);
}
__device__ __forceinline__ uint32_t merge(uint32_t m,uint32_t low) {
    uint32_t r;
    asm("v_and_or_b32 %0, %1, %3, %2" : "=v"(r) : "v"(m), "v"(low), "s"(0x44444444u));
    return r;
}
template<int Bits> __device__ __forceinline__ int dot(uint4 w,uint4 a) {
    uint32_t p=w.x&0x33333333, q=(w.x>>2)&0x33333333;
    uint32_t r=w.y&0x33333333, s=(w.y>>2)&0x33333333;
    if constexpr(Bits==3) {
        p=merge(w.z<<2,p); q=merge(w.z<<1,q);
        r=merge(w.z,r); s=merge(w.z>>1,s);
    }
    int v=__builtin_amdgcn_sdot8(p,compact(a.x,a.y),0,false);
    v=__builtin_amdgcn_sdot8(q,compact(a.x>>4,a.y>>4),v,false);
    v=__builtin_amdgcn_sdot8(r,compact(a.z,a.w),v,false);
    return __builtin_amdgcn_sdot8(s,compact(a.z>>4,a.w>>4),v,false);
}
template<int Bits,bool Xor=false> __device__ __forceinline__ void run(vegajet::sub4::Kernarg k) {
    uint32_t row=blockIdx.x, m=blockIdx.y, lane=threadIdx.x;
    if(row>=k.rows || m>=k.M || k.M>8) return;
    const auto* w=reinterpret_cast<const uint32_t*>(uniform_u64(k.weights+uint64_t(row)*(k.cols/32)*Bits*4));
    const auto* dm=reinterpret_cast<const uint32_t*>(uniform_u64(k.dm+uint64_t(row)*(Xor ? k.cols/32 : k.cols/k.group)*4));
    const auto* x=reinterpret_cast<const uint4*>(uniform_u64(k.activations+uint64_t(m)*(k.cols/2)));
    const auto* ds=reinterpret_cast<const float2*>(uniform_u64(k.activation_ds+uint64_t(m)*(k.cols/32)*8));
    auto resource=[](const void* p) {
        return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(p),static_cast<short>(0),0x7fffffff,0x00027000);
    };
    auto xr=resource(x), dr=resource(ds), mr=resource(dm);
    // Uniform row base lives in SGPRs; only the byte offset varies by lane.
    using rsrc_t = uint32_t __attribute__((ext_vector_type(4)));
    uint64_t wb=reinterpret_cast<uint64_t>(w);
    rsrc_t wr={uint32_t(wb),uint32_t(wb>>32),0x7fffffff,0x00027000};
    uint32_t nb=k.cols/32; float acc=0;
    for(uint32_t base=0;base<nb;base+=256) {
        uint32_t b=base+lane*4;
        auto index=[&](uint32_t raw) { return min(raw,nb-1); };
        auto phys=[&](uint32_t logical) { return Xor ? logical^(row&15) : logical; };
        uint32_t b0=phys(index(b)), b1=phys(index(b+1)), b2=phys(index(b+2)), b3=phys(index(b+3));
        if(b<nb) { // Mask idle lanes instead of rereading the clamped tail four times.
        // Four independent 16-byte vector requests before first consumption (ILP=4).
        // Byte offsets support Q3 block alignment; four guard words protect the final vector request.
        using v4 = uint32_t __attribute__((ext_vector_type(4)));
        v4 v0,v1,v2,v3;
        asm volatile(
            "buffer_load_dwordx4 %0, %4, %8, 0 offen\n"
            "buffer_load_dwordx4 %1, %5, %8, 0 offen\n"
            "buffer_load_dwordx4 %2, %6, %8, 0 offen\n"
            "buffer_load_dwordx4 %3, %7, %8, 0 offen\n"
            "s_waitcnt vmcnt(0)"
            : "=&v"(v0), "=&v"(v1), "=&v"(v2), "=&v"(v3)
            : "v"(b0*Bits*4), "v"(b1*Bits*4), "v"(b2*Bits*4), "v"(b3*Bits*4), "s"(wr)
            : "memory");
        uint4 ws[4]={make_uint4(v0[0],v0[1],v0[2],v0[3]),make_uint4(v1[0],v1[1],v1[2],v1[3]),
                     make_uint4(v2[0],v2[1],v2[2],v2[3]),make_uint4(v3[0],v3[1],v3[2],v3[3])};
        #pragma unroll
        for(uint32_t j=0;j<4;++j) {
            uint32_t bj=index(b+j);
            uint4 xa=__builtin_bit_cast(uint4,__builtin_amdgcn_raw_buffer_load_b128(xr,bj*16,0,0));
            int v=dot<Bits>(ws[j],xa);
            uint32_t meta=__builtin_amdgcn_raw_buffer_load_b32(mr,(Xor ? phys(bj) : bj*32/k.group)*4,0,0);
            float2 a=__builtin_bit_cast(float2,__builtin_amdgcn_raw_buffer_load_b64(dr,bj*8,0,0));
            float scale=__half2float(__ushort_as_half(meta&65535));
            float zero=__half2float(__ushort_as_half(meta>>16));
            if(b+j<nb) acc+=scale*(a.x*v-zero*a.y);
        }
        }
    }
    for(int d=32;d;d>>=1) acc+=__shfl_xor(acc,d,64);
    if(lane==0) reinterpret_cast<float*>(k.output)[uint64_t(m)*k.rows+row]=acc;
}
}
#define ENTRY(name,bits) extern "C" __global__ __launch_bounds__(64) void name(vegajet::sub4::Kernarg k) { run<bits>(k); }
ENTRY(vj_sub4_q2_gemv,2)
ENTRY(vj_sub4_q3_gemv,3)
ENTRY(vj_sub4_q2_gemm,2)
ENTRY(vj_sub4_q3_gemm,3)

#define XOR_ENTRY(name,bits) extern "C" __global__ __launch_bounds__(64) void name(vegajet::sub4::Kernarg k) { run<bits,true>(k); }
XOR_ENTRY(vj_sub4_q2_gemv_xor,2)
XOR_ENTRY(vj_sub4_q3_gemv_xor,3)
XOR_ENTRY(vj_sub4_q2_gemm_xor,2)
XOR_ENTRY(vj_sub4_q3_gemm_xor,3)

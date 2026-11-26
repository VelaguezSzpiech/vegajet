// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include <hip/hip_runtime.h>
#include "ops/milo_launch.hpp"
namespace {
__device__ __forceinline__ float factor(uint64_t base,uint64_t scale,uint32_t i,uint32_t cols) {
    const auto* q=reinterpret_cast<const uint32_t*>(base);
    uint32_t bit=i*3, shift=bit&31;
    uint64_t pair=uint64_t(q[bit/32])|(uint64_t(q[bit/32+1])<<32);
    int v=int(((pair>>shift)&7)^4)-4;
    return v*reinterpret_cast<const float*>(scale)[i/cols];
}
}
extern "C" __global__ __launch_bounds__(64)
void vj_milo_vx(vegajet::milo::Kernarg k) {
    uint32_t r=blockIdx.x,m=blockIdx.y,lane=threadIdx.x;
    if(r>=k.rank || m>=k.M || k.M>8) return;
    const auto* x=reinterpret_cast<const int8_t*>(k.x)+m*k.cols;
    const auto* ds=reinterpret_cast<const float2*>(k.xds)+m*(k.cols/32);
    float a=0;
    for(uint32_t c=lane;c<k.cols;c+=64) a+=factor(k.v,k.vs,r*k.cols+c,k.cols)*(x[c]*ds[c/32].x);
    for(int d=32;d;d>>=1) a+=__shfl_xor(a,d,64);
    if(lane==0) reinterpret_cast<float*>(k.tmp)[m*k.rank+r]=a;
}
extern "C" __global__ __launch_bounds__(64)
void vj_milo_apply(vegajet::milo::Kernarg k) {
    uint32_t row=blockIdx.x*64+threadIdx.x,m=blockIdx.y;
    if(row>=k.rows || m>=k.M || k.M>8) return;
    float a=reinterpret_cast<float*>(k.y)[m*k.rows+row];
    for(uint32_t r=0;r<k.rank;++r) a+=factor(k.u,k.us,row*k.rank+r,k.rank)*reinterpret_cast<const float*>(k.tmp)[m*k.rank+r];
    reinterpret_cast<float*>(k.y)[m*k.rows+row]=a;
}

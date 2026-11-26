// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/offload/cuda_wire.hpp"
#include <cuda/atomic>
#include <cuda_fp16.h>
using namespace vegajet::offload;
__device__ unsigned load(unsigned& p) { return cuda::atomic_ref<unsigned,cuda::thread_scope_system>(p).load(cuda::memory_order_acquire); }
__device__ void store(unsigned& p,unsigned v) { cuda::atomic_ref<unsigned,cuda::thread_scope_system>(p).store(v,cuda::memory_order_release); }
extern "C" __global__ void vj_offload_worker(CudaRing* ring,const unsigned char* weights,
    const unsigned char* scales,unsigned rows,unsigned cols,unsigned long long ws,unsigned long long ss,
    unsigned format,unsigned long long ticks) {
    __shared__ int work;
    const unsigned long long begin=clock64();
    unsigned slot=0;
    if(threadIdx.x==0) store(ring->alive,1);
    __syncthreads();
    while(true) {
        if(threadIdx.x==0) {
            work=(load(ring->stop) || (ticks && clock64()-begin>=ticks))?-1:(load(ring->commands[slot].state)==1?1:0);
        }
        __syncthreads();
        if(work<0) break;
        if(work) {
            auto& c=ring->commands[slot];
            for(unsigned r=threadIdx.x;r<rows;r+=blockDim.x) {
                float sum=0;
                for(unsigned b=0;b<cols/32;++b) {
                    auto p=weights+r*ws+b*(format?16:18);
                    float d;
                    if(format) d=*reinterpret_cast<const float*>(scales+r*ss+b*4);
                    else { d=__half2float(*reinterpret_cast<const __half*>(p)); p+=2; }
                    int dot=0;
                    for(unsigned j=0;j<32;++j) {
                        unsigned code=(p[j%16]>>(j/16*4))&15;
                        int v;
                        if(format) { const int mag[8]={4,12,20,28,36,44,52,62}; v=mag[code&7]*(code&8?-1:1); }
                        else v=int(code)-8;
                        dot+=v*int(c.x[b*32+j]);
                    }
                    sum+=(d*c.scales[b])*float(dot);
                }
                c.y[r]=sum;
            }
            __syncthreads();
            if(threadIdx.x==0) { __threadfence_system(); store(c.state,2); }
            slot=(slot+1)%cuda_slots;
        } else __nanosleep(1000);
        __syncthreads();
    }
    if(threadIdx.x==0) store(ring->alive,0);
}

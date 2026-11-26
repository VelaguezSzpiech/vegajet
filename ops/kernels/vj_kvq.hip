// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
// ABI: ops/kvq_launch.hpp. D=128 or 256, block=256. No runtime API calls.
#include "ops/kernels/vj_kvq_device.hiph"
using namespace vegajet::kvq_dev;
extern "C" __global__ __launch_bounds__(256)
void vj_kvq_append_kernel(const float* input,uint8_t* cache,const int8_t* signs,
    unsigned dim,unsigned kv_heads,unsigned token,unsigned format) {
    unsigned tid=threadIdx.x, h=blockIdx.x;
    __shared__ float x[256];
    if(tid<dim) x[tid]=input[h*dim+tid]*(format==3?signs[tid]:1);
    __syncthreads();
    if(format==3) transform(x,dim,tid);
    uint8_t* p=cache+(uint64_t(token)*kv_heads+h)*stride(format,dim);
    append_vector(x,p,dim,format,tid);
}
// ABI and grid unchanged: (splits, KV heads), 256 threads. Hoist each query's
// quantization/rotation out of the token loop. Accumulate rotated Turbo3 V and
// inverse-rotate only the finished output (linearity), rather than every token.
extern "C" __global__ __launch_bounds__(256)
void vj_kvq_grouped_partial_kernel(const float* q,const uint8_t* k,const uint8_t* v,
    const uint2* ranges,const int8_t* signs,float* partial,float* lse,
    unsigned dim,unsigned q_heads,unsigned kv_heads,unsigned tokens,unsigned splits,unsigned format,float attention_scale) {
    unsigned tid=threadIdx.x, kh=blockIdx.y, sp=blockIdx.x;
    unsigned begin=__builtin_amdgcn_readfirstlane(ranges[kh].x), end=__builtin_amdgcn_readfirstlane(ranges[kh].y);
    __shared__ float qq[256],red[4],qs[8];
    __shared__ uint32_t qc[32];
    unsigned t0=uint64_t(tokens)*sp/splits,t1=uint64_t(tokens)*(sp+1)/splits;
    for(unsigned h=begin;h<end;++h) {
        if(tid<dim) qq[tid]=q[uint64_t(h)*dim+tid]*(format==3?signs[tid]:1);
        __syncthreads();
        if(format==3) transform(qq,dim,tid);
        if(format==2) {
            if(tid<dim/32) { float am=0;for(unsigned j=0;j<32;++j) am=fmaxf(am,fabsf(qq[tid*32+j]));qs[tid]=am/7; }
            __syncthreads();
            if(tid<dim/8) { unsigned word=0;float s=qs[tid/4];for(unsigned j=0;j<8;++j) word|=(unsigned(max(-7,min(7,int(roundf(s>0?qq[tid*8+j]/s:0)))))&15)<<(4*j);qc[tid]=word; }
            __syncthreads();
        }
        const float qv=tid<dim?qq[tid]:0;
        // Keep int4 operands in registers: eight lanes own one 32-value block each.
        const unsigned b=tid%(dim/32);
        const uint32_t q0=format==2?qc[4*b]:0, q1=format==2?qc[4*b+1]:0;
        const uint32_t q2=format==2?qc[4*b+2]:0, q3=format==2?qc[4*b+3]:0;
        const float qscale=format==2?qs[b]:0;
        float m=-1e30f,l=0,o=0;
        for(unsigned t=t0;t<t1;++t) {
            const uint64_t off=(uint64_t(t)*kv_heads+kh)*stride(format,dim);
            const uint8_t* kp=k+off; const uint8_t* vp=v+off;
            float dot=0;
            if(format==2) {
                if(tid<dim/32) {
                    const uint32_t* codes=(const uint32_t*)(kp+2*(dim/32))+4*tid;
                    int acc=__builtin_amdgcn_sdot8(q0,codes[0],0,false);
                    acc=__builtin_amdgcn_sdot8(q1,codes[1],acc,false);
                    acc=__builtin_amdgcn_sdot8(q2,codes[2],acc,false);
                    acc=__builtin_amdgcn_sdot8(q3,codes[3],acc,false);
                    dot=float(acc)*qscale*halfval((const uint16_t*)kp+tid);
                }
            } else if(tid<dim) dot=qv*elem(kp,format,dim,tid);
            const float vv=tid<dim?elem(vp,format,dim,tid):0;
            dot=sum64(dot); if((tid&63)==0) red[tid/64]=dot;
            __syncthreads();
            const float score=((red[0]+red[1])+red[2]+red[3])*attention_scale;
            const float nm=fmaxf(m,score),a=__expf(m-nm),beta=__expf(score-nm);
            o=fmaf(beta,vv,o*a); l=l*a+beta; m=nm;
            // red must be consumed by every wave before the next token overwrites it.
            __syncthreads();
        }
        if(format==3) {
            if(tid<dim) qq[tid]=l>0?o/l:0;
            __syncthreads();
            transform(qq,dim,tid);
            if(tid<dim) o=qq[tid]*signs[tid];
        } else o=l>0?o/l:0;
        const uint64_t row=uint64_t(h)*splits+sp;
        if(tid<dim) partial[row*dim+tid]=o;
        if(tid==0) lse[row]=l>0?m+__logf(l):-INFINITY;
        __syncthreads();
    }
}

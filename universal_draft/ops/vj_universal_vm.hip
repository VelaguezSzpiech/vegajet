// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>
#include "vj_vm_desc.hpp"

namespace {
using namespace vegajet::vm;
constexpr uint32_t kThreads = 256;


__device__ __forceinline__ float wave_sum(float x) {
    x += __shfl_xor(x, 32, 64); x += __shfl_xor(x, 16, 64);
    x += __shfl_xor(x, 8, 64);  x += __shfl_xor(x, 4, 64);
    x += __shfl_xor(x, 2, 64);  x += __shfl_xor(x, 1, 64);
    return x;
}
__device__ __forceinline__ float warp32_max(float x) {
    x = fmaxf(x, __shfl_xor(x, 16, 32)); x = fmaxf(x, __shfl_xor(x, 8, 32));
    x = fmaxf(x, __shfl_xor(x, 4, 32)); x = fmaxf(x, __shfl_xor(x, 2, 32));
    return fmaxf(x, __shfl_xor(x, 1, 32));
}
// Cached mode uses ordinary flat loads; streaming mode issues glc=1,slc=1 loads.
__device__ __forceinline__ uint32_t load_u32(const uint32_t* p, bool stream) {
    uint32_t v;
    if (stream) {
        asm volatile("global_load_dword %0, %1, off glc slc\n\ts_waitcnt vmcnt(0)"
                     : "=v"(v) : "v"(p) : "memory");
    } else v = *p;
    return v;
}
__device__ __forceinline__ uint32_t packed_load(const void* p, bool stream) {
    return load_u32(reinterpret_cast<const uint32_t*>(p), stream);
}
__device__ __forceinline__ int32_t dot4(uint32_t a, uint32_t b, int32_t c=0) {
    return __builtin_amdgcn_sdot4(static_cast<int32_t>(a), static_cast<int32_t>(b), c, false);
}
__device__ __forceinline__ int32_t dot8(uint32_t a, uint32_t b, int32_t c=0) {
    return __builtin_amdgcn_sdot8(static_cast<int32_t>(a), static_cast<int32_t>(b), c, false);
}
__device__ __forceinline__ float f16(uint16_t x) { return __half2float(__ushort_as_half(x)); }

// ptr0=row-major signed Q8 weights; ptr1=FP16 scales per 32 K values/row;
// ptr2=signed Q8 activations; ptr3=FP32 scales per 32 K; ptr4=FP32 outputs.
__device__ void gemv_q8(const VjVmOpDesc& op, uint32_t wg, uint32_t nwg, uint32_t tid) {
    const auto* w = reinterpret_cast<const int8_t*>(op.ptr0);
    const auto* wd = reinterpret_cast<const uint16_t*>(op.ptr1);
    const auto* x = reinterpret_cast<const int8_t*>(op.ptr2);
    const auto* xd = reinterpret_cast<const float*>(op.ptr3);
    auto* y = reinterpret_cast<float*>(op.ptr4);
    const uint32_t lane=tid&63u, wave=tid>>6, ns=(op.cols+31u)>>5;
    const bool stream=(op.flags&(NonTemporalWeights|DynamicCacheBypass))!=0;
    for (uint32_t r=wg*4u+wave; r<op.rows; r+=nwg*4u) {
        float a=0.0f;
        for (uint32_t i=lane*4u; i+4u<=op.cols; i+=256u) {
            const uint64_t off=static_cast<uint64_t>(r)*op.cols+i;
            const uint32_t ww=packed_load(w+off,stream), xx=*reinterpret_cast<const uint32_t*>(x+i);
            const uint32_t b=i>>5;
            const float s=f16(wd[static_cast<uint64_t>(r)*ns+b])*xd[b];
            a=fmaf(static_cast<float>(dot4(ww,xx)),s,a);
        }
        a=wave_sum(a); if (lane==0) y[r]=a;
    }
}

// Q4A8/Q4DOT8: ptr0 holds signed two's-complement Q4, low nibble first;
// Q4A8 uses Q8 activation bytes, Q4DOT8 uses 4-bit activation codes in the
// same nibble order. Scale layout matches GEMV_Q8.
__device__ void gemv_q4(const VjVmOpDesc& op, uint32_t wg, uint32_t nwg, uint32_t tid, bool act4) {
    const auto* w=reinterpret_cast<const uint32_t*>(op.ptr0);
    const auto* wd=reinterpret_cast<const uint16_t*>(op.ptr1);
    const auto* x=reinterpret_cast<const uint8_t*>(op.ptr2);
    const auto* xd=reinterpret_cast<const float*>(op.ptr3);
    auto* y=reinterpret_cast<float*>(op.ptr4);
    const uint32_t lane=tid&63u, wave=tid>>6, nb=(op.cols+31u)>>5, ns=nb;
    const bool stream=(op.flags&(NonTemporalWeights|DynamicCacheBypass))!=0;
    for (uint32_t r=wg*4u+wave; r<op.rows; r+=nwg*4u) {
        float a=0.0f;
        for(uint32_t b=lane;b<nb;b+=64u) {
            const uint64_t wo=static_cast<uint64_t>(r)*((op.cols+7u)>>3)+static_cast<uint64_t>(b)*4u;
            const uint32_t w0=load_u32(w+wo,stream), w1=load_u32(w+wo+1u,stream);
            const uint32_t w2=load_u32(w+wo+2u,stream), w3=load_u32(w+wo+3u,stream);
            const uint32_t wx[4]={w0,w1,w2,w3};
            if(act4) {
                const uint32_t* xp=reinterpret_cast<const uint32_t*>(x)+static_cast<uint64_t>(b)*4u;
                const uint32_t x0=load_u32(xp+0,false);
                const uint32_t x1=load_u32(xp+1,false);
                const uint32_t x2=load_u32(xp+2,false);
                const uint32_t x3=load_u32(xp+3,false);
                int32_t d=dot8(w0,x0)+dot8(w1,x1)+dot8(w2,x2)+dot8(w3,x3);
                a=fmaf(static_cast<float>(d),f16(wd[static_cast<uint64_t>(r)*ns+b])*xd[b],a);
            } else {
                const int8_t* xa=reinterpret_cast<const int8_t*>(x)+static_cast<uint64_t>(b)*32u;
                int32_t d=0;
                for(uint32_t j=0;j<32u;j+=4u) {
                    const uint32_t packed=wx[j>>3], sh=(j&7u)*4u;
                    uint32_t q=0, xv=*reinterpret_cast<const uint32_t*>(xa+j);
                    #pragma unroll
                    for(uint32_t z=0;z<4u;++z) { int32_t n=(packed>>(sh+z*4u))&15u; n=(n^8)-8; q|=(static_cast<uint32_t>(n)&255u)<<(z*8u); }
                    d+=dot4(q,xv);
                }
                a=fmaf(static_cast<float>(d),f16(wd[static_cast<uint64_t>(r)*ns+b])*xd[b],a);
            }
        }
        a=wave_sum(a); if(lane==0)y[r]=a;
    }
}

__device__ float activation_f32(float g,uint32_t flags) {
    return (flags&GeGluActivation)?0.5f*g*(1.0f+tanhf(0.7978845608f*(g+0.044715f*g*g*g))):g/(1.0f+__expf(-g));
}
__device__ void element_f32(const VjVmOpDesc& op,uint32_t wg,uint32_t nwg,uint32_t tid) {
    const auto* a=reinterpret_cast<const float*>(op.ptr0);
    const auto* b=reinterpret_cast<const float*>(op.ptr1);
    for(uint64_t i=uint64_t(wg)*kThreads+tid;i<uint64_t(op.rows)*op.cols;i+=uint64_t(nwg)*kThreads) {
        if(op.opcode==ConvertF32ToF16)reinterpret_cast<uint16_t*>(op.ptr1)[i]=__half_as_ushort(__float2half_rn(a[i]));
        else reinterpret_cast<float*>(op.ptr2)[i]=op.opcode==ResidualAddF32?a[i]+b[i]:activation_f32(a[i],op.flags)*b[i];
    }
}
__device__ void quantize_f32(const VjVmOpDesc& op,uint32_t wg,uint32_t nwg,uint32_t tid) {
    const uint64_t nb=(uint64_t(op.cols)+31u)/32u;const unsigned lane=tid&31u;
    for(uint64_t b=uint64_t(wg)*8u+(tid>>5);b<uint64_t(op.rows)*nb;b+=uint64_t(nwg)*8u) {
        const uint64_t c=(b%nb)*32u+lane,i=(b/nb)*op.cols+c;
        const float v=c<op.cols?reinterpret_cast<const float*>(op.ptr0)[i]:0;
        const float mx=warp32_max(fabsf(v)),scale=mx/127.0f,inverse=mx>0?127.0f/mx:0;
        if(c<op.cols){const float scaled=isfinite(inverse)?v*inverse:(v/mx)*127.0f;const int8_t q=static_cast<int8_t>(max(-127,min(127,__float2int_rn(scaled))));
            reinterpret_cast<int8_t*>(op.ptr1)[i]=q;if(op.ptr3)reinterpret_cast<float*>(op.ptr3)[i]=q*scale;}
        if(lane==0)reinterpret_cast<float*>(op.ptr2)[b]=scale;
    }
}
__device__ void gemv_f32(const VjVmOpDesc& op,uint32_t wg,uint32_t nwg,uint32_t tid) {
    for(uint64_t r=uint64_t(wg)*kThreads+tid;r<op.rows;r+=uint64_t(nwg)*kThreads){
        float sum=op.ptr3?reinterpret_cast<const float*>(op.ptr3)[r]:0;
        for(uint32_t c=0;c<op.cols;++c)sum=fmaf(reinterpret_cast<const float*>(op.ptr0)[r*op.cols+c],reinterpret_cast<const float*>(op.ptr1)[c],sum);
        reinterpret_cast<float*>(op.ptr2)[r]=sum;
    }
}
__device__ void moe_gate_up_f32(const MoeF32Desc& m,uint32_t wg,uint32_t nwg,uint32_t tid,uint32_t* status) {
    const uint64_t matrix=uint64_t(m.hidden)*m.intermediate,slots=uint64_t(m.top_k)*m.intermediate;
    for(uint64_t i=uint64_t(wg)*kThreads+tid;i<slots;i+=uint64_t(nwg)*kThreads){
        const uint32_t e=reinterpret_cast<const uint32_t*>(m.ids)[i/m.intermediate];
        if(e>=m.experts){if(status)atomicOr(status,1u);continue;}
        const uint64_t base=uint64_t(e)*matrix+(i%m.intermediate)*m.hidden;float g=0,u=0;
        for(uint32_t h=0;h<m.hidden;++h){const float x=reinterpret_cast<const float*>(m.input)[h];
            g=fmaf(reinterpret_cast<const float*>(m.weights)[base+h],x,g);u=fmaf(reinterpret_cast<const float*>(m.up_weights)[base+h],x,u);}
        reinterpret_cast<float*>(m.output)[i]=g+(m.shared?reinterpret_cast<const float*>(m.shared)[i%m.intermediate]:0);
        reinterpret_cast<float*>(m.output)[slots+i]=u;
    }
}
__device__ void moe_down_f32(const MoeF32Desc& m,uint32_t wg,uint32_t nwg,uint32_t tid,uint32_t* status) {
    const uint64_t matrix=uint64_t(m.hidden)*m.intermediate;
    for(uint64_t h=uint64_t(wg)*kThreads+tid;h<m.hidden;h+=uint64_t(nwg)*kThreads){float sum=0;
        for(uint32_t j=0;j<m.top_k;++j){const uint32_t e=reinterpret_cast<const uint32_t*>(m.ids)[j];
            if(e>=m.experts){if(status)atomicOr(status,1u);continue;}float v=0;
            for(uint32_t c=0;c<m.intermediate;++c)v=fmaf(reinterpret_cast<const float*>(m.weights)[uint64_t(e)*matrix+h*m.intermediate+c],reinterpret_cast<const float*>(m.input)[uint64_t(j)*m.intermediate+c],v);
            sum=fmaf(v,reinterpret_cast<const float*>(m.mixture)[j],sum);}
        reinterpret_cast<float*>(m.output)[h]=sum;
    }
}
__device__ void rms_f32(const VjVmOpDesc& op,unsigned wg,unsigned nwg,unsigned tid){
    const auto* input=reinterpret_cast<const float*>(op.ptr0);
    const auto* weights=reinterpret_cast<const float*>(op.ptr1);
    auto* output=reinterpret_cast<float*>(op.ptr2);
    unsigned lane=tid%64;
    for(unsigned row=4*wg+tid/64;row<op.rows;row+=4*nwg){
        auto off=static_cast<uint64_t>(row)*op.cols;float ss=0;
        for(unsigned c=lane;c<op.cols;c+=64){float x=input[off+c];ss=fmaf(x,x,ss);}
        float inv=rsqrtf(wave_sum(ss)/op.cols+op.eps);
        for(unsigned c=lane;c<op.cols;c+=64)output[off+c]=input[off+c]*inv*weights[c];
    }
}
// Residual + RMS norm + per-block Q8 quantizer. Single workgroup owns the vector.
__device__ void residual_rms_q8(const VjVmOpDesc& op,uint32_t tid) {
    __shared__ float sums[4];
    const auto* x=reinterpret_cast<const float*>(op.ptr0); auto* residual=reinterpret_cast<float*>(op.ptr1);
    const auto* g=reinterpret_cast<const float*>(op.ptr2); auto* q=reinterpret_cast<int8_t*>(op.ptr3);
    auto* scales=reinterpret_cast<float*>(op.ptr4); const uint32_t D=op.cols, wave=tid>>6,lane=tid&63u;
    float ss=0;
    for(uint32_t i=tid;i<D;i+=kThreads){float v=x[i]+residual[i];residual[i]=v;ss=fmaf(v,v,ss);}
    ss=wave_sum(ss); if(lane==0)sums[wave]=ss; __syncthreads();
    const float inv=rsqrtf((sums[0]+sums[1]+sums[2]+sums[3])/static_cast<float>(D)+op.eps);
    for(uint32_t b=tid>>5;b<(D+31u)/32u;b+=8u) {
        const uint32_t i=b*32u+(tid&31u); const float v=(i<D)?residual[i]*inv*g[i]:0.0f;
        const float mx=warp32_max(fabsf(v)); const float d=mx*(1.0f/127.0f), id=mx>0?127.0f/mx:0;
        if(i<D)q[i]=static_cast<int8_t>(max(-127,min(127,__float2int_rn(v*id))));
        if((tid&31u)==0)scales[b]=d;
    }
}

// QK norm + split-half RoPE; head vectors are FP32 and modified in place.
__device__ void qk_norm_rope(const VjVmOpDesc& op,uint32_t wg,uint32_t nwg,uint32_t tid) {
    const auto* v=reinterpret_cast<const float*>(op.ptr0); const auto* g=reinterpret_cast<const float*>(op.ptr1);
    auto* out=reinterpret_cast<float*>(op.ptr2?op.ptr2:op.ptr0);const bool skip=(op.flags&SkipRms)!=0;
    const uint32_t wave=tid>>6,lane=tid&63u,hd=op.cols,half=hd>>1;
    for(uint32_t h=wg*4u+wave;h<op.rows;h+=nwg*4u) {
        const float* vh=v+static_cast<uint64_t>(h)*hd; float ss=0;
        for(uint32_t i=lane;i<hd;i+=64u)ss=fmaf(vh[i],vh[i],ss);
        const float inv=skip?1.0f:rsqrtf(wave_sum(ss)/static_cast<float>(hd)+op.eps);
        const float theta=static_cast<float>(op.aux0);
        for(uint32_t p=lane;p<half;p+=64u) {
            const float a=vh[p]*inv*(skip?1.0f:g[p]), b=vh[p+half]*inv*(skip?1.0f:g[p+half]);
            const float ang=theta*__powf(10000.0f,-2.0f*static_cast<float>(p)/hd);
            float s,c;__sincosf(ang,&s,&c);out[static_cast<uint64_t>(h)*hd+p]=a*c-b*s;out[static_cast<uint64_t>(h)*hd+p+half]=a*s+b*c;
        }
    }
}

// Streaming blockwise stable softmax over paged FP16 KV. One WG per Q head;
// pages are page_table[page] => physical page with [tokens,kv_heads,head_dim] K then V.
__device__ void flash_gqa(const VjVmFlashDesc& f,uint32_t tid) {
    const uint32_t lane=tid&63u, h=blockIdx.x*4u+(tid>>6);
    if(h>=f.q_heads||f.head_dim>256u||f.kv_heads==0u)return;
    const auto* q=reinterpret_cast<const float*>(f.q)+static_cast<uint64_t>(h)*f.head_dim;
    const auto* table=reinterpret_cast<const uint32_t*>(f.page_table);
    const uint32_t kh=h/(f.q_heads/f.kv_heads), groups=f.page_tokens*f.kv_heads*f.head_dim;
    float acc[4]={0,0,0,0}; const uint32_t ng=(f.head_dim+63u)/64u;
    float m=-INFINITY,l=0.0f;
    for(uint32_t t=0;t<f.context_tokens;++t) {
        const uint32_t p=t/f.page_tokens,pt=t%f.page_tokens;
        if(p>=f.page_count)break;
        const uint32_t phys=table[p]; const uint64_t base=(static_cast<uint64_t>(phys)*groups+static_cast<uint64_t>(pt)*f.kv_heads*f.head_dim+static_cast<uint64_t>(kh)*f.head_dim);
        const auto* k=reinterpret_cast<const uint16_t*>(f.k_cache)+base;
        const auto* v=reinterpret_cast<const uint16_t*>(f.v_cache)+base;
        float d=0; for(uint32_t g=0;g<ng;++g){uint32_t i=g*64u+lane;if(i<f.head_dim)d=fmaf(q[i],f16(k[i]),d);}
        d=wave_sum(d)*f.scale;
        const float mn=fmaxf(m,d), alpha=isinf(m)?0.0f:__expf(m-mn), beta=__expf(d-mn);
        l=l*alpha+beta;m=mn;
        for(uint32_t g=0;g<ng;++g){uint32_t i=g*64u+lane;if(i<f.head_dim)acc[g]=fmaf(beta,f16(v[i]),acc[g]*alpha);}
    }
    const uint64_t out=static_cast<uint64_t>(h)*f.head_dim;
    for(uint32_t g=0;g<ng;++g){uint32_t i=g*64u+lane;if(i<f.head_dim)reinterpret_cast<float*>(f.out)[out+i]=l>0?acc[g]/l:0.0f;}
    if(lane==0&&f.lse)reinterpret_cast<float*>(f.lse)[h]=l>0?m+__logf(l):-INFINITY;
}

// Router top-k: deterministic insertion sort, stable ties preserve smaller id.
__device__ void router_topk(const VjVmOpDesc& op,uint32_t tid) {
    if((tid&255u)!=0)return;
    const auto* logits=reinterpret_cast<const float*>(op.ptr0);auto* ids=reinterpret_cast<uint32_t*>(op.ptr1);
    auto* weights=reinterpret_cast<float*>(op.ptr2);const uint32_t experts=op.cols,k=min(op.aux0,experts);
    for(uint32_t t=blockIdx.x;t<op.rows;t+=gridDim.x) {
        float mx=-INFINITY;for(uint32_t e=0;e<experts;++e)mx=fmaxf(mx,logits[static_cast<uint64_t>(t)*experts+e]);
        for(uint32_t j=0;j<k;++j){uint32_t best=0;float bv=-INFINITY;
            for(uint32_t e=0;e<experts;++e){float z=logits[static_cast<uint64_t>(t)*experts+e];bool used=false;for(uint32_t q=0;q<j;++q)used|=ids[static_cast<uint64_t>(t)*k+q]==e;
                if(!used&&(z>bv||(z==bv&&e<best))){bv=z;best=e;}}
            ids[static_cast<uint64_t>(t)*k+j]=best;
        }
        float denom=0;for(uint32_t j=0;j<k;++j)denom+=__expf(logits[static_cast<uint64_t>(t)*experts+ids[static_cast<uint64_t>(t)*k+j]]-mx);
        for(uint32_t j=0;j<k;++j)weights[static_cast<uint64_t>(t)*k+j]=__expf(logits[static_cast<uint64_t>(t)*experts+ids[static_cast<uint64_t>(t)*k+j]]-mx)/denom;
    }
}

// SiLU/GELU gate*up followed by independent symmetric signed Q8 blocks.
__device__ void act_mul_q8(const VjVmOpDesc& op,uint32_t wg,uint32_t nwg,uint32_t tid) {
    const auto* gate=reinterpret_cast<const float*>(op.ptr0);const auto* up=reinterpret_cast<const float*>(op.ptr1);
    auto* q=reinterpret_cast<int8_t*>(op.ptr2);auto* ds=reinterpret_cast<float*>(op.ptr3);
    const uint32_t nb=(op.cols+31u)/32u, sub=tid&31u;
    for(uint32_t b=wg*8u+(tid>>5);b<nb;b+=nwg*8u) {
        const uint32_t i=b*32u+sub;float v=0;
        if(i<op.cols){float g=gate[i];float act=(op.flags&GeGluActivation)?(0.5f*g*(1.0f+tanhf(0.7978845608f*(g+0.044715f*g*g*g)))):(g/(1.0f+__expf(-g)));v=act*up[i];}
        float mx=warp32_max(fabsf(v));float d=mx/127.0f,id=mx>0?127.0f/mx:0;
        if(i<op.cols)q[i]=static_cast<int8_t>(max(-127,min(127,__float2int_rn(v*id))));
        if(sub==0)ds[b]=d;
    }
}

// Expert-combine: ptr0 expert output [token,top_k,hidden], ptr1 IDs (unused
// by already-gathered slots), ptr2 routing weights, ptr3 output.
__device__ void moe_combine(const VjVmOpDesc& op,uint32_t tid) {
    const auto* x=reinterpret_cast<const float*>(op.ptr0);const auto* w=reinterpret_cast<const float*>(op.ptr2);auto* y=reinterpret_cast<float*>(op.ptr3);
    const uint32_t hidden=op.cols,k=op.aux0;
    for(uint32_t i=blockIdx.x*kThreads+tid;i<op.rows*hidden;i+=gridDim.x*kThreads){uint32_t t=i/hidden,d=i%hidden;float a=0;
        for(uint32_t j=0;j<k;++j)a=fmaf(w[static_cast<uint64_t>(t)*k+j],x[(static_cast<uint64_t>(t)*k+j)*hidden+d],a);y[i]=a;}
}

__device__ __forceinline__ void unknown(uint32_t* status,uint32_t tid) { if(tid==0&&status)atomicOr(status,1u); }
} // namespace

// Ordered launches provide inter-workgroup visibility.
extern "C" __global__ __launch_bounds__(kThreads,4)
void vj_vm_attn_block(const VjVmOpDesc* __restrict__ program,uint32_t num_ops,uint32_t cache_policy,uint32_t* __restrict__ status){
    (void)cache_policy;
    const uint32_t tid=threadIdx.x,wg=blockIdx.x,nwg=gridDim.x;
    for(uint32_t pc=0;pc<num_ops;++pc){const VjVmOpDesc& op=program[pc];
        if(op.opcode==ArResRmsQ8){if(wg==0)residual_rms_q8(op,tid);}
        else if(op.opcode==GemvQ8Dp4a)gemv_q8(op,wg,nwg,tid);
        else if(op.opcode==GemvQ4A8)gemv_q4(op,wg,nwg,tid,false);
        else if(op.opcode==QkNormRopeKv)qk_norm_rope(op,wg,nwg,tid);
        else if(op.opcode==FlashDecodeGqa){const auto& f=*reinterpret_cast<const VjVmFlashDesc*>(op.ptr0);flash_gqa(f,tid);}
        else if(op.opcode==GridBarrier){__syncthreads();return;}
        else unknown(status,tid);
        __syncthreads();if(op.flags&BarrierAfter)return;
    }
}
extern "C" __global__ __launch_bounds__(kThreads,4)
void vj_vm_ffn_block(const VjVmOpDesc* __restrict__ program,uint32_t num_ops,uint32_t cache_policy,uint32_t* __restrict__ status){
    (void)cache_policy;
    const uint32_t tid=threadIdx.x,wg=blockIdx.x,nwg=gridDim.x;
    for(uint32_t pc=0;pc<num_ops;++pc){const VjVmOpDesc& op=program[pc];
        if(op.opcode==ArResRmsQ8){if(wg==0)residual_rms_q8(op,tid);}
        else if(op.opcode==GemvQ4A8)gemv_q4(op,wg,nwg,tid,false);
        else if(op.opcode==GemvQ4Dot8)gemv_q4(op,wg,nwg,tid,true);
        else if(op.opcode==MoeRouterTopK)router_topk(op,tid);
        else if(op.opcode==ActMulQ8)act_mul_q8(op,wg,nwg,tid);
        else if(op.opcode==GridBarrier){__syncthreads();return;}
        else unknown(status,tid);
        __syncthreads();if(op.flags&BarrierAfter)return;
    }
}
extern "C" __global__ __launch_bounds__(kThreads,4)
void vj_vm_aux_block(const VjVmOpDesc* __restrict__ program,uint32_t num_ops,uint32_t cache_policy,uint32_t* __restrict__ status){
    (void)cache_policy;
    const uint32_t tid=threadIdx.x,wg=blockIdx.x,nwg=gridDim.x;
    if(num_ops!=1){unknown(status,tid);return;}
    {const VjVmOpDesc& op=program[0];
        if(op.opcode==RmsF32)rms_f32(op,wg,nwg,tid);
        else if(op.opcode==ResidualAddF32||op.opcode==ActMulF32||op.opcode==ConvertF32ToF16)element_f32(op,wg,nwg,tid);
        else if(op.opcode==QuantizeF32)quantize_f32(op,wg,nwg,tid);
        else if(op.opcode==GemvF32)gemv_f32(op,wg,nwg,tid);
        else if(op.opcode==PagedKvStore){const auto& s=*reinterpret_cast<const KvStoreDesc*>(op.ptr0);
            if(s.page_tokens&&s.position/s.page_tokens<s.pages){auto pg=reinterpret_cast<const unsigned*>(s.page_table)[s.position/s.page_tokens];
                if(pg<s.pages){uint64_t width=uint64_t(s.heads)*s.dim,base=(uint64_t(pg)*s.page_tokens+s.position%s.page_tokens)*width;
                    for(uint64_t i=uint64_t(wg)*blockDim.x+tid;i<width;i+=uint64_t(nwg)*blockDim.x){reinterpret_cast<uint16_t*>(s.k_cache)[base+i]=(op.flags&F32KvInput)?__half_as_ushort(__float2half_rn(reinterpret_cast<const float*>(s.k)[i])):reinterpret_cast<const uint16_t*>(s.k)[i];reinterpret_cast<uint16_t*>(s.v_cache)[base+i]=(op.flags&F32KvInput)?__half_as_ushort(__float2half_rn(reinterpret_cast<const float*>(s.v)[i])):reinterpret_cast<const uint16_t*>(s.v)[i];}}
                else unknown(status,tid);}else unknown(status,tid);}
        else if(op.opcode==MoeGateUpF32)moe_gate_up_f32(*reinterpret_cast<const MoeF32Desc*>(op.ptr0),wg,nwg,tid,status);
        else if(op.opcode==MoeDownF32)moe_down_f32(*reinterpret_cast<const MoeF32Desc*>(op.ptr0),wg,nwg,tid,status);
        else if(op.opcode==LatentAttention){const auto& a=*reinterpret_cast<const LatentDesc*>(op.ptr0);
            const auto* q=reinterpret_cast<const float*>(a.query);const auto* proj=reinterpret_cast<const float*>(a.value_proj);
            for(uint64_t j=uint64_t(wg)*blockDim.x+tid;j<uint64_t(a.heads)*a.output_dim;j+=uint64_t(nwg)*blockDim.x){uint64_t h=j/a.output_dim;float mx=-INFINITY,den=0,num=0;
                for(uint64_t t=0;t<a.tokens;++t){float score=0,val=0;for(unsigned k=0;k<a.rank;++k){float v=(op.flags&F16LatentCache)?f16(reinterpret_cast<const uint16_t*>(a.latent)[t*a.rank+k]):reinterpret_cast<const float*>(a.latent)[t*a.rank+k];score=fmaf(q[h*a.rank+k],v,score);val=fmaf(proj[j*a.rank+k],v,val);}
                    for(unsigned k=0;k<a.rot_dim;++k){float v=(op.flags&F16LatentCache)?f16(reinterpret_cast<const uint16_t*>(a.rot_key)[t*a.rot_dim+k]):reinterpret_cast<const float*>(a.rot_key)[t*a.rot_dim+k];score=fmaf(reinterpret_cast<const float*>(a.rot_query)[h*a.rot_dim+k],v,score);}
                    score*=a.scale;float next=fmaxf(mx,score),keep=__expf(mx-next),w=__expf(score-next);den=den*keep+w;num=fmaf(w,val,num*keep);mx=next;}
                reinterpret_cast<float*>(a.out)[j]=num/den;}}
        else if(op.opcode==GatedDelta){const auto& s=*reinterpret_cast<const DeltaDesc*>(op.ptr0);
            const auto* q=reinterpret_cast<const float*>(s.q);const auto* k=reinterpret_cast<const float*>(s.k);const auto* v=reinterpret_cast<const float*>(s.v);
            const auto* gates=reinterpret_cast<const float*>(s.gates);auto* state=reinterpret_cast<float*>(s.state);
            for(uint64_t c=uint64_t(wg)*blockDim.x+tid;c<s.value_dim;c+=uint64_t(nwg)*blockDim.x){float pred=0,sum=0;
                for(unsigned i=0;i<s.key_dim;++i)pred=fmaf(state[uint64_t(i)*s.value_dim+c],k[i],pred);
                float delta=gates[1]*(v[c]-gates[0]*pred);
                for(unsigned i=0;i<s.key_dim;++i){auto& x=state[uint64_t(i)*s.value_dim+c];x=fmaf(k[i],delta,gates[0]*x);sum=fmaf(x,q[i],sum);}reinterpret_cast<float*>(s.out)[c]=sum;}}
        else if(op.opcode==MoeExpertCombine)moe_combine(op,tid);
        else if(op.opcode==GridBarrier){__syncthreads();return;}
        else unknown(status,tid);

    }
}

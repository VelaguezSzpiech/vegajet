// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace vegajet::vm {
struct alignas(64) VjVmOpDesc { uint32_t opcode,flags,rows,cols; uint64_t ptr0,ptr1,ptr2,ptr3,ptr4; uint32_t aux0; float eps; };
static_assert(sizeof(VjVmOpDesc)==64 && alignof(VjVmOpDesc)==64,"VM descriptor ABI");
static_assert(offsetof(VjVmOpDesc,ptr0)==16 && offsetof(VjVmOpDesc,aux0)==56,"VM descriptor offsets");
static_assert(std::is_standard_layout<VjVmOpDesc>::value,"VM descriptor must be standard layout");

enum Opcode : uint32_t { ArResRmsQ8=1,GemvQ4A8=2,GemvQ8Dp4a=3,QkNormRopeKv=4,FlashDecodeGqa=5,GridBarrier=6,GemvQ4Dot8=7,MoeRouterTopK=8,ActMulQ8=9,MoeExpertCombine=10,RmsF32=11,PagedKvStore=12,GatedDelta=13,LatentAttention=14,ResidualAddF32=15,QuantizeF32=16,GemvF32=17,ActMulF32=18,MoeGateUpF32=19,MoeDownF32=20,ConvertF32ToF16=21 };
enum Flag : uint32_t { NonTemporalWeights=1u<<0,BarrierAfter=1u<<2,DynamicCacheBypass=1u<<3,GeGluActivation=1u<<4,F32KvInput=1u<<5,SkipRms=1u<<6,F16LatentCache=1u<<7 };
enum CachePolicy : uint32_t { CacheDefault=0,CacheStreaming=1,CacheCoherent=2 };
struct alignas(16) VjVmLaunchDesc { uint32_t num_ops,cache_policy,flags,reserved; };
static_assert(sizeof(VjVmLaunchDesc)==16,"VM launch ABI");

// Independent store also usable after a rotary descriptor via ordered dispatch.
// One token, one head. State layout [key_dim,value_dim]; g and beta in gates[2].
struct DeltaDesc { uint64_t q,k,v,gates,state,out; uint32_t key_dim,value_dim; };
// Absorbed MLA: per-head projected latent queries and rotary queries, shared
// latent/rotary cache, per-head value up-projection [heads,output_dim,rank].
struct LatentDesc {uint64_t query,rot_query,latent,rot_key,value_proj,out;uint32_t heads,tokens,rank,rot_dim,output_dim;float scale;};
// F16LatentCache selects FP16 latent and rot_key arrays; other fields stay FP32.
// Latent arrays are logically contiguous; page-table indirection is not supported.
struct KvStoreDesc { uint64_t k,v,k_cache,v_cache,page_table; uint32_t position,page_tokens,pages,heads,dim; };
struct VjVmFlashDesc {
    uint64_t q,k_cache,v_cache,out,lse;
    uint32_t q_heads,kv_heads,head_dim,page_tokens;
    uint32_t context_tokens,page_count,gqa_group,reserved;
    uint64_t page_table; // u32 page IDs; physical KV pages are contiguous
    float scale;
};

// Single-token routed FP32 projections. Gate/up matrices are
// [experts,intermediate,hidden]; input[hidden], IDs[top_k]. Output contains
// all gate slots followed by all up slots: [2,top_k,intermediate]. Optional
// shared[intermediate] is added to gates only, before separate activation.
// Down matrices are [experts,hidden,intermediate]; input[top_k,intermediate],
// mixture[top_k]; output[hidden] is the weighted sum of selected projections.
struct MoeF32Desc {
    uint64_t weights,up_weights,input,ids,mixture,output;
    uint32_t experts,hidden,intermediate,top_k;
    uint64_t shared;
};
static_assert(sizeof(MoeF32Desc)==72 && offsetof(MoeF32Desc,shared)==64,"routed FP32 ABI");
// ResidualAddF32 / ActMulF32: ptr0/ptr1 inputs, ptr2 output [rows,cols].
// QuantizeF32: ptr0 FP32, ptr1 Q8, ptr2 scales [rows,ceil(cols/32)],
// optional ptr3 dequantized FP32. Partial last blocks never cross row bounds.
// GemvF32: ptr0 weights[rows,cols], ptr1 x[cols], ptr2 y[rows], optional ptr3 bias.
// ConvertF32ToF16: ptr0 FP32 input, ptr1 FP16 output [rows,cols], round-nearest-even.
// MoeGateUpF32 / MoeDownF32: ptr0 points to MoeF32Desc.
// QkNormRopeKv: optional ptr2 out (otherwise in-place); SkipRms means RoPE only.
// PagedKvStore: F32KvInput converts input K/V FP32 to the unchanged FP16 cache.

// Main descriptor pointer ABI:
// ArResRmsQ8: ptr0 x FP32, ptr1 residual FP32 in/out, ptr2 norm weights FP32,
// ptr3 signed Q8 output, ptr4 FP32 scales [ceil(cols/32)]. TP1 local only;
// collective is host-decomposed.
// GEMV Q8/Q4A8/Q4DOT8: ptr0 row-major signed Q8 or low-nibble-first signed Q4
// weights; ptr1 FP16 row scales [rows,ceil(cols/32)]; ptr2 signed Q8 activations
// except Q4DOT8 (signed packed Q4); ptr3 FP32 activation scales per 32 values
// (Q4DOT8 scales each 32 values); ptr4 FP32 y[rows].
// QkNormRopeKv: ptr0 FP32 vectors in/out, ptr1 FP32 norm weights; rows=heads,
// cols=head dim, aux0=position, eps=RMS epsilon; split-half RoPE.
// FlashDecodeGqa: ptr0 -> VjVmFlashDesc. q is FP32 [q_heads,head_dim], K/V
// caches are FP16 [physical_page,page_tokens,kv_heads,head_dim], page_table maps
// logical pages to physical pages; out FP32 [q_heads,head_dim], optional LSE.
// GridBarrier is host-only. Split program at opcode/BarrierAfter and launch the
// next segment after ordered completion; there is intentionally no grid spin.
// MoeRouterTopK: ptr0 logits FP32 [rows,cols], ptr1 IDs u32 [rows,aux0],
// ptr2 FP32 normalized selected weights [rows,aux0]; rows=tokens, cols=experts.
// ActMulQ8: ptr0 gate FP32, ptr1 up FP32, ptr2 signed Q8 output, ptr3 FP32
// scales per 32; GeGluActivation chooses tanh-approximate GELU, otherwise SiLU.
// MoeExpertCombine: ptr0 already gathered expert outputs FP32 [rows,aux0,cols],
// ptr2 weights FP32 [rows,aux0], ptr3 FP32 output [rows,cols]. ptr1 reserved.
} // namespace vegajet::vm

// RmsF32: rows vectors of cols FP32. ptr0=input, ptr1=norm[cols], ptr2=output.

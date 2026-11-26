// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
#include "core/quant_repacker.hpp"
#include "core/kfd_runtime.hpp"
#include <cstdio>
namespace vegajet::ornith::sub4_integration {
// Keep the Q4 fallback for larger prefill and fused shared-expert kernels. All sidecar
// planes are real arena allocations, including guards and expanded XOR scale planes.
template<class Weight> bool upload(kfd::Runtime& rt,size_t gpu,const std::vector<float>& original,
                                   const quant::PackedVjQ4A8& q4,Weight& w,std::string* err) {
    size_t bytes=0;
    auto put=[&](const auto& v,uint64_t& va) {
        size_t n=v.size()*sizeof(v[0]); bytes+=n;
        va=rt.vram_arena(gpu).alloc(n,256);
        return va && rt.upload_to_vram(gpu,va,v.data(),n);
    };
    const uint32_t bits=quant::sub4_dot8_bits();
    const bool xor_on=quant::tcc_xor_enabled() && w.cols%512==0 && (bits || w.cols!=512); // Q4 K=512 keeps the W16 reduction.
    bool ok=true;
    if(bits) {
        auto p=quant::pack_f32_to_vj_sub4(original.data(),w.rows,w.cols,
                bits==2?quant::NativeFormat::VJ_Q2_DOT8:quant::NativeFormat::VJ_Q3_DOT8);
        if(quant::milo_enabled()) {
            auto f=quant::fit_milo(original.data(),p,16,24,quant::MiloStorage::Int3);
            ok=put(f.u3,w.milo_u) && put(f.v3,w.milo_v) && put(f.us,w.milo_us) && put(f.vs,w.milo_vs);
            w.milo_rank=f.rank;
        }
        if(xor_on) p=quant::repack_tcc_xor(p);
        ok=ok && put(p.qs,w.sub4_qs) && put(p.dm,w.sub4_dm);
        w.sub4_bits=bits; w.sub4_xor=xor_on;
    } else if(xor_on) {
        auto p=quant::repack_tcc_xor(q4);
        ok=put(p.qs,w.xor_qs) && put(p.dm,w.xor_dm);
    }
    w.extra_bytes=bytes;
    if(!ok && err) *err="SUB4/TCC/MiLo sidecar allocation or upload failed";
    return ok;
}
}

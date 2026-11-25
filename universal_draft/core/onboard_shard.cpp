#include "core/onboard_shard.hpp"
#include "core/workspace_packer.hpp"
#include "core/universal_quant.hpp"
#include "core/ggml_types.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace universal::onboard {
namespace {
constexpr uint64_t kMax=std::numeric_limits<uint64_t>::max();
bool mul(uint64_t a,uint64_t b,uint64_t& out){if(b&&a>kMax/b)return false;out=a*b;return true;}
uint64_t tensor_bytes(const Tensor& t){
    uint64_t n=1; for(uint64_t d:t.shape.dims){if(!d||!mul(n,d,n))return 0;}
    const auto tr=vegajet::ggml_type_traits(t.type);
    if(!tr.block_elems||n%tr.block_elems)return 0;
    uint64_t out=0;return mul(n/tr.block_elems,tr.block_bytes,out)?out:0;
}
std::vector<uint64_t> apport(uint64_t total,const std::vector<Device>& ds){
    std::vector<uint64_t> out(ds.size());long double sum=0;
    for(const auto& d:ds)sum+=static_cast<long double>(d.speed);
    if(!(sum>0)||!std::isfinite(sum))return {};
    struct R{size_t i;long double fraction;};std::vector<R> rem;uint64_t used=0;
    for(size_t i=0;i<ds.size();++i){long double exact=static_cast<long double>(total)*ds[i].speed/sum;
        out[i]=static_cast<uint64_t>(std::floor(exact));used+=out[i];rem.push_back({i,exact-out[i]});}
    std::stable_sort(rem.begin(),rem.end(),[](const R&a,const R&b){return a.fraction>b.fraction;});
    for(uint64_t n=total-used;n;--n)++out[rem[static_cast<size_t>(total-used-n)%rem.size()].i];
    return out;
}
}
uint64_t full_weight_bytes(const Model& m,const Profile& p){
    uint64_t actual=0;
    for(const auto& t:m.tensors){uint64_t b=tensor_bytes(t);if(!b)return p.parameters?kMax:0;if(actual>kMax-b)return kMax;actual+=b;}
    if(!p.parameters)return actual;
    if(!(p.bits_per_weight>0)||!std::isfinite(p.bits_per_weight))return 0;
    const long double bytes=std::ceil(static_cast<long double>(p.parameters)*p.bits_per_weight/8.0L);
    if(bytes>=static_cast<long double>(kMax))return kMax;
    return std::max(static_cast<uint64_t>(bytes),actual);
}
Plan plan(const Model& m,const Profile& p,const std::vector<Device>& ds){
    Plan r; r.full_resident=true;
    if(!m.valid()){r.error="invalid compiler model";return r;}
    if(ds.empty()||p.context_tokens==0){r.error="missing devices or context";return r;}
    for(const auto& d:ds)if(!d.vram_bytes||!(d.speed>0)||!std::isfinite(d.speed)){r.error="invalid device capacity or speed";return r;}
    r.model_weight_bytes=full_weight_bytes(m,p);
    if(!r.model_weight_bytes){r.error="model weight footprint is unknown or zero";return r;}
    const bool mla=m.family==Family::SparseMLAMoE;
    const uint64_t layers=p.layers?p.layers:m.layers;
    const uint64_t kvheads=mla?(p.kv_heads?p.kv_heads:1):(p.kv_heads?p.kv_heads:m.kv_heads);
    const uint64_t hd=p.head_dim?p.head_dim:(m.head_dim?m.head_dim:1);
    const uint64_t qh64=p.heads?p.heads:m.heads;
    if(!layers||!kvheads||!hd||(!mla&&(!qh64||qh64%kvheads))||
       qh64>std::numeric_limits<uint32_t>::max()||kvheads>std::numeric_limits<uint32_t>::max()){
        r.error="invalid full-model attention geometry";return r;
    }
    uint64_t per_head=0,kv=0;
    if(!mul(layers,p.context_tokens,per_head)||!mul(per_head,(hd+31)/32,per_head)||
       !mul(per_head,2,per_head)||!mul(per_head,kvheads,kv)){
        r.error="KV footprint overflow";return r;
    }
    r.kv_bytes=kv;
    auto wb=apport(r.model_weight_bytes,ds);
    if(wb.size()!=ds.size()){r.error="invalid speed weights";return r;}
    r.gpus.resize(ds.size());
    const uint32_t qheads=static_cast<uint32_t>(qh64);
    const uint32_t kvh=static_cast<uint32_t>(kvheads);
    const uint32_t group=mla?0:qheads/kvh;
    std::vector<uint64_t> group_counts;
    if(!mla)group_counts=apport(kvh,ds);
    uint32_t gcur=0;
    for(size_t i=0;i<ds.size();++i){auto& s=r.gpus[i];s.weight_bytes=wb[i];
        if(mla){s.sequence_fraction=ds[i].speed;}
        else {uint32_t cnt=static_cast<uint32_t>(group_counts[i]);s.kv_head_begin=gcur;s.kv_head_end=gcur+cnt;
            s.q_head_begin=gcur*group;s.q_head_end=(gcur+cnt)*group;gcur+=cnt;s.kv_bytes=per_head*cnt;}
    }
    if(mla){long double speeds=0;for(const auto&d:ds)speeds+=d.speed;
        for(size_t i=0;i<ds.size();++i)r.gpus[i].sequence_fraction=static_cast<double>(ds[i].speed/speeds);
        auto splits=apport(kv,ds);for(size_t i=0;i<ds.size();++i)r.gpus[i].kv_bytes=splits[i];
    }
    const auto arena=pack_workspace(m,{1,std::max(512u,p.context_tokens)});
    if(!arena.valid()){r.error=arena.error;return r;}
    r.feasible=true;r.kv_bytes=0;
    for(size_t i=0;i<ds.size();++i){auto& s=r.gpus[i];const auto cap=ds[i].vram_bytes;
        s.arena_bytes=arena.arena_bytes; const auto blocks=s.kv_bytes;s.kv_bytes=0;
        if(s.weight_bytes>kMax-s.arena_bytes){r.feasible=false;r.error="fixed footprint overflow";return r;}
        s.resident_bytes=s.weight_bytes+s.arena_bytes;
        if(s.resident_bytes>cap){r.feasible=false;continue;}
        if(!blocks)continue;
        auto choice=vegajet::universal::co_budget(cap-s.resident_bytes,blocks,p.kv_block_bytes,{});
        if(!choice.fits){r.feasible=false;continue;}
        s.kv_bytes=choice.kv_bytes;s.kv_tier=static_cast<uint32_t>(choice.tier);
        s.resident_bytes+=s.kv_bytes;
        if(r.kv_bytes>kMax-s.kv_bytes){r.feasible=false;r.error="aggregate KV overflow";return r;}
        r.kv_bytes+=s.kv_bytes;
    }
    if(!r.feasible)r.error="weights, workspace and minimum KV tier exceed rank capacity";
    return r;
}
} // namespace universal::onboard

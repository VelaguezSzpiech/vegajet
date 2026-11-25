#include "core/fusion_pass.hpp"
#include <algorithm>
namespace universal { namespace {
bool consumes(const Op& op,size_t value){return std::find(op.inputs.begin(),op.inputs.end(),value)!=op.inputs.end();}
size_t output(const Op& op){return op.outputs.empty()?0:op.outputs.front();}
}
FusionStats PassManager::run(Model& m,const PassOptions& options) const {
 FusionStats s;const bool temporal=!options.prefill&&options.m_hit==1;
 for(auto&o:m.ops){const bool weight_read=o.kind==OpKind::Linear||o.kind==OpKind::MoEFFN;o.cache=weight_read?(temporal?CachePolicy::Streaming:CachePolicy::Temporal):CachePolicy::Default;o.glc=weight_read&&temporal;o.slc=weight_read&&temporal;if(weight_read){o.grid_x=4*options.n_cu;o.split_k=static_cast<uint32_t>(std::max<uint64_t>(1,(m.hidden+256ull*options.n_cu-1)/(256ull*options.n_cu)));o.grid_y=std::max<uint32_t>(1,options.m_hit)*o.split_k;}}
 if(options.tp_size==1)for(size_t i=0;i<m.ops.size();){auto& ar=m.ops[i];if(ar.kind!=OpKind::AllReduce){++i;continue;}if(ar.inputs.size()!=1||ar.outputs.size()!=1)return s;const size_t from=ar.inputs[0],to=ar.outputs[0];for(size_t j=0;j<m.ops.size();++j)if(j!=i)for(auto&in:m.ops[j].inputs)if(in==to)in=from;m.ops.erase(m.ops.begin()+static_cast<std::ptrdiff_t>(i));++s.elided_allreduce;}
 if(options.tp_size==1)for(size_t i=0;i+2<m.ops.size();++i){const auto&a=m.ops[i];const auto&b=m.ops[i+1];const auto&c=m.ops[i+2];if(a.kind==OpKind::ResidualAdd&&b.kind==OpKind::RMSNorm&&c.kind==OpKind::Quantize&&consumes(b,output(a))&&consumes(c,output(b))){for(size_t k=i;k<i+3;++k)m.ops[k].tag="fused_local_res_rms_quant";++s.reductions;}}
 for(size_t i=0;i+3<m.ops.size();++i){const auto&a=m.ops[i];const auto&b=m.ops[i+1];const auto&c=m.ops[i+2];const auto&d=m.ops[i+3];if(a.kind==OpKind::AllReduce&&b.kind==OpKind::ResidualAdd&&c.kind==OpKind::RMSNorm&&d.kind==OpKind::Quantize&&consumes(b,output(a))&&consumes(c,output(b))&&consumes(d,output(c))){for(size_t k=i;k<i+4;++k)m.ops[k].tag="fused_ar_res_rms_quant";++s.reductions;}}
 for(size_t i=0;i<m.ops.size();++i)if(m.ops[i].kind==OpKind::Linear&&m.ops[i].tag=="qkv"){size_t j=i+1;size_t v=output(m.ops[i]);if(j<m.ops.size()&&m.ops[j].kind==OpKind::QKNorm&&consumes(m.ops[j],v))v=output(m.ops[j++]);if(j+1<m.ops.size()&&m.ops[j].kind==OpKind::RoPE&&consumes(m.ops[j],v)&&m.ops[j+1].kind==OpKind::PagedKVWrite&&consumes(m.ops[j+1],output(m.ops[j]))){for(size_t k=i;k<=j+1;++k)m.ops[k].tag="fused_qkv_rope_kv";++s.qkv_cache;}}
 for(size_t i=0;i<m.ops.size();++i)if(m.ops[i].kind==OpKind::MoERouter){size_t gate=m.ops.size(),shared=m.ops.size();for(size_t j=i+1;j<m.ops.size()&&m.ops[j].kind!=OpKind::MoERouter;++j){if(m.ops[j].kind==OpKind::Linear&&m.ops[j].tag=="shared_expert"&&consumes(m.ops[j],output(m.ops[i])))shared=j;if(m.ops[j].kind==OpKind::Linear&&m.ops[j].tag=="gate_up"&&consumes(m.ops[j],output(m.ops[i]))){gate=j;break;}}if(gate==m.ops.size())continue;if(shared!=m.ops.size()&&(!consumes(m.ops[gate],output(m.ops[shared]))||shared>gate))continue;const size_t act=gate+1,down=gate+2;if(down>=m.ops.size()||m.ops[act].kind!=OpKind::ActMul||!consumes(m.ops[act],output(m.ops[gate]))||(m.ops[down].kind!=OpKind::Linear&&m.ops[down].kind!=OpKind::MoEFFN)||m.ops[down].tag!="down"||!consumes(m.ops[down],output(m.ops[act])))continue;for(size_t k=i;k<=down;++k)m.ops[k].tag="fused_mono_moe";++s.moe;}
 return s;
}
} // namespace universal

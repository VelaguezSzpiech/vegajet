#include "core/universal_ir.hpp"
#include <algorithm>
#include <limits>
#include <string>

namespace universal {
uint64_t Shape::elements() const { uint64_t n=1;for(auto d:dims){if(d&&n>std::numeric_limits<uint64_t>::max()/d)return 0;n*=d;}return n;}
namespace {
bool get(const vegajet::GgufFile& g,const std::string& k,uint64_t& v){return g.get_arch_u64(k,v);}
bool has(const vegajet::GgufFile& g,const std::string& needle){for(const auto&t:g.tensors())if(t.name.find(needle)!=std::string::npos)return true;return false;}
bool layer_has(const vegajet::GgufFile& g,uint64_t layer,const std::string& role){const std::string prefix="blk."+std::to_string(layer)+".";for(const auto&t:g.tensors())if(t.name.compare(0,prefix.size(),prefix)==0&&t.name.find(role)!=std::string::npos)return true;return false;}
}
Model lower_gguf(const vegajet::GgufFile& g,const Options& opt){
 Model m;if(!opt.tp_size||opt.tp_size>4||!opt.n_cu||!(opt.m==1||(opt.m>=2&&opt.m<=16)||(opt.m>=64&&opt.m<=512))){m.error="invalid compile options";return m;}
 if(g.architecture().empty()){m.error="missing general.architecture";return m;}
 get(g,"embedding_length",m.hidden);get(g,"block_count",m.layers);get(g,"attention.head_count",m.heads);get(g,"attention.head_count_kv",m.kv_heads);get(g,"expert_count",m.experts);
 if(!m.hidden||!m.layers){m.error="required hidden-size or layer-count metadata missing";return m;}
 if(!m.heads||m.hidden%m.heads||(m.kv_heads&&(m.kv_heads>m.heads||m.heads%m.kv_heads))){m.error="invalid attention head metadata";return m;}m.head_dim=m.hidden/m.heads;
 for(const auto&t:g.tensors()){Tensor x;x.name=t.name;x.type=t.type;for(uint32_t i=0;i<t.n_dims;++i)x.shape.dims.push_back(t.ne[i]);if(x.shape.dims.empty()||!x.shape.elements()){m.error="invalid tensor shape: "+t.name;return m;}m.tensors.push_back(std::move(x));}
 const bool mla=has(g,"kv_a_proj")||has(g,"kv_lora");const bool linear=has(g,"linear_attn")||has(g,"in_proj_qkv")||has(g,"conv1d")||has(g,"gated_delta");
 m.family=mla?Family::SparseMLAMoE:(linear?Family::HybridGDNMoE:(m.experts>1?Family::PureMoE:Family::DenseGQA));
 for(uint64_t l=0;l<m.layers;++l){const bool ssm=layer_has(g,l,"linear_attn")||layer_has(g,l,"gated_delta")||layer_has(g,l,"conv1d");const bool mla_layer=layer_has(g,l,"kv_a_proj")&&layer_has(g,l,"kv_b_proj");
  if(!ssm&&!layer_has(g,l,"attn_q")&&!layer_has(g,l,"q_proj")){m.error="missing attention/linear tensor role for layer "+std::to_string(l);return m;}
  if(m.family==Family::SparseMLAMoE&&!ssm&&!mla_layer){m.error="incomplete MLA tensor roles for layer "+std::to_string(l);return m;}
  if(m.experts>1&&!(layer_has(g,l,"ffn_gate")||layer_has(g,l,"ffn_up")||layer_has(g,l,"gate_exps"))){m.error="missing MoE gate tensor role for layer "+std::to_string(l);return m;}
  if(m.experts>1&&!(layer_has(g,l,"ffn_down")||layer_has(g,l,"down_exps"))){m.error="missing MoE down tensor role for layer "+std::to_string(l);return m;}
  if(m.experts<=1&&!(layer_has(g,l,"ffn_gate")||layer_has(g,l,"gate_proj")||layer_has(g,l,"ffn_up"))){m.error="missing dense feed-forward gate tensor role for layer "+std::to_string(l);return m;}
  if(m.experts<=1&&!(layer_has(g,l,"ffn_down")||layer_has(g,l,"down_proj"))){m.error="missing dense feed-forward down tensor role for layer "+std::to_string(l);return m;}
 }
 size_t value=1;auto add=[&](OpKind k,const std::string& tag,std::vector<size_t> ins){Op o;o.kind=k;o.tag=tag;o.inputs=std::move(ins);o.outputs.push_back(value);m.ops.push_back(std::move(o));return value++;};
 size_t residual=m.input_value;
 for(uint64_t l=0;l<m.layers;++l){const bool ssm=layer_has(g,l,"linear_attn")||layer_has(g,l,"gated_delta")||layer_has(g,l,"conv1d");const bool mla_layer=layer_has(g,l,"kv_a_proj")&&layer_has(g,l,"kv_b_proj");
  size_t norm=add(OpKind::RMSNorm,"attn_norm",{residual});size_t attended;
  if(ssm){attended=add(OpKind::SSMRecurrence,"linear_attention",{norm});}
  else{size_t qkv=add(OpKind::Linear,"qkv",{norm});size_t q=qkv;if(layer_has(g,l,"q_norm"))q=add(OpKind::QKNorm,"qk_norm",{qkv});size_t rope=add(OpKind::RoPE,"rope",{q});size_t kv=add(OpKind::PagedKVWrite,"paged_kv",{rope});attended=add(mla_layer?OpKind::AttentionMLA:OpKind::AttentionGQA,mla_layer?"mla":"gqa",{qkv,kv});}
  size_t attn_res=add(OpKind::ResidualAdd,"attn_residual",{residual,attended});size_t ffn_input=add(OpKind::RMSNorm,"ffn_norm",{attn_res});size_t ffn;
  if(m.experts>1){size_t route=add(OpKind::MoERouter,"router",{ffn_input});std::vector<size_t> gate_inputs{ffn_input,route};if(layer_has(g,l,"shared_expert")){size_t shared=add(OpKind::Linear,"shared_expert",{ffn_input,route});gate_inputs.push_back(shared);}size_t gate=add(OpKind::Linear,"gate_up",gate_inputs);size_t act=add(OpKind::ActMul,"expert_actmul",{gate});ffn=add(OpKind::MoEFFN,"down",{act});}
  else{size_t gate=add(OpKind::Linear,"gate_up",{ffn_input});size_t act=add(OpKind::ActMul,"actmul",{gate});ffn=add(OpKind::Linear,"down",{act});}
  size_t reduced=add(OpKind::AllReduce,"tp_reduce",{ffn});size_t post=add(OpKind::ResidualAdd,"ffn_residual",{attn_res,reduced});size_t out=add(OpKind::RMSNorm,"post_norm",{post});residual=add(OpKind::Quantize,"activation",{out});add(OpKind::GridBarrier,"layer_boundary",{residual});
 }
 for(auto&o:m.ops)if(o.kind==OpKind::Linear||o.kind==OpKind::MoEFFN){o.grid_x=4*opt.n_cu;o.grid_y=opt.m;o.split_k=static_cast<uint32_t>(std::max<uint64_t>(1,(m.hidden+255)/256));}
 return m;
}
} // namespace universal

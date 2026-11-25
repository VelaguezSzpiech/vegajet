#include "core/workspace_packer.hpp"
#include <algorithm>
#include <limits>
#include <unordered_map>
namespace universal { namespace {
constexpr uint64_t kBufferAlignment=256;
bool align_up(uint64_t x,uint64_t a,uint64_t& y){if(!a||x>std::numeric_limits<uint64_t>::max()-(a-1))return false;y=((x+a-1)/a)*a;return true;}
}
WorkspacePlan pack_workspace(const std::vector<BufferRequest>& req,const WorkspaceShape& sh,uint64_t alignment){WorkspacePlan p;if(!(sh.m==1||(sh.m>=2&&sh.m<=16)||(sh.m>=64&&sh.m<=512))||sh.seq_len<512||sh.seq_len>131072||!alignment){p.error="unsupported symbolic shape or alignment";return p;}
 for(const auto& r:req){if(r.birth>r.death){p.error="invalid buffer lifetime";return p;}uint64_t n=r.shape.bytes_per_token;if(r.shape.scales_with_seq){if(sh.seq_len&&n>std::numeric_limits<uint64_t>::max()/sh.seq_len){p.error="buffer size overflow";return p;}n*=sh.seq_len;}if(!r.shape.scales_with_seq||r.shape.scales_with_m){if(sh.m&&n>std::numeric_limits<uint64_t>::max()/sh.m){p.error="buffer size overflow";return p;}n*=sh.m;}if(r.shape.element_bytes&&n>std::numeric_limits<uint64_t>::max()/r.shape.element_bytes){p.error="buffer size overflow";return p;}n*=r.shape.element_bytes;if(n>std::numeric_limits<uint64_t>::max()-p.naive_bytes){p.error="naive workspace size overflow";return p;}p.naive_bytes+=n;p.buffers.push_back({r.name,0,n,r.birth,r.death});}
 // First-fit interval coloring. Candidate offsets are arena end or ends of buffers that overlap in time.
 for(size_t i=0;i<p.buffers.size();++i){auto& b=p.buffers[i];std::vector<uint64_t> candidates{0};for(size_t j=0;j<i;++j){const auto& q=p.buffers[j];if(!(b.death<q.birth||q.death<b.birth))candidates.push_back(q.offset+q.bytes);}std::sort(candidates.begin(),candidates.end());candidates.erase(std::unique(candidates.begin(),candidates.end()),candidates.end());bool placed=false;for(uint64_t c:candidates){uint64_t off=0;if(!align_up(c,kBufferAlignment,off)||b.bytes>std::numeric_limits<uint64_t>::max()-off)continue;bool clash=false;for(size_t j=0;j<i;++j){const auto& q=p.buffers[j];const bool life=!(b.death<q.birth||q.death<b.birth);const bool mem=off<q.offset+q.bytes&&q.offset<off+b.bytes;if(life&&mem){clash=true;break;}}if(!clash){b.offset=off;p.arena_bytes=std::max(p.arena_bytes,off+b.bytes);placed=true;break;}}if(!placed){p.error="workspace placement overflow";return p;}}
 uint64_t aligned=0;if(!align_up(p.arena_bytes,alignment,aligned)){p.error="arena alignment overflow";return p;}p.arena_bytes=aligned;
 return p;}

WorkspacePlan pack_workspace(const Model& model,const WorkspaceShape& shape,uint64_t alignment){
 if(!model.error.empty()||model.ops.empty()||!model.hidden){WorkspacePlan p;p.error="invalid graph for workspace planning";return p;}
 std::unordered_map<size_t,size_t> index;std::unordered_map<size_t,BufferRequest> requests;std::unordered_map<size_t,bool> used;
 for(size_t i=0;i<model.ops.size();++i)for(size_t out:model.ops[i].outputs){if(!out||!index.emplace(out,i).second){WorkspacePlan p;p.error="duplicate or reserved graph output ID";return p;}requests.emplace(out,BufferRequest{"value_"+std::to_string(out),i,i,{model.hidden,4,false,false}});used[out]=false;}
 for(size_t i=0;i<model.ops.size();++i)for(size_t in:model.ops[i].inputs){if(in==model.input_value)continue;const auto producer=index.find(in);if(producer==index.end()){WorkspacePlan p;p.error="graph use has no producer";return p;}requests[in].death=i;used[in]=true;}
 const size_t end=model.ops.size()-1;std::vector<size_t> ids;for(const auto&entry:requests)ids.push_back(entry.first);std::sort(ids.begin(),ids.end());std::vector<BufferRequest> req;for(size_t id:ids){auto&entry=*requests.find(id);const size_t producer=index[id];const auto&op=model.ops[producer];if(op.kind==OpKind::PagedKVWrite||op.kind==OpKind::GridBarrier)continue;if(!used[id])entry.second.death=end;req.push_back(entry.second);const auto kind=op.kind;if(kind==OpKind::AttentionGQA||kind==OpKind::AttentionMLA)req.push_back({"attention_scratch_"+std::to_string(producer),producer,producer,{model.hidden,4,true,true}});}
 return pack_workspace(req,shape,alignment);
}
bool workspace_overlap_free(const WorkspacePlan& p,size_t step){if(!p.valid())return false;for(size_t i=0;i<p.buffers.size();++i)for(size_t j=i+1;j<p.buffers.size();++j){const auto&a=p.buffers[i];const auto&b=p.buffers[j];if(step>=a.birth&&step<=a.death&&step>=b.birth&&step<=b.death&&a.offset<b.offset+b.bytes&&b.offset<a.offset+a.bytes)return false;}return true;}
} // namespace universal

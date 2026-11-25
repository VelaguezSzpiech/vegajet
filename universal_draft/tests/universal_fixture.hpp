#pragma once
#include "tests/test_common.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>
namespace universal_test {
enum class FixtureFamily { Dense, MoE, Hybrid, MLA };
inline vjtest::GgufWriter tiny_gguf(FixtureFamily family){
 vjtest::GgufWriter w;const std::string a="fixture";w.kv_str("general.architecture",a);w.kv_u32(a+".embedding_length",64);w.kv_u32(a+".block_count",2);w.kv_u32(a+".attention.head_count",4);w.kv_u32(a+".attention.head_count_kv",2);if(family!=FixtureFamily::Dense)w.kv_u32(a+".expert_count",4);
 auto tensor=[&](const std::string& n,std::vector<uint64_t> dims){uint64_t count=1;for(auto d:dims)count*=d;w.tensor(n,std::move(dims),0,std::vector<uint8_t>(static_cast<size_t>(count*4),0));};
 for(unsigned l=0;l<2;++l){const std::string p="blk."+std::to_string(l)+".";if(family==FixtureFamily::Hybrid&&l==0)tensor(p+"linear_attn.in_proj_qkv.weight",{64,64});else tensor(p+"attn_q.weight",{64,64});if(family==FixtureFamily::MLA){tensor(p+"attn_kv_a_proj.weight",{64,64});tensor(p+"attn_kv_b_proj.weight",{64,64});}if(family==FixtureFamily::Dense&&l==1)tensor(p+"attn_q_norm.weight",{16});
  if(family!=FixtureFamily::Dense){tensor(p+"ffn_gate_exps.weight",{64,64,4});tensor(p+"ffn_down_exps.weight",{64,64,4});if(family==FixtureFamily::MoE&&l==0)tensor(p+"ffn_shared_expert.weight",{64,64});}else{tensor(p+"ffn_gate.weight",{64,64});tensor(p+"ffn_down.weight",{64,64});}
 }
 return w;
}
inline std::string write_gguf(const vjtest::GgufWriter& writer){const char* env=std::getenv("TMPDIR");std::string dir=(env&&*env)?env:".";if(dir.back()!='/')dir+='/';std::string pattern=dir+"universal-gguf-XXXXXX";std::vector<char> name(pattern.begin(),pattern.end());name.push_back('\0');int fd=::mkstemp(name.data());if(fd<0)return {};::close(fd);auto bytes=writer.build();FILE*f=std::fopen(name.data(),"wb");if(!f){::unlink(name.data());return {};}const bool wrote=std::fwrite(bytes.data(),1,bytes.size(),f)==bytes.size();const bool closed=std::fclose(f)==0;if(!wrote||!closed){::unlink(name.data());return {};}return name.data();}
inline std::string write_tiny_gguf(FixtureFamily family){return write_gguf(tiny_gguf(family));}
} // namespace universal_test

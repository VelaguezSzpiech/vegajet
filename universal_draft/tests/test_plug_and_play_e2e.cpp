#include "core/tiny_execution.hpp"
#include "core/fusion_pass.hpp"
#include "core/onboard_shard.hpp"
#include "core/universal_quant.hpp"
#include "tests/universal_fixture.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <unistd.h>

#define CHECK(x) do { if(!(x)){std::fprintf(stderr,"check failed: %s:%d %s\n",__FILE__,__LINE__,#x);return 1;} } while(0)

namespace {
using universal::onboard::Device;
constexpr uint64_t GiB=uint64_t{1}<<30;
struct Case {
    const char* name; uint64_t params, layers; uint32_t heads, kv_heads, head_dim;
    universal::Family family; bool feasible[4];
};
std::vector<std::vector<Device>> hardware(){
    return {
        {{"MI50",16*GiB,1.0}},
        {{"MI50-1080",32*GiB,1080.0},{"MI50-1140",32*GiB,1140.0}},
        {{"MI50-0",32*GiB,1.0},{"MI50-1",32*GiB,1.0},{"MI50-2",32*GiB,1.0}},
        {{"MI60-0",32*GiB,1.0},{"MI60-1",32*GiB,1.0}}
    };
}

}
int main(){
    using namespace universal;
    const Case cases[]={
        {"Dense-32B-GQA",32'000'000'000ull,64,64,8,128,Family::DenseGQA,{false,true,true,true}},
        {"Pure-MoE-80B",80'000'000'000ull,64,64,8,128,Family::PureMoE,{false,true,true,true}},
        {"Hybrid-GDN-MoE-122B",122'000'000'000ull,80,64,8,128,Family::HybridGDNMoE,{false,false,true,false}},
        {"DeepSeek-MLA-MoE-320B",320'000'000'000ull,64,128,1,576,Family::SparseMLAMoE,{false,false,false,false}}
    };
    const auto devices=hardware();unsigned combinations=0, sharded_cases=0, reference_families=0;
    const universal_test::FixtureFamily fixtures[]={universal_test::FixtureFamily::Dense,universal_test::FixtureFamily::MoE,
        universal_test::FixtureFamily::Hybrid,universal_test::FixtureFamily::MLA};
    for(size_t f=0;f<4;++f){
        const std::string path=universal_test::write_tiny_gguf(fixtures[f]);CHECK(!path.empty());
        vegajet::GgufFile gg;CHECK(gg.open(path));
        for(size_t h=0;h<devices.size();++h){
            Options opt;opt.tp_size=static_cast<uint32_t>(devices[h].size());opt.n_cu=(h==3?64:60);
            Model m=lower_gguf(gg,opt);CHECK(m.valid()&&m.family==cases[f].family);
            onboard::Profile profile;profile.name=cases[f].name;profile.parameters=cases[f].params;profile.layers=cases[f].layers;
            profile.bits_per_weight=4.5;profile.context_tokens=4096;profile.heads=cases[f].heads;
            profile.kv_heads=cases[f].kv_heads;profile.head_dim=cases[f].head_dim;
            auto p=onboard::plan(m,profile,devices[h]);CHECK(p.full_resident);
            CHECK(p.model_weight_bytes==static_cast<uint64_t>(std::ceil(cases[f].params*4.5L/8.0L)));
            CHECK(p.gpus.size()==devices[h].size());uint64_t weight_sum=0,kv_sum=0;
            for(size_t i=0;i<p.gpus.size();++i){const auto&s=p.gpus[i];CHECK(s.arena_bytes>0&&s.arena_bytes%(2u<<20)==0&&s.resident_bytes==s.weight_bytes+s.arena_bytes+s.kv_bytes);
                CHECK(s.resident_bytes<=devices[h][i].vram_bytes||!p.feasible);weight_sum+=s.weight_bytes;kv_sum+=s.kv_bytes;}
            CHECK(weight_sum==p.model_weight_bytes&&kv_sum==p.kv_bytes);CHECK(p.feasible==cases[f].feasible[h]);
            if(h==1)CHECK(p.gpus[1].weight_bytes>p.gpus[0].weight_bytes);
            if(!p.feasible)CHECK(!p.error.empty()&&p.model_weight_bytes>0);
            if(cases[f].family==Family::DenseGQA){uint32_t nextq=0,nextkv=0;for(const auto&s:p.gpus){
                CHECK(s.q_head_begin==nextq&&s.kv_head_begin==nextkv);
                CHECK(s.q_head_end-s.q_head_begin==8*(s.kv_head_end-s.kv_head_begin));
                nextq=s.q_head_end;nextkv=s.kv_head_end;}
                CHECK(nextq==cases[f].heads&&nextkv==cases[f].kv_heads);
            }
            if(cases[f].family==Family::SparseMLAMoE){double fractions=0;for(const auto&s:p.gpus){CHECK(s.sequence_fraction>0);fractions+=s.sequence_fraction;}
                CHECK(std::fabs(fractions-1.0)<1e-12);}
            PassManager{}.run(m,{opt.tp_size,opt.n_cu,1,false});
            std::string error;
            if(h==0){
                // This is explicitly a TP1 family reference, not distributed evidence.
                TinyExecutionResult numeric;
                if(!run_tiny_execution(m,numeric,&error)){std::fprintf(stderr,"TP1 family %zu: %s\n",f,error.c_str());return 1;}
                CHECK(numeric.emitted_ops>2&&numeric.checked_values>=m.hidden&&numeric.output.size()==m.hidden);
                CHECK(numeric.routing_ids==numeric.reference_routing_ids);++reference_families;
            }
            if(f==0&&(h==1||h==2)){
                // Numeric fixture uses its OWN geometry/footprint, not the 32B capacity profile.
                onboard::Profile execution_profile;execution_profile.context_tokens=3;execution_profile.bits_per_weight=0;
                const auto execution_plan=onboard::plan(m,execution_profile,devices[h]);CHECK(execution_plan.feasible);
                ShardedTinyResult distributed;
                if(!run_sharded_tiny_execution(m,execution_plan,distributed,&error)){
                    std::fprintf(stderr,"distributed dense TP%zu: %s\n",devices[h].size(),error.c_str());return 1;
                }
                CHECK(distributed.rank_ops.size()==devices[h].size());
                CHECK(distributed.host_reductions==2&&distributed.attention_gathers==2);
                CHECK(distributed.numeric.output.size()==m.hidden&&distributed.numeric.checked_values>=m.hidden);
                size_t next=0,total_heads=0;
                for(size_t r=0;r<distributed.rank_ops.size();++r){
                    CHECK(distributed.rank_ops[r]==m.ops.size());
                    CHECK(distributed.ffn_begin[r]==next&&distributed.ffn_end[r]>next);
                    next=distributed.ffn_end[r];total_heads+=distributed.q_heads[r];
                }
                CHECK(next==m.hidden&&total_heads==m.heads);
                if(h==1)CHECK(distributed.ffn_end[0]<distributed.ffn_end[1]-distributed.ffn_begin[1]);
                if(h==2)CHECK(distributed.q_heads[2]==0); // Q4/KV2: only two whole attention groups.
                ShardedTinyResult wrong;
                CHECK(!run_sharded_tiny_execution(m,execution_plan,wrong,&error,true));
                CHECK(error.find("sharded numeric mismatch")!=std::string::npos);
                auto invalid=execution_plan;invalid.gpus[1].q_head_begin=0;
                CHECK(!run_sharded_tiny_execution(m,invalid,wrong,&error));
                CHECK(error=="invalid tiny rank head slices");
                std::printf("[PASS] %s / TP%zu: CAPACITY + synthetic sharded dense graph, %zu reductions, %zu gathers, max error %.9g\n",
                    cases[f].name,devices[h].size(),distributed.host_reductions,distributed.attention_gathers,
                    static_cast<double>(distributed.numeric.max_abs_error));
                ++sharded_cases;
            }else{
                std::printf("[CAPACITY-ONLY] %s / profile %zu (%s)\n",cases[f].name,h,p.feasible?"fits":"infeasible");
            }
            ++combinations;
        }
        gg.close();::unlink(path.c_str());
    }
    CHECK(combinations==16);
    CHECK(sharded_cases==2&&reference_families==4);
    const std::string path=universal_test::write_tiny_gguf(universal_test::FixtureFamily::Dense);CHECK(!path.empty());
    vegajet::GgufFile gg;CHECK(gg.open(path));Model tiny=lower_gguf(gg,{});CHECK(tiny.valid());
    onboard::Profile small;small.context_tokens=32;small.bits_per_weight=0;
    const auto bytes=onboard::full_weight_bytes(tiny,small);CHECK(bytes>0&&bytes<1024*1024);
    auto fits=onboard::plan(tiny,small,{{"tiny-device",64*1024*1024,1.0}});CHECK(fits.feasible&&fits.model_weight_bytes==bytes);
    // Different mixed-quant tensor table, same 122B parameter count. No external weights.
    auto mixed=tiny;mixed.tensors.clear();mixed.tensors.push_back({"experts",{{256,437500000}},22});
    mixed.tensors.push_back({"dense",{{32,312500000}},8});
    onboard::Profile mix;mix.context_tokens=4096;mix.layers=80;mix.heads=64;mix.kv_heads=8;mix.head_dim=128;
    const auto concrete=onboard::plan(mixed,mix,devices[1]);CHECK(concrete.feasible);
    CHECK(concrete.model_weight_bytes==437500000ull*82+312500000ull*34);
    auto oom=onboard::plan(tiny,small,{{"undersized",bytes-1,1.0}});CHECK(!oom.feasible&&!oom.error.empty());
    CHECK(!onboard::plan(tiny,small,{}).feasible);gg.close();::unlink(path.c_str());
    std::printf("[PASS] plug-and-play: %u capacity cells; %u synthetic dense rank executions (TP2 asymmetric, TP3), wrong-slice controls; %u independent TP1 family checks. Other cells CAPACITY-ONLY. QKV replicated; TP3 has one idle GQA rank, all FFN ranks active.\n",combinations,sharded_cases,reference_families);
    return 0;
}

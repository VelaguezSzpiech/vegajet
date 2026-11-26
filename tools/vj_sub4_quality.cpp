// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// CPU only: deterministic sampled-row weight error and synthetic activation KL proxies.
#include "core/gguf_reader.hpp"
#include "core/quant_repacker.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
using namespace vegajet;
using namespace vegajet::quant;
static double kl(const std::vector<float>& a,const std::vector<float>& b) {
    double ma=*std::max_element(a.begin(),a.end()),mb=*std::max_element(b.begin(),b.end()),sa=0,sb=0;
    for(size_t i=0;i<a.size();++i) { sa+=std::exp(a[i]-ma); sb+=std::exp(b[i]-mb); }
    double out=0;
    for(size_t i=0;i<a.size();++i) out+=std::exp(a[i]-ma)/sa*((a[i]-ma-std::log(sa))-(b[i]-mb-std::log(sb)));
    return std::max(0.,out);
}
int main(int argc,char** argv) try {
    if(argc<2) { std::fprintf(stderr,"Usage: vj_sub4_quality <model-file> [--max-tensors N]\n"); return 2; }
    unsigned max_tensors=4;
    for(int i=2;i<argc;++i) {
        if(std::string(argv[i])!="--max-tensors" || ++i==argc) throw std::runtime_error("expected --max-tensors N");
        std::string n=argv[i]; size_t used=0; auto value=std::stoul(n,&used);
        if(used!=n.size() || !value || value>10000) throw std::runtime_error("invalid tensor limit");
        max_tensors=unsigned(value);
    }
    GgufFile f; if(!f.open(argv[1])) throw std::runtime_error(f.error());
    auto start=std::chrono::steady_clock::now(); unsigned count=0;
    std::mt19937 rng(12345); std::normal_distribution<float> normal;
    std::puts("CPU proxies only; 64 evenly spaced rows/tensor, 8 Gaussian probes, softmax logits /sqrt(K); not held-out model KL.");
    for(const auto& t:f.tensors()) {
        if(count>=max_tensors) break;
        if(std::chrono::steady_clock::now()-start>std::chrono::seconds(240)) { std::puts("Stopped at 240s sample budget"); break; }
        // Dense trunk/shared projections, excluding nextn, routed experts, embedding and head.
        if(t.n_dims!=2 || t.type!=uint32_t(GgmlType::Q4_K) || t.name.rfind("blk.",0)!=0 ||
           t.name.rfind("blk.48.",0)==0 || t.ne[0]%64 || t.ne[0]>32768 || !t.ne[1]) continue;
        unsigned R=unsigned(std::min<uint64_t>(64,t.ne[1])),C=unsigned(t.ne[0]);
        auto* raw=f.tensor_data(t); if(!raw) throw std::runtime_error(f.error());
        auto& reg=DecoderRegistry::instance(); uint64_t rb=reg.row_bytes(GgmlType::Q4_K,C);
        std::vector<float> w(size_t(R)*C),dq(w.size());
        for(unsigned r=0;r<R;++r) {
            uint64_t src=R==1?0:uint64_t(r)*(t.ne[1]-1)/(R-1);
            if(!reg.decode_row_f32(GgmlType::Q4_K,raw+src*rb,w.data()+size_t(r)*C,C)) throw std::runtime_error("decode failed");
        }
        f.release(raw,t.n_bytes()); ++count;
        std::vector<float> probes(8*C); for(auto& v:probes) v=normal(rng)/std::sqrt(float(C));
        double norm=0; for(float v:w) norm+=double(v)*v;
        auto report=[&](const char* label,const MiloResidual* correction) {
            double error=0,kld=0;
            // Reconstruct factor product directly for Frobenius; no R*C temporary correction matrix.
            for(unsigned r=0;r<R;++r) for(unsigned c=0;c<C;++c) {
                double value=dq[size_t(r)*C+c];
                if(correction) for(unsigned k=0;k<correction->rank;++k)
                    value+=double(milo_factor(*correction,true,size_t(r)*correction->rank+k))*milo_factor(*correction,false,size_t(k)*C+c);
                double e=w[size_t(r)*C+c]-value; error+=e*e;
            }
            for(unsigned probe=0;probe<8;++probe) {
                const float* x=probes.data()+probe*C; std::vector<float> a(R),b(R);
                for(unsigned r=0;r<R;++r) {
                    double av=0,bv=0; for(unsigned c=0;c<C;++c) { av+=double(w[size_t(r)*C+c])*x[c]; bv+=double(dq[size_t(r)*C+c])*x[c]; }
                    a[r]=float(av); b[r]=float(bv);
                }
                if(correction) apply_milo(*correction,x,b.data());
                kld+=kl(a,b)/8;
            }
            std::printf("%s sampled=%ux%u %-15s rel_frob=%.8g KL_proxy=%.8g\n",t.name.c_str(),R,C,label,std::sqrt(error/std::max(norm,1e-30)),kld);
        };
        dequantize_vj_q4_a8(pack_f32_to_vj_q4_a8(w.data(),R,C),dq.data()); report("Q4",nullptr);
        for(auto fmt:{NativeFormat::VJ_Q3_DOT8,NativeFormat::VJ_Q2_DOT8}) {
            auto p=pack_f32_to_vj_sub4(w.data(),R,C,fmt); dequantize_vj_sub4(p,dq.data());
            std::string label=fmt==NativeFormat::VJ_Q3_DOT8?"Q3":"Q2"; report(label.c_str(),nullptr);
            for(unsigned rank:{16u,32u}) for(auto storage:{MiloStorage::Fp16,MiloStorage::Int3}) {
                auto factors=fit_milo(w.data(),p,rank,24,storage);
                auto name=label+"+MiLo"+std::to_string(rank)+(storage==MiloStorage::Int3?"/i3":"/f16"); report(name.c_str(),&factors);
            }
        }
    }
    if(!count) throw std::runtime_error("no compatible trunk Q4_K matrices found");
    return 0;
} catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }

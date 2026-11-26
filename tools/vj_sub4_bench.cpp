// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// GPU benchmark: BUILD ONLY during CPU/offline development. Never a ctest.
#include "core/kfd_runtime.hpp"
#include "core/quant_repacker.hpp"
#include "ops/sub4_launch.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
using namespace vegajet;
static void check(bool ok) { if(!ok) throw std::runtime_error("KFD operation failed"); }
int main(int argc,char** argv) try {
    if(argc!=2) { std::fprintf(stderr,"Usage: vj_sub4_bench HSACO_DIRECTORY\n"); return 2; }
    constexpr unsigned N=4096,K=4096,repeats=20;
    kfd::Runtime rt; std::string err;
    if(!rt.init(.5,16,&err)) throw std::runtime_error(err);
    for(const char* file:{"vj_kfd_base.hsaco","vj_quant_ops.hsaco","vj_sub4_dot8.hsaco"})
        if(!rt.load_hsaco(0,std::string(argv[1])+"/"+file,&err)) throw std::runtime_error(err);
    auto& arena=rt.vram_arena(0);
    auto upload=[&](const auto& v) {
        size_t bytes=v.size()*sizeof(v[0]); uint64_t va=arena.alloc(bytes,256);
        check(va!=0); check(rt.upload_to_vram(0,va,v.data(),bytes)); return va;
    };
    std::vector<float> w(size_t(N)*K);
    for(size_t i=0;i<w.size();++i) w[i]=std::sin(float(i)*.007f);
    for(unsigned M:{1u,6u}) {
        std::vector<int8_t> x(M*K); std::vector<uint32_t> xp(M*K/8);
        std::vector<float> sc(M*K/32,.03125f),ds(M*K/16),ref(M*N),out(M*N);
        for(size_t i=0;i<x.size();++i) { x[i]=int(i%15)-7; xp[i/8]|=uint32_t(x[i]&15)<<(4*(i%8)); }
        for(size_t b=0;b<sc.size();++b) {
            int sum=0; for(unsigned j=0;j<32;++j) sum+=x[b*32+j];
            ds[b*2]=sc[b]; ds[b*2+1]=sc[b]*sum;
        }
        auto xv=upload(xp),dv=upload(ds),sv=upload(sc); uint64_t yv=arena.alloc(out.size()*4,256);
        for(unsigned bits:{4u,3u,2u}) {
            uint64_t wv,mv; size_t bytes;
            if(bits==4) {
                auto p=quant::pack_f32_to_vj_q4_dot8(w.data(),N,K);
                wv=upload(p.qs); mv=upload(p.ds); bytes=p.total_bytes();
                std::vector<float> dq(w.size()); quant::dequantize_vj_q4_dot8(p,dq.data());
                for(unsigned m=0;m<M;++m) for(unsigned r=0;r<N;++r) {
                    double a=0; for(unsigned c=0;c<K;++c) a+=double(dq[size_t(r)*K+c])*x[m*K+c]*sc[m*K/32+c/32];
                    ref[m*N+r]=float(a);
                }
            } else {
                auto p=quant::pack_f32_to_vj_sub4(w.data(),N,K,bits==2?quant::NativeFormat::VJ_Q2_DOT8:quant::NativeFormat::VJ_Q3_DOT8);
                wv=upload(p.qs); mv=upload(p.dm); bytes=p.total_bytes()-16;
                quant::gemv_vj_sub4(p,x.data(),sc.data(),ref.data(),M);
            }
            std::string name=bits==4?(M==1?"vj_gemv_q4_dot8_kernel":"vj_gemm_q4_dot8_tiled_kernel"):
                "vj_sub4_q"+std::to_string(bits)+(M==1?"_gemv":"_gemm");
            auto* kernel=rt.get_kernel(0,name); check(kernel!=nullptr);
            auto launch=[&] {
                if(bits==4) {
                    struct Args { uint64_t w,d,x,s,y; uint32_t n,k,m,pad; } a{wv,mv,xv,sv,yv,N,K,M,0};
                    check(rt.enqueue_kernel(0,*kernel,M==1?((N+3)/4)*256:((N+31)/32)*256,1,1,256,1,1,&a,M==1?48:56));
                } else {
                    sub4::Kernarg a{wv,mv,xv,dv,yv,N,K,64,M};
                    check(rt.enqueue_kernel(0,*kernel,N*64,M,1,64,1,1,&a,sizeof(a)));
                }
                check(rt.submit_batch(0)); check(rt.wait_idle(0));
            };
            launch(); // warmup, outside timing
            auto start=std::chrono::steady_clock::now();
            for(unsigned it=0;it<repeats;++it) launch();
            double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
            check(rt.download_from_vram(0,out.data(),yv,out.size()*4));
            float error=0;
            for(size_t i=0;i<out.size();++i) {
                if(!std::isfinite(out[i]) || !std::isfinite(ref[i]))
                    throw std::runtime_error("nonfinite GPU or CPU output");
                error=std::max(error,std::abs(out[i]-ref[i]));
            }
            std::printf("Q%u M=%u weight-effective GB/s=%.3f latency_us=%.3f GPU-vs-CPU max_abs_error=%.9g\n",bits,M,bytes/seconds/1e9,seconds*1e6,error);
        }
    }
    return 0;
} catch(const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }

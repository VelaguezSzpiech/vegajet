// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Live GPU tool. BUILD ONLY during offline development; never registered in ctest.
#include "core/kfd_runtime.hpp"
#include "core/kv_quant_oracle.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
using namespace vegajet;
using namespace vegajet::kvq;
static void check(bool ok) { if(!ok) throw std::runtime_error("KFD operation failed"); }
int main(int argc,char** argv) {
    if(argc<3 || std::strcmp(argv[1],"--run-gpu")!=0) {
        std::fprintf(stderr,"usage: vj_kvq_bench --run-gpu BUILD/hsaco [--65536] [--dim256] [--chronological]\n");return 2;
    }
    try {
        bool large=false,chronological=false;unsigned d=128;
        for(int i=3;i<argc;++i) {if(!std::strcmp(argv[i],"--65536"))large=true;else if(!std::strcmp(argv[i],"--dim256"))d=256;else if(!std::strcmp(argv[i],"--chronological"))chronological=true;else throw std::runtime_error("unknown option");}
        kfd::Runtime rt;std::string err;
        if(!rt.init(0.5,16,&err)) throw std::runtime_error(err);
        check(rt.load_hsaco(0,std::string(argv[2])+"/vj_kfd_base.hsaco",&err));
        check(rt.load_hsaco(0,std::string(argv[2])+"/vj_kvq.hsaco",&err));
        check(rt.load_hsaco(0,std::string(argv[2])+"/vj_kv_flash_ops.hsaco",&err));
        auto* part=rt.get_kernel(0,"vj_kvq_grouped_partial_kernel");
        auto* merge=rt.get_kernel(0,"vj_flash_decode_lse_merge_kernel");
        auto* append=rt.get_kernel(0,"vj_kvq_append_kernel");
        auto* clock=rt.get_kernel(0,"kfd_stamp_clock_u64");
        check(part && merge && append && clock);
        constexpr unsigned nh=12;const unsigned ns=split_count(1,chronological), max_t=large?65536:16384;
        auto& arena=rt.vram_arena(0);
        auto alloc=[&](uint64_t n){auto a=arena.alloc(n,256);check(a!=0);return a;};
        const auto qva=alloc(nh*d*4),kva=alloc(uint64_t(max_t)*2*d),vva=alloc(uint64_t(max_t)*2*d);
        const auto pva=alloc(nh*ns*d*4),lva=alloc(nh*ns*4),ova=alloc(nh*d*4),rva=alloc(8),sva=alloc(d),iva=alloc(d*4);
        const auto clockva=alloc(16);
        HeadRange range{0,nh};auto signs=turbo3_signs(d,0x5455524230u);
        check(rt.upload_to_vram(0,rva,&range,sizeof(range)));check(rt.upload_to_vram(0,sva,signs.data(),d));
        std::mt19937 rng(41);std::normal_distribution<float> normal;
        std::vector<float> q(nh*d),k(size_t(max_t)*d),v(k.size()),gpu(nh*d);
        for(auto& x:q)x=normal(rng);for(auto& x:k)x=normal(rng);for(auto& x:v)x=normal(rng);
        check(rt.upload_to_vram(0,qva,q.data(),q.size()*4));
        const char* names[]={"fp16","q8_0","q4_0","turbo3"};
        std::printf("format dim context splits latency_us gpu_cpu_max_abs gpu_fp32_max_abs append_max_abs\n");
        for(unsigned fmt=0;fmt<4;++fmt) {
            auto f=DeviceFormat(fmt);const unsigned bytes=vector_bytes(f,d);
            std::vector<uint8_t> pk(size_t(max_t)*bytes),pv(pk.size());
            for(unsigned t=0;t<max_t;++t) {pack_device_vector(k.data()+size_t(t)*d,d,f,pk.data()+size_t(t)*bytes);pack_device_vector(v.data()+size_t(t)*d,d,f,pv.data()+size_t(t)*bytes);}
            check(rt.upload_to_vram(0,kva,pk.data(),pk.size()));check(rt.upload_to_vram(0,vva,pv.data(),pv.size()));
            // Exercise append on last row, outside the attention timing region.
            check(rt.upload_to_vram(0,iva,k.data()+size_t(max_t-1)*d,d*4));
            AppendArgs aa{iva,kva,sva,d,1,max_t-1,fmt};
            check(rt.dispatch_sync(0,*append,256,1,1,256,1,1,&aa,sizeof(aa)));
            std::vector<uint8_t> ap(bytes);check(rt.download_from_vram(0,ap.data(),kva+size_t(max_t-1)*bytes,bytes));
            std::vector<float> ax(d),cx(d);unpack_device_vector(ap.data(),d,f,ax.data());unpack_device_vector(pk.data()+size_t(max_t-1)*bytes,d,f,cx.data());
            double ae=0;for(unsigned i=0;i<d;++i)ae=std::max(ae,double(std::fabs(ax[i]-cx[i])));
            // Restore CPU-packed row so CPU/GPU attention compares identical cache bytes.
            check(rt.upload_to_vram(0,kva+size_t(max_t-1)*bytes,pk.data()+size_t(max_t-1)*bytes,bytes));
            for(unsigned t:{512u,4096u,16384u,65536u}) {
                if(t>max_t)continue;
                PartialArgs pa{qva,kva,vva,rva,sva,pva,lva,d,nh,1,t,ns,fmt,1/std::sqrt(float(d))};
                MergeArgs ma{pva,lva,ova,nh,ns,d};
                auto run=[&](){check(rt.enqueue_kernel(0,*part,ns*256,1,1,256,1,1,&pa,sizeof(pa),true));check(rt.enqueue_kernel(0,*merge,nh*64,1,1,64,1,1,&ma,sizeof(ma),true));};
                check(valid_shape(d,nh,1,t,ns,f,&range));
                for(int i=0;i<3;++i) {run();check(rt.submit_batch(0)&&rt.wait_idle(0));}
                struct ClockArgs {uint64_t dst;uint32_t index,pad;};
                ClockArgs ca{clockva,0,0},cz{clockva,1,0};
                check(rt.enqueue_kernel(0,*clock,64,1,1,64,1,1,&ca,sizeof(ca),true));
                for(int i=0;i<10;++i)run();
                check(rt.enqueue_kernel(0,*clock,64,1,1,64,1,1,&cz,sizeof(cz),true));
                check(rt.submit_batch(0)&&rt.wait_idle(0));
                uint64_t ts[2]={};check(rt.download_from_vram(0,ts,clockva,sizeof(ts)));
                // gfx906 s_memrealtime: 25 MHz, same conversion as Ornith's existing microbench.
                double us=double(ts[1]-ts[0])*0.04/10;
                check(rt.download_from_vram(0,gpu.data(),ova,gpu.size()*4));
                double error=0,fp32err=0;
                for(unsigned h=0;h<nh;++h) {
                    auto cpu=device_attention_reference(q.data()+h*d,pk.data(),pv.data(),d,t,f);
                    // Independent fp32 two-pass attention on original input.
                    std::vector<double> scores(t);double peak=-INFINITY,z=0;
                    for(unsigned j=0;j<t;++j) {double s=0;for(unsigned i=0;i<d;++i)s+=double(q[h*d+i])*k[size_t(j)*d+i];scores[j]=s/std::sqrt(double(d));peak=std::max(peak,scores[j]);}
                    for(auto& s:scores){s=std::exp(s-peak);z+=s;}
                    for(unsigned i=0;i<d;++i) {double out=0;for(unsigned j=0;j<t;++j)out+=scores[j]/z*v[size_t(j)*d+i];error=std::max(error,double(std::fabs(gpu[h*d+i]-cpu[i])));fp32err=std::max(fp32err,std::fabs(gpu[h*d+i]-out));}
                }
                std::printf("%s %u %u %u %.3f %.8g %.8g %.8g\n",names[fmt],d,t,ns,us,error,fp32err,ae);
                if(!std::isfinite(error)||error>0.01)throw std::runtime_error("GPU/CPU error exceeds 0.01");
            }
        }
    } catch(const std::exception& e) {std::fprintf(stderr,"vj_kvq_bench: %s\n",e.what());return 1;}
}

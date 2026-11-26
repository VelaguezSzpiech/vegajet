// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/offload/host.hpp"
#include "core/async_table_prefetch.hpp"
#include "core/q4_dot8_repack.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <linux/mempolicy.h>
#if defined(__x86_64__)
#include <immintrin.h>
#endif
namespace vegajet::offload {
Switches Switches::environment() {
    auto on=[](const char* p){ return p && std::strcmp(p,"1")==0; };
    return {on(std::getenv("VEGAJET_OFFLOAD_L3_CPU")),on(std::getenv("VEGAJET_OFFLOAD_L2_CUDA")),
        on(std::getenv("VEGAJET_ENGRAM_IOURING")),on(std::getenv("VEGAJET_BORROWED_KV")),
        on(std::getenv("VEGAJET_SPEC_HOBBIT"))};
}
HostArena::HostArena(size_t bytes,int node,bool pin) {
    if(!bytes || bytes>std::numeric_limits<size_t>::max()-2*huge_bytes) throw std::invalid_argument("arena size");
    bytes_=(bytes+huge_bytes-1)&~(huge_bytes-1);
    base_=mmap(nullptr,bytes_,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS|MAP_HUGETLB|(21<<MAP_HUGE_SHIFT),-1,0);
    huge_=base_!=MAP_FAILED;
    if(!huge_) {
        void* raw=mmap(nullptr,bytes_+huge_bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        if(raw==MAP_FAILED) throw std::bad_alloc();
        auto a=(reinterpret_cast<uintptr_t>(raw)+huge_bytes-1)&~(huge_bytes-1);
        size_t lead=a-reinterpret_cast<uintptr_t>(raw);
        if(lead) munmap(raw,lead);
        munmap(reinterpret_cast<void*>(a+bytes_),huge_bytes-lead);
        base_=reinterpret_cast<void*>(a); madvise(base_,bytes_,MADV_HUGEPAGE);
    }
    if(node>=0 && node<4096) {
        std::vector<unsigned long> mask(size_t(node)/(8*sizeof(unsigned long))+1);
        mask[size_t(node)/(8*sizeof(unsigned long))] |= 1ul<<(node%(8*sizeof(unsigned long)));
        numa_=syscall(SYS_mbind,base_,bytes_,MPOL_BIND,mask.data(),node+1,0)==0;
    }
    if(pin) pinned_=mlock(base_,bytes_)==0;
}
HostArena::~HostArena() { if(pinned_) munlock(base_,bytes_); if(base_) munmap(base_,bytes_); }
bool Matrix::valid() const {
    return (format==Format::Q4_0 || format==Format::Dot8Iq3Sub32) && weights && rows && cols && cols%32==0 && weight_stride>=size_t(cols/32)*(format==Format::Q4_0?18:16)
        && (format==Format::Q4_0 || (scales && scale_stride>=size_t(cols/32)*4));
}
namespace {
float block(Matrix m,uint32_t r,uint32_t b,int8_t* w) {
    auto p=m.weights+r*m.weight_stride+b*(m.format==Format::Q4_0?18:16);
    float d;
    if(m.format==Format::Q4_0) { uint16_t h; std::memcpy(&h,p,2); d=prefetch::f16_to_f32(h); p+=2; }
    else std::memcpy(&d,m.scales+r*m.scale_stride+b*4,4);
    for(int j=0;j<32;++j) {
        unsigned c=(p[j%16]>>(j/16*4))&15;
        w[j]=m.format==Format::Q4_0?int(c)-8:dot8_iq3_level(c);
    }
    return d;
}
int dot_scalar(const int8_t* w,const int8_t* x) { int s=0; for(int j=0;j<32;++j) s+=int(w[j])*int(x[j]); return s; }
#if defined(__x86_64__)
__attribute__((target("avx2"))) int dot_avx(const uint8_t* packed,Format format,const int8_t* x) {
    const auto codes=_mm_loadu_si128(reinterpret_cast<const __m128i*>(packed));
    const auto mask=_mm_set1_epi8(15);
    const auto palette=_mm_setr_epi8(4,12,20,28,36,44,52,62,-4,-12,-20,-28,-36,-44,-52,-62);
    __m256i sum=_mm256_setzero_si256();
    for(int half=0;half<2;++half) {
        auto nibbles=_mm_and_si128(half?_mm_srli_epi16(codes,4):codes,mask);
        auto levels=format==Format::Q4_0?_mm_sub_epi8(nibbles,_mm_set1_epi8(8)):_mm_shuffle_epi8(palette,nibbles);
        auto w=_mm256_cvtepi8_epi16(levels);
        auto a=_mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(x+half*16)));
        sum=_mm256_add_epi32(sum,_mm256_madd_epi16(w,a));
    }
    alignas(32) int lanes[8]; _mm256_store_si256(reinterpret_cast<__m256i*>(lanes),sum);
    int total=0; for(int value:lanes) total+=value; return total;
}
#endif
void run(Matrix m,Activation x,float* y,bool avx) {
    if(!m.valid() || !x.values || !x.scales || !y) throw std::invalid_argument("GEMV shape/pointers");
    for(uint32_t r=0;r<m.rows;++r) {
        float sum=0;
        for(uint32_t b=0;b<m.cols/32;++b) {
            float d; int dot;
#if defined(__x86_64__)
            if(avx) {
                auto p=m.weights+r*m.weight_stride+b*(m.format==Format::Q4_0?18:16);
                if(m.format==Format::Q4_0) { uint16_t h; std::memcpy(&h,p,2); d=prefetch::f16_to_f32(h); p+=2; }
                else std::memcpy(&d,m.scales+r*m.scale_stride+b*4,4);
                dot=dot_avx(p,m.format,x.values+b*32);
            } else
#else
            (void)avx;
#endif
            { int8_t w[32]; d=block(m,r,b,w); dot=dot_scalar(w,x.values+b*32); }
            sum+=(d*x.scales[b])*float(dot);
        }
        y[r]=sum;
    }
}
}
void gemv_reference(Matrix m,Activation x,float* y) { run(m,x,y,false); }
void gemv(Matrix m,Activation x,float* y) {
#if defined(__x86_64__)
    run(m,x,y,__builtin_cpu_supports("avx2"));
#else
    run(m,x,y,false);
#endif
}
std::vector<Node> host_nodes() {
    cpu_set_t allowed; CPU_ZERO(&allowed); if(sched_getaffinity(0,sizeof(allowed),&allowed)) throw std::runtime_error("sched_getaffinity");
    std::vector<Node> out;
    for(auto& e:std::filesystem::directory_iterator("/sys/devices/system/node")) {
        auto name=e.path().filename().string();
        if(name.size()<5 || name.substr(0,4)!="node" || name.find_first_not_of("0123456789",4)!=std::string::npos) continue;
        Node n{std::stoi(name.substr(4)),{}};
        for(int c=0;c<CPU_SETSIZE;++c) if(CPU_ISSET(c,&allowed) && std::filesystem::exists(e.path()/("cpu"+std::to_string(c)))) n.cpus.push_back(c);
        if(!n.cpus.empty()) out.push_back(std::move(n));
    }
    if(out.empty()) { Node n{0,{}}; for(int c=0;c<CPU_SETSIZE;++c) if(CPU_ISSET(c,&allowed)) n.cpus.push_back(c); out.push_back(n); }
    std::sort(out.begin(),out.end(),[](auto& a,auto& b){return a.id<b.id;}); return out;
}
struct CpuTier::Worker {
    Ring<Job*,64> ring; std::atomic<bool> stop{false}, pinned{false}, ready{false}; std::thread thread;
    explicit Worker(int cpu) { thread=std::thread([this,cpu] {
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu,&set);
        pinned.store(pthread_setaffinity_np(pthread_self(),sizeof(set),&set)==0);
        ready.store(true,std::memory_order_release);
        for(;;) { Job* j=nullptr;
            if(ring.pop(j)) { gemv(j->matrix,j->x,j->y); j->done.store(true,std::memory_order_release); }
            else if(stop.load(std::memory_order_acquire)) break;
            else std::this_thread::yield();
        }
    }); }
    ~Worker() { stop.store(true,std::memory_order_release); thread.join(); }
};
CpuTier::CpuTier(std::vector<Node> nodes) {
    for(auto& n:nodes) if(!n.cpus.empty()) workers_.push_back(std::make_unique<Worker>(n.cpus.front()));
    if(workers_.empty()) throw std::invalid_argument("no worker CPUs");
    for(auto& w:workers_) while(!w->ready.load(std::memory_order_acquire)) std::this_thread::yield();
}
CpuTier::~CpuTier()=default;
bool CpuTier::affinity_ok(size_t w) const { return workers_.at(w)->pinned.load(); }
bool CpuTier::submit(size_t w,Job& j) {
    if(w>=workers_.size() || !j.matrix.valid() || !j.x.values || !j.x.scales || !j.y) return false;
    j.done.store(false,std::memory_order_relaxed); return workers_[w]->ring.push(&j);
}
} // namespace vegajet::offload

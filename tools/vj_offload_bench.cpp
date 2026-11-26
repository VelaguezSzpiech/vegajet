// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/offload/host.hpp"
#include "core/offload/cuda.hpp"
#include "core/offload/engram.hpp"
#include <chrono>
#include <cstring>
#include <iostream>
#include <unistd.h>
using namespace vegajet::offload;
using Clock=std::chrono::steady_clock;
double us(Clock::time_point start) { return std::chrono::duration<double,std::micro>(Clock::now()-start).count(); }
int main(int argc,char** argv) {
    bool cuda=argc==2 && std::strcmp(argv[1],"--cuda")==0;
    const auto switches=Switches::environment();
    constexpr unsigned rows=512,cols=4096,iterations=100;
    HostArena arena(rows*(cols/32*18),host_nodes().front().id);
    auto w=static_cast<uint8_t*>(arena.data());
    for(unsigned i=0;i<rows*cols/32;++i) { w[i*18]=0; w[i*18+1]=0x3c; for(unsigned j=0;j<16;++j) w[i*18+2+j]=uint8_t(i+j); }
    Matrix m{w,nullptr,rows,cols,cols/32*18,0,Format::Q4_0};
    std::vector<int8_t> x(cols,1); std::vector<float> scales(cols/32,0.125f),y(rows),ref(rows);
    gemv_reference(m,{x.data(),scales.data()},ref.data());
    if(switches.l3) {
        CpuTier cpu; Job job; job.matrix=m; job.x={x.data(),scales.data()}; job.y=y.data();
        auto start=Clock::now();
        for(unsigned i=0;i<iterations;++i) { if(!cpu.submit(0,job)) return 2; while(!job.done.load(std::memory_order_acquire)) std::this_thread::yield(); }
        std::cout<<"L3 queued us/expert="<<us(start)/iterations<<" rows="<<rows<<" cols="<<cols
            <<" hugetlb="<<arena.hugetlb()<<" numa_bound="<<arena.numa_bound()<<" affinity="<<cpu.affinity_ok(0)<<'\n';
        if(y!=ref) return 3;
        start=Clock::now(); for(unsigned i=0;i<iterations;++i) gemv(m,{x.data(),scales.data()},y.data());
        std::cout<<"L3 GEMV us/expert="<<us(start)/iterations<<'\n';
    }
    if(switches.engram) {
        char path[]="/tmp/vj-offload-bench-XXXXXX"; int fd=mkstemp(path); if(fd<0) return 4;
        std::vector<uint8_t> file(4096*256,42);
        if(write(fd,file.data(),file.size())!=ssize_t(file.size()) || fsync(fd)) {close(fd);unlink(path);return 5;}
        try {
            Engram io(path,4096,256,16,true); std::vector<uint64_t> ids={0,17,33,49,65,81,97,255};
            auto start=Clock::now();
            for(unsigned i=0;i<iterations;++i) { if(!io.submit(ids)) return 6; while(!io.poll()) std::this_thread::yield(); if(!io.success()) return 7; }
            std::cout<<"Engram us/step="<<us(start)/iterations<<" rows/step="<<ids.size()<<" backend="<<io.backend()<<" direct="<<io.direct()<<" pinned="<<io.pinned()<<" registered="<<io.registered()<<'\n';
        } catch(...) { close(fd); unlink(path); throw; }
        close(fd); unlink(path);
    }
    if(cuda && switches.l2) {
        CudaTier gpu(m,0,250);
        if(!gpu.ok()) { std::cout<<"CUDA unavailable: "<<gpu.error()<<'\n'; return 0; }
        auto start=Clock::now();
        if(!gpu.submit({x.data(),scales.data()})) { std::cout<<"CUDA worker not accepting jobs\n"; return 8; }
        while(!gpu.poll(y.data())) { if(!gpu.ok() || us(start)>1000000) { std::cout<<"CUDA timeout\n"; return 9; } std::this_thread::yield(); }
        std::cout<<"L2 CUDA ring us/expert="<<us(start)<<" (one cold round trip)\n";
        if(y!=ref) return 10;
    }
}

// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/offload/executor.hpp"
#include <cassert>
#include <cmath>
#include <cstring>
#include <unistd.h>
using namespace vegajet::offload;
int main() {
    uint8_t w[18]={0,0x3c}; for(int i=2;i<18;++i) w[i]=0x99;
    int8_t x[32]; for(auto& v:x) v=1;
    float scale=1,y=0;
    Matrix matrix{w,nullptr,1,32,18,0,Format::Q4_0}; Activation act{x,&scale};
    MissBackends b;
    bool fetch=true,hot=false,cuda=false,cpu=true;
    b.fetch=[&](auto){return fetch;};
    auto fake=[&](bool& available) { return [&available](auto,Matrix m,Activation a,float* out) {
        if(!available) return false;
        gemv_reference(m,a,out); return true;
    }; };
    b.hot=fake(hot); b.cuda=fake(cuda); b.cpu=fake(cpu);
    ExpertExecutor e({true,true,false,true,false},1,b);
    assert(e.miss(0,matrix,act,&y) && y==32 && e.stats().cpu==1);
    e.end_step(); hot=true;
    assert(e.miss(0,matrix,act,&y) && e.stats().hot==1);
    hot=false; cuda=true;
    assert(e.miss(1,matrix,act,&y) && e.stats().cuda==1);
    fetch=false;
    assert(!e.miss(1,matrix,act,&y) && e.stats().failed==1);
    ExpertExecutor disabled({false,false,false,false,false},1,b);
    assert(!disabled.miss(0,matrix,act,&y));
    assert((cold_experts({5,0,2,0},0.5)==std::vector<uint32_t>{1,3}));
    assert(cold_experts({1,2},0).empty()); assert(cold_experts({1,2},1).size()==2);
    bool caught=false; try { cold_experts({0},NAN); } catch(...) { caught=true; } assert(caught);
    // Real CPU instantiation remains device-free and agrees with the scalar oracle.
    ExpertExecutor cpu_only({true,false,false,false,false});
    assert(cpu_only.miss(0,matrix,act,&y) && y==32);
    // Instantiate the actual table -> read -> CPU miss pipeline, with both
    // requested I/O backends (io_uring may use the documented sandbox fallback).
    char path[]="/tmp/vj-executor-XXXXXX";
    int fd=mkstemp(path); assert(fd>=0);
    std::vector<uint8_t> row(4096,0); std::memcpy(row.data(),w,sizeof(w));
    assert(write(fd,row.data(),row.size())==static_cast<ssize_t>(row.size()));
    assert(fsync(fd)==0);
    for(bool uring:{false,true}) {
        TableMatrix layout{4096,0,0,1,32,18,0,Format::Q4_0};
        TableExecutor table(path,1,layout,{true,false,uring,false,false});
        assert(table.miss(0,act,&y) && y==32);
        assert(!table.miss(1,act,&y));
    }
    close(fd); unlink(path);
}

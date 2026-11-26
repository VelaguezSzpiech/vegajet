// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/offload/host.hpp"
#include "core/q4_dot8_repack.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
using namespace vegajet;
using namespace vegajet::offload;
int main() {
    Ring<uint64_t,8> ring; uint64_t v;
    for(unsigned k=0;k<100;++k) { for(unsigned i=0;i<8;++i) assert(ring.push(k*8+i)); assert(!ring.push(0));
        for(unsigned i=0;i<8;++i) { assert(ring.pop(v)); assert(v==k*8+i); } assert(!ring.pop(v)); }
    Ring<uint64_t,16> concurrent;
    std::thread producer([&]{ for(uint64_t i=0;i<100000;++i) while(!concurrent.push(i)) std::this_thread::yield(); });
    for(uint64_t i=0;i<100000;++i) { while(!concurrent.pop(v)) std::this_thread::yield(); assert(v==i); } producer.join();
    HostArena arena(12345,-1,true); assert(uintptr_t(arena.data())%huge_bytes==0); assert(arena.size()==huge_bytes);
    std::cout<<"hugetlb="<<arena.hugetlb()<<" pinned="<<arena.pinned()<<'\n';
    constexpr unsigned rows=5,cols=64;
    std::vector<uint8_t> w(rows*36); int8_t x[cols]; float xd[2]={0.25f,0.5f}, y[rows], ref[rows];
    for(unsigned j=0;j<cols;++j) x[j]=int8_t(j*17); // includes -128; signed integer dot must not saturate
    for(unsigned r=0;r<rows;++r) for(unsigned b=0;b<2;++b) {
        auto p=w.data()+r*36+b*18; p[0]=0; p[1]=0x3c;
        for(unsigned i=0;i<16;++i) p[i+2]=uint8_t(i*19+r*3+b);
    }
    Matrix m{w.data(),nullptr,rows,cols,36,0,Format::Q4_0};
    gemv_reference(m,{x,xd},ref); gemv(m,{x,xd},y);
    for(unsigned r=0;r<rows;++r) {
        float independent=0;
        for(unsigned b=0;b<2;++b) { int dot=0; for(unsigned j=0;j<32;++j) dot+=(int((w[r*36+b*18+2+j%16]>>(j/16*4))&15)-8)*int(x[b*32+j]); independent+=xd[b]*dot; }
        assert(y[r]==ref[r]); assert(y[r]==independent);
    }
    CpuTier cpu; assert(cpu.workers()>0); assert(cpu.affinity_ok(0));
    for(int repeat=0;repeat<200;++repeat) {
        Job job; job.matrix=m; job.x={x,xd}; job.y=y;
        assert(cpu.submit(0,job)); while(!job.done.load(std::memory_order_acquire)) std::this_thread::yield();
        assert(std::memcmp(y,ref,sizeof(y))==0);
    }
    auto layout=dot8_iq3_sub32_layout(rows,cols);
    std::vector<uint8_t> codes(layout.w_plane_bytes()), scales(rows*8);
    for(unsigned r=0;r<rows;++r) for(unsigned b=0;b<2;++b) {
        uint8_t c[32]; for(unsigned j=0;j<32;++j) c[j]=(r+b+j)%16;
        dot8_pack_codes(c,codes.data()+r*layout.w_row_stride_bytes+b*16);
        float d=float(b+1)/8; std::memcpy(scales.data()+r*8+b*4,&d,4);
    }
    m={codes.data(),scales.data(),rows,cols,layout.w_row_stride_bytes,8,Format::Dot8Iq3Sub32};
    gemv_reference(m,{x,xd},ref); gemv(m,{x,xd},y);
    for(unsigned r=0;r<rows;++r) {
        float independent=0;
        for(unsigned b=0;b<2;++b) { int dot=0; for(unsigned j=0;j<32;++j) dot+=int(dot8_iq3_level((r+b+j)%16))*int(x[b*32+j]); independent+=(float(b+1)/8*xd[b])*dot; }
        assert(y[r]==ref[r] && y[r]==independent);
    }
    Job invalid; assert(!cpu.submit(100000,invalid)); assert(!cpu.submit(0,invalid));
}

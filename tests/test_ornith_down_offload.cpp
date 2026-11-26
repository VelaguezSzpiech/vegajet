// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "models/ornith/ornith_offload.hpp"
#include <cassert>
#include <cstdlib>
using namespace vegajet;
int main() {
    setenv("VEGAJET_OFFLOAD_L3_CPU","1",1);
    unsetenv("VEGAJET_OFFLOAD_L2_CUDA");
    auto layout=dot8_iq3_sub32_layout(256*2,32);
    std::vector<uint8_t> weights(layout.w_plane_bytes(),0),scales(layout.s_plane_bytes(),0);
    float one=1;
    for(uint32_t r=0;r<512;++r) {
        auto* p=layout.scale_embedded?weights.data()+r*layout.w_row_stride_bytes+layout.s_offset_bytes:scales.data()+r*layout.s_row_stride_bytes;
        std::memcpy(p,&one,4);
    }
    ornith::DownOffload offload(0.1);
    offload.compact(0,0,weights,scales,layout,2);
    assert(layout.rows==(256-25+1)*2);
    std::vector<int32_t> ids={0,255,25,26,27,28,29,30};
    std::vector<int8_t> q(8*32,1);
    std::vector<float> ds(8,1),tw(8,0.5);
    auto read=[&](void* dst,uint64_t src,size_t n) { std::memcpy(dst,reinterpret_cast<const void*>(src),n); };
    auto write=[&](uint64_t dst,const void* src,size_t n) { std::memcpy(reinterpret_cast<void*>(dst),src,n); };
    std::vector<float> correction; std::vector<int32_t> original;
    offload.prepare(0,0,1,reinterpret_cast<uint64_t>(ids.data()),reinterpret_cast<uint64_t>(q.data()),
        reinterpret_cast<uint64_t>(ds.data()),reinterpret_cast<uint64_t>(tw.data()),read,write,correction,original);
    assert(original[0]==0 && original[1]==255);
    assert(ids[0]==231 && ids[1]==230 && ids[2]==0);
    assert(correction.size()==2 && correction[0]==64 && correction[1]==64);
    // Cold-expert zero slot must not contribute on the GPU.
    for(size_t i=231*2*layout.w_row_stride_bytes;i<weights.size();++i) assert(weights[i]==0);
}

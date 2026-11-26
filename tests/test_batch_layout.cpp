// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/sequence_layout.hpp"
#include <cassert>
#include <algorithm>
#include <array>
#include <vector>
using namespace vegajet::batch;
int main() {
    // Independent scalar enumeration of the M=1 kernel's head/block layouts.
    for(auto heads:std::array<std::array<uint32_t,2>,3>{{{{12,1}},{{10,2}},{{10,1}}}}) {
        uint32_t nq=heads[0],nkv=heads[1]; AttentionRows layout(nq,nkv,32768);
        const size_t row_floats=nq*512u+nkv*256u*2u;
        std::vector<float> packed(row_floats*8u);
        for(size_t i=0;i<packed.size();++i) packed[i]=static_cast<float>(i);
        for(uint32_t m=0;m<8;++m) {
            auto* one=reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(packed.data())+m*layout.qkv);
            for(uint32_t h=0;h<nq;++h) for(uint32_t d=0;d<256;++d) {
                assert(one[h*512+d]==packed[m*row_floats+h*512+d]);
                assert(one[h*512+256+d]==packed[m*row_floats+h*512+256+d]);
                assert((m*layout.qgate)/4+h*256+d==(m*nq+h)*256+d);
                assert(m*layout.quant+h*256+d==(m*nq+h)*256+d);
            }
            for(uint32_t h=0;h<nkv;++h) for(uint32_t pos:{0u,1u,4095u}) {
                uint64_t off=m*layout.kv_segment+(uint64_t(pos)*nkv+h)*272;
                assert(off==(uint64_t(m)*4096*nkv+uint64_t(pos)*nkv+h)*272);
                assert(off+272<=(m+1)*layout.kv_segment);
            }
            assert(layout.scales==uint64_t(nq)*256/32*8);
        }
    }
    for(uint32_t b=1;b<=8;++b) {
        assert(!last_row_logits(true,false,true,b));
        assert(last_row_logits(true,false,false,b)==(b>1));
    }
    // Noncontiguous rows, reordered rows, and singleton tails decode the same
    // physical slots in the convolution and recurrence kernels.
    for (const std::vector<uint32_t>& slots: {std::vector<uint32_t>{7,0,5,2}, {4}, {0,1,2,3,4,5,6,7}}) {
        const auto packed=pack_slots(slots);
        for (uint32_t row=0;row<slots.size();++row) assert(((packed>>(3u*row))&7u)==slots[row]);
    }
    bool caught=false; try { pack_slots({1,1}); } catch (...) { caught=true; } assert(caught);
}

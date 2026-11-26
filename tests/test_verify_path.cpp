// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// CPU oracle: independent grouping, signed Q3 dot products, FP32 per-lane
// accumulation, sum16 tree and b32 quantization for both members of each pair.
#include "ops/verify_routes.hpp"
#include "tests/test_common.hpp"
#include <array>
#include <algorithm>
#include <cmath>
#include <map>
#include <random>

static uint32_t bits(float x) { uint32_t u; std::memcpy(&u,&x,4); return u; }
static float reduce(std::array<float,16> a) {
    for (unsigned d : {8u,4u,2u,1u}) { auto b=a; for(unsigned i=0;i<16;++i) a[i]=b[i]+b[i^d]; }
    return a[0];
}
int main() {
    std::mt19937 rng(193);
    for(unsigned m=2;m<=16;++m) for(unsigned mode=0;mode<4;++mode) {
        std::vector<int32_t> ids(m*8);
        for(unsigned r=0;r<ids.size();++r)
            ids[r]=mode==0 ? r%8 : mode==1 ? r : mode==2 ? (r/8+r%8)%16 : rng()%256;
        std::map<int32_t,std::vector<unsigned>> oracle;
        for(unsigned r=0;r<ids.size();++r) oracle[ids[r]].push_back(r);
        std::vector<unsigned> expected(ids.size(),ids.size()), writes(ids.size());
        for(auto& kv:oracle) for(size_t j=0;j<kv.second.size();j+=2)
            expected[kv.second[j]]=kv.second[std::min(j+1,kv.second.size()-1)];
        for(unsigned r=0;r<ids.size();++r) {
            unsigned p=vegajet::verify::partner(ids.data(),ids.size(),r);
            VJ_CHECK(p==expected[r]);
            if(p<ids.size()) { ++writes[r]; if(p!=r) ++writes[p]; }
        }
        for(unsigned w:writes) VJ_CHECK(w==1);
    }
    // 32 output neurons; one weight packet serves two different activation rows.
    // Oracle multiplies decoded signed levels independently; paired body decodes
    // packed two-bit magnitude + sign bytes and shares packets across both rows.
    float out[2][32], ref[2][32];
    for(unsigned row=0;row<32;++row) {
        std::array<float,16> acc[2]={}, oracle[2]={};
        for(unsigned it=0;it<6;++it) for(unsigned lane=0;lane<16;++lane) {
            unsigned mag[2][16], signs[2]; int weight[2][16], x[2][2][16]; float scale[2][2];
            for(unsigned half=0;half<2;++half) {
                signs[half]=rng();
                for(unsigned j=0;j<16;++j) {
                    mag[half][j]=rng()%3;
                    int v=mag[half][j]==0?8:mag[half][j]==1?25:43;
                    weight[half][j]=(signs[half]>>j&1)?-v:v;
                    for(unsigned t=0;t<2;++t) x[t][half][j]=int(rng()%255)-127;
                }
            }
            for(auto& t:scale) for(float& v:t) v=float(1+rng()%100)*0.00001f;
            // Packet-first schedule: low half then high half, independent chains.
            for(unsigned half=0;half<2;++half) for(unsigned t=0;t<2;++t) {
                int dp=0; for(unsigned j=0;j<16;++j) {
                    static const int levels[]={8,25,43};
                    int v=levels[mag[half][j]];
                    if(signs[half]>>j&1) v=-v;
                    dp+=v*x[t][half][j];
                }
                acc[t][lane]=std::fma(scale[t][half],float(dp),acc[t][lane]);
            }
            // Independent row-first oracle, with the same mandated half order.
            for(unsigned t=0;t<2;++t) for(unsigned half=0;half<2;++half) {
                int dp=0; for(unsigned j=0;j<16;++j) dp+=weight[half][j]*x[t][half][j];
                oracle[t][lane]=std::fma(scale[t][half],float(dp),oracle[t][lane]);
            }
        }
        for(unsigned t=0;t<2;++t) {
            for(unsigned lane=0;lane<16;++lane) VJ_CHECK(bits(acc[t][lane])==bits(oracle[t][lane]));
            out[t][row]=reduce(acc[t]); ref[t][row]=reduce(oracle[t]);
        }
    }
    for(unsigned t=0;t<2;++t) {
        float a=0,b=0; for(unsigned r=0;r<32;++r) { a=std::max(a,std::fabs(out[t][r])); b=std::max(b,std::fabs(ref[t][r])); }
        VJ_CHECK(bits(a/127.f)==bits(b/127.f));
        for(unsigned r=0;r<32;++r) VJ_CHECK(std::nearbyint(out[t][r]*127.f/a)==std::nearbyint(ref[t][r]*127.f/b));
    }
    std::puts("verify pairs: exhaustive M=2..16 writer coverage + bit-exact Q3 pair arithmetic oracle PASS");
}

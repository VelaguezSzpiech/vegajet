// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/quant_repacker.hpp"
#include "tests/test_common.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
using namespace vegajet::quant;
int main() {
    for(auto fmt:{NativeFormat::VJ_Q2_DOT8,NativeFormat::VJ_Q3_DOT8}) for(unsigned group:{64u,128u}) {
        const unsigned top=fmt==NativeFormat::VJ_Q2_DOT8?3:7, bits=top==3?2:3;
        std::vector<float> w(5*256), dq(w.size());
        for(size_t i=0;i<w.size();++i) w[i]=float(i%(top+1));
        auto p=pack_f32_to_vj_sub4(w.data(),5,256,fmt,group);
        dequantize_vj_sub4(p,dq.data());
        VJ_CHECK(w==dq);
        for(size_t i=0;i<w.size();++i) VJ_CHECK(sub4_code(p,i)==unsigned(w[i]));
        // Independent expected word construction, including each MSB position.
        for(unsigned k=0;k<32;++k) {
            std::vector<float> one(128,0); one[k]=float(top);
            auto a=pack_f32_to_vj_sub4(one.data(),1,128,fmt,group);
            for(unsigned j=0;j<bits;++j) {
                unsigned expect=0;
                if(j==k/16) expect=3u<<(4*((k%16)/2)+2*(k%2));
                if(j==2) expect=1u<<(4*((k%16)/2)+2*(k/16)+k%2);
                VJ_CHECK(a.qs[j]==expect);
            }
        }
        std::vector<uint16_t> h(w.size());
        for(size_t i=0;i<w.size();++i) h[i]=f32_to_fp16(w[i]);
        VJ_CHECK(pack_f16_to_vj_sub4(h.data(),5,256,fmt,group).qs==p.qs);
        for(size_t i=0;i<w.size();++i) w[i]=std::sin(float(i)*0.31f)*2+0.17f;
        p=pack_f32_to_vj_sub4(w.data(),5,256,fmt,group); dequantize_vj_sub4(p,dq.data());
        auto again=pack_f32_to_vj_sub4(dq.data(),5,256,fmt,group);
        VJ_CHECK(again.qs==p.qs);
        // Emulate kernel SWAR parity operands, independent of scalar sub4_code decode.
        auto compact=[](uint32_t a,uint32_t b) {
            a&=0x0f0f0f0f; b&=0x0f0f0f0f;
            a=(a|(a>>4))&0x00ff00ff; b=(b|(b>>4))&0x00ff00ff;
            return ((a|(a>>8))&65535)|((b|(b>>8))<<16);
        };
        for(unsigned block=0;block<5*8;++block) {
            uint32_t a[4]={}; int expected=0;
            for(unsigned j=0;j<32;++j) {
                int q=int((j*7+block)%16)-8;
                a[j/8]|=uint32_t(q&15)<<(4*(j%8));
                expected+=q*int(sub4_code(p,block*32+j));
            }
            int got=0;
            for(unsigned stripe=0;stripe<4;++stripe) {
                uint32_t lo=(p.qs[block*bits+stripe/2]>>(2*(stripe%2)))&0x33333333;
                if(bits==3) {
                    uint32_t msb=p.qs[block*bits+2];
                    msb=stripe<=2 ? msb<<(2-stripe) : msb>>1;
                    lo|=msb&0x44444444;
                }
                uint32_t act=compact(a[2*(stripe/2)]>>(4*(stripe%2)),a[2*(stripe/2)+1]>>(4*(stripe%2)));
                for(unsigned j=0;j<8;++j) {
                    int q=int((act>>(4*j))&15); if(q>=8) q-=16;
                    got+=q*int((lo>>(4*j))&15);
                }
            }
            VJ_CHECK(got==expected);
        }
        for(unsigned M=1;M<=8;++M) {
            std::vector<int8_t> x(M*256); std::vector<float> sc(M*8), y(M*5);
            for(size_t i=0;i<x.size();++i) x[i]=int(i%16)-8;
            for(size_t i=0;i<sc.size();++i) sc[i]=0.01f*(1+i%7);
            gemv_vj_sub4(p,x.data(),sc.data(),y.data(),M);
            for(unsigned m=0;m<M;++m) for(unsigned r=0;r<5;++r) {
                double ref=0; for(unsigned c=0;c<256;++c) ref+=double(dq[r*256+c])*x[m*256+c]*sc[m*8+c/32];
                VJ_CHECK(std::abs(y[m*5+r]-ref)<2e-5*(1+std::abs(ref)));
            }
        }
    }
    // Non-square residual matrix and both ranks; fp16 factor serialization included.
    std::vector<float> w(48*128),d(w.size());
    for(size_t i=0;i<w.size();++i) w[i]=std::sin(float(i)*.13f)+.4f*std::cos(float(i)*.073f);
    auto p=pack_f32_to_vj_sub4(w.data(),48,128,NativeFormat::VJ_Q3_DOT8);
    dequantize_vj_sub4(p,d.data());
    for(unsigned rank:{16u,32u}) {
        auto f=fit_milo(w.data(),p,rank); auto f2=fit_milo(w.data(),p,rank);
        VJ_CHECK(f.u==f2.u && f.v==f2.v);
        double before=0,after=0;
        for(unsigned c=0;c<128;++c) {
            std::vector<float> x(128),y(48); x[c]=1; apply_milo(f,x.data(),y.data());
            for(unsigned r=0;r<48;++r) {
                double e=w[r*128+c]-d[r*128+c]; before+=e*e; after+=(e-y[r])*(e-y[r]);
            }
        }
        std::printf("MiLo rank %u residual SSE %.9g -> %.9g\n",rank,before,after);
        VJ_CHECK(after<before*.8);
    }
    setenv("VEGAJET_MILO","0",1); VJ_CHECK(!milo_enabled());
    setenv("VEGAJET_MILO","1",1); VJ_CHECK(milo_enabled()); unsetenv("VEGAJET_MILO");
    bool threw=false; try { pack_f32_to_vj_sub4(w.data(),1,32,NativeFormat::VJ_Q2_DOT8); } catch(const std::invalid_argument&) { threw=true; }
    VJ_CHECK(threw);
    puts("sub4 CPU checks passed");
}

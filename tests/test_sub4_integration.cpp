// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/quant_repacker.hpp"
#include "tests/test_common.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
using namespace vegajet::quant;
// Kernel lane/block partition and butterfly reduction, independent scalar per-block dot.
static std::vector<float> kernel(const PackedVjSub4& p,const std::vector<int8_t>& x,unsigned M) {
    std::vector<float> y(M*p.rows);
    for(unsigned m=0;m<M;++m) for(unsigned r=0;r<p.rows;++r) {
        std::array<float,64> a{};
        for(unsigned lane=0;lane<64;++lane) for(unsigned base=0;base<p.cols/32;base+=256)
            for(unsigned j=0;j<4;++j) {
                unsigned b=base+lane*4+j; if(b>=p.cols/32) continue;
                int dot=0,sum=0;
                for(unsigned c=0;c<32;++c) { int v=x[m*p.cols+b*32+c]; sum+=v; dot+=v*int(sub4_code(p,size_t(r)*p.cols+b*32+c)); }
                unsigned meta=p.tcc_xor?p.dm[size_t(r)*(p.cols/32)+(b^(r&15))]:p.dm[size_t(r)*(p.cols/p.group)+b*32/p.group];
                a[lane]+=fp16_to_f32(meta&65535)*(.03125f*dot-fp16_to_f32(meta>>16)*(.03125f*sum));
            }
        for(unsigned d=32;d;d>>=1) { auto old=a; for(unsigned l=0;l<64;++l) a[l]+=old[l^d]; }
        y[m*p.rows+r]=a[0];
    }
    return y;
}
// W32 family oracle: integer dot per block and nested FMAs in ascending logical b.
static std::vector<float> q4_kernel(const PackedVjQ4A8& p,bool xored) {
    std::vector<float> y(p.rows);
    for(unsigned r=0;r<p.rows;++r) {
        std::array<float,32> acc{};
        for(unsigned l=0;l<32;++l) for(unsigned b=l;b<p.cols/32;b+=32) {
            unsigned phys=xored?b^(r&15):b; int dot=0,sum=0;
            for(unsigned j=0;j<32;++j) {
                int x=int((b*32+j)*11%255)-127; sum+=x;
                auto q=p.qs[size_t(r)*p.qs_row_stride_bytes+phys*16+j%16];
                dot+=x*((j<16?q:q>>4)&15);
            }
            auto dm=p.dm[size_t(r)*p.dm_row_stride_u32+phys];
            acc[l]=std::fma(fp16_to_f32(dm&65535)*.003f,float(dot),std::fma(fp16_to_f32(dm>>16),sum*.003f,acc[l]));
        }
        for(unsigned d=16;d;d>>=1) { auto old=acc; for(unsigned l=0;l<32;++l) acc[l]+=old[l^d]; }
        y[r]=acc[0];
    }
    return y;
}
int main() {
    constexpr unsigned R=35,C=1024;
    std::vector<float> w(R*C),plain(w.size()),decoded(w.size());
    for(size_t i=0;i<w.size();++i) w[i]=std::sin(float(i)*.113f)+.37f*std::cos(float(i)*.071f);
    for(unsigned cols:{512u,3072u,4096u}) {
        std::vector<float> source(R*cols);
        for(size_t i=0;i<source.size();++i) source[i]=std::sin(float(i)*.071f);
        auto q=pack_f32_to_vj_q4_a8(source.data(),R,cols);
        VJ_CHECK(q4_kernel(q,false)==q4_kernel(repack_tcc_xor(q),true));
    }
    auto q4=pack_f32_to_vj_q4_a8(w.data(),R,C); auto qx=repack_tcc_xor(q4); auto qback=repack_tcc_xor(qx);
    VJ_CHECK(q4.qs==qback.qs && q4.dm==qback.dm);
    for(unsigned r=0;r<R;++r) for(unsigned b=0;b<C/32;++b) {
        unsigned phys=b^(r&15); VJ_CHECK(qx.dm[r*q4.dm_row_stride_u32+phys]==q4.dm[r*q4.dm_row_stride_u32+b]);
        for(unsigned j=0;j<16;++j) VJ_CHECK(qx.qs[r*q4.qs_row_stride_bytes+phys*16+j]==q4.qs[r*q4.qs_row_stride_bytes+b*16+j]);
    }
    for(auto fmt:{NativeFormat::VJ_Q2_DOT8,NativeFormat::VJ_Q3_DOT8}) for(unsigned group:{64u,128u}) {
        auto p=pack_f32_to_vj_sub4(w.data(),R,C,fmt,group); auto px=repack_tcc_xor(p);
        dequantize_vj_sub4(p,plain.data()); dequantize_vj_sub4(px,decoded.data()); VJ_CHECK(plain==decoded);
        VJ_CHECK(px.total_bytes()==p.total_bytes()+(size_t(R)*C/32-p.dm.size())*4);
        for(unsigned M=1;M<=8;++M) {
            std::vector<int8_t> x(M*C); for(size_t i=0;i<x.size();++i) x[i]=int(i*7%16)-8;
            auto y=kernel(p,x,M),yx=kernel(px,x,M); VJ_CHECK(y==yx);
            for(unsigned m=0;m<M;++m) for(unsigned r=0;r<R;++r) {
                double oracle=0; for(unsigned c=0;c<C;++c) oracle+=double(plain[r*C+c])*x[m*C+c]*.03125;
                VJ_CHECK(std::abs(y[m*R+r]-oracle)<2e-4*(1+std::abs(oracle)));
            }
        }
    }
    auto p=pack_f32_to_vj_sub4(w.data(),R,C,NativeFormat::VJ_Q3_DOT8);
    for(unsigned rank:{16u,32u}) {
        auto f=fit_milo(w.data(),p,rank,4,MiloStorage::Int3);
        VJ_CHECK(f.u.empty() && f.v.empty());
        const size_t expected_words=(size_t(R)*rank*3+31)/32+1+(size_t(rank)*C*3+31)/32+1;
        VJ_CHECK(f.total_bytes()==4*(expected_words+R+rank));
        VJ_CHECK(f.total_bytes() < (size_t(R)*rank+size_t(rank)*C)*2);
        // Exercise sign extension and cross-dword INT3 indices with independent decoder.
        auto decode=[](const std::vector<uint32_t>& q,size_t i) {
            unsigned code=0; for(unsigned b=0;b<3;++b) code|=((q[(i*3+b)/32]>>((i*3+b)%32))&1)<<b;
            return code>=4?int(code)-8:int(code);
        };
        for(size_t i=0;i<size_t(rank)*C;++i) VJ_CHECK(milo_factor(f,false,i)==decode(f.v3,i)*f.vs[i/C]);
        for(size_t i=0;i<size_t(R)*rank;++i) VJ_CHECK(milo_factor(f,true,i)==decode(f.u3,i)*f.us[i/rank]);
        for(unsigned M=1;M<=8;++M) for(unsigned m=0;m<M;++m) {
            std::vector<float> x(C),y(R,.25f),ref(R,.25f);
            for(unsigned c=0;c<C;++c) x[c]=float(int((c*7+m)%255)-127)*.003f;
            apply_milo(f,x.data(),y.data());
            // GPU oracle: wave64 strided V*x reduction then the row-wise rank epilogue.
            std::vector<float> tmp(rank);
            for(unsigned k=0;k<rank;++k) {
                std::array<float,64> a{};
                for(unsigned l=0;l<64;++l) for(unsigned c=l;c<C;c+=64) a[l]+=milo_factor(f,false,k*C+c)*x[c];
                for(unsigned d=32;d;d>>=1) { auto old=a; for(unsigned l=0;l<64;++l) a[l]+=old[l^d]; }
                tmp[k]=a[0];
            }
            for(unsigned r=0;r<R;++r) for(unsigned k=0;k<rank;++k) ref[r]+=milo_factor(f,true,r*rank+k)*tmp[k];
            for(unsigned r=0;r<R;++r) VJ_CHECK(std::abs(y[r]-ref[r])<1e-4f*(1+std::abs(y[r])));
        }
    }
    bool threw=false; try { repack_tcc_xor(pack_f32_to_vj_sub4(w.data(),1,128,NativeFormat::VJ_Q2_DOT8)); } catch(const std::invalid_argument&) { threw=true; } VJ_CHECK(threw);
    unsetenv("VEGAJET_SUB4_DOT8"); VJ_CHECK(!sub4_dot8_bits());
    setenv("VEGAJET_SUB4_DOT8","q3",1); VJ_CHECK(sub4_dot8_bits()==3);
    setenv("VEGAJET_SUB4_DOT8","q2",1); VJ_CHECK(sub4_dot8_bits()==2);
    setenv("VEGAJET_SUB4_DOT8","invalid",1); VJ_CHECK(!sub4_dot8_bits()); unsetenv("VEGAJET_SUB4_DOT8");
    puts("SUB4 XOR, INT3 MiLo and M=1..8 kernel CPU oracles passed");
}

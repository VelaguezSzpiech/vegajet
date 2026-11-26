// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "models/ornith/ornith_l4.hpp"
#include "core/kv_quant_oracle.hpp"
#include "ops/wire_normal_launch.hpp"
#include "ops/ornith_gqa_kvq_launch.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
using namespace vegajet;
static void check(bool x) { if(!x) { std::fprintf(stderr,"L4 contract failure\n"); std::abort(); } }
// Independent two-pass softmax vs chronological/split online recurrence and LSE merge.
static void attention(const std::vector<float>& q,const std::vector<float>& k,const std::vector<float>& v,
                      uint32_t tokens,uint32_t splits) {
    std::vector<double> scores(tokens),want(256),got(256),parts(splits*256),ls(splits);
    double mx=-INFINITY,den=0;
    for(uint32_t t=0;t<tokens;++t) {
        for(unsigned d=0;d<256;++d) scores[t]+=double(q[d])*k[t*256+d]/16;
        mx=std::max(mx,scores[t]);
    }
    for(uint32_t t=0;t<tokens;++t) {
        double w=std::exp(scores[t]-mx); den+=w;
        for(unsigned d=0;d<256;++d) want[d]+=w*v[t*256+d];
    }
    for(auto& x:want) x/=den;
    for(uint32_t s=0;s<splits;++s) {
        double m=-INFINITY,l=0;
        for(uint32_t t=flash_decode_splitkv::split_begin(tokens,s,splits);t<flash_decode_splitkv::split_end(tokens,s,splits);++t) {
            double nm=std::max(m,scores[t]),a=std::exp(m-nm),b=std::exp(scores[t]-nm);
            for(unsigned d=0;d<256;++d) parts[s*256+d]=a*parts[s*256+d]+b*v[t*256+d];
            l=a*l+b; m=nm;
        }
        ls[s]=l>0?m+std::log(l):-INFINITY;
        if(l>0) for(unsigned d=0;d<256;++d) parts[s*256+d]/=l;
    }
    double lm=*std::max_element(ls.begin(),ls.end()),z=0;
    for(unsigned s=0;s<splits;++s) {
        double w=std::exp(ls[s]-lm); z+=w;
        for(unsigned d=0;d<256;++d) got[d]+=w*parts[s*256+d];
    }
    for(unsigned d=0;d<256;++d) check(std::abs(got[d]/z-want[d])<1e-10);
}
int main() {
    // TP=3 straddles the global 16-head groups at Q offsets 12 and 22.
    const auto a=ornith::l4::ranges(12,1,0,0,16);
    const auto b=ornith::l4::ranges(10,2,12,0,16);
    const auto c=ornith::l4::ranges(10,1,22,1,16);
    check(a[0].begin==0 && a[0].end==12);
    check(b[0].end==4 && b[1].begin==4 && b[1].end==10);
    check(c[0].begin==0 && c[0].end==10);
    for(auto r:{a,b,c}) {
        uint32_t heads=r.back().end;
        for(uint32_t h=0;h<heads;++h) {
            unsigned count=0; for(auto p:r) count+=h>=p.begin && h<p.end;
            check(count==1);
        }
    }
    // Model the padded producer for every real rank/group and compare with the KV oracle's
    // independent int8 query quantizer. Sine inputs avoid rounding ties (the device uses RNE).
    for (const auto& rank : {a,b,c}) {
        std::vector<float> query(rank.back().end*256);
        for(unsigned i=0;i<query.size();++i) query[i]=std::sin(float(i+1)*0.17f);
        for(auto range:rank) {
            std::vector<uint8_t> tile(16*384,0xa5);
            for(unsigned slot=0;slot<16;++slot) {
                float zero[256]={};
                const float* source=slot<range.end-range.begin ? query.data()+(range.begin+slot)*256 : zero;
                for(unsigned block=0;block<8;++block)
                    wire_normal::quantize_q8_block(source+block*32,tile.data()+slot*384+block*48);
                int8_t codes[256]; float scales[8];
                kvq::query_pack_i8(source,256,codes,scales);
                for(unsigned d=0;d<256;++d) {
                    const auto* wire=tile.data()+slot*384+d/32*48;
                    check(int8_t(wire[16+d%32])==codes[d]);
                    check((uint16_t(wire[0])|(uint16_t(wire[1])<<8))==kvq::f32_to_f16(scales[d/32]));
                }
                if(slot>=range.end-range.begin)
                    for(unsigned i=0;i<384;++i) check(tile[slot*384+i]==0);
            }
            // Padded heads cannot enter the real-head scatter or its block32 metadata.
            std::vector<int8_t> scatter(query.size()+256,42);
            for(unsigned slot=0;slot<range.end-range.begin;++slot)
                for(unsigned d=0;d<256;++d) scatter[(range.begin+slot)*256+d]=0;
            for(unsigned h=0;h<=rank.back().end;++h)
                for(unsigned d=0;d<256;++d) check(scatter[h*256+d]==(h>=range.begin && h<range.end?0:42));
        }
    }
    const auto empty=ornith::l4::ranges(4,2,16,0,16);
    check(empty[0].begin==0 && empty[0].end==0 && empty[1].end==4);
    for(unsigned h=0;h<2;++h) for(unsigned row=0;row<8;++row) {
        const auto off=ornith::l4::wire_offset(32768,h,row,4096);
        check(off==(uint64_t(h)*32768+row*4096)*384);
        check(off+4096*384<=uint64_t(h+1)*32768*384);
        for(auto f:{kvq::DeviceFormat::Q8,kvq::DeviceFormat::Q4,kvq::DeviceFormat::Turbo3}) {
            const auto native=ornith::l4::native_offset(row,4096,2,f);
            check(native==ornith::l4::cache_bytes(row*4096,2,f));
            check(native+ornith::l4::cache_bytes(4096,2,f)<=ornith::l4::cache_bytes(32768,2,f));
        }
    }
    // FP64 prep oracle: RMSNorm then split-half RoPE preserves the normalized norm.
    // Its FP32 image is the direct append input; gates and raw V bypass both transforms.
    for(unsigned pos:{0u,17u,32767u}) {
        double raw[256],sq=0,prepared[256],rotated[256];
        for(unsigned d=0;d<256;++d) { raw[d]=std::sin((d+1)*0.14); sq+=raw[d]*raw[d]; }
        double inv=1/std::sqrt(sq/256+1e-6);
        for(unsigned d=0;d<256;++d) prepared[d]=raw[d]*inv;
        for(unsigned d=0;d<256;++d) {
            if(d>=64) rotated[d]=prepared[d];
            else {
                unsigned p=d%32; double theta=pos*std::pow(10000000.,-2.*p/64);
                rotated[d]=d<32 ? prepared[p]*std::cos(theta)-prepared[p+32]*std::sin(theta)
                                        : prepared[p]*std::sin(theta)+prepared[p+32]*std::cos(theta);
            }
        }
        double nr=0; for(double x:rotated) nr+=x*x;
        check(std::abs(nr-sq*inv*inv)<1e-10);
    }
    // Full epilogue oracle also checks saturated codes, sigmoid gates, scales and sum metadata.
    {
        std::vector<float> output(16*256),gate(16*256),ds(16*16);
        std::vector<int8_t> codes(16*256);
        for(unsigned i=0;i<output.size();++i) { output[i]=std::sin(float(i)*0.31f); gate[i]=float(int(i%13)-6); }
        wire_normal::epilogue_reference(output.data(),gate.data(),1,codes.data(),ds.data());
        for(unsigned block=0;block<16*8;++block) {
            float am=0,sum=0;
            for(unsigned j=0;j<32;++j) {
                unsigned i=block*32+j; float y=output[i]/(1+std::exp(-gate[i]));
                am=std::max(am,std::abs(y)); sum+=y;
            }
            check(ds[block*2]==am*(1.f/127.f) && ds[block*2+1]==sum);
            for(unsigned j=0;j<32;++j) {
                unsigned i=block*32+j;
                int expected=int(std::nearbyint(output[i]/(1+std::exp(-gate[i]))*(am>0?127/am:0)));
                check(codes[i]==std::max(-127,std::min(127,expected)));
            }
        }
    }
    bool rejected=false;
    try { ornith::l4::ranges(17,1,0,0,32); } catch(const std::invalid_argument&) { rejected=true; }
    check(rejected);
    for(unsigned kh:{1u,2u}) {
        check(ornith::l4::splits(32768,kh,false)*kh*4>=240);
        check(ornith::l4::splits(32768,kh,true)==1);
        check(ornith::l4::splits(1,kh,false)==1);
        const auto sp=flash_decode_splitkv::plan_scratch(1,kvq::split_count(kh));
        check(!sp.overflow && sp.partial_bytes==16ULL*kvq::split_count(kh)*256*4);
        check(sp.lse_offset>=sp.partial_bytes && sp.output_offset>=sp.lse_offset+sp.lse_bytes);
        check(sp.total_bytes>=sp.output_lse_offset+sp.output_lse_bytes);
    }
    for(auto f:{kvq::DeviceFormat::Q8,kvq::DeviceFormat::Q4,kvq::DeviceFormat::Turbo3}) {
        const uint32_t stride=kvq::vector_bytes(f,256),T=19;
        check(ornith::l4::cache_bytes(32768,2,f)==uint64_t(32768)*2*stride);
        std::vector<uint8_t> cache(T*2*stride+16,0xa5);
        std::vector<float> keys(T*256),values(T*256),q(256),x(256);
        for(unsigned d=0;d<256;++d) q[d]=std::sin(float(d)*0.31f);
        for(unsigned t=0;t<T;++t) for(unsigned h=0;h<2;++h) {
            for(unsigned d=0;d<256;++d) x[d]=std::sin(float(t*7+h*17+d)*0.13f);
            auto p=cache.data()+(t*2+h)*stride;
            kvq::pack_device_vector(x.data(),256,f,p);
            kvq::unpack_device_vector(p,256,f,x.data());
            for(unsigned d=0;d<256;++d) {
                check(std::isfinite(x[d]));
                if(h==1) { keys[t*256+d]=x[d]; values[t*256+d]=x[255-d]; }
            }
            if(f==kvq::DeviceFormat::Q8) {
                uint8_t wire[384]; wire_normal::q8_wire_from_native_row(p,wire);
                for(unsigned d=0;d<256;++d) {
                    const uint8_t* w=wire+d/32*48;
                    float y=wire_normal::f16_bits_to_f32(uint16_t(w[0])|(uint16_t(w[1])<<8))*int8_t(w[16+d%32]);
                    check(y==x[d]);
                }
            }
        }
        for(unsigned i=T*2*stride;i<cache.size();++i) check(cache[i]==0xa5);
        for(unsigned s:{1u,30u,60u}) attention(q,keys,values,T,s);
    }
    // Exercise the super flash's real chunk boundaries against an independent two-pass oracle.
    // Includes a partial four-token tile and the 32K crossover workload.
    for (uint32_t span : {1u, 33u, 1921u, 7681u, 32768u}) {
        std::vector<float> q(256),k(size_t(span)*256),v(k.size());
        for (unsigned d=0;d<256;++d) q[d]=std::sin(d*0.17f);
        for (size_t i=0;i<k.size();++i) { k[i]=std::sin(float(i%1009)*0.11f);v[i]=std::cos(float(i%997)*0.07f); }
        const uint32_t len=ornith::kvq_super::chunk_length(span,false);
        attention(q,k,v,span,(span+len-1)/len);
        check(ornith::kvq_super::chunk_length(span,true)==span);
    }
    std::puts("L4 TP ranges, padding, KV oracle layouts, sizing, 240-wave selection and split/LSE CPU reference passed; no device execution");
}

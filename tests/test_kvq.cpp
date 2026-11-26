// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/kv_quant_oracle.hpp"
#include "tests/test_common.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
using namespace vegajet::kvq;
int main() {
    static_assert(split_count(1)*4>=240 && split_count(2)*2*4>=240);
    static_assert(split_count(2,true)==1);
    HeadRange ranges[]={{0,5},{5,12}};
    VJ_CHECK(valid_shape(128,12,2,4096,30,DeviceFormat::Q4,ranges));
    VJ_CHECK(valid_shape(256,12,2,65536,30,DeviceFormat::Turbo3,ranges));
    VJ_CHECK(!valid_shape(128,12,2,UINT32_MAX,30,DeviceFormat::Q4,ranges));
    VJ_CHECK(!valid_shape(64,12,2,4096,30,DeviceFormat::Q4,ranges));
    ranges[1].begin=4;VJ_CHECK(!valid_shape(128,12,2,4096,30,DeviceFormat::Q4,ranges));
    DeviceFormat parsed=DeviceFormat::Fp16;
    VJ_CHECK(parse_format("q8_0",parsed) && parsed==DeviceFormat::Q8);
    VJ_CHECK(parse_format("q4_0",parsed) && parsed==DeviceFormat::Q4);
    VJ_CHECK(parse_format("turbo3",parsed) && parsed==DeviceFormat::Turbo3);
    VJ_CHECK(parse_format("fp16",parsed) && parsed==DeviceFormat::Fp16);
    VJ_CHECK(!parse_format("invalid",parsed) && split_count(0)==0);
    VJ_CHECK(split_count(UINT32_MAX)==1);
    VJ_CHECK(!valid_shape(256,12,UINT32_MAX,UINT32_MAX,30,DeviceFormat::Fp16,ranges));
    std::mt19937 rng(19);std::normal_distribution<float> normal;
    const char* names[]={"fp16","q8_0","q4_0","turbo3"};
    for(unsigned d:{128u,256u}) for(unsigned ff=0;ff<4;++ff) {
        auto f=DeviceFormat(ff);const unsigned n=97,bytes=vector_bytes(f,d);
        VJ_CHECK(bytes==(ff==0?d*2:ff==1?d+d/16:ff==2?d*18/32:d*14/32));
        std::vector<float> k(n*d),v(n*d),q(d),decoded(d);
        for(auto& x:k)x=normal(rng);
        for(auto& x:v)x=normal(rng);
        for(auto& x:q)x=normal(rng);
        // Deliberately offset byte arrays to verify unaligned CPU references.
        std::vector<uint8_t> pk(n*bytes+1),pv(n*bytes+1);
        for(unsigned t=0;t<n;++t) {pack_device_vector(k.data()+t*d,d,f,pk.data()+1+t*bytes);pack_device_vector(v.data()+t*d,d,f,pv.data()+1+t*bytes);}
        auto out=device_attention_reference(q.data(),pk.data()+1,pv.data()+1,d,n,f);
        // Independent fp64 full precision attention and KL against the packed/query-quantized distribution.
        std::vector<float> qd=q;
        if(f==DeviceFormat::Q4) {
            std::vector<uint32_t> c(d/8);std::vector<float> s(d/32);query_pack_i4(q.data(),d,c.data(),s.data());
            for(unsigned j=0;j<d;++j) {unsigned nib=(c[j/8]>>(4*(j%8)))&15;qd[j]=float(nib>=8?int(nib)-16:int(nib))*s[j/32];}
        }
        std::vector<double> score(n),packed_score(n),probs(n),packed_probs(n);
        double peak=-INFINITY,pp=-INFINITY;
        for(unsigned t=0;t<n;++t) {
            unpack_device_vector(pk.data()+1+t*bytes,d,f,decoded.data());
            for(unsigned j=0;j<d;++j) {score[t]+=double(q[j])*k[t*d+j];packed_score[t]+=double(qd[j])*decoded[j];}
            score[t]/=std::sqrt(double(d));packed_score[t]/=std::sqrt(double(d));peak=std::max(peak,score[t]);pp=std::max(pp,packed_score[t]);
        }
        double z=0,zp=0;
        for(unsigned t=0;t<n;++t) {probs[t]=std::exp(score[t]-peak);packed_probs[t]=std::exp(packed_score[t]-pp);z+=probs[t];zp+=packed_probs[t];}
        double kl=0,maxerr=0;
        for(unsigned t=0;t<n;++t) {probs[t]/=z;packed_probs[t]/=zp;kl+=probs[t]*std::log(probs[t]/packed_probs[t]);}
        for(unsigned j=0;j<d;++j) {double expected=0;for(unsigned t=0;t<n;++t)expected+=probs[t]*v[t*d+j];maxerr=std::max(maxerr,std::fabs(expected-out[j]));}
        VJ_CHECK(std::isfinite(kl)&&kl<0.1 && maxerr<0.3);
        if(ff<2) VJ_CHECK(kl<0.001 && maxerr<0.02);
        std::printf("kvq D=%u %s output_max_abs=%.8g KL=%.8g\n",d,names[ff],maxerr,kl);
        // CPU online partition/LSE-merge model versus the independent two-pass oracle.
        // Include more splits than tokens, so all-empty partitions and tails participate.
        double split_error=0;
        for(unsigned splits:{1u,30u,60u,128u}) {
            std::vector<double> po(size_t(splits)*d,0),lse(splits,-INFINITY);
            for(unsigned sp=0;sp<splits;++sp) {
                unsigned begin=uint64_t(n)*sp/splits,end=uint64_t(n)*(sp+1)/splits;
                double peak=-INFINITY,den=0;
                for(unsigned t=begin;t<end;++t) {
                    double m=std::max(peak,packed_score[t]),a=std::exp(peak-m),b=std::exp(packed_score[t]-m);
                    den=den*a+b;peak=m;
                    unpack_device_vector(pv.data()+1+t*bytes,d,f,decoded.data());
                    for(unsigned j=0;j<d;++j)po[size_t(sp)*d+j]=po[size_t(sp)*d+j]*a+b*decoded[j];
                }
                if(den>0) {lse[sp]=peak+std::log(den);for(unsigned j=0;j<d;++j)po[size_t(sp)*d+j]/=den;}
            }
            double peak=*std::max_element(lse.begin(),lse.end()),den=0;
            std::vector<double> merged(d,0);
            for(unsigned sp=0;sp<splits;++sp) {
                double w=lse[sp]==-INFINITY?0:std::exp(lse[sp]-peak);den+=w;
                for(unsigned j=0;j<d;++j)merged[j]+=w*po[size_t(sp)*d+j];
            }
            for(unsigned j=0;j<d;++j)split_error=std::max(split_error,std::fabs(merged[j]/den-out[j]));
        }
        VJ_CHECK(split_error<1e-7);
        std::printf("  split/LSE reference max_abs=%.8g (S=1,30,60,128)\n",split_error);
        // Turbo3 optimization oracle: move inverse rotation past the online
        // weighted sum, independently of the unpacked-V attention oracle above.
        // Exercise single-token partitions, empty partitions, and nontrivial tails.
        if(f==DeviceFormat::Turbo3) {
            auto signs=turbo3_signs(d,0x5455524230u);
            std::vector<float> qr(d),kr(d),vr(d),rot(d);
            turbo3_rotate(q.data(),d,signs.data(),qr.data());
            double domain_error=0;
            for(unsigned splits:{1u,30u,128u}) {
                std::vector<double> result(d,0),weights(splits,0);
                std::vector<std::vector<float>> values(splits,std::vector<float>(d,0));
                std::vector<double> ls(splits,-INFINITY);
                for(unsigned sp=0;sp<splits;++sp) {
                    const unsigned begin=uint64_t(n)*sp/splits,end=uint64_t(n)*(sp+1)/splits;
                    float m=-1e30f,l=0;
                    std::fill(rot.begin(),rot.end(),0);
                    for(unsigned t=begin;t<end;++t) {
                        // Decode the bit planes without undoing the rotation.
                        std::vector<uint32_t> kc(3*d/32),vc(3*d/32);
                        std::vector<uint16_t> ks(d/32),vs(d/32);
                        std::memcpy(ks.data(),pk.data()+1+t*bytes,d/16);
                        std::memcpy(vs.data(),pv.data()+1+t*bytes,d/16);
                        std::memcpy(kc.data(),pk.data()+1+t*bytes+d/16,3*d/8);
                        std::memcpy(vc.data(),pv.data()+1+t*bytes+d/16,3*d/8);
                        turbo3_unpack_rotated(kc.data(),ks.data(),d,kr.data());
                        turbo3_unpack_rotated(vc.data(),vs.data(),d,vr.data());
                        float score=0;
                        for(unsigned j=0;j<d;++j) score+=qr[j]*kr[j];
                        score/=std::sqrt(float(d));
                        const float nm=std::max(m,score),a=std::exp(m-nm),b=std::exp(score-nm);
                        l=l*a+b;m=nm;
                        for(unsigned j=0;j<d;++j)rot[j]=std::fma(b,vr[j],rot[j]*a);
                    }
                    if(l>0) {
                        for(auto& x:rot)x/=l;
                        turbo3_unrotate(rot.data(),d,signs.data(),values[sp].data());
                        ls[sp]=m+std::log(l);
                    }
                }
                const double peak=*std::max_element(ls.begin(),ls.end());
                double den=0;
                for(unsigned sp=0;sp<splits;++sp) {
                    weights[sp]=ls[sp]==-INFINITY?0:std::exp(ls[sp]-peak);den+=weights[sp];
                    for(unsigned j=0;j<d;++j)result[j]+=weights[sp]*values[sp][j];
                }
                for(unsigned j=0;j<d;++j)domain_error=std::max(domain_error,std::fabs(result[j]/den-out[j]));
            }
            VJ_CHECK(domain_error<2e-6);
            std::printf("  rotate-once attention max_abs=%.8g (S=1,30,128)\n",domain_error);
        }
        // Zero query must give exactly the arithmetic mean of decoded V, including turbo's inverse rotation.
        std::fill(q.begin(),q.end(),0);
        out=device_attention_reference(q.data(),pk.data()+1,pv.data()+1,d,n,f);
        std::vector<double> mean(d,0);
        for(unsigned t=0;t<n;++t) {unpack_device_vector(pv.data()+1+t*bytes,d,f,decoded.data());for(unsigned j=0;j<d;++j)mean[j]+=decoded[j]/double(n);}
        for(unsigned j=0;j<d;++j)VJ_CHECK(std::fabs(out[j]-mean[j])<1e-7);
        out=device_attention_reference(q.data(),pk.data()+1,pv.data()+1,d,0,f);
        for(auto x:out)VJ_CHECK(x==0);
        // Zero input is representable in all four formats (turbo's zero scale overrides nonzero centroids).
        std::vector<uint8_t> zero(bytes);pack_device_vector(q.data(),d,f,zero.data());unpack_device_vector(zero.data(),d,f,decoded.data());
        for(auto x:decoded)VJ_CHECK(x==0);
        // Native Q8 and fp16 scale planes: explicit independent byte decoder.
        if(ff<2) for(unsigned j=0;j<d;++j) {
            uint16_t h;std::memcpy(&h,pk.data()+1+(ff==0?j*2:(j/32)*2),2);
            float expected=f16_to_f32(h)*(ff==0?1:int8_t(pk[1+d/16+j]));
            unpack_device_vector(pk.data()+1,d,f,decoded.data());VJ_CHECK(decoded[j]==expected);
        }
    }
    std::puts("PASS: device layouts and independent attention/zero/empty references (CPU only)");
}

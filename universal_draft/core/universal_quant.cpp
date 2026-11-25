#include "core/universal_quant.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace vegajet::universal {
namespace {
constexpr Format formats[]={{256,50,"IQ1_S"},{256,56,"IQ1_M"},{256,54,"TQ1_0"},{256,66,"TQ2_0"},{32,17,"MXFP4"}};
size_t odd256(size_t n) noexcept { const size_t tiles=n/256+(n%256!=0); return (tiles+(tiles%2==0))*256; }
float half(uint16_t h) noexcept {uint32_t s=uint32_t(h&0x8000)<<16,e=(h>>10)&31,m=h&1023,u;if(!e){if(!m)u=s;else{uint32_t mm=m,ee=113;while(!(mm&1024)){mm<<=1;--ee;}u=s|(ee<<23)|((mm&1023)<<13);}}else if(e==31)u=s|0x7f800000|(m<<13);else u=s|((e+112)<<23)|(m<<13);float f;std::memcpy(&f,&u,4);return f;}
}


const Format* format(Type t) noexcept {switch(t){case Type::IQ1_S:return &formats[0];case Type::IQ1_M:return &formats[1];case Type::TQ1_0:return &formats[2];case Type::TQ2_0:return &formats[3];case Type::MXFP4:return &formats[4];}return nullptr;}
const Format* format(std::string_view n) noexcept {for(const auto& f:formats)if(n==f.name)return &f;return nullptr;}
bool decode(Type t,const uint8_t* src,size_t bytes,float* dst,size_t cap,size_t n,const float* table,size_t table_n) noexcept {
 const auto* f=format(t);if(!f||n%f->block_elems||n>cap||n/f->block_elems>std::numeric_limits<size_t>::max()/f->block_bytes||bytes<n/f->block_elems*f->block_bytes)return false;
 if(!n)return true;
 if(!src||!dst)return false;
 // The canonical signed IQ1 grid is injected by the caller; never invent one.
 if(t==Type::IQ1_S){if(!table||table_n!=2048*8)return false;for(size_t i=0;i<table_n;++i)if(!std::isfinite(table[i]))return false;for(size_t b=0;b<n/256;++b){const uint8_t*p=src+b*50;const float d=half(uint16_t(p[0])|(uint16_t(p[1])<<8));const uint8_t*qs=p+2;for(size_t ib=0;ib<8;++ib){const uint16_t h=uint16_t(p[34+2*ib])|(uint16_t(p[35+2*ib])<<8);const float scale=d*float(2*((h>>12)&7)+1),delta=(h&0x8000)?-.125f:.125f;for(size_t l=0;l<4;++l){const size_t idx=qs[4*ib+l]|(size_t((h>>(3*l))&7)<<8);for(size_t j=0;j<8;++j)dst[b*256+ib*32+l*8+j]=scale*(table[idx*8+j]+delta);}}}return true;}
 if(t==Type::IQ1_M){if(!table||table_n!=2048*8)return false;for(size_t i=0;i<table_n;++i)if(!std::isfinite(table[i]))return false;for(size_t b=0;b<n/256;++b){const uint8_t*p=src+b*56;const uint8_t*qs=p,*qh=p+32;uint16_t sc[4];for(size_t i=0;i<4;++i)sc[i]=uint16_t(p[48+2*i])|(uint16_t(p[49+2*i])<<8);const uint16_t base_bits=uint16_t((sc[0]>>12)|((sc[1]>>8)&0x00f0)|((sc[2]>>4)&0x0f00)|(sc[3]&0xf000));const float base=half(base_bits);for(size_t ib=0;ib<8;++ib)for(size_t l=0;l<4;++l){const uint8_t nib=uint8_t((qh[2*ib+l/2]>>(4*(l%2)))&15);const size_t idx=qs[4*ib+l]|(size_t(nib&7)<<8);const float local=base*float(2*((sc[ib/2]>>(6*(ib%2)+3*(l/2)))&7)+1),delta=(nib&8)?-.125f:.125f;for(size_t j=0;j<8;++j)dst[b*256+ib*32+l*8+j]=local*(table[idx*8+j]+delta);}}return true;}
 if(t==Type::TQ1_0){for(size_t b=0;b<n/256;++b){const uint8_t*p=src+b*54;const float d=half(uint16_t(p[52])|(uint16_t(p[53])<<8));size_t o=b*256;for(size_t g=0;g<5;++g){for(size_t j=0;j<32;++j){const uint16_t v=uint16_t(uint8_t(uint16_t(p[j])*uint16_t(g==0?1:g==1?3:g==2?9:g==3?27:81)))*3;const unsigned t=v>>8;dst[o++]=(int(t)-1)*d;}}for(size_t g=0;g<5;++g){for(size_t j=0;j<16;++j){const uint16_t v=uint16_t(uint8_t(uint16_t(p[32+j])*uint16_t(g==0?1:g==1?3:g==2?9:g==3?27:81)))*3;const unsigned t=v>>8;dst[o++]=(int(t)-1)*d;}}for(size_t g=0;g<4;++g){for(size_t j=0;j<4;++j){const uint16_t v=uint16_t(uint8_t(uint16_t(p[48+j])*uint16_t(g==0?1:g==1?3:g==2?9:27)))*3;const unsigned t=v>>8;dst[o++]=(int(t)-1)*d;}}}return true;}
 if(t==Type::TQ2_0){for(size_t b=0;b<n/256;++b){const uint8_t*p=src+b*66;const float d=half(uint16_t(p[64])|(uint16_t(p[65])<<8));for(size_t half_i=0;half_i<2;++half_i)for(size_t shift_group=0;shift_group<4;++shift_group)for(size_t j=0;j<32;++j){const uint8_t q=(p[half_i*32+j]>>(shift_group*2))&3;dst[b*256+half_i*128+shift_group*32+j]=(int(q)-1)*d;}}return true;}
 if(t!=Type::MXFP4)return false;
 for(size_t b=0;b<n/32;++b){const uint8_t*p=src+b*17;const float scale=std::ldexp(1.0f,int(p[0])-127);if(!std::isfinite(scale))return false;static constexpr float v[8]={0.f,.5f,1.f,1.5f,2.f,3.f,4.f,6.f};for(size_t i=0;i<32;++i){const uint8_t q=(p[1+i%16]>>(4*(i/16)))&15;const float x=v[q&7]*scale;dst[b*32+i]=(q&8)?-x:x;}}
 return true;
}
const std::vector<NativeFormat>& native_formats(){static const std::vector<NativeFormat>v={{0,"VJ_Q4_A8"},{1,"VJ_Q8_DP4A"},{2,"VJ_Q4_DOT8"},{3,"VJ_IQ_PACKED"}};return v;}
std::vector<size_t> select_upgrades(const std::vector<Upgrade>&a,uint64_t cap){struct S{uint64_t c;double g;std::vector<size_t> p;};std::vector<S>d{{0,0,{}}};for(size_t i=0;i<a.size();++i){if(!a[i].extra_bytes||!(a[i].saved_ms>0))continue;auto n=d;for(const auto&s:d)if(s.c<=cap&&a[i].extra_bytes<=cap-s.c){auto p=s.p;p.push_back(i);n.push_back({s.c+a[i].extra_bytes,s.g+a[i].saved_ms,std::move(p)});}std::sort(n.begin(),n.end(),[](const S&x,const S&y){return x.c<y.c||(x.c==y.c&&x.g>y.g);});d.clear();double best=-1;for(auto&s:n)if(s.g>best){best=s.g;d.push_back(std::move(s));}}return d.back().p;}
bool crossover_wins(double old_bw,double new_bw,double old_bpw,double new_bpw) noexcept {return std::isfinite(old_bw)&&std::isfinite(new_bw)&&std::isfinite(old_bpw)&&std::isfinite(new_bpw)&&old_bw>0&&old_bpw>0&&new_bpw>0&&new_bw/old_bw>new_bpw/old_bpw;}
double roi_ms_per_gib(double saved,uint64_t extra) noexcept {return extra&&std::isfinite(saved)&&saved>0?saved*1073741824.0/double(extra):0.0;}
double min_i_star(uint32_t ncu,double sclk,double eta,double bw) noexcept {return ncu&&std::isfinite(sclk)&&std::isfinite(eta)&&std::isfinite(bw)&&sclk>0&&eta>0&&bw>0?double(ncu)*64.0*sclk*eta/bw:0.0;}
uint64_t profile_budget_bytes(uint32_t gib) noexcept {return (gib==16||gib==32||gib==96)?uint64_t(gib)*1024*1024*1024:0;}
Strides odd_plane_strides(size_t q,size_t d) noexcept{return {odd256(q),odd256(d)};}
void pack_odd_planes(const uint8_t*q,size_t qb,const uint8_t*d,size_t db,size_t rows,uint8_t*qo,uint8_t*doo,Strides s) noexcept {if((rows&&(!q||!d||!qo||!doo))||s.qs_bytes<qb||s.dm_bytes<db)return;for(size_t r=0;r<rows;++r){std::memcpy(qo+r*s.qs_bytes,q+r*qb,qb);std::memcpy(doo+r*s.dm_bytes,d+r*db,db);}}
void unpack_odd_planes(uint8_t*q,size_t qb,uint8_t*d,size_t db,size_t rows,const uint8_t*qi,const uint8_t*di,Strides s) noexcept {if((rows&&(!q||!d||!qi||!di))||s.qs_bytes<qb||s.dm_bytes<db)return;for(size_t r=0;r<rows;++r){std::memcpy(q+r*qb,qi+r*s.qs_bytes,qb);std::memcpy(d+r*db,di+r*s.dm_bytes,db);}}
uint32_t plane_channel_spread(size_t stride,size_t rows,size_t tile,size_t tile_bytes,size_t channels) noexcept {if(!tile_bytes||!channels||channels>32)return 0;uint32_t mask=0;for(size_t r=0;r<rows;++r)mask|=uint32_t(1)<<(((r*stride+tile*tile_bytes)/tile_bytes)%channels);return static_cast<uint32_t>(__builtin_popcount(mask));}
double fleet_crossover(const std::vector<BoardSample>& ranks) noexcept {
 if(ranks.empty()||ranks.size()>8)return 0;
 double peak=0,bw=0;
 for(const auto& r:ranks){if(!r.usable_bytes||!r.cap_w||r.eta>1||!min_i_star(r.cu,r.clock_ghz,r.eta,r.hbm_gbs))return 0;
  peak+=r.cu*64.0*r.clock_ghz*r.eta;bw+=r.hbm_gbs;}
 return peak/bw;
}
Candidate bandwidth_candidate(uint64_t a,uint64_t b,double old_bw,double new_bw) noexcept {
 if(!a||b<=a||!(old_bw>0)||!(new_bw>0)||!std::isfinite(old_bw)||!std::isfinite(new_bw))return {};
 // For the same tensor, packed byte ratio is precisely its bits/weight ratio.
 return {b-a,old_bw,new_bw,1.0,double(b)/double(a),double(a)/(old_bw*1e6)-double(b)/(new_bw*1e6)};
}
Residency co_budget(uint64_t free,uint64_t blocks,const uint32_t (&sizes)[3],const std::vector<Candidate>& candidates){
 Residency r{false,KvTier::Q8_0,0,0,{}};
 if(!blocks||!sizes[2]||sizes[1]<sizes[2]||sizes[0]<sizes[1])return r;
 for(unsigned i=0;i<3;++i){if(blocks>free/sizes[i])continue;
  r.fits=true;r.tier=static_cast<KvTier>(i);r.kv_bytes=blocks*sizes[i];std::vector<Upgrade> u;
  for(const auto& c:candidates)u.push_back({c.extra,crossover_wins(c.old_bw,c.new_bw,c.old_bpw,c.new_bpw)?c.saved_ms:0.});
  r.chosen=select_upgrades(u,free-r.kv_bytes);for(auto idx:r.chosen)r.upgrade_bytes+=candidates[idx].extra;return r;
 }return r;
}
KvPlan plan_profile_kv(uint64_t budget,uint32_t requested,uint32_t layers,uint32_t heads,uint32_t dim,uint32_t bytes,double fraction) noexcept {if(!layers||!heads||!dim||!bytes||!std::isfinite(fraction)||fraction<=0)return {0,0};fraction=std::min(fraction,1.0);const long double per=2.0L*layers*heads*dim*bytes;const uint64_t usable=static_cast<uint64_t>(budget*fraction);const uint64_t fit=static_cast<uint64_t>(usable/per);const uint32_t tok=static_cast<uint32_t>(std::min<uint64_t>(requested,fit));return {static_cast<uint64_t>(per*tok),tok};}
double cosine_similarity(const float*a,const float*b,size_t n) noexcept {if((!a||!b)&&n)return 0;double dot=0,x=0,y=0;for(size_t i=0;i<n;++i){dot+=double(a[i])*b[i];x+=double(a[i])*a[i];y+=double(b[i])*b[i];}return x==0||y==0?0:dot/std::sqrt(x*y);}
} // namespace vegajet::universal

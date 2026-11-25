// assert() must stay live under any build type (RelWithDebInfo / an explicit -DNDEBUG would otherwise make every
// check below a no-op and the test pass vacuously).
#undef NDEBUG
#include <cassert>
#include "core/universal_quant.hpp"
#include <cmath>
#include <cstdint>
#include <vector>
#include <string_view>
using namespace vegajet::universal;
int main(){
 for(unsigned n=1;n<=8;++n)for(unsigned cap:{150u,200u}){
  std::vector<BoardSample> boards;double peak=0,bw=0;
  for(unsigned i=0;i<n;++i){BoardSample s{i%2?64u:60u,cap,(i%2?32ull:16ull)<<30,1.08+.06*(i%2),.8,700.+10*i};boards.push_back(s);peak+=s.cu*64*s.clock_ghz*s.eta;bw+=s.hbm_gbs;}
  assert(std::abs(fleet_crossover(boards)-peak/bw)<1e-12);
 }
 assert(fleet_crossover({})==0);
 auto win=bandwidth_candidate(100000000,150000000,100,200),lose=bandwidth_candidate(100000000,150000000,100,125);
 assert(win.saved_ms==.25&&lose.saved_ms<0&&roi_ms_per_gib(win.saved_ms,win.extra)>0);
 const uint32_t sizes[3]={34,18,14};const std::vector<Candidate> offers={{10,100,101,2,4,9},{4,100,200,4,5,3}};for(unsigned i=0;i<3;++i){const auto r=co_budget(100*sizes[i]+4,100,sizes,offers);assert(r.fits&&static_cast<unsigned>(r.tier)==i&&r.kv_bytes==100*sizes[i]&&r.chosen==std::vector<size_t>{1}&&r.upgrade_bytes==4);}assert(!co_budget(1399,100,sizes,offers).fits);

 assert(format(Type::MXFP4)->block_elems==32&&format("IQ1_S")->block_bytes==50);
 assert(native_formats().size()==4&&std::string_view(native_formats()[0].name)=="VJ_Q4_A8");
 assert(crossover_wins(100,150,4,5));assert(!crossover_wins(100,120,4,5));
 // Real 0/1 choice: taking the locally best density item is suboptimal here.
 const std::vector<Upgrade> u={{6,30},{5,24},{5,24}};
 const auto chosen=select_upgrades(u,10);double saved=0;uint64_t used=0;for(size_t i:chosen){saved+=u[i].saved_ms;used+=u[i].extra_bytes;}assert(used<=10&&saved==48&&chosen.size()==2);
 const Strides s=odd_plane_strides(1536,2048);assert(s.qs_bytes==1792&&s.dm_bytes==2304);
 // Q4A8: sixteen packed bytes and four scale/minimum bytes per K32.
 for(size_t k:{1536u,2048u,2560u,3072u,4096u,5120u,6144u,8192u}){
  const size_t qb=k/2, db=k/8;const Strides padded=odd_plane_strides(qb,db);
  for(auto plane:{std::pair<size_t,size_t>{qb,padded.qs_bytes},{db,padded.dm_bytes}}){
   assert(plane.second>=plane.first&&plane.second%512==256);
   for(size_t off=0;off<plane.first;off+=256)assert(plane_channel_spread(plane.second,16,off/256)==16);
  }
  if(k==4096){assert(db==512&&plane_channel_spread(db,16,0)==8&&padded.dm_bytes==768);}
 }
 constexpr size_t rows=8,qb=1536,db=2048;const Strides st=odd_plane_strides(qb,db);std::vector<uint8_t>q(rows*qb),d(rows*db),qp(rows*st.qs_bytes),dp(rows*st.dm_bytes),qr(q.size()),dr(d.size());for(size_t i=0;i<q.size();++i)q[i]=uint8_t(i*19);for(size_t i=0;i<d.size();++i)d[i]=uint8_t(i*37);pack_odd_planes(q.data(),qb,d.data(),db,rows,qp.data(),dp.data(),st);unpack_odd_planes(qr.data(),qb,dr.data(),db,rows,qp.data(),dp.data(),st);assert(q==qr&&d==dr);assert(plane_channel_spread(st.qs_bytes,16,0)==16);assert(plane_channel_spread(st.dm_bytes,16,0)==16);
 const auto kv=plan_profile_kv(1ull<<30,8192,32,8,128,2,0.9);assert(kv.tokens>0&&kv.tokens<=8192&&kv.bytes<=uint64_t((1ull<<30)*0.9));
 const float a[]={1,2,3},b[]={2,4,6};assert(std::abs(cosine_similarity(a,b,3)-1.0)<1e-12);const float c[]={1,0,0},e[]={0,1,0};assert(cosine_similarity(c,e,3)==0);const float ca[]={1,.5f,.25f},cb[]={1,.5001f,.25f};assert(cosine_similarity(ca,cb,3)>=.999);assert(roi_ms_per_gib(2.0,1ull<<30)==2.0);assert(std::abs(min_i_star(60,1.0,.8,800.0)-3.84)<1e-12);assert(profile_budget_bytes(16)==16ull*1024*1024*1024&&profile_budget_bytes(96)==96ull*1024*1024*1024&&profile_budget_bytes(48)==0);
 // MXFP4 one block: scale=1, values 0, .5, 1, 1.5, 2, 3, 4, 6 and signs.
 uint8_t block[17]={127};for(int i=0;i<16;++i)block[i+1]=uint8_t((i%8)|(((i%8)|8)<<4));float out[32];assert(decode(Type::MXFP4,block,sizeof block,out,32,32));assert(out[1]==.5f&&out[16]==0&&out[17]==-.5f&&out[31]==-6.f);
 assert(!decode(Type::IQ1_S,block,sizeof block,out,32,32));assert(!decode(Type::TQ1_0,block,sizeof block,out,32,32));
 // A deliberately synthetic table tests index/scaling wiring, not canonical IQ accuracy.
 std::vector<float> codebook(2048*8);uint8_t iq1block[50]={};float iqtemporary[256];assert(!decode(Type::IQ1_S,iq1block,sizeof iq1block,iqtemporary,256,256));assert(!decode(Type::IQ1_S,iq1block,sizeof iq1block,iqtemporary,256,256,codebook.data(),codebook.size()-1));for(size_t i=0;i<2048;++i)for(size_t j=0;j<8;++j)codebook[i*8+j]=float(int(i%11)-5)*.125f+float(j)*.03125f;
 uint8_t iq1s[50]={};iq1s[0]=0x00;iq1s[1]=0x3c;for(size_t ib=0;ib<8;++ib){const uint16_t h=uint16_t((ib&7)<<12)|((ib&1)?0x8000:0);iq1s[34+2*ib]=uint8_t(h);iq1s[35+2*ib]=uint8_t(h>>8);for(size_t l=0;l<4;++l)iq1s[2+4*ib+l]=uint8_t(ib*4+l);}float iqout[256];assert(decode(Type::IQ1_S,iq1s,sizeof iq1s,iqout,256,256,codebook.data(),codebook.size()));
 for(size_t ib=0;ib<8;++ib)for(size_t l=0;l<4;++l)for(size_t j=0;j<8;++j){const size_t idx=ib*4+l;const float scale=float(2*(ib&7)+1);const float delta=(ib&1)?-.125f:.125f;assert(iqout[ib*32+l*8+j]==scale*(codebook[idx*8+j]+delta));}
 uint8_t iq1m[56]={};for(size_t ib=0;ib<8;++ib)for(size_t l=0;l<4;++l){const uint8_t idx=uint8_t(ib*4+l);iq1m[4*ib+l]=idx;uint8_t& packed=iq1m[32+2*ib+l/2];packed|=uint8_t((idx>>8)<<((l%2)*4));}iq1m[52]=0;iq1m[53]=0xc0;iq1m[54]=0;iq1m[55]=0x30; // reconstruct FP16 1.0 from packed top nibbles
 for(size_t ib=0;ib<8;++ib)for(size_t l=0;l<4;++l){if((ib+l)&1)iq1m[32+2*ib+l/2]|=uint8_t(8<<((l%2)*4));}
 float iqmout[256];assert(decode(Type::IQ1_M,iq1m,sizeof iq1m,iqmout,256,256,codebook.data(),codebook.size()));for(size_t ib=0;ib<8;++ib)for(size_t l=0;l<4;++l)for(size_t j=0;j<8;++j){const size_t idx=ib*4+l;const float delta=((ib+l)&1)?-.125f:.125f;assert(iqmout[ib*32+l*8+j]==codebook[idx*8+j]+delta);}
 uint8_t tq1[54]={};tq1[52]=0;tq1[53]=0x3c;for(size_t i=0;i<32;++i)tq1[i]=uint8_t((i*73+19)&255);for(size_t i=0;i<16;++i)tq1[32+i]=uint8_t((i*47+101)&255);for(size_t i=0;i<4;++i)tq1[48+i]=uint8_t((i*61+37)&255);float ternary1[256];assert(decode(Type::TQ1_0,tq1,sizeof tq1,ternary1,256,256));size_t pos=0;for(size_t g=0;g<5;++g)for(size_t j=0;j<32;++j){const uint16_t v=uint16_t(uint8_t(uint16_t(tq1[j])*(g==0?1:g==1?3:g==2?9:g==3?27:81)))*3;assert(ternary1[pos++]==float(int(v>>8)-1));}for(size_t g=0;g<5;++g)for(size_t j=0;j<16;++j){const uint16_t v=uint16_t(uint8_t(uint16_t(tq1[32+j])*(g==0?1:g==1?3:g==2?9:g==3?27:81)))*3;assert(ternary1[pos++]==float(int(v>>8)-1));}for(size_t g=0;g<4;++g)for(size_t j=0;j<4;++j){const uint16_t v=uint16_t(uint8_t(uint16_t(tq1[48+j])*(g==0?1:g==1?3:g==2?9:27)))*3;assert(ternary1[pos++]==float(int(v>>8)-1));}
 uint8_t tq2[66]={};tq2[64]=0;tq2[65]=0x3c;for(size_t i=0;i<64;++i)tq2[i]=0xe4;float ternary[256];assert(decode(Type::TQ2_0,tq2,sizeof tq2,ternary,256,256));for(size_t i=0;i<256;++i)assert(ternary[i]==float(int((i/32)%4)-1));
}

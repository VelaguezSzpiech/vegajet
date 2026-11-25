// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
// Reference: archive vocab kernel, one lane per 16 bytes; ascending xor tree.
int main(){constexpr int D=3072,N=53,grid=5,pitch=D+256;std::mt19937 rng(865);
 std::vector<int8_t>w(N*pitch),x(D);std::vector<_Float16>d(N*96);std::array<float,96>xd;
 for (auto& z : w) z = int(rng() % 255) - 127;
 for (auto& z : x) z = int(rng() % 255) - 127;
 for (auto& z : d) z = (_Float16)((int(rng() % 2001) - 1000) / 999.3f);
 for (auto& z : xd) z = (int(rng() % 2001) - 1000) / 997.1f;
 std::array<float,N> ref{},got{};for(int ties:{0,1}){
 if(ties){for(int r=1;r<N;++r){std::memcpy(&w[r*pitch],w.data(),D);std::memcpy(&d[r*96],d.data(),96*sizeof(d[0]));}}
 for(int r=0;r<N;++r){std::array<float,64>a{},b{};
 for(int l=0;l<64;++l){for(int i=l;i<D/16;i+=64){int dp=0;for(int j=0;j<16;++j)dp+=int(w[r*pitch+i*16+j])*x[16*i+j];a[l]=std::fma(float(d[r*96+i/2])*xd[i/2],float(dp),a[l]);}
 int dots[3]={};float scales[3];for(int t=0;t<3;++t){int i=l+64*t;scales[t]=float(d[r*96+i/2])*xd[i/2];for(int word=0;word<4;++word)for(int byte=0;byte<4;++byte){int j=4*word+byte;dots[t]+=int(w[r*pitch+i*16+j])*x[16*i+j];}}
 for(int t=0;t<3;++t)b[l]=std::fma(scales[t],float(dots[t]),b[l]);}
 for(int k=1;k<64;k*=2){auto p=a;for(int l=0;l<64;++l)a[l]=p[l]+p[l^k];for(int l=0;l<64;l+=2*k)b[l]+=b[l+k];}
 ref[r]=a[0];got[r]=b[0];if(std::memcmp(&ref[r],&got[r],4))return 1;}
 for(int wg=0;wg<grid;++wg){float best=-1e30f;int id=-1;for(int wave=0;wave<4;++wave)for(int r=4*wg+wave;r<N;r+=grid*4)if(ref[r]>best){best=ref[r];id=r+71;}
 std::array<float,4>v;v.fill(-1e30f);std::array<int,4> ix;ix.fill(-1);for(int r=4*wg;r<N;r+=4*grid)for(int j=0;j<4&&r+j<N;++j)if(got[r+j]>v[j]){v[j]=got[r+j];ix[j]=r+j+71;}
 int pick=0;for(int j=1;j<4;++j)if(v[j]>v[pick])pick=j;if(ix[pick]!=id||std::memcmp(&best,&v[pick],4))return 2;}}
 std::puts("106 logit bits comparisons and 10 WG max/id pairs passed (random and tied, padded rows)");}

// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/acceptmoe.hpp"
#include "core/moe_offload.hpp"
#include "core/spec_cycle.hpp"
#include "tests/test_common.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <random>
using namespace vegajet;
using Row = std::array<float, 256>;
using Route = std::array<int, 8>;

// Independent scalar oracle: stable sort ranks probabilities, then applies the
// union/cold score only to the unprotected ranks. No device helpers used.
std::vector<Route> oracle(const std::vector<Row>& logits, unsigned keep, float acceptance,
                         float penalty, float cold_penalty, const std::array<uint32_t,256>& last) {
    std::array<bool,256> used{};
    std::vector<Route> result;
    for (size_t m=0; m<logits.size(); ++m) {
        const auto& x=logits[m];
        float mx=*std::max_element(x.begin(),x.end());
        Row p{}; for (int e=0;e<256;++e) p[e]=std::exp(x[e]-mx);
        Route row{}; std::array<bool,256> selected{};
        for (unsigned r=0;r<8;++r) {
            int best=-1; float score=-1;
            for (int e=0;e<256;++e) if (!selected[e]) {
                float v=p[e];
                if (m && r>=keep) {
                    if (!used[e]) v*=1-std::pow(acceptance,float(m))*(1-penalty);
                    if (!last[e] || 20-last[e]>8) v*=cold_penalty;
                }
                if (v>score) { score=v;best=e; }
            }
            row[r]=best;selected[best]=true;
        }
        for (int e:row) used[e]=true;
        result.push_back(row);
    }
    return result;
}
// Kernel model: four experts per lane, two key sets, wave maximum with lower
// expert id winning ties. Tests the actual streaming selection against sort.
std::vector<Route> kernel_model(const std::vector<Row>& x, unsigned keep, float acceptance,
                               float penalty, float cold_penalty, const std::array<uint32_t,256>& last) {
    std::array<bool,256> used{};std::vector<Route> out;float commitment=1;
    for (size_t m=0;m<x.size();++m) {
        if (m) commitment*=acceptance;
        Row p{}, biased{}; const float mx=*std::max_element(x[m].begin(),x[m].end());
        for (int e=0;e<256;++e) {
            p[e]=std::exp(x[m][e]-mx);
            biased[e]=p[e]*acceptmoe::multiplier(used[e],m && acceptmoe::cold(last[e],20,8),commitment,penalty,cold_penalty);
        }
        std::array<bool,256> selected{};Route row{};
        for (unsigned r=0;r<8;++r) {
            std::array<int,64> lane{};lane.fill(-1);
            auto better=[&](int a,int b){if(a<0)return b;if(b<0)return a;const Row& v=(!m||r<keep)?p:biased;return v[a]>v[b] || (v[a]==v[b]&&a<b)?a:b;};
            for(int l=0;l<64;++l) for(int c=0;c<4;++c) if(!selected[l+64*c]) lane[l]=better(lane[l],l+64*c);
            for(int d=32;d;d/=2) for(int l=0;l<d;++l) lane[l]=better(lane[l],lane[l+d]);
            row[r]=lane[0];selected[row[r]]=true;
        }
        for(int e:row)used[e]=true;
        out.push_back(row);
    }
    return out;
}
// Archived plain policy expressed independently as a stable sort of remaining
// experts at each rank. The union is frozen until the entire row is selected.
std::vector<Route> plain_oracle(const std::vector<Row>& logits, unsigned keep, float penalty) {
    std::array<bool,256> used{};
    std::vector<Route> out;
    for (size_t m=0;m<logits.size();++m) {
        Row prob{};
        float mx=*std::max_element(logits[m].begin(),logits[m].end());
        for (int e=0;e<256;++e) prob[e]=std::exp(logits[m][e]-mx);
        std::vector<int> remaining(256);
        for (int e=0;e<256;++e) remaining[e]=e;
        Route row{};
        for (unsigned r=0;r<8;++r) {
            auto score=[&](int e) { return (!m || r<keep || used[e]) ? prob[e] : prob[e]*penalty; };
            std::stable_sort(remaining.begin(),remaining.end(),[&](int a,int b) {
                return score(a)>score(b) || (score(a)==score(b) && a<b);
            });
            row[r]=remaining.front();remaining.erase(remaining.begin());
        }
        for (int e:row) used[e]=true;
        out.push_back(row);
    }
    return out;
}
void test_hobbit() {
    using namespace moeoffload;
    char path[]="./vj-hobbit-XXXXXX";int fd=mkstemp(path);VJ_CHECK(fd>=0);unlink(path);
    std::array<uint8_t,64> full{};full.fill(0x55);std::array<uint8_t,32> low{};low.fill(0x22);
    VJ_CHECK(write(fd,full.data(),full.size())==64);VJ_CHECK(write(fd,low.data(),low.size())==32);
    OffloadConfig cfg;cfg.staging_slots=4;cfg.io=IoKind::Preadv;cfg.hot_per_layer=1;
    MoeOffload high({fd,0,1,4,16},cfg), compact({fd,64,1,4,8},cfg);
    VJ_CHECK(high.ok() && compact.ok());
    std::array<uint8_t,16> dst{};std::string err;offload::FetchPrecision precision;
    VJ_CHECK(compact.fetch(0,2,dst.data(),&err));
    setenv("VEGAJET_SPEC_HOBBIT","1",1);
    auto fetch=[&](unsigned e,unsigned row,unsigned rank){dst.fill(0);VJ_CHECK(high.fetch_spec(0,e,row,rank,0.1,&compact,dst.data(),dst.size(),&precision,&err));};
    fetch(2,1,1);VJ_CHECK(precision==offload::FetchPrecision::Low && dst[0]==0x22 && dst[8]==0);
    fetch(2,0,1);VJ_CHECK(precision==offload::FetchPrecision::Full && dst[15]==0x55);
    fetch(2,1,0);VJ_CHECK(precision==offload::FetchPrecision::Full);
    VJ_CHECK(compact.fetch(0,0,dst.data(),&err));fetch(0,1,2);VJ_CHECK(precision==offload::FetchPrecision::Full);
    fetch(3,1,1);VJ_CHECK(precision==offload::FetchPrecision::Full); // low not staged
    VJ_CHECK(high.fetch_spec(0,2,1,1,0.1,&high,dst.data(),dst.size(),&precision,&err));
    VJ_CHECK(precision==offload::FetchPrecision::Full); // aliased low table is never Low
    VJ_CHECK(high.fetch_spec(0,2,1,1,0.1,nullptr,dst.data(),dst.size(),&precision,&err));
    VJ_CHECK(precision==offload::FetchPrecision::Full);
    setenv("VEGAJET_SPEC_HOBBIT","0",1);fetch(2,1,1);VJ_CHECK(precision==offload::FetchPrecision::Full);
    VJ_CHECK(!high.fetch_spec(0,2,1,1,0.1,&compact,dst.data(),7,&precision,&err));
    unsetenv("VEGAJET_SPEC_HOBBIT");close(fd);
}
int main() {
    VJ_CHECK(!acceptmoe::commit_mode(nullptr));
    VJ_CHECK(!acceptmoe::commit_mode("plain"));
    VJ_CHECK(!acceptmoe::commit_mode("invalid"));
    VJ_CHECK(acceptmoe::commit_mode("commit"));
    VJ_CHECK(acceptmoe::plain_equivalent(1,1,8));
    VJ_CHECK(acceptmoe::plain_equivalent(1,0.5f,0));
    VJ_CHECK(!acceptmoe::plain_equivalent(0.75f,1,0));
    VJ_CHECK(!acceptmoe::plain_equivalent(1,0.5f,8));
    // Endpoint must preserve arbitrary penalty bits, not 1-(1-penalty).
    for (float penalty : {0.0f,0.1f,0.25f,0.9f,1.0f})
        VJ_CHECK(acceptmoe::multiplier(false,false,1,penalty,1)==penalty);
    VJ_CHECK(acceptmoe::multiplier(false,false,0,0.25f,0.5f)==1);
    VJ_CHECK(acceptmoe::multiplier(false,false,1,0.25f,0.5f)==0.25f);
    VJ_CHECK(acceptmoe::multiplier(true,true,1,0.25f,0.5f)==0.5f);
    VJ_CHECK(!acceptmoe::cold(0,20,0));VJ_CHECK(acceptmoe::cold(0,20,8));
    VJ_CHECK(!acceptmoe::cold(12,20,8));VJ_CHECK(acceptmoe::cold(11,20,8));
    std::mt19937 rng(123);std::uniform_real_distribution<float> dist(-4,4);
    for (unsigned trial=0;trial<32;++trial) {
        std::vector<Row> x(16);std::array<uint32_t,256> last{};
        for(auto& row:x)for(float& v:row)v=dist(rng);
        if (trial==0) for(auto& row:x)row.fill(0); // deterministic tie handling
        for(unsigned e=0;e<256;++e)last[e]=(e%3==0)?19:0;
        for (float pen : {0.0f,0.1f,0.25f,1.0f}) for (unsigned keep : {1u,2u,8u}) {
            const auto plain=plain_oracle(x,keep,pen);
            VJ_CHECK(plain==kernel_model(x,keep,1,pen,1,last));
        }
        for(float a:{0.0f,0.75f,1.0f})for(float pen:{0.0f,0.1f,1.0f})for(float cold:{0.0f,0.25f,1.0f})for(unsigned keep:{1u,2u,8u}) {
            auto ref=oracle(x,keep,a,pen,cold,last), model=kernel_model(x,keep,a,pen,cold,last);
            VJ_CHECK(ref==model);
            VJ_CHECK(model[0]==oracle({x[0]},8,1,1,1,last)[0]);
            // Device weights must remain true selected probabilities, not biased
            // ranking scores. Model float softmax+renormalization vs double oracle.
            for (size_t m=0; m<x.size(); ++m) {
                const float mx=*std::max_element(x[m].begin(),x[m].end());
                Row p{};float sum=0;
                for(int e=0;e<256;++e) {p[e]=std::exp(x[m][e]-mx);sum+=p[e];}
                float chosen=0;double exact=0;
                for(int e:model[m]) {p[e]/=sum;chosen+=p[e];exact+=std::exp(double(x[m][e])-double(mx));}
                float normalized=0;
                for(int e:model[m]) {
                    const float w=p[e]/chosen;
                    const double refw=std::exp(double(x[m][e])-double(mx))/exact;
                    VJ_CHECK(std::abs(double(w)-refw)<2e-6);
                    normalized+=w;
                }
                VJ_CHECK(std::abs(normalized-1.0f)<2e-6f);
            }
            for(auto row:model) {std::sort(row.begin(),row.end());VJ_CHECK(std::adjacent_find(row.begin(),row.end())==row.end());}
        }
    }
    // A cold, slightly higher-probability expert loses to a recently used expert
    // only in the biased ranks. Own top-1 and context row stay unchanged.
    std::vector<Row> x(2);for(auto& r:x)r.fill(-10);
    x[1][0]=5;x[1][1]=1;x[1][2]=0.9f;
    std::array<uint32_t,256> last{};last[2]=19;
    auto biased=oracle(x,1,0,1,0.25f,last), neutral=oracle(x,1,0,1,1,last);
    VJ_CHECK(biased[1][0]==0 && biased[1][1]==2 && neutral[1][1]==1);
    test_hobbit();
    spec::CycleTimes t;t.add(1,2,3,4);t.add(2,3,4,5);
    VJ_CHECK(t.cycles==2 && t.draft+t.verify+t.catchup+t.recompute==24);
    std::puts("AcceptMoE oracle, commitment, recency, HOBBIT fake fetch, cycle accounting passed");
}

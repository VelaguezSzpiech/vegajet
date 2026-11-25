// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// C++17 CPU algebra/rounding verification, not device-kernel parity evidence.
// Requires chronological absolute error <= 1e-5 and exact durable commits
// after clearing/poisoning temporary sources, for every accepted prefix.
#include "ops/kernels/vj_gdn_treewy_verify.hiph"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

using vegajet::treewy::Record;
using Vec=std::vector<float>;
constexpr int H=32,D=128,K=8,State=H*D*D;
static void require(bool b,const char* why) {
    if (!b) { std::fprintf(stderr,"FAIL: %s\n",why); std::exit(1); }
}
static uint32_t bits(float f) { uint32_t b; std::memcpy(&b,&f,4); return b; }
static float mul(float a,float b) { volatile float r=a*b; return r; }
static float sub(float a,float b) { volatile float r=a-b; return r; }
struct Inputs { int m; Vec s,q,k,v,g,b; std::vector<int> parent; };
struct Oracle { std::vector<Vec> states; Vec output; };

// Independent transcription of pristine production arithmetic, not the WY
// helper: 8 groups of 16 FMAs, sum groups left-to-right, beta*(v-g*pred),
// then fmaf(k,dv,g*S), then the same grouped dot with q. The unchanged
// parent state is selected first; each root-to-node path is chronological.
// Scope: raw recurrence only, already prepared FP32 q/k/v/g/b inputs.
static Oracle chronological(const Inputs& in) {
    Oracle ref; ref.states.reserve(in.m); ref.output.resize(in.m*H*D);
    for (int n=0;n<in.m;++n) {
        Vec s=in.parent[n]<0 ? in.s : ref.states[in.parent[n]];
        for (int h=0;h<H;++h) for (int c=0;c<D;++c) {
            float p[8]={},o[8]={};
            for (int group=0;group<8;++group) for (int r=0;r<16;++r) {
                int row=group*16+r;
                p[group]=std::fma(s[h*D*D+row*D+c],in.k[(n*K+h%K)*D+row],p[group]);
            }
            float pred=p[0]; for (int group=1;group<8;++group) pred=pred+p[group];
            float dv=mul(in.b[n*H+h],sub(in.v[(n*H+h)*D+c],mul(in.g[n*H+h],pred)));
            for (int group=0;group<8;++group) for (int r=0;r<16;++r) {
                int row=group*16+r,idx=h*D*D+row*D+c;
                s[idx]=std::fma(in.k[(n*K+h%K)*D+row],dv,mul(in.g[n*H+h],s[idx]));
                o[group]=std::fma(s[idx],in.q[(n*K+h%K)*D+row],o[group]);
            }
            float out=o[0]; for (int group=1;group<8;++group) out=out+o[group];
            ref.output[(n*H+h)*D+c]=out;
        }
        ref.states.push_back(std::move(s));
    }
    return ref;
}
static float reconstruct(const Record& r,int a,int row,int c) {
    float z=r.initial[row*32+c];
    if (a) {
        const int i=a-1;
        z=mul(r.root[i],z);
        for (int j=0;j<=i;++j) if (r.path[i*12+j]!=0.0f)
            z=std::fma(r.key[j*128+row],mul(r.path[i*12+j],r.delta[j*32+c]),z);
    }
    return z;
}
static Vec commit(const std::vector<Record>& rec,int a) {
    Vec state(State);
    for (int wg=0;wg<128;++wg) for (int row=0;row<D;++row) for (int c=0;c<32;++c)
        state[(wg/4)*D*D+row*D+(wg%4)*32+c]=reconstruct(rec[wg],a,row,c);
    return state;
}

// CPU model of the header's actual unit-triangular inverse-by-doubling solve.
// Temporary matrices are deliberately separate from durable Record storage.
static Vec treewy(const Inputs& in,std::vector<Record>& records,Vec& scratch) {
    const int m=in.m;
    Vec out(m*H*D);
    scratch.resize(2*144+2*384);
    for (int wg=0;wg<128;++wg) {
        int h=wg/4,part=wg%4;
        Record& r=records[wg];
        for (int row=0;row<D;++row) for (int c=0;c<32;++c)
            r.initial[row*32+c]=in.s[h*D*D+row*D+part*32+c];
        for (int i=0;i<m;++i) {
            std::copy_n(in.k.data()+(i*K+h%K)*D,D,r.key+i*D);
            float d=1;
            for (int j=0;j<m;++j) r.path[i*12+j]=0;
            for (int j=i;j>=0;j=in.parent[j]) { r.path[i*12+j]=d; d=mul(d,in.g[j*H+h]); }
            r.root[i]=d;
        }
        float* b=scratch.data(); float* bn=b+144; float* x=bn+144; float* xn=x+384;
        std::fill(scratch.begin(),scratch.end(),0.0f);
        for (int i=0;i<m;++i) {
            for (int j=0;j<i;++j) if (r.path[i*12+j]!=0.0f)
                b[i*12+j]=-mul(mul(in.b[i*H+h],r.path[i*12+j]),vegajet::treewy::dot128(r.key+i*D,r.key+j*D));
            for (int c=0;c<32;++c) {
                float column[D]; for (int row=0;row<D;++row) column[row]=r.initial[row*32+c];
                float pred=vegajet::treewy::dot128(column,r.key+i*D);
                x[i*32+c]=mul(in.b[i*H+h],sub(in.v[(i*H+h)*D+part*32+c],mul(r.root[i],pred)));
            }
        }
        for (int span=1;span<m;span*=2) {
            for (int i=0;i<m;++i) for (int c=0;c<32;++c) {
                float z=x[i*32+c];
                for (int j=0;j<i;++j) z=std::fma(b[i*12+j],x[j*32+c],z);
                xn[i*32+c]=z;
            }
            for (int i=0;i<m;++i) for (int j=0;j<m;++j) {
                float z=0;
                for (int l=j+1;l<i;++l) z=std::fma(b[i*12+l],b[l*12+j],z);
                bn[i*12+j]=z;
            }
            std::copy_n(xn,m*32,x); std::copy_n(bn,144,b);
        }
        std::copy_n(x,m*32,r.delta);
        for (int i=0;i<m;++i) for (int c=0;c<32;++c) {
            float column[D]; for (int row=0;row<D;++row) column[row]=reconstruct(r,i+1,row,c);
            out[(i*H+h)*D+part*32+c]=vegajet::treewy::dot128(column,in.q.data()+(i*K+h%K)*D);
        }
    }
    return out;
}
struct Difference { size_t count=0; double max_abs=0, max_rel=0; };
static Difference compare(const Vec& a,const Vec& b) {
    require(a.size()==b.size(),"size mismatch"); Difference d;
    for (size_t i=0;i<a.size();++i) {
        require(std::isfinite(a[i])&&std::isfinite(b[i]),"nonfinite result");
        d.count+=bits(a[i])!=bits(b[i]);
        const double error=std::abs(double(a[i])-b[i]);
        // True relative error: |actual-reference|/|reference|, without a floor.
        // Zero reference: 0 for zero error, +infinity otherwise.
        const double relative=b[i]!=0.0f ? error/std::abs(double(b[i])) :
            (error==0.0 ? 0.0 : std::numeric_limits<double>::infinity());
        d.max_abs=std::max(d.max_abs,error);
        d.max_rel=std::max(d.max_rel,relative);
    }
    return d;
}
static void accumulate(Difference& total,const Difference& d) {
    total.count+=d.count;
    total.max_abs=std::max(total.max_abs,d.max_abs);
    total.max_rel=std::max(total.max_rel,d.max_rel);
}
static void poison(Vec& v) { std::fill(v.begin(),v.end(),std::numeric_limits<float>::quiet_NaN()); }
int main() {
    Difference all_outputs,all_states;
    std::printf("True relative error = abs(error)/abs(reference); zero reference: 0 if exact, +inf otherwise.\n");
    // Decimal scaling/division generates rounded non-dyadic real inputs (not
    // exactly representable integer/dyadic arithmetic fixtures). Every stored
    // FP32 number is of course itself dyadic; exact dots are not assumed.
    for (int m: {5,8,12}) {
        Difference m_outputs,m_states;
        for (int tree: {0,1}) for (uint32_t seed: {17u,91u}) {
        Inputs in{m,Vec(State),Vec(m*K*D),Vec(m*K*D),Vec(m*H*D),Vec(m*H),Vec(m*H),std::vector<int>(m)};
        std::mt19937 rng(seed+uint32_t(m*101+tree));
        auto random=[&]() { return float(int(rng()%2000001)-1000000)/1000003.0f; };
        for (float& z:in.s) z=random()*0.31f;
        for (float& z:in.q) z=random()*0.21f;
        for (float& z:in.k) z=random()*0.13f;
        for (float& z:in.v) z=random()*0.73f;
        for (float& z:in.g) z=0.87f+random()*0.09f;
        for (float& z:in.b) z=0.45f+random()*0.31f;
        for (int i=0;i<m;++i) in.parent[i]=i==0 ? -1 : (tree ? (i-1)/2 : i-1);
        auto baseline=chronological(in);
        std::vector<Record> records(128); Vec scratch;
        Vec output=treewy(in,records,scratch);
        auto od=compare(output,baseline.output);
        accumulate(m_outputs,od);
        std::printf("M=%d tree=%d seed=%u output bit_mismatches=%zu/%zu max_abs=%.9g max_true_rel=%.9g\n",m,tree,seed,od.count,output.size(),od.max_abs,od.max_rel);
        require(od.max_abs<=1e-5,"output absolute error exceeds 1e-5");
        Vec first_output(output.begin(),output.begin()+H*D);
        Vec first_expected(baseline.output.begin(),baseline.output.begin()+H*D);
        require(compare(first_output,first_expected).count==0,"one-node output ordering must remain exact");
        std::vector<Vec> clean;
        for (int a=0;a<=m;++a) clean.push_back(commit(records,a));
        Vec saved_initial=in.s;
        // Clear EVERY temporary source and work matrix, then poison them.
        // Commit has no access to these objects; only durable records survive.
        for (Vec* v: {&in.s,&in.q,&in.k,&in.v,&in.g,&in.b,&scratch,&output})
            std::fill(v->begin(),v->end(),0.0f);
        std::fill(in.parent.begin(),in.parent.end(),0);
        for (int a=0;a<=m;++a)
            require(compare(commit(records,a),clean[a]).count==0,"cleared scratch changed durable commit bits");
        poison(in.s); poison(in.q); poison(in.k); poison(in.v); poison(in.g); poison(in.b); poison(scratch); poison(output);
        std::fill(in.parent.begin(),in.parent.end(),-99);
        for (int a=0;a<=m;++a) {
            Vec after=commit(records,a);
            auto pd=compare(after,clean[a]); require(pd.count==0,"poisoned scratch changed durable commit bits");
            const Vec& expected=a ? baseline.states[a-1] : saved_initial;
            auto d=compare(after,expected); accumulate(m_states,d);
            std::printf("M=%d tree=%d seed=%u a=%d state bit_mismatches=%zu/%zu max_abs=%.9g max_true_rel=%.9g clear_and_poison=exact\n",m,tree,seed,a,d.count,after.size(),d.max_abs,d.max_rel);
            require(d.max_abs<=1e-5,"state absolute error exceeds 1e-5");
            if (a<=1) require(d.count==0,"zero/one-node ordering must remain exact");
        }
    }
        accumulate(all_outputs,m_outputs); accumulate(all_states,m_states);
        std::printf("M=%d aggregate both trees/seeds: output max_abs=%.9g max_true_rel=%.9g; state max_abs=%.9g max_true_rel=%.9g\n",m,m_outputs.max_abs,m_outputs.max_rel,m_states.max_abs,m_states.max_rel);
    }
    std::printf("M=5/8/12 aggregate both trees/seeds: output max_abs=%.9g max_true_rel=%.9g; state max_abs=%.9g max_true_rel=%.9g\n",all_outputs.max_abs,all_outputs.max_rel,all_states.max_abs,all_states.max_rel);
    std::printf("PASS CPU chronological accuracy and exact clear/poison commits for all prefixes; GPU parity UNTESTED.\n");
    return 0;
}

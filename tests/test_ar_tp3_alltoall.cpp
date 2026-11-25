// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "ops/kernels/vj_ar_tp3_alltoall.hiph"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <thread>
#include <vector>

using namespace vj_tp3;
static double even(double x) {
    double lo=std::floor(x), f=x-lo;
    return lo+(f>0.5 || (f==0.5 && std::fmod(lo,2.0)!=0.0));
}
static float half_oracle(float x) {
    double a=std::fabs(double(x));
    if(a==0) return x;
    int e=0; std::frexp(a,&e);
    double step=std::ldexp(1.0,std::max(-24,e-11));
    return std::copysign(float(even(a/step)*step),x);
}
// Independent byte parser: no production unpack, bits, or half helpers.
static unsigned u16(const uint8_t* p) { return unsigned(p[0])+256u*p[1]; }
static uint32_t u32(const uint8_t* p) { return u16(p)+(uint32_t(u16(p+2))<<16); }
static float parse_half(unsigned h) {
    unsigned e=(h>>10)&31, m=h&1023;
    assert(e!=31);
    float v=e ? std::ldexp(float(1024+m),int(e)-25) : std::ldexp(float(m),-24);
    return h&32768 ? -v : v;
}
static float parse(const uint8_t* p,unsigned i,Wire w) {
    if(w==Wire::fp16) return parse_half(u16(p+2*i));
    const uint8_t* block=p+(i/32)*34;
    int q=block[i%32];
    if(q>=128) q-=256;
    return float(q)*parse_half(u16(block+32));
}
static std::array<float,32> oracle(const float* x,Wire w) {
    std::array<float,32> out{};
    float a=0;
    for(unsigned i=0;i<32;++i) a=std::max(a,std::fabs(x[i]));
    float scale=half_oracle(a/127.0f);
    for(unsigned i=0;i<32;++i) {
        if(w==Wire::fp16) out[i]=half_oracle(x[i]);
        else {
            float ratio=scale==0 ? 0 : x[i]/scale;
            int q=int(std::copysign(even(std::fabs(double(ratio))),ratio));
            out[i]=scale*float(std::max(-127,std::min(127,q)));
        }
    }
    return out;
}
// CPU lane model of the register push: shuffled adjacent bytes, then 17
// disjoint halfword writers for int8 (including an unaligned next tile).
static bool register_pack(const float* x,uint16_t* out,Wire wire) {
    uint16_t reference[32]{};
    if(!pack(x,reference,wire)) return false;
    std::array<float,32> maxima{}, next{};
    for(unsigned lane=0;lane<32;++lane) maxima[lane]=std::fabs(x[lane]);
    for(unsigned delta=16;delta;delta>>=1) {
        for(unsigned lane=0;lane<32;++lane) next[lane]=std::max(maxima[lane],maxima[lane^delta]);
        maxima=next;
    }
    uint16_t h=to_half(maxima[0]/127.0f);
    std::array<unsigned,32> q{}, pair{};
    for(unsigned lane=0;lane<32;++lane) q[lane]=uint8_t(quantize(x[lane],from_half(h)));
    for(unsigned lane=0;lane<31;++lane) pair[lane]=q[lane]|(q[lane+1]<<8);
    for(unsigned lane=0;lane<block_bytes(wire)/2;++lane)
        out[lane]=wire==Wire::fp16 ? to_half(x[lane]) : lane==16 ? h : uint16_t(pair[2*lane]);
    assert(std::equal(out,out+block_bytes(wire)/2,reference));
    return true;
}
struct Mail {
    std::vector<uint8_t> guarded;
    std::atomic<unsigned> epoch{0}; // Host release/acquire for the wire's final word.
    unsigned publications=0;
    uint8_t* data() { return guarded.data()+64; }
    void reset(unsigned bytes) {
        guarded.assign(bytes+128,0xa5);
        std::fill(data(),data()+bytes,0);
        epoch.store(0);
        publications=0;
    }
    void check(unsigned bytes) const {
        for(unsigned i=0;i<64;++i) {
            assert(guarded[i]==0xa5);
            assert(guarded[bytes+64+i]==0xa5);
        }
    }
};
struct Simulation {
    std::array<std::array<Mail,3>,3> inbox;
    std::array<std::array<float,length>,3> input{},output{};
    std::array<std::atomic<unsigned>,3> error{};
    unsigned last_epoch=0;
    void run(unsigned epoch,Wire wire,Partition partition,int missing=-1,
             int stale=-1,unsigned polls=20000) {
        assert(epoch>last_epoch && epoch!=0);
        unsigned payload=wire==Wire::fp16 ? 6144 : 3264;
        unsigned frame=((payload+4+63)/64)*64;
        assert(payload_bytes(wire)==payload && lines(wire)*64==frame);
        assert(epoch_word(wire)*4==frame-4);
        for(auto& dest:inbox) for(auto& mail:dest) mail.reset(frame);
        for(auto& e:error) e.store(0);
        for(auto& o:output) o.fill(12345.0f);
        std::array<std::thread,3> threads;
        for(unsigned rank=0;rank<3;++rank) threads[rank]=std::thread([&,rank] {
            if(int(rank)!=missing) {
                std::vector<uint16_t> packed(payload/2,0);
                std::array<unsigned,tiles> seen{};
                // A wave visits one tile at a time; lanes 0..31 own its values.
                for(unsigned phase=0;phase<3;++phase) {
                    unsigned owner=(rank+phase)%3;
                    for(unsigned tile=range_begin(owner,partition);
                        tile<range_begin(owner+1,partition);++tile) {
                        ++seen[tile];
                        assert(register_pack(input[rank].data()+tile*32,
                                             packed.data()+tile*(wire==Wire::fp16 ? 32 : 17),wire));
                    }
                }
                for(unsigned n:seen) assert(n==1);
                // Exact halfword pushes, including alternating 34-byte alignment.
                for(unsigned i=0;i<payload/2;++i)
                    for(unsigned dest=0;dest<3;++dest) {
                        inbox[dest][rank].data()[2*i]=uint8_t(packed[i]);
                        inbox[dest][rank].data()[2*i+1]=uint8_t(packed[i]>>8);
                    }
                for(unsigned dest=0;dest<3;++dest) {
                    auto& m=inbox[dest][rank];
                    unsigned e=int(rank)==stale ? epoch-1 : epoch;
                    for(unsigned b=0;b<4;++b) m.data()[frame-4+b]=uint8_t(e>>(8*b));
                    ++m.publications;
                    m.epoch.store(e,std::memory_order_release);
                }
            }
            bool ready=false;
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
            for(unsigned n=0;n<(polls==20000 ? 100000000u : polls);++n) {
                if(std::chrono::steady_clock::now()>deadline) break;
                unsigned a=inbox[rank][0].epoch.load(std::memory_order_acquire);
                unsigned b=inbox[rank][1].epoch.load(std::memory_order_acquire);
                unsigned c=inbox[rank][2].epoch.load(std::memory_order_acquire);
                if(a==epoch && b==epoch && c==epoch) { ready=true; break; }
                if(a>epoch || b>epoch || c>epoch) break;
                std::this_thread::yield();
            }
            if(!ready) error[rank].fetch_or(4);
            if(ready) for(unsigned r=0;r<3;++r) assert(u32(inbox[rank][r].data()+frame-4)==epoch);
            for(unsigned i=0;i<length;++i) {
                float result=std::numeric_limits<float>::quiet_NaN();
                if(ready) {
                    float a=parse(inbox[rank][0].data(),i,wire);
                    float b=parse(inbox[rank][1].data(),i,wire);
                    float c=parse(inbox[rank][2].data(),i,wire);
                    volatile float ab=a+b; result=ab+c;
                }
                output[rank][i]=result;
            }
        });
        for(auto& t:threads) t.join(); // Independent single-epoch wire fixtures only.
        last_epoch=epoch;
        for(auto& dest:inbox) for(unsigned r=0;r<3;++r) {
            auto& m=dest[r];
            m.check(frame);
            assert(m.publications==(int(r)==missing ? 0u : 1u));
            for(unsigned i=payload;i<frame-4;++i) assert(m.data()[i]==0);
        }
        bool fail=missing>=0 || stale>=0 || polls==0;
        for(unsigned r=0;r<3;++r) {
            assert((error[r].load()!=0)==fail);
            if(fail) for(float f:output[r]) assert(std::isnan(f));
        }
        if(!fail) for(unsigned tile=0;tile<tiles;++tile) {
            auto a=oracle(input[0].data()+tile*32,wire);
            auto b=oracle(input[1].data()+tile*32,wire);
            auto c=oracle(input[2].data()+tile*32,wire);
            for(unsigned i=0;i<32;++i) {
                volatile float ab=a[i]+b[i]; float expected=ab+c[i];
                for(unsigned r=0;r<3;++r) assert(bits(output[r][tile*32+i])==bits(expected));
            }
        }
    }
};

// Three persistent rank threads, four consecutive epochs, no global barriers.
// Immutable expected bytes are separate from the transport. Each source owns
// its mailbox; release/acquire protects the non-atomic payload just as the GPU
// system release/coherent load does. Two banks prevent overwriting a slow read.
static void asynchronous_epochs(Wire wire,Partition partition) {
    constexpr unsigned epochs=4;
    const unsigned frame=lines(wire)*64, payload=payload_bytes(wire);
    std::array<std::array<std::array<Mail,3>,2>,3> inbox;
    for(auto& rank:inbox) for(auto& bank:rank) for(auto& mail:bank) mail.reset(frame);
    std::array<std::array<std::vector<uint8_t>,3>,epochs+1> expected;
    std::array<std::array<std::array<float,length>,3>,epochs+1> decoded{};
    for(unsigned e=1;e<=epochs;++e) for(unsigned r=0;r<3;++r) {
        auto& bytes=expected[e][r]; bytes.assign(frame,0);
        for(unsigned tile=0;tile<tiles;++tile) {
            float x[32]; uint16_t packed[32];
            for(unsigned lane=0;lane<32;++lane)
                x[lane]=float(int((tile*13+lane*7+r*3)%29)-14)*float(e*4+r+1)/16.0f;
            assert(register_pack(x,packed,wire));
            auto ref=oracle(x,wire);
            for(unsigned lane=0;lane<32;++lane) decoded[e][r][tile*32+lane]=ref[lane];
            for(unsigned h=0;h<block_bytes(wire)/2;++h) {
                unsigned off=tile*block_bytes(wire)+h*2;
                bytes[off]=uint8_t(packed[h]); bytes[off+1]=uint8_t(packed[h]>>8);
            }
        }
        for(unsigned b=0;b<4;++b) bytes[frame-4+b]=uint8_t(e>>(8*b));
        if(e>1) assert(!std::equal(bytes.begin(),bytes.begin()+payload,expected[e-1][r].begin()));
    }
    std::array<std::atomic<unsigned>,3> published{};
    std::array<std::thread,3> threads;
    for(unsigned rank=0;rank<3;++rank) threads[rank]=std::thread([&,rank] {
        unsigned consumed=0;
        for(unsigned e=1;e<=epochs;++e) {
            assert(consumed+1==e);
            unsigned bank=e&1;
            // Mirror the kernel's rotated partition and halfword writer layout.
            for(unsigned phase=0;phase<3;++phase) {
                unsigned owner=(rank+phase)%3;
                for(unsigned tile=range_begin(owner,partition);tile<range_begin(owner+1,partition);++tile)
                    for(unsigned lane=0;lane<block_bytes(wire)/2;++lane) {
                        unsigned off=tile*block_bytes(wire)+2*lane;
                        for(unsigned dest=0;dest<3;++dest) {
                            auto* p=inbox[dest][bank][rank].data();
                            p[off]=expected[e][rank][off]; p[off+1]=expected[e][rank][off+1];
                        }
                    }
            }
            for(unsigned dest=0;dest<3;++dest) {
                auto& mail=inbox[dest][bank][rank];
                std::copy(expected[e][rank].begin()+payload,expected[e][rank].end(),mail.data()+payload);
                ++mail.publications;
                mail.epoch.store(e,std::memory_order_release);
            }
            published[rank].store(e,std::memory_order_release);
            auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
            // Pause only the slow rank AFTER publish and BEFORE consume. The
            // fast rank must finish e and publish e+1 to release this gate.
            if(rank==2 && e<epochs) {
                bool advanced=false;
                for(unsigned n=0;n<100000000u;++n) {
                    if(published[0].load(std::memory_order_acquire)>=e+1) { advanced=true; break; }
                    if(std::chrono::steady_clock::now()>deadline) break;
                    std::this_thread::yield();
                }
                assert(advanced);
                assert(inbox[rank][(e+1)&1][0].epoch.load(std::memory_order_acquire)==e+1);
            }
            bool ready=false;
            for(unsigned n=0;n<100000000u;++n) {
                unsigned a=inbox[rank][bank][0].epoch.load(std::memory_order_acquire);
                unsigned b=inbox[rank][bank][1].epoch.load(std::memory_order_acquire);
                unsigned c=inbox[rank][bank][2].epoch.load(std::memory_order_acquire);
                if(a==e && b==e && c==e) { ready=true; break; }
                assert(a<=e && b<=e && c<=e);
                if(std::chrono::steady_clock::now()>deadline) break;
                std::this_thread::yield();
            }
            assert(ready);
            for(unsigned r=0;r<3;++r) {
                const auto& bytes=expected[e][r];
                assert(std::equal(bytes.begin(),bytes.end(),inbox[rank][bank][r].data()));
                inbox[rank][bank][r].check(frame);
            }
            for(unsigned i=0;i<length;++i) {
                volatile float ab=parse(inbox[rank][bank][0].data(),i,wire)+parse(inbox[rank][bank][1].data(),i,wire);
                float actual=ab+parse(inbox[rank][bank][2].data(),i,wire);
                volatile float refab=decoded[e][0][i]+decoded[e][1][i];
                assert(bits(actual)==bits(refab+decoded[e][2][i]));
            }
            consumed=e; // Rank-local reads finish before its next publish.
        }
    });
    for(auto& thread:threads) thread.join();
    for(auto& rank:inbox) for(auto& bank:rank) for(auto& mail:bank) {
        mail.check(frame); assert(mail.publications==2);
    }
}

static void epoch_controls() {
    // Exact match only, and ONLY the requested bank. Future e+2 cannot pass e;
    // when waiting e+1 that same future tag is in the other, ignored bank.
    std::array<std::array<unsigned,3>,2> tags{};
    unsigned error=0, polls=0;
    auto poll=[&](unsigned epoch,unsigned budget) {
        if(error) return false;
        for(unsigned n=0;n<budget;++n) {
            ++polls;
            const auto& bank=tags[epoch&1];
            if(bank[0]==epoch && bank[1]==epoch && bank[2]==epoch) return true;
            if(bank[0]>epoch || bank[1]>epoch || bank[2]>epoch) break;
        }
        error=4; return false;
    };
    tags[1]={3,1,1}; tags[0]={2,2,2};
    assert(bank_frame(0,1)==bank_frame(0,3));
    assert(!poll(1,8)); assert(polls==1);
    tags[1]={1,1,1}; assert(!poll(1,8)); assert(polls==1); // Sticky fail-stop.
    error=0; tags[1]={3,3,3}; assert(poll(2,8)); // Future wrong bank ignored.
    error=0; tags[0]={0,0,0}; assert(!poll(2,8)); // Wrong bank is not readiness.
    error=0; tags[0]={2,2,2}; assert(!poll(2,0));
    error=0; tags[0]={0,2,2}; assert(!poll(2,8)); // Missing/stale source.
    auto admissible=[](unsigned epoch,unsigned other,unsigned current) {
        return epoch && other==epoch-1 && current==(epoch>2 ? epoch-2 : 0);
    };
    assert(admissible(1,0,0) && admissible(2,1,0) && admissible(3,2,1));
    assert(!admissible(0,0,0) && !admissible(1,0,1)); // Zero and replay.
    assert(!admissible(3,1,0) && !admissible(2,3,2)); // Skip and regression.
}
int main() {
    assert(std::fesetround(FE_TONEAREST)==0);
    for(auto p:{Partition::symmetric,Partition::weighted}) {
        assert(range_begin(0,p)==0 && range_begin(3,p)==96);
        for(unsigned r=0;r<3;++r)
            assert(range_begin(r+1,p)-range_begin(r,p)==(p==Partition::symmetric ? 32u : r==0 ? 30u : 33u));
    }
    assert(payload_bytes(Wire::fp16)==6144 && lines(Wire::fp16)==97);
    assert(payload_bytes(Wire::int8)==3264 && lines(Wire::int8)==52);
    auto sim=std::make_unique<Simulation>();
    std::mt19937 rng(0x38a7);
    std::uniform_real_distribution<float> dist(-9.713f,10.119f);
    unsigned epoch=1;
    for(unsigned trial=0;trial<8;++trial) {
        for(auto& row:sim->input) for(float& x:row) x=dist(rng);
        const float special[]={0.0f,-0.0f,0x1p-25f,0x1.8p-24f,0x1p-14f,
            1.00048828125f,-1.00146484375f,65504.0f};
        for(unsigned r=0;r<3;++r) {
            for(unsigned i=0;i<8;++i) sim->input[r][i]=special[i]*(r==1 ? -1.0f : 1.0f);
            // Whole zero/tiny blocks exercise scale underflow, not just tiny
            // values in a block whose maximum is large.
            for(unsigned i=32;i<64;++i) sim->input[r][i]=0;
            for(unsigned i=64;i<96;++i) sim->input[r][i]=(i&1 ? -1 : 1)*0x1p-25f;
        }
        for(auto w:{Wire::fp16,Wire::int8}) for(auto p:{Partition::symmetric,Partition::weighted})
            sim->run(epoch++,w,p);
    }
    for(auto w:{Wire::fp16,Wire::int8}) {
        for(int peer=0;peer<3;++peer) {
            sim->run(epoch++,w,Partition::weighted,peer,-1,8);
            sim->run(epoch++,w,Partition::symmetric,-1,peer,8);
            sim->run(epoch++,w,Partition::weighted);
        }
        sim->run(epoch++,w,Partition::symmetric,-1,-1,0);
    }
    float invalid[32]{}; uint16_t packet[32];
    invalid[0]=std::numeric_limits<float>::infinity(); assert(!pack(invalid,packet,Wire::fp16));
    invalid[0]=std::numeric_limits<float>::quiet_NaN(); assert(!pack(invalid,packet,Wire::int8));
    invalid[0]=65505; assert(!pack(invalid,packet,Wire::int8));
    for(auto w:{Wire::fp16,Wire::int8}) for(auto p:{Partition::symmetric,Partition::weighted})
        asynchronous_epochs(w,p);
    epoch_controls();
    std::cout << "tp3 compact wire, partitions, four asynchronous epochs, parity isolation, fail-stop: PASS\n";
}

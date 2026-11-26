// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/slot_batch_engine.hpp"
#include <cassert>
#include <set>
using namespace vegajet::sched;
int main() {
    SchedulerConfig c;
    c.max_slots=4; c.speculative=false; c.prefix_reuse=false;
    c.prefill_chunk=2; c.sequence_context=64;
    RequestScheduler scheduler(c);
    struct State { int32_t sum=0; uint32_t pos=0; } states[4];
    uint32_t widest=0, resets=0;
    SlotBatchEngine::Step stepper=[&](const auto& slots,const auto& pos,const auto& tokens,const auto& fresh) {
        widest=std::max<uint32_t>(widest,slots.size());
        std::set<uint32_t> unique;
        std::vector<int32_t> out;
        for(size_t i=0;i<slots.size();++i) {
            assert(unique.insert(slots[i]).second);
            auto& s=states[slots[i]];
            if(fresh[i]) { s={}; ++resets; }
            assert(pos[i]==s.pos++);
            s.sum+=tokens[i]; out.push_back(s.sum%101);
        }
        return out;
    };
    uint32_t chunked=0;
    SlotBatchEngine engine(4,64,stepper,[&](uint32_t slot,uint32_t pos,const std::vector<int32_t>& input,bool fresh,bool sample) {
        chunked+=input.size()>1;
        int32_t prediction=-1;
        for(auto token:input) {
            prediction=stepper({slot},{pos++},{token},{fresh})[0]; fresh=false;
        }
        return sample?prediction:-1;
    });
    std::map<uint64_t,std::vector<int32_t>> results;
    auto drain=[&] {
        for(unsigned it=0;!scheduler.idle();++it) {
            assert(it<1000);
            std::vector<Delivery> d;
            auto plan=scheduler.schedule({},d);
            if(!plan.items.empty()) scheduler.commit(plan,engine.step(plan),d);
            for(auto& v:d) results[v.req].insert(results[v.req].end(),v.tokens.begin(),v.tokens.end());
            assert(scheduler.check_invariants());
        }
    };
    std::vector<std::vector<int32_t>> prompts={{1,2,3},{8},{2,4,5,6,7},{9,9}};
    std::vector<uint64_t> ids;
    for(auto p:prompts) ids.push_back(scheduler.submit(p,6));
    drain();
    assert(widest>1 && resets==4 && chunked>0);
    for(size_t i=0;i<ids.size();++i) {
        int32_t sum=0; for(auto t:prompts[i]) sum+=t;
        std::vector<int32_t> reference{sum%101};
        for(int j=1;j<6;++j) { sum+=reference.back(); reference.push_back(sum%101); }
        assert(results[ids[i]]==reference);
    }
    auto cancelled=scheduler.submit({1},9); scheduler.cancel(cancelled);
    auto reused=scheduler.submit({7,8},2); drain();
    assert(results[cancelled].empty()); assert((results[reused]==std::vector<int32_t>{15,30}));
    assert(resets==5);
    auto active_cancelled=scheduler.submit({6},8);
    std::vector<Delivery> first_delivery;
    auto first_plan=scheduler.schedule({},first_delivery);
    scheduler.commit(first_plan,engine.step(first_plan),first_delivery);
    scheduler.cancel(active_cancelled);
    auto after_cancel=scheduler.submit({4,5},2); drain();
    assert((results[after_cancel]==std::vector<int32_t>{9,18}));
    // Admission and generation obey physical context even with ample virtual blocks.
    RequestScheduler bounded(c); bounded.submit(std::vector<int32_t>(65,1),2);
    std::vector<Delivery> d; assert(bounded.schedule({},d).items.empty()); assert(d[0].end==EndReason::Failed);
}

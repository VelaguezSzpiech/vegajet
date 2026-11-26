// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
#include "core/request_scheduler.hpp"
#include <algorithm>
#include <stdexcept>

namespace vegajet::sched {
// The executor owns fixed physical state slots. Row i updates slots[i] in
// place, independent of row compaction or decode order. No paged-KV aliasing.
class SlotBatchEngine final : public BatchEngine {
public:
    using Step = std::function<std::vector<int32_t>(const std::vector<uint32_t>& slots,
        const std::vector<uint32_t>& positions, const std::vector<int32_t>& tokens,
        const std::vector<bool>& reset)>;
    using Prefill = std::function<int32_t(uint32_t,uint32_t,const std::vector<int32_t>&,bool,bool)>;
    SlotBatchEngine(uint32_t capacity, uint32_t context, Step step, Prefill prefill={})
        : slots_(capacity), context_(context), step_(std::move(step)), prefill_(std::move(prefill)) {
        if (!capacity || capacity > 8 || !context || !step_) throw std::invalid_argument("slot batch config");
    }
    std::vector<StepOutput> step(const StepPlan& plan) override {
        for (auto& s : slots_) if (s.used && std::find(plan.resident.begin(), plan.resident.end(), s.seq) == plan.resident.end()) s = {};
        std::vector<StepOutput> out;
        std::vector<uint32_t> decode_slots, positions;
        std::vector<int32_t> tokens;
        std::vector<bool> resets;
        std::vector<uint64_t> requests;
        for (const auto& w : plan.items) {
            if (w.tokens.empty() || w.pos0 >= context_ || w.tokens.size() > context_ - w.pos0)
                throw std::runtime_error("sequence exceeds physical slot context");
            uint32_t slot = 0;
            while (slot < slots_.size() && !(slots_[slot].used && slots_[slot].seq == w.seq)) ++slot;
            bool fresh = slot == slots_.size();
            if (fresh) {
                slot = 0;
                while (slot < slots_.size() && slots_[slot].used) ++slot;
                if (slot == slots_.size() || w.pos0 != 0) throw std::runtime_error("missing physical sequence state");
                slots_[slot] = {w.seq, 0, true};
            }
            auto& s = slots_[slot];
            if (s.position != w.pos0) throw std::runtime_error("physical slot position mismatch");
            if (w.decode) {
                if (w.tokens.size() != 1) throw std::runtime_error("slot batching requires greedy decode");
                decode_slots.push_back(slot); positions.push_back(w.pos0); tokens.push_back(w.tokens[0]);
                resets.push_back(fresh); requests.push_back(w.req); ++s.position;
            } else {
                // The GPU hook ingests a chunk in scratch state and commits it
                // into this slot. The scalar fallback is useful for CPU executors.
                int32_t pred = -1;
                if (prefill_) {
                    pred=prefill_(slot,s.position,w.tokens,fresh,w.sample);
                    s.position+=static_cast<uint32_t>(w.tokens.size());
                } else for (int32_t tok : w.tokens) {
                    auto p = step_({slot}, {s.position}, {tok}, {fresh});
                    if (p.size() != 1) throw std::runtime_error("bad prefill prediction count");
                    pred = p[0]; ++s.position; fresh = false;
                }
                if (w.sample) out.push_back({w.req, {pred}});
            }
        }
        if (!tokens.empty()) {
            auto p = step_(decode_slots, positions, tokens, resets);
            if (p.size() != tokens.size()) throw std::runtime_error("bad decode prediction count");
            for (size_t i=0; i<p.size(); ++i) out.push_back({requests[i], {p[i]}});
        }
        return out;
    }
private:
    struct Slot { paged::SeqId seq=0; uint32_t position=0; bool used=false; };
    std::vector<Slot> slots_;
    uint32_t context_;
    Step step_;
    Prefill prefill_;
};
} // namespace vegajet::sched

// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/engine_factory.hpp"

#include "core/kfd_runtime.hpp"

namespace vegajet::serve {

EngineRegistry& EngineRegistry::instance() {
    static EngineRegistry r;
    return r;
}

void EngineRegistry::add(const std::string& architecture, EngineProvider p) {
    std::lock_guard<std::mutex> lk(mu_);
    providers_[architecture] = std::move(p);
}

bool EngineRegistry::has(const std::string& architecture) const {
    std::lock_guard<std::mutex> lk(mu_);
    return providers_.count(architecture) != 0;
}

std::vector<std::string> EngineRegistry::architectures() const {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<std::string> v;
    for (const auto& kv : providers_) v.push_back(kv.first);
    return v;
}

bool EngineRegistry::create(const EngineSpec& spec, EngineHandle& out, std::string* err) const {
    EngineProvider p;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = providers_.find(spec.architecture);
        if (it != providers_.end()) p = it->second;
    }
    if (!p) {
        std::string known;
        for (const auto& a : architectures()) known += (known.empty() ? "" : ", ") + a;
        if (err) {
            if (!kfd::Runtime::compiled_with_gpu())
                *err = "no engine for architecture '" + spec.architecture +
                       "' in this build: it was built with VEGAJET_CPU_ONLY=ON, so GPU engines are not compiled in "
                       "(use --mock for CPU testing, or a GPU build)";
            else
                *err = "no streaming engine is registered for architecture '" + spec.architecture +
                       "': this build's GPU entry points for it are bench / parity only (vegajet bench / run); "
                       "registered architectures: " + (known.empty() ? std::string("none") : known);
        }
        return false;
    }
    std::string why;
    if (!p(spec, out, &why) || !out.engine) {
        if (err) *err = "engine for '" + spec.architecture + "' failed to start: " + (why.empty() ? "no engine returned" : why);
        out = EngineHandle{};
        return false;
    }
    return true;
}

MockBatchEngine::MockBatchEngine(MockEngine& m) : eos_(m.tokenizer().eos_id()) {
    GenerationParams gp;
    gp.max_tokens = 256;
    m.generate({0}, gp, [&](const TokenEvent& ev) {
        if (ev.is_eos) return false;
        if (ev.token_id >= 0) reply_.push_back(ev.token_id);
        return true;
    });
}

std::vector<sched::StepOutput> MockBatchEngine::step(const sched::StepPlan& plan) {
    std::vector<sched::StepOutput> out;
    for (const auto& w : plan.items) {
        if (!w.sample) continue;
        sched::StepOutput o;
        o.req = w.req;
        if (!w.decode) {
            const uint32_t p = w.pos0 + static_cast<uint32_t>(w.tokens.size());  // position of the sampled token
            auto it = prompt_len_.emplace(w.req, p).first;                       // first sample: p == prompt length
            o.tokens.push_back(canned(p - it->second));
        } else {
            auto it = prompt_len_.find(w.req);
            const uint32_t base = it == prompt_len_.end() ? 0 : it->second;
            for (uint32_t i = 0;; ++i) {  // accept matching drafts, then the bonus token
                const uint32_t p = w.pos0 + 1 + i;
                const int32_t t = canned(p >= base ? p - base : 0);
                o.tokens.push_back(t);
                if (i >= w.n_draft || t == eos_ || w.tokens[1 + i] != t) break;
            }
        }
        if (o.tokens.back() == eos_) prompt_len_.erase(w.req);
        out.push_back(std::move(o));
    }
    return out;
}

}  // namespace vegajet::serve

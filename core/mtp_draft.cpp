// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/mtp_draft.hpp"

#include <algorithm>
#include <cstdio>
#include <unordered_set>

namespace vegajet::mtp_draft {

namespace {
uint32_t lm_rows_of(const ChainConfig& cfg) {
    if (cfg.shortlist) return cfg.shortlist->rows_per_rank;
    uint32_t m = 0;
    for (uint32_t r : cfg.voc_rows) m = std::max(m, r);
    return m;
}
}  // namespace

ChainPlan plan_chain(const ChainConfig& cfg, uint32_t anchor_pos) {
    ChainPlan p;
    if (cfg.k == 0u || cfg.k > kMaxChain) { p.error = "chain length must be 1.." + std::to_string(kMaxChain); return p; }
    if (static_cast<uint64_t>(anchor_pos) + cfg.k > cfg.max_ctx) { p.error = "chain runs past the MTP cache (max_ctx)"; return p; }
    if (!cfg.shortlist && cfg.voc_rows.size() != cfg.n_ranks) { p.error = "full-vocabulary chain needs voc_rows per rank"; return p; }
    if (cfg.shortlist && cfg.shortlist->n_ranks != cfg.n_ranks) { p.error = "shortlist rank count differs"; return p; }
    for (uint32_t j = 0; j < cfg.k; ++j) {
        StepDesc d;
        d.step = j;
        d.pos = anchor_pos + j;
        d.tok_src = static_cast<uint32_t>(j == 0u ? TokSrc::kHostSlot : TokSrc::kDeviceArgmax);
        d.tok_slot = j == 0u ? 0u : j - 1u;
        d.h_src = static_cast<uint32_t>(j == 0u ? HSrc::kPend : HSrc::kMtpOut);
        d.h_slot = j == 0u ? 0u : j - 1u;
        d.h_out_slot = j;
        d.lm_mode = static_cast<uint32_t>(cfg.shortlist ? LmMode::kShortlist : LmMode::kFullVocab);
        d.lm_rows = lm_rows_of(cfg);
        d.argmax_slot = j;
        d.host_read = (j + 1u == cfg.k) ? 1u : 0u;
        d.attend_hi = d.pos + 1u;
        p.steps.push_back(d);
    }
    p.host_waits = 1;
    p.ok = true;
    return p;
}

bool check_chain(const ChainPlan& p, const ChainConfig& cfg, uint32_t anchor_pos, std::string* err) {
    auto fail = [&](const std::string& m) { if (err) *err = m; return false; };
    if (!p.ok) return fail("plan not ok: " + p.error);
    if (p.steps.size() != cfg.k) return fail("step count");
    uint32_t reads = 0;
    for (uint32_t j = 0; j < p.steps.size(); ++j) {
        const StepDesc& d = p.steps[j];
        if (d.pos != anchor_pos + j) return fail("step " + std::to_string(j) + ": position");
        if (d.attend_hi != d.pos + 1u) return fail("step " + std::to_string(j) + ": attends past its own row");
        if (j > 0u) {
            const StepDesc& prev = p.steps[j - 1u];
            if (d.tok_src != static_cast<uint32_t>(TokSrc::kDeviceArgmax) || d.tok_slot != prev.argmax_slot)
                return fail("step " + std::to_string(j) + ": token must come from the previous step's device argmax");
            if (d.h_src != static_cast<uint32_t>(HSrc::kMtpOut) || d.h_slot != prev.h_out_slot)
                return fail("step " + std::to_string(j) + ": hidden state must come from the previous step");
            if (d.h_out_slot == d.h_slot) return fail("step " + std::to_string(j) + ": reads and writes the same h slot");
        } else if (d.tok_src != static_cast<uint32_t>(TokSrc::kHostSlot) || d.h_src != static_cast<uint32_t>(HSrc::kPend)) {
            return fail("step 0 must read the committed token and h_pend");
        }
        if (d.h_out_slot >= kMaxChain || d.argmax_slot >= kMaxChain) return fail("slot out of range");
        reads += d.host_read;
        if (d.host_read && j + 1u != p.steps.size()) return fail("host read before the last step");
    }
    if (reads != 1u || p.host_waits != 1u) return fail("exactly one host wait per chain");
    return true;
}

int32_t resolve_draft_id(int32_t kernel_id, const ChainConfig& cfg) {
    if (kernel_id < 0) return -1;
    if (!cfg.shortlist) return (cfg.vocab == 0u || static_cast<uint32_t>(kernel_id) < cfg.vocab) ? kernel_id : -1;
    const spec_replay::Shortlist& s = *cfg.shortlist;
    const uint32_t idx = static_cast<uint32_t>(kernel_id);
    const uint32_t g = idx / s.rows_per_rank;
    const uint32_t local = idx - g * s.rows_per_rank;
    if (g >= s.n_ranks || local >= s.tok_map[g].size()) return -1;
    return s.tok_map[g][local];
}

// ---- ledger --------------------------------------------------------------------------------------------------------
bool MtpLedger::prefill(uint32_t n, std::string* err) {
    if (n > rows_.size()) { if (err) *err = "prefill past max_ctx"; return false; }
    for (uint32_t p = 0; p < n; ++p) rows_[p] = Row::kExact;
    return true;
}

bool MtpLedger::draft(uint32_t anchor, uint32_t k, std::string* err) {
    if (static_cast<uint64_t>(anchor) + k > rows_.size()) { if (err) *err = "draft past max_ctx"; return false; }
    const uint32_t ex = exact_prefix();
    if (ex < anchor) {
        if (err) *err = "row " + std::to_string(ex) + " is not exact before a chain at anchor " + std::to_string(anchor);
        return false;
    }
    rows_[anchor] = Row::kExact;
    for (uint32_t j = 1; j < k; ++j) rows_[anchor + j] = Row::kProvisional;
    return true;
}

VerifyPlan MtpLedger::after_verify(const std::vector<int32_t>& vtoks, uint32_t anchor, uint32_t acc, bool neural) {
    VerifyPlan v;
    // Suffix / plain cycle: the chain did not ingest vtoks[0], so the anchor row is written now from h_pend
    // (trunk_row = UINT32_MAX marks "h_pend" here); then vtoks[j] at anchor + j from the trunk row of vtoks[j - 1].
    if (!neural && !vtoks.empty()) v.catchup.push_back(CatchupStep{vtoks[0], anchor, UINT32_MAX});
    for (uint32_t j = 1; j <= acc && j < vtoks.size(); ++j) v.catchup.push_back(CatchupStep{vtoks[j], anchor + j, j - 1u});
    v.h_pend_trunk_row = acc;
    v.next_anchor = anchor + acc + 1u;
    return v;
}

void MtpLedger::apply(const VerifyPlan& v) {
    for (const auto& c : v.catchup) if (c.pos < rows_.size()) rows_[c.pos] = Row::kExact;
    // rows from next_anchor on stay provisional / empty: the next chain overwrites them before reading
}

uint32_t MtpLedger::exact_prefix() const {
    uint32_t n = 0;
    while (n < rows_.size() && rows_[n] == Row::kExact) ++n;
    return n;
}

// ---- suffix bypass ---------------------------------------------------------------------------------------------------
namespace {
sfx::CascadeConfig bypass_cfg(uint32_t max_draft) {
    sfx::CascadeConfig c;
    c.min_match = kSuffixBypassMin;
    c.min_chain = kSuffixBypassMin;
    c.max_draft = max_draft;
    return c;
}
}  // namespace

SuffixBypass::SuffixBypass(uint32_t max_draft) : max_draft_(max_draft), cascade_(bypass_cfg(max_draft)) {}
void SuffixBypass::commit(int32_t tok) { cascade_.commit(tok); }
void SuffixBypass::commit(const std::vector<int32_t>& toks) { cascade_.commit(toks); }

BypassDecision SuffixBypass::decide() {
    BypassDecision d;
    const auto dec = cascade_.decide();
    d.match_len = dec.match.length;
    if (dec.source == sfx::DraftSource::kSuffix && dec.draft.size() >= kSuffixBypassMin && dec.match.length >= kSuffixBypassMin) {
        d.bypass = true;
        d.draft.assign(dec.draft.begin(), dec.draft.begin() + std::min<size_t>(dec.draft.size(), max_draft_));
        ++hits;
    } else {
        ++misses;
    }
    return d;
}

// ---- stats -------------------------------------------------------------------------------------------------------------
void SpecStats::add(const CycleSample& c) {
    ++cycles;
    (c.suffix ? suffix : neural)++;
    drafted += c.depth;
    accepted += c.accepted;
    committed += c.accepted + 1u;
    draft_ms += c.draft_ms;
    verify_ms += c.verify_ms;
    if (!c.experts.empty()) {
        double sum = 0.0;
        for (const auto& layer : c.experts) {
            std::unordered_set<int32_t> u;
            for (int32_t e : layer) if (e >= 0) u.insert(e);
            sum += static_cast<double>(u.size());
        }
        unique_experts_sum += sum / static_cast<double>(c.experts.size());
        ++expert_cycles;
    }
}

std::string format_stats(const std::string& label, const SpecStats& s) {
    char buf[512];
    char uq[32];
    if (s.unique_experts() < 0.0) std::snprintf(uq, sizeof(uq), "n/a");
    else std::snprintf(uq, sizeof(uq), "%.1f", s.unique_experts());
    std::snprintf(buf, sizeof(buf),
                  "[SPEC-STATS %s] cycles=%llu acc/cyc=%.3f tok/cyc=%.3f depth=%.2f sfx_hit=%.1f%% uniq_exp=%s draft_ms=%.3f verify_ms=%.3f acc_tok_s=%.2f",
                  label.c_str(), static_cast<unsigned long long>(s.cycles), s.accepted_per_cycle(), s.tokens_per_cycle(),
                  s.mean_depth(), 100.0 * s.suffix_hit_rate(), uq,
                  s.cycles ? s.draft_ms / static_cast<double>(s.cycles) : 0.0,
                  s.cycles ? s.verify_ms / static_cast<double>(s.cycles) : 0.0, s.accepted_tok_s());
    return buf;
}

}  // namespace vegajet::mtp_draft

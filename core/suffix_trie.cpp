// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/suffix_trie.hpp"

#include <algorithm>

namespace vegajet::sfx {

SuffixIndex::SuffixIndex() { clear(); }

void SuffixIndex::clear() {
    st_.assign(1, State{});
    out_.assign(1, {});
    tr_.clear();
    last_ = 0;
    hist_.clear();
}

void SuffixIndex::reserve(size_t tokens) {
    st_.reserve(2 * tokens + 1);
    out_.reserve(2 * tokens + 1);
    tr_.reserve(3 * tokens + 1);  // a suffix automaton has < 3n transitions
    hist_.reserve(tokens);
}

int32_t SuffixIndex::next(int32_t s, int32_t tok) const {
    const auto it = tr_.find(key(s, tok));
    return it == tr_.end() ? -1 : it->second;
}

void SuffixIndex::set_next(int32_t s, int32_t tok, int32_t to) {
    const auto r = tr_.insert({key(s, tok), to});
    if (r.second) out_[static_cast<size_t>(s)].push_back(tok);
    else r.first->second = to;
}

void SuffixIndex::push(int32_t c) {
    const int64_t pos = static_cast<int64_t>(hist_.size());
    hist_.push_back(c);
    const int32_t cur = static_cast<int32_t>(st_.size());
    st_.push_back({-1, st_[last_].len + 1, pos});
    out_.emplace_back();
    int32_t p = last_;
    while (p != -1 && next(p, c) == -1) {
        set_next(p, c, cur);
        p = st_[p].link;
    }
    if (p == -1) {
        st_[cur].link = 0;
    } else {
        const int32_t q = next(p, c);
        if (st_[p].len + 1 == st_[q].len) {
            st_[cur].link = q;
        } else {
            const int32_t clone = static_cast<int32_t>(st_.size());
            st_.push_back({st_[q].link, st_[p].len + 1, st_[q].first_end});
            out_.emplace_back();
            const std::vector<int32_t> q_out = out_[static_cast<size_t>(q)];  // copy: set_next may reallocate out_
            for (int32_t tok : q_out) set_next(clone, tok, next(q, tok));
            while (p != -1 && next(p, c) == q) {
                set_next(p, c, clone);
                p = st_[p].link;
            }
            st_[q].link = clone;
            st_[cur].link = clone;
        }
    }
    last_ = cur;
}

Match SuffixIndex::longest_match() const {
    Match m;
    if (last_ == 0) return m;
    const int32_t l = st_[last_].link;
    if (l <= 0) return m;
    m.length = st_[l].len;
    m.end_pos = st_[l].first_end;
    return m;
}

std::vector<int32_t> SuffixIndex::draft(uint32_t max_tokens) const {
    std::vector<int32_t> out;
    const Match m = longest_match();
    if (m.length == 0) return out;
    for (int64_t i = m.end_pos + 1; i < static_cast<int64_t>(hist_.size()) && out.size() < max_tokens; ++i)
        out.push_back(hist_[static_cast<size_t>(i)]);
    return out;
}

std::vector<uint32_t> SuffixIndex::continuation_degrees(const std::vector<int32_t>& chain) const {
    std::vector<uint32_t> deg;
    if (last_ == 0) return deg;
    int32_t s = st_[last_].link;
    if (s <= 0) return deg;
    deg.reserve(chain.size());
    for (int32_t tok : chain) {
        deg.push_back(static_cast<uint32_t>(out_[static_cast<size_t>(s)].size()));
        s = next(s, tok);
        if (s < 0) break;
    }
    return deg;
}

CascadeDecision CascadeHybrid::decide() const {
    CascadeDecision d;
    d.match = idx_.longest_match();
    if (d.match.length >= cfg_.min_match) {
        auto chain = idx_.draft(cfg_.max_draft);
        if (chain.size() >= cfg_.min_chain) {
            d.source = DraftSource::kSuffix;
            d.draft = std::move(chain);
        }
    }
    return d;
}

} // namespace vegajet::sfx

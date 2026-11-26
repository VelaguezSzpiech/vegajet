// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#pragma once
#include "core/json_lite.hpp"
#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <set>
#include <stdexcept>

namespace vegajet::tp::detail {
inline void require(bool ok, const std::string& message) {
    if (!ok) throw std::invalid_argument(message);
}
inline json::Value parse(const std::string& text) {
    json::Value v;
    require(json::parse(text, v) && v.is_object(), "expected JSON object");
    return v;
}
inline void keys(const json::Value& v, std::initializer_list<std::string> allowed, bool strict = true) {
    require(v.is_object(), "expected object");
    std::set<std::string> seen;
    for (const auto& p : v.obj) {
        require(seen.insert(p.first).second, "duplicate key: " + p.first);
        require(!strict || std::find(allowed.begin(), allowed.end(), p.first) != allowed.end(),
                "unknown key: " + p.first);
    }
}
inline const json::Value& field(const json::Value& v, std::string_view name) {
    auto p = v.get(name);
    require(p != nullptr, "missing field: " + std::string(name));
    return *p;
}
inline uint64_t integer(const json::Value& v) {
    require(v.is_number() && std::isfinite(v.num) && v.num >= 0 &&
            v.num <= 9007199254740991.0 && std::floor(v.num) == v.num,
            "expected exactly representable unsigned integer (at most 2^53-1)");
    return static_cast<uint64_t>(v.num);
}
inline double number(const json::Value& v) {
    require(v.is_number() && std::isfinite(v.num), "expected finite number");
    return v.num;
}
inline std::string string(const json::Value& v) {
    require(v.is_string(), "expected string");
    return v.str;
}
inline bool boolean(const json::Value& v) {
    require(v.is_bool(), "expected boolean");
    return v.b;
}
inline uint64_t add(uint64_t a, uint64_t b) {
    if (b > std::numeric_limits<uint64_t>::max() - a) throw std::overflow_error("TP size addition overflow");
    return a + b;
}
inline uint64_t mul(uint64_t a, uint64_t b) {
    if (a && b > std::numeric_limits<uint64_t>::max() / a) throw std::overflow_error("TP size multiplication overflow");
    return a * b;
}
} // namespace vegajet::tp::detail

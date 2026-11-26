// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "models/ornith/ornith_dot8_route.hpp"

#include <cstdlib>

namespace vegajet::ornith {

namespace {

bool ends_with(const std::string& s, const char* suf) {
    const std::string t(suf);
    return s.size() >= t.size() && s.compare(s.size() - t.size(), t.size(), t) == 0;
}

bool fail(std::string* err, const std::string& msg) {
    if (err) *err = "VEGAJET_V13_DOT8_PROMOTE: " + msg;
    return false;
}

} // namespace

bool dot8_promote_enabled() {
    const char* v = std::getenv("VEGAJET_V13_DOT8_PROMOTE");
    return v && v[0] == '1' && v[1] == '\0';
}

bool resolve_dot8_upload_plan(const std::vector<TensorPlan>& plan, uint32_t n_trunk, bool mtp_uploaded,
                              const Dot8KernelRoutes& routes, Dot8UploadPlan& out, std::string* err) {
    out = Dot8UploadPlan{};
    out.gate_up_dot8.assign(n_trunk, 0);
    std::vector<uint8_t> gate(n_trunk, 0), up(n_trunk, 0);
    for (const auto& p : plan) {
        if (p.keep_float || p.format != quant::NativeFormat::VJ_Q4_DOT8) continue;
        if (p.name.rfind("draft/", 0) == 0) continue;  // the drafter uploads its own DOT8 weights
        if (p.layer < 0) return fail(err, p.name + " promoted to VJ_Q4_DOT8 has no upload route");
        const auto il = static_cast<uint32_t>(p.layer);
        if (il >= n_trunk) {
            if (mtp_uploaded) return fail(err, p.name + " promoted to VJ_Q4_DOT8: the MTP head uploads VJ_Q4_A8 only");
            continue;
        }
        const bool routed = p.role == quant::TensorRole::RoutedMoE;
        if (routed && p.disk == quant::DiskQuantType::IQ3_XXS && ends_with(p.name, "ffn_down_exps.weight")) {
            if (!routes.down_dot8) return fail(err, p.name + " promoted to VJ_Q4_DOT8 but the DOT8 Down kernel route is missing");
            ++out.down_layers;
            continue;
        }
        const bool is_gate = ends_with(p.name, "ffn_gate_exps.weight");
        const bool is_up = ends_with(p.name, "ffn_up_exps.weight");
        if (routed && p.disk == quant::DiskQuantType::IQ2_S && (is_gate || is_up)) {
            if (!routes.gate_up_dot8) return fail(err, p.name + " promoted to VJ_Q4_DOT8 but the DOT8 Gate+Up kernel route is missing");
            if (!routes.gate_up_geometry) return fail(err, p.name + " promoted to VJ_Q4_DOT8 but the DOT8 Gate+Up layout does not support this shape");
            (is_gate ? gate : up)[il] = 1;
            continue;
        }
        return fail(err, p.name + " (" + ggml_type_name(static_cast<uint32_t>(p.disk)) +
                             ") promoted to VJ_Q4_DOT8 has no upload/kernel route");
    }
    for (uint32_t il = 0; il < n_trunk; ++il) {
        if (gate[il] != up[il])
            return fail(err, "blk." + std::to_string(il) + " " + (gate[il] ? "ffn_gate_exps" : "ffn_up_exps") +
                                 " promoted to VJ_Q4_DOT8 without its merged Gate+Up pair");
        out.gate_up_dot8[il] = gate[il];
        out.gate_up_layers += gate[il];
    }
    return true;
}

} // namespace vegajet::ornith

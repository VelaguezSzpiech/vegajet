// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// v1.3 dot8-wire: TensorPlan::format == VJ_Q4_DOT8 -> GPU upload route resolution
// (models/ornith/ornith_dot8_route.*). Checks the env toggle, the per-layer Gate+Up mask, the Down route,
// and that every promoted tensor without an upload + kernel route is refused (fail closed). Pure CPU.
#include "models/ornith/ornith_dot8_route.hpp"
#include "tests/test_common.hpp"

#include <cstdlib>
#include <string>
#include <vector>

using namespace vegajet;
using namespace vegajet::ornith;
using quant::DiskQuantType;
using quant::NativeFormat;
using quant::TensorRole;

namespace {

TensorPlan tp(const std::string& name, int32_t layer, DiskQuantType disk, NativeFormat fmt,
              TensorRole role = TensorRole::RoutedMoE) {
    TensorPlan p;
    p.name = name;
    p.layer = layer;
    p.disk = disk;
    p.format = fmt;
    p.role = role;
    return p;
}

// Trunk of n layers: IQ2_S Gate/Up + IQ3_XXS Down, all VJ_IQ_PACKED, plus an attention projection.
std::vector<TensorPlan> base_plan(uint32_t n) {
    std::vector<TensorPlan> v;
    for (uint32_t il = 0; il < n; ++il) {
        const std::string p = "blk." + std::to_string(il) + ".";
        const auto l = static_cast<int32_t>(il);
        v.push_back(tp(p + "ffn_gate_exps.weight", l, DiskQuantType::IQ2_S, NativeFormat::VJ_IQ_PACKED));
        v.push_back(tp(p + "ffn_up_exps.weight", l, DiskQuantType::IQ2_S, NativeFormat::VJ_IQ_PACKED));
        v.push_back(tp(p + "ffn_down_exps.weight", l, DiskQuantType::IQ3_XXS, NativeFormat::VJ_IQ_PACKED));
        v.push_back(tp(p + "attn_q.weight", l, DiskQuantType::IQ3_XXS, NativeFormat::VJ_Q4_A8, TensorRole::DenseProj));
    }
    return v;
}

void promote(std::vector<TensorPlan>& v, const std::string& name) {
    for (auto& p : v)
        if (p.name == name) { p.format = NativeFormat::VJ_Q4_DOT8; return; }
    VJ_CHECK(false);
}

Dot8KernelRoutes all_routes() {
    Dot8KernelRoutes r;
    r.gate_up_dot8 = r.down_dot8 = r.gate_up_geometry = true;
    return r;
}

bool refused(const std::vector<TensorPlan>& v, uint32_t n, bool mtp, const Dot8KernelRoutes& r, const char* needle) {
    Dot8UploadPlan out;
    std::string err;
    if (resolve_dot8_upload_plan(v, n, mtp, r, out, &err)) return false;
    if (err.find(needle) == std::string::npos) {
        std::fprintf(stderr, "unexpected error text: %s\n", err.c_str());
        return false;
    }
    return err.rfind("VEGAJET_V13_DOT8_PROMOTE: ", 0) == 0;
}

} // namespace

int main() {
    // Env toggle: exactly "1" enables it; default off.
    VJ_CHECK(unsetenv("VEGAJET_V13_DOT8_PROMOTE") == 0 && !dot8_promote_enabled());
    VJ_CHECK(setenv("VEGAJET_V13_DOT8_PROMOTE", "0", 1) == 0 && !dot8_promote_enabled());
    VJ_CHECK(setenv("VEGAJET_V13_DOT8_PROMOTE", "10", 1) == 0 && !dot8_promote_enabled());
    VJ_CHECK(setenv("VEGAJET_V13_DOT8_PROMOTE", "1", 1) == 0 && dot8_promote_enabled());
    VJ_CHECK(unsetenv("VEGAJET_V13_DOT8_PROMOTE") == 0);

    constexpr uint32_t n = 4;
    {   // Nothing promoted: empty mask, no error.
        Dot8UploadPlan out;
        std::string err;
        VJ_CHECK(resolve_dot8_upload_plan(base_plan(n), n, true, Dot8KernelRoutes{}, out, &err) && err.empty());
        VJ_CHECK(out.gate_up_dot8.size() == n && out.gate_up_layers == 0 && out.down_layers == 0);
    }
    {   // Gate+Up pairs on layers 1 and 3 and every Down: mask follows the plan exactly.
        auto v = base_plan(n);
        for (uint32_t il : {1u, 3u}) {
            promote(v, "blk." + std::to_string(il) + ".ffn_gate_exps.weight");
            promote(v, "blk." + std::to_string(il) + ".ffn_up_exps.weight");
        }
        for (uint32_t il = 0; il < n; ++il) promote(v, "blk." + std::to_string(il) + ".ffn_down_exps.weight");
        // Float tensors and drafter tensors never need a trunk route.
        auto f = tp("blk.0.attn_norm.weight", 0, DiskQuantType::F32, NativeFormat::VJ_Q4_DOT8, TensorRole::Other);
        f.keep_float = true;
        v.push_back(f);
        v.push_back(tp("draft/blk.0.attn_q.weight", 0, DiskQuantType::Q8_0, NativeFormat::VJ_Q4_DOT8, TensorRole::MtpDraft));
        Dot8UploadPlan out;
        std::string err;
        VJ_CHECK(resolve_dot8_upload_plan(v, n, false, all_routes(), out, &err));
        VJ_CHECK((out.gate_up_dot8 == std::vector<uint8_t>{0, 1, 0, 1}));
        VJ_CHECK(out.gate_up_layers == 2 && out.down_layers == n);
        // Down only needs the Down route.
        Dot8KernelRoutes down_only;
        down_only.down_dot8 = true;
        auto d = base_plan(n);
        promote(d, "blk.2.ffn_down_exps.weight");
        VJ_CHECK(resolve_dot8_upload_plan(d, n, false, down_only, out, &err) && out.down_layers == 1 && out.gate_up_layers == 0);
    }
    {   // Fail closed: half of the merged pair.
        auto v = base_plan(n);
        promote(v, "blk.2.ffn_gate_exps.weight");
        VJ_CHECK(refused(v, n, false, all_routes(), "blk.2 ffn_gate_exps promoted to VJ_Q4_DOT8 without its merged Gate+Up pair"));
        auto u = base_plan(n);
        promote(u, "blk.0.ffn_up_exps.weight");
        VJ_CHECK(refused(u, n, false, all_routes(), "blk.0 ffn_up_exps promoted"));
    }
    {   // Fail closed: missing kernel routes / unsupported geometry.
        auto v = base_plan(n);
        promote(v, "blk.1.ffn_gate_exps.weight");
        promote(v, "blk.1.ffn_up_exps.weight");
        auto r = all_routes();
        r.gate_up_dot8 = false;
        VJ_CHECK(refused(v, n, false, r, "DOT8 Gate+Up kernel route is missing"));
        r = all_routes();
        r.gate_up_geometry = false;
        VJ_CHECK(refused(v, n, false, r, "layout does not support this shape"));
        auto d = base_plan(n);
        promote(d, "blk.3.ffn_down_exps.weight");
        r = all_routes();
        r.down_dot8 = false;
        VJ_CHECK(refused(d, n, false, r, "DOT8 Down kernel route is missing"));
    }
    {   // Fail closed: promoted tensors outside the routed MoE routes.
        auto v = base_plan(n);
        promote(v, "blk.0.attn_q.weight");
        VJ_CHECK(refused(v, n, false, all_routes(), "blk.0.attn_q.weight (IQ3_XXS) promoted to VJ_Q4_DOT8 has no upload/kernel route"));
        auto s = base_plan(n);
        s.push_back(tp("blk.0.ffn_gate_shexp.weight", 0, DiskQuantType::IQ2_S, NativeFormat::VJ_Q4_DOT8, TensorRole::SharedExpert));
        VJ_CHECK(refused(s, n, false, all_routes(), "ffn_gate_shexp.weight (IQ2_S)"));
        auto g = base_plan(n);  // IQ3_XXS Gate has no DOT8 Gate+Up route (the merged route is IQ2_S)
        g.push_back(tp("blk.0.ffn_gate_exps.weight", 0, DiskQuantType::IQ3_XXS, NativeFormat::VJ_Q4_DOT8));
        VJ_CHECK(refused(g, n, false, all_routes(), "(IQ3_XXS) promoted"));
        auto o = base_plan(n);
        o.push_back(tp("output.weight", -1, DiskQuantType::IQ3_XXS, NativeFormat::VJ_Q4_DOT8, TensorRole::VocabHead));
        VJ_CHECK(refused(o, n, false, all_routes(), "output.weight promoted to VJ_Q4_DOT8 has no upload route"));
    }
    {   // nextn / MTP layer (index == n_trunk): refused only when the MTP head is uploaded.
        auto v = base_plan(n);
        v.push_back(tp("blk.4.ffn_down_exps.weight", 4, DiskQuantType::IQ3_XXS, NativeFormat::VJ_Q4_DOT8));
        VJ_CHECK(refused(v, n, true, all_routes(), "the MTP head uploads VJ_Q4_A8 only"));
        Dot8UploadPlan out;
        std::string err;
        VJ_CHECK(resolve_dot8_upload_plan(v, n, false, all_routes(), out, &err) && out.down_layers == 0);
    }
    std::printf("test_v13_dot8_route: OK\n");
    return 0;
}

#pragma once

namespace vegajet {
struct SuperVariant {
    const char* object;
    const char* suffix;
    bool valid;
};
inline SuperVariant select_super_variant(bool vol, bool iq2s) {
    if (vol && iq2s) return {nullptr, "", false};
    if (vol) return {"vj_ornith_super_vol.hsaco", "_vol", true};
    if (iq2s) return {"vj_ornith_super_iq2sreg.hsaco", "_iq2sreg", true};
    return {nullptr, "", true};
}
} // namespace vegajet

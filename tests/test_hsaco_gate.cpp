// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Code-object gate decisions (class limits, waiver ratchet, stale waiver, unlisted kernel, code size, spill),
// the live PCIe link classifier and its sysfs string parsers. Pure logic, no GPU toolchain.
#include "core/doctor.hpp"
#include "core/hsaco_gate.hpp"
#include "tests/test_common.hpp"

#include <set>

using namespace vegajet::gate;
namespace doc = vegajet::doctor;

int main() {
    const GateEntry stream{"s", KernelClass::Stream, 0};
    const GateEntry tiled{"t", KernelClass::Tiled, 0};
    const GateEntry waived{"w", KernelClass::Tiled, 96};
    std::string why;

    VJ_CHECK(evaluate(&stream, "s", 48, 0, 1000, &why) == GateResult::Ok);
    VJ_CHECK(evaluate(&stream, "s", 49, 0, 1000, &why) == GateResult::Fail);
    VJ_CHECK(evaluate(&tiled, "t", 64, 0, 24575, &why) == GateResult::Ok);
    VJ_CHECK(evaluate(&tiled, "t", 65, 0, 1000, &why) == GateResult::Fail);
    VJ_CHECK(evaluate(&tiled, "t", 20, 0, 24576, &why) == GateResult::Fail);          // code size is strict <
    VJ_CHECK(evaluate(&tiled, "t", 20, 16, 1000, &why) == GateResult::Fail);          // any spill
    VJ_CHECK(evaluate(nullptr, "new_kernel", 8, 0, 100, &why) == GateResult::Fail);   // unlisted
    VJ_CHECK(why.find("new_kernel") != std::string::npos);
    VJ_CHECK(evaluate(&waived, "w", 96, 0, 1000, &why) == GateResult::OkWaived);
    VJ_CHECK(evaluate(&waived, "w", 80, 0, 1000, &why) == GateResult::OkWaived);
    VJ_CHECK(evaluate(&waived, "w", 97, 0, 1000, &why) == GateResult::Fail);          // ratchet
    VJ_CHECK(evaluate(&waived, "w", 64, 0, 1000, &why) == GateResult::Fail);          // stale waiver
    VJ_CHECK(evaluate(&waived, "w", 96, 8, 1000, &why) == GateResult::Fail);          // waiver never covers spill

    // The compiled-in manifest is well formed: unique names, waivers above their class limit.
    std::set<std::string> seen;
    VJ_CHECK(!manifest().empty());
    for (const auto& e : manifest()) {
        VJ_CHECK(seen.insert(e.name).second);
        VJ_CHECK(e.waiver_vgpr == 0u || e.waiver_vgpr > class_limit(e.klass));
        VJ_CHECK(find_entry(e.name) == &e);
    }
    VJ_CHECK(find_entry("no_such_kernel") == nullptr);
    VJ_CHECK(elf_function_sizes(nullptr, 0).empty());
    const uint8_t junk[64] = {0x7f, 'E', 'L', 'F', 2};
    VJ_CHECK(elf_function_sizes(junk, sizeof(junk)).empty());

    // ---- doctor link parsers + classifier ----
    VJ_CHECK(doc::parse_link_speed_gts("2.5 GT/s PCIe") == 2.5);
    VJ_CHECK(doc::parse_link_speed_gts("8.0 GT/s PCIe") == 8.0);
    VJ_CHECK(doc::parse_link_speed_gts("16.0 GT/s PCIe") == 16.0);
    VJ_CHECK(doc::parse_link_speed_gts("Unknown") == 0.0);
    doc::PcieDpmLevel lv;
    VJ_CHECK(doc::parse_pp_dpm_pcie_line("0: 2.5GT/s, x8 80Mhz *", lv) && lv.index == 0 && lv.gts == 2.5 && lv.width == 8 && lv.active);
    VJ_CHECK(doc::parse_pp_dpm_pcie_line("1: 8.0GT/s, x16 773Mhz ", lv) && lv.index == 1 && lv.gts == 8.0 && lv.width == 16 && !lv.active);
    VJ_CHECK(!doc::parse_pp_dpm_pcie_line("garbage", lv));

    auto mk = [](int n, int busy, double gts, int level) { return std::vector<doc::LinkSample>(static_cast<size_t>(n), doc::LinkSample{busy, gts, level}); };
    VJ_CHECK(doc::classify_live_link(mk(10, 0, 2.5, 0)).verdict == doc::LinkVerdict::NoLoadSeen);
    VJ_CHECK(doc::classify_live_link(mk(10, 99, 8.0, 1)).verdict == doc::LinkVerdict::UpshiftsUnderLoad);
    VJ_CHECK(doc::classify_live_link(mk(10, 99, 2.5, 0)).verdict == doc::LinkVerdict::StuckGen1);
    auto mixed = mk(5, 99, 8.0, 1);
    const auto slow = mk(5, 99, 2.5, 0);
    mixed.insert(mixed.end(), slow.begin(), slow.end());
    const auto c = doc::classify_live_link(mixed);
    VJ_CHECK(c.verdict == doc::LinkVerdict::Partial && c.busy_samples == 10 && c.busy_fast_samples == 5 && c.dpm_level_changed);

    std::printf("test_hsaco_gate: OK\n");
    return 0;
}

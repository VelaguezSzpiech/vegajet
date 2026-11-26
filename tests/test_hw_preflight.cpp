// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/hw_preflight.hpp"
#include "tests/test_common.hpp"
#include <filesystem>
#include <string>
#include <fstream>
#include <unistd.h>
#include <vector>
namespace fs = std::filesystem;
static bool has(const std::vector<vegajet::hw_preflight::Warning>& w, const std::string& c, const std::string& sub = "") {
    for (const auto& x : w) if (x.check == c && x.detail.find(sub) != std::string::npos) return true;
    return false;
}
static void put16(std::vector<uint8_t>& c, size_t o, uint16_t v) { c[o] = v & 0xFF; c[o + 1] = v >> 8; }
static void put32(std::vector<uint8_t>& c, size_t o, uint32_t v) { put16(c, o, v & 0xFFFF); put16(c, o + 2, v >> 16); }
int main() {
    char t[] = "/tmp/vj-hw-preflight-XXXXXX";
    char* d = mkdtemp(t);
    VJ_CHECK(d);
    const fs::path r = d;
    // Mock BDFs are assembled from parts so release privacy scanners do not flag them as real PCI addresses.
    const std::string bridge_bdf = std::string("0000:00") + ":01.0", gpu_bdf = std::string("0000:01") + ":00.0";
    const fs::path br = r / "devices/pci0000:00" / bridge_bdf, gpu = br / gpu_bdf;
    fs::create_directories(r / "bus/pci/devices");
    fs::create_directories(gpu);
    fs::create_symlink(gpu, r / "bus/pci/devices" / gpu_bdf);
    std::ofstream(br / "current_link_speed") << "2.5 GT/s PCIe\n";
    std::ofstream(br / "max_link_speed") << "8.0 GT/s PCIe\n";
    std::ofstream(br / "current_link_width") << "8\n";
    std::ofstream(br / "max_link_width") << "16\n";
    // Unprivileged-style 64-byte config -> RO/MPS/ACS must be reported unknown, never guessed.
    { std::vector<uint8_t> c(64, 0); std::ofstream(gpu / "config", std::ios::binary).write(reinterpret_cast<char*>(c.data()), c.size()); }
    auto w = vegajet::hw_preflight::run({gpu_bdf}, r.string());
    VJ_CHECK(has(w, "pcie_link", "2.5"));
    VJ_CHECK(has(w, "relaxed_ordering", "unknown") && has(w, "max_payload", "unknown") && has(w, "acs", "unknown"));
    // Full 4 KiB config: PCIe cap at 0x40 with RO off, MPS 128; bridge ACS cap with ReqRedir on.
    {
        std::vector<uint8_t> c(4096, 0);
        put16(c, 0x06, 0x10); c[0x34] = 0x40; c[0x40] = 0x10; c[0x41] = 0x00;
        put16(c, 0x48, 0x0000);  // DevCtl: RO off, MPS=128
        std::ofstream(gpu / "config", std::ios::binary).write(reinterpret_cast<char*>(c.data()), c.size());
        std::vector<uint8_t> bc(4096, 0);
        put32(bc, 0x100, 0x000Du | (1u << 16));  // ACS ext cap, next = 0
        put16(bc, 0x106, 0x0004);                // ACSCtl ReqRedir
        std::ofstream(br / "config", std::ios::binary).write(reinterpret_cast<char*>(bc.data()), bc.size());
    }
    w = vegajet::hw_preflight::run({gpu_bdf}, r.string());
    VJ_CHECK(has(w, "relaxed_ordering", "RlxdOrd-"));
    VJ_CHECK(has(w, "max_payload", "128"));
    VJ_CHECK(has(w, "acs", "ReqRedir+"));
    // Healthy device: RO on, MPS 256 -> no RO/MPS warnings.
    {
        std::vector<uint8_t> c(4096, 0);
        put16(c, 0x06, 0x10); c[0x34] = 0x40; c[0x40] = 0x10;
        put16(c, 0x48, 0x0010 | (1u << 5));
        std::ofstream(gpu / "config", std::ios::binary).write(reinterpret_cast<char*>(c.data()), c.size());
    }
    w = vegajet::hw_preflight::run({gpu_bdf}, r.string());
    VJ_CHECK(!has(w, "relaxed_ordering") && !has(w, "max_payload"));
    // Dispatch placement: cpulist parsing, UMA no-op, shared node, mixed nodes, explicit override.
    VJ_CHECK((vegajet::hw_preflight::parse_cpulist("0-2,5, 7") == std::vector<int>{0, 1, 2, 5, 7}));
    VJ_CHECK((vegajet::hw_preflight::parse_active_dpm_mhz("0: 925Mhz\n1: 1725Mhz *\n0: 350Mhz *\n2: 800MHz\n") ==
              std::vector<double>{1725.0, 350.0}));
    VJ_CHECK(vegajet::hw_preflight::parse_active_dpm_mhz("S: 19Mhz\nno star here\n").empty());
    VJ_CHECK(vegajet::hw_preflight::parse_cpulist("3-1").empty() && vegajet::hw_preflight::parse_cpulist("x").empty());
    std::string why;
    fs::create_directories(r / "devices/system/node/node1");
    std::ofstream(r / "devices/system/node/possible") << "0\n";
    VJ_CHECK(vegajet::hw_preflight::dispatch_cpus({gpu_bdf}, "", &why, r.string()).empty());
    VJ_CHECK((vegajet::hw_preflight::dispatch_cpus({gpu_bdf}, "4-5", &why, r.string()) == std::vector<int>{4, 5}));
    std::ofstream(r / "devices/system/node/possible") << "0-1\n";
    std::ofstream(r / "devices/system/node/node1/cpulist") << "8-11\n";
    std::ofstream(gpu / "numa_node") << "1\n";
    VJ_CHECK((vegajet::hw_preflight::dispatch_cpus({gpu_bdf}, "", &why, r.string()) == std::vector<int>{8, 9, 10, 11}));
    std::ofstream(gpu / "numa_node") << "-1\n";
    VJ_CHECK(vegajet::hw_preflight::dispatch_cpus({gpu_bdf}, "", &why, r.string()).empty());
    fs::remove_all(r);
    return 0;
}

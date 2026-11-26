// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Versioned hardware profile for the topology planner: per-GPU CU count, VRAM capacity, memory kind and
// theoretical bus, calibrated HBM2 streaming ceiling, sustained SCLK/MCLK, power cap, and the P2P link matrix.
//
// The on-disk form is hw_profile.json (schema 1) under the user cache directory. It is written once on first
// boot by load_or_calibrate()/calibrate_and_cache() and reused afterwards: a later boot validates the version
// and every required field instead of probing the machine again. Measurement is injectable through
// MeasurementBackend, so a CPU mock supplies the topology and streaming numbers in tests and this translation
// unit never opens a device node.
//
// Calibration policy: a measured value outside [expected_min_bw_gbs, expected_max_bw_gbs] is treated as a
// failed measurement. A failed measurement is replaced by a datasheet estimate from the memory clock
//     bw_gbs = mclk_mhz * 2 (DDR) * bus_bits / 8 / 1000 * kEstimatedPeakFraction
// when allow_estimate is set, and the profile is then marked measured=false. A failed topology probe is fatal.
//
// Compatibility: core/quant_repacker's HardwareRooflineProfile reads a flat subset of this file with a
// line-oriented parser (keys bdf, n_cu, sclk_ghz, mclk_mhz, bw_hbm_gbs, power_cap_w, eta). The serializer
// therefore emits those alias keys next to the canonical ones, and the parser accepts either spelling. The
// alias keys exist only for that reader; new code uses the canonical fields.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace vegajet::hw {

constexpr uint32_t kSchemaVersion = 1;
constexpr uint32_t kMaxGpus = 8;              // topology planner range N=1..8
constexpr uint32_t kHbm2BusBits = 4096;       // full HBM2 stack width of the reference part
constexpr double kEstimatedPeakFraction = 0.80;  // sustained streaming / theoretical peak

enum class MemKind : uint8_t { Unknown = 0, HBM2, HBM2e, GDDR6, DDR4 };
const char* mem_kind_name(MemKind m);
bool mem_kind_from_name(const std::string& s, MemKind& out);

enum class LinkKind : uint8_t {
    Unknown = 0,  // not probed: the planner treats it as a host bounce
    None,         // probed: no direct peer access
    Xgmi,
    Pcie,
};
const char* link_kind_name(LinkKind k);
bool link_kind_from_name(const std::string& s, LinkKind& out);

// One direction of one peer link. gbs is sustained unidirectional bandwidth, latency_us per message.
struct P2pLink {
    LinkKind kind = LinkKind::Unknown;
    double gbs = 0.0;
    double latency_us = 0.0;
    bool direct() const { return kind == LinkKind::Xgmi || kind == LinkKind::Pcie; }
};

struct Gpu {
    uint32_t index = 0;
    std::string id;        // stable label, e.g. "gpu0"
    std::string bdf;       // optional runtime label; may be empty
    std::string unique_id; // 16-char lowercase hex hardware unique_id from sysfs/KFD (optional)
    uint32_t cu = 0;
    uint64_t vram_bytes = 0;
    MemKind mem = MemKind::HBM2;
    uint32_t bus_bits = kHbm2BusBits;
    double mem_clock_mhz = 0.0;
    double sclk_ghz = 0.0;
    double bw_gbs = 0.0;   // calibrated sustained streaming ceiling
    double power_cap_w = 0.0;
    double eta = 0.55;     // sustained VALU issue efficiency
    std::vector<P2pLink> p2p;  // p2p[j] = link to the GPU at position j (size == gpus.size() after normalize)
};

struct Profile {
    uint32_t schema = kSchemaVersion;
    std::string arch = "gfx906";
    std::string source = "unknown";  // "measured", "estimated", "cached", "mock", ...
    std::string generated_at;        // free text; empty when the writer does not stamp a time
    bool measured = false;
    std::vector<Gpu> gpus;

    size_t size() const { return gpus.size(); }
    const Gpu* find(uint32_t index) const;
    // Peer link; out-of-range and self links are an Unknown/empty entry.
    P2pLink link(size_t from, size_t to) const;

    double aggregate_bw_gbs() const;
    double min_bw_gbs() const;
    double max_bw_gbs() const;
    uint64_t total_vram_bytes() const;
    uint64_t min_vram_bytes() const;

    // Sorts by index, fills empty ids ("gpu<N>"), permutes each P2P row with its GPU, sizes every row to
    // gpus.size(), zeroes the diagonal and makes the matrix symmetric. Conflicting directions resolve
    // conservatively: an explicit None wins, two direct kinds resolve to Xgmi if either is Xgmi, and
    // bandwidth/latency take the pessimistic value.
    void normalize();

    // Structural and physical sanity: known schema, 1..kMaxGpus unique GPUs, positive CU/VRAM/bus/clocks/
    // bandwidth, finite values, consistent p2p row sizes, and no direct link with a non-positive bandwidth.
    bool valid(std::string* why = nullptr) const;
};

// Deterministic JSON (schema 1). Emits one GPU object per line (plus compatibility aliases) so the legacy
// line-oriented roofline reader still sees the same file. `to_json` normalizes a copy first.
std::string to_json(const Profile& p);
bool from_json(const std::string& text, Profile& out, std::string* err = nullptr);

bool load(const std::string& path, Profile& out, std::string* err = nullptr);
bool save(const std::string& path, const Profile& p, std::string* err = nullptr);
std::string default_path();  // $HOME/.config/vegajet/hw_profile.json, or /tmp/vegajet_hw_profile.json

double estimate_stream_gbs(double mem_clock_mhz, uint32_t bus_bits);

// Machine probe. Implemented outside this module (a device backend in production, a mock in tests).
class MeasurementBackend {
public:
    virtual ~MeasurementBackend() = default;
    // One entry per visible GPU with index/id/cu/vram/mem/bus/clocks/power/eta. False = topology unavailable.
    virtual bool topology(std::vector<Gpu>& out, std::string* err) = 0;
    // Calibrated sustained streaming HBM2 GB/s for one GPU. False = measurement failed (estimate may apply).
    virtual bool measure_stream(const Gpu& gpu, double& gbs, std::string* err) = 0;
    // Optional: fills the p2p rows. Default leaves them Unknown, which the planner treats as a host bounce.
    virtual bool links(std::vector<Gpu>& gpus, std::string* err) {
        (void)gpus;
        if (err) err->clear();
        return true;
    }
    // Optional: ISA string stored in the profile. Defaults to the gfx906 target of this build.
    virtual std::string arch() const { return "gfx906"; }
};

struct CalibrationConfig {
    std::string cache_path;             // empty -> default_path()
    bool force = false;                 // ignore a valid cache and measure again
    bool allow_estimate = true;         // fall back to estimate_stream_gbs() when a measurement fails
    uint32_t max_gpus = kMaxGpus;
    double expected_min_bw_gbs = 50.0;
    double expected_max_bw_gbs = 4000.0;
    std::string generated_at;           // stamped into the profile verbatim (empty by default)
};

enum class CalibrationStatus : uint8_t { Cached = 0, Measured, Estimated, Failed };
const char* calibration_status_name(CalibrationStatus s);

struct CalibrationResult {
    CalibrationStatus status = CalibrationStatus::Failed;
    Profile profile;
    std::string detail;
    std::string cache_path;
    bool wrote_cache = false;
};

// Load a valid cache (schema match and valid()), otherwise measure and save. Never throws.
CalibrationResult load_or_calibrate(MeasurementBackend& backend, const CalibrationConfig& cfg = {});
// Always measure (even with a valid cache present), then save.
CalibrationResult calibrate_and_cache(MeasurementBackend& backend, const CalibrationConfig& cfg = {});

// Deterministic plain-text report (GPU table plus P2P matrix).
std::string format_profile(const Profile& p);

} // namespace vegajet::hw

// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// Contiguous routed-expert pack for host-side expert streaming: one file per model, a 4 KiB header, a 16-byte-per-
// expert index and the expert records back to back, every record padded to 4 KiB so a record read is a single
// O_DIRECT-eligible request (offset, length and buffer all 4 KiB aligned).
//
// Record (layer, expert) = gate [gate_elems f32], up [up_elems f32], down [down_elems f32], zero padding to the
// next 4 KiB. Index entry i (layer-major, i = layer * n_experts + expert) is {u64 offset, u32 bytes, u32 flags},
// so a worker that knows (layer, expert) only needs the header to address the record; the index makes the format
// self-describing and lets a packer append without rewriting data.
//
// I/O: io_uring (IORING_OP_READ, raw syscalls, no liburing) when the kernel allows it and VJ_IO_URING != 0,
// otherwise pread. Both paths use O_DIRECT when the filesystem accepts it and fall back to the page-cache path if
// the open or the first read is rejected (some filesystems reject O_DIRECT on overlay/network mounts); the
// destination is copied through an aligned bounce buffer when the caller's buffer is not 4 KiB aligned, so a
// read is never silently short or misaligned.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vegajet::expertpack {

constexpr uint64_t kAlign = 4096;
constexpr uint32_t kHeaderBytes = 4096;
constexpr uint32_t kIndexEntryBytes = 16;

struct PackDesc {
    uint32_t n_layers = 0;
    uint32_t n_experts = 0;
    uint32_t gate_elems = 0;
    uint32_t up_elems = 0;
    uint32_t down_elems = 0;
};

uint64_t round_up(uint64_t v, uint64_t a);
uint64_t payload_bytes(const PackDesc& d);   // (gate + up + down) * 4
uint64_t record_bytes(const PackDesc& d);    // payload rounded up to kAlign
uint64_t index_bytes(const PackDesc& d);     // n_layers * n_experts * 16

struct PackOptions {
    bool use_uring = true;          // ignored (off) when VJ_IO_URING=0
    bool use_direct = true;         // O_DIRECT when the filesystem supports it
    bool simulate_ring_failure = false;  // tests: break the ring fd so the automatic pread fallback runs
    uint32_t ring_entries = 8;
};

// Writes a pack. `src` holds n_layers * n_experts records in layer-major order, each gate, up, down contiguous.
bool write_pack(const std::string& path, const PackDesc& d, const float* src, std::string* err);

class PackReader {
public:
    PackReader();
    ~PackReader();
    PackReader(const PackReader&) = delete;
    PackReader& operator=(const PackReader&) = delete;

    bool open(const std::string& path, std::string* err);
    bool open(const std::string& path, const PackOptions& opt, std::string* err);
    bool is_open() const { return fd_ >= 0; }
    const PackDesc& desc() const { return desc_; }

    uint64_t expert_offset(uint32_t layer, uint32_t expert) const;
    // One record into three FP32 buffers (each must hold desc().<part>_elems floats).
    bool read_expert(uint32_t layer, uint32_t expert, float* gate, float* up, float* down, std::string* err);
    // Raw record bytes (record_bytes() of them) into dst; dst may be any alignment.
    bool read_raw(uint32_t layer, uint32_t expert, void* dst, size_t dst_bytes, std::string* err);

    const char* io_mode() const;   // "io_uring+O_DIRECT", "pread+O_DIRECT", "io_uring", "pread"
    struct Stats {
        uint64_t reads = 0, uring_reads = 0, pread_reads = 0, bounced = 0, direct = 0, retries = 0;
        uint64_t uring_failures = 0;  // times an io_uring submission failed and the read went to pread
    };
    const Stats& stats() const { return st_; }

private:
    bool parse_header(const std::vector<uint8_t>& h, std::string* err);
    bool ensure_scratch(size_t n);
    // Direct read into an aligned buffer, n a multiple of kAlign; retries once without O_DIRECT on EINVAL.
    bool read_irect(uint64_t off, void* buf, size_t n, std::string* err);
    // Arbitrary destination: aligned destinations read in place, others through the aligned scratch window.
    bool read_at(uint64_t off, void* dst, size_t n, std::string* err);
    bool load_index(std::string* err);
    bool reopen(bool direct);
    void close();

    struct Ring;
    std::unique_ptr<Ring> ring_;
    std::string path_;
    int fd_ = -1;
    bool direct_ = false;
    bool try_uring_ = true;
    PackDesc desc_;
    uint64_t record_bytes_ = 0, data_offset_ = 0, total_bytes_ = 0;
    std::vector<uint8_t> index_;
    uint8_t* scratch_ = nullptr;
    size_t scratch_bytes_ = 0;
    Stats st_;
};

}  // namespace vegajet::expertpack

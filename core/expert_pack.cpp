// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#include "core/expert_pack.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>

#if defined(__linux__)
#include <sys/syscall.h>
#endif

#if defined(__linux__) && defined(__NR_io_uring_setup) && defined(__NR_io_uring_enter)
#include <linux/io_uring.h>
#include <sys/mman.h>
#define VJ_HAVE_IO_URING 1
#endif

namespace vegajet::expertpack {

namespace {
constexpr char kMagic[8] = {'V', 'J', 'E', 'X', 'P', 'A', 'C', 'K'};
constexpr uint32_t kVersion = 1;

void put_u32(std::vector<uint8_t>& b, size_t off, uint32_t v) { std::memcpy(b.data() + off, &v, 4); }
void put_u64(std::vector<uint8_t>& b, size_t off, uint64_t v) { std::memcpy(b.data() + off, &v, 8); }
uint32_t get_u32(const uint8_t* b, size_t off) {
    uint32_t v;
    std::memcpy(&v, b + off, 4);
    return v;
}
uint64_t get_u64(const uint8_t* b, size_t off) {
    uint64_t v;
    std::memcpy(&v, b + off, 8);
    return v;
}

bool uring_enabled_by_env() {
    const char* v = std::getenv("VJ_IO_URING");
    return !(v && std::strcmp(v, "0") == 0);
}

bool write_all(int fd, const void* p, size_t n) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    size_t off = 0;
    while (off < n) {
        const ssize_t w = ::write(fd, b + off, n - off);
        if (w < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        off += static_cast<size_t>(w);
    }
    return true;
}

bool aligned_ptr(const void* p) { return (reinterpret_cast<uintptr_t>(p) & (kAlign - 1)) == 0; }
}  // namespace

uint64_t round_up(uint64_t v, uint64_t a) { return a ? ((v + a - 1) / a) * a : v; }
uint64_t payload_bytes(const PackDesc& d) { return (uint64_t(d.gate_elems) + d.up_elems + d.down_elems) * 4; }
uint64_t record_bytes(const PackDesc& d) { return round_up(payload_bytes(d), kAlign); }
uint64_t index_bytes(const PackDesc& d) { return uint64_t(d.n_layers) * d.n_experts * kIndexEntryBytes; }

// ------------------------------------------------------------------------------------------------- writer
bool write_pack(const std::string& path, const PackDesc& d, const float* src, std::string* err) {
    auto fail = [&](const std::string& s) {
        if (err) *err = s;
        return false;
    };
    if (!d.n_layers || !d.n_experts || !d.gate_elems || !d.up_elems || !d.down_elems) return fail("empty pack description");
    if (!src) return fail("null source");
    const uint64_t rec = record_bytes(d), idx = index_bytes(d);
    const uint64_t idx_off = kHeaderBytes;
    const uint64_t data_off = round_up(idx_off + idx, kAlign);
    const uint64_t total = data_off + rec * uint64_t(d.n_layers) * d.n_experts;

    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return fail(std::string("open for write: ") + std::strerror(errno));

    std::vector<uint8_t> h(kHeaderBytes, 0);
    std::memcpy(h.data(), kMagic, 8);
    put_u32(h, 8, kVersion);
    put_u32(h, 12, kHeaderBytes);
    put_u32(h, 16, d.n_layers);
    put_u32(h, 20, d.n_experts);
    put_u32(h, 24, d.gate_elems);
    put_u32(h, 28, d.up_elems);
    put_u32(h, 32, d.down_elems);
    put_u32(h, 36, static_cast<uint32_t>(kAlign));
    put_u32(h, 40, 0);  // dtype: 0 = f32
    put_u64(h, 48, rec);
    put_u64(h, 56, idx_off);
    put_u64(h, 64, uint64_t(d.n_layers) * d.n_experts);
    put_u64(h, 72, data_off);
    put_u64(h, 80, total);

    std::vector<uint8_t> index(size_t(idx), 0);
    const uint64_t payload = payload_bytes(d);
    for (uint32_t l = 0, i = 0; l < d.n_layers; ++l)
        for (uint32_t e = 0; e < d.n_experts; ++e, ++i) {
            const uint64_t off = data_off + uint64_t(i) * rec;
            std::memcpy(index.data() + size_t(i) * kIndexEntryBytes, &off, 8);
            const uint32_t nb = static_cast<uint32_t>(payload);
            std::memcpy(index.data() + size_t(i) * kIndexEntryBytes + 8, &nb, 4);
            put_u32(index, size_t(i) * kIndexEntryBytes + 12, 0);
        }

    bool ok = write_all(fd, h.data(), h.size()) && write_all(fd, index.data(), index.size());
    std::vector<uint8_t> pad(kAlign, 0);
    if (ok && data_off > idx_off + idx) ok = write_all(fd, pad.data(), size_t(data_off - idx_off - idx));
    const std::vector<uint8_t> zeros(kAlign, 0);
    const size_t plen = size_t(payload), zero_pad = size_t(rec - payload);
    for (uint32_t l = 0, i = 0; ok && l < d.n_layers; ++l)
        for (uint32_t e = 0; e < d.n_experts; ++e, ++i) {
            const float* r = src + size_t(i) * size_t(d.gate_elems + d.up_elems + d.down_elems);
            ok = write_all(fd, r, plen);
            if (ok && zero_pad) ok = write_all(fd, zeros.data(), zero_pad);
        }
    if (::fsync(fd) != 0 && errno != EINVAL) ok = false;
    if (::close(fd) != 0) ok = false;
    if (!ok) {
        ::unlink(path.c_str());
        return fail("write pack failed");
    }
    return true;
}

// ------------------------------------------------------------------------------------------------- io_uring
#if defined(VJ_HAVE_IO_URING)
struct PackReader::Ring {
    int fd = -1;
    void* sq_ptr = MAP_FAILED;
    void* cq_ptr = MAP_FAILED;
    size_t sq_size = 0, cq_size = 0, sqes_size = 0;
    io_uring_sqe* sqes = nullptr;
    uint32_t *sq_head = nullptr, *sq_tail = nullptr, *sq_array = nullptr;
    uint32_t *cq_head = nullptr, *cq_tail = nullptr;
    uint32_t sq_mask = 0, sq_entries = 0, cq_mask = 0;
    io_uring_cqe* cqes = nullptr;

    ~Ring() {
        if (sqes) ::munmap(sqes, sqes_size);
        if (cq_ptr != MAP_FAILED && cq_ptr != sq_ptr) ::munmap(cq_ptr, cq_size);
        if (sq_ptr != MAP_FAILED) ::munmap(sq_ptr, sq_size);
        if (fd >= 0) ::close(fd);
    }

    bool init(uint32_t entries) {
        io_uring_params p{};
        p.flags = IORING_SETUP_CLAMP | IORING_SETUP_CQSIZE;
        p.cq_entries = 2 * entries;
        fd = static_cast<int>(::syscall(__NR_io_uring_setup, entries, &p));
        if (fd < 0) return false;
        sq_size = p.sq_off.array + p.sq_entries * sizeof(uint32_t);
        cq_size = p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe);
        const bool single = (p.features & IORING_FEAT_SINGLE_MMAP) != 0;
        if (single) sq_size = cq_size = std::max(sq_size, cq_size);
        sq_ptr = ::mmap(nullptr, sq_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQ_RING);
        if (sq_ptr == MAP_FAILED) return false;
        cq_ptr = single ? sq_ptr
                        : ::mmap(nullptr, cq_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd,
                                 IORING_OFF_CQ_RING);
        if (cq_ptr == MAP_FAILED) return false;
        sqes_size = p.sq_entries * sizeof(io_uring_sqe);
        void* sq = ::mmap(nullptr, sqes_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQES);
        if (sq == MAP_FAILED) return false;
        sqes = static_cast<io_uring_sqe*>(sq);
        auto* sb = static_cast<uint8_t*>(sq_ptr);
        auto* cb = static_cast<uint8_t*>(cq_ptr);
        sq_head = reinterpret_cast<uint32_t*>(sb + p.sq_off.head);
        sq_tail = reinterpret_cast<uint32_t*>(sb + p.sq_off.tail);
        sq_mask = *reinterpret_cast<uint32_t*>(sb + p.sq_off.ring_mask);
        sq_entries = *reinterpret_cast<uint32_t*>(sb + p.sq_off.ring_entries);
        sq_array = reinterpret_cast<uint32_t*>(sb + p.sq_off.array);
        cq_head = reinterpret_cast<uint32_t*>(cb + p.cq_off.head);
        cq_tail = reinterpret_cast<uint32_t*>(cb + p.cq_off.tail);
        cq_mask = *reinterpret_cast<uint32_t*>(cb + p.cq_off.ring_mask);
        cqes = reinterpret_cast<io_uring_cqe*>(cb + p.cq_off.cqes);
        for (uint32_t i = 0; i < sq_entries; ++i) sq_array[i] = i;
        return true;
    }

    // One synchronous IORING_OP_READ; returns bytes in *res or a negative errno.
    bool read_once(int file_fd, void* buf, size_t len, uint64_t off, int* res) {
        const uint32_t tail = __atomic_load_n(sq_tail, __ATOMIC_ACQUIRE);
        const uint32_t idx = tail & sq_mask;
        io_uring_sqe* sqe = &sqes[idx];
        std::memset(sqe, 0, sizeof *sqe);
        sqe->opcode = IORING_OP_READ;
        sqe->fd = file_fd;
        sqe->addr = reinterpret_cast<uint64_t>(buf);
        sqe->len = static_cast<uint32_t>(len);
        sqe->off = off;
        sqe->user_data = tail;
        sq_array[idx] = idx;
        __atomic_store_n(sq_tail, tail + 1, __ATOMIC_RELEASE);
        if (::syscall(__NR_io_uring_enter, fd, 1, 1, IORING_ENTER_GETEVENTS, nullptr, 0) < 0) return false;
        for (;;) {
            const uint32_t head = __atomic_load_n(cq_head, __ATOMIC_ACQUIRE);
            if (head != __atomic_load_n(cq_tail, __ATOMIC_ACQUIRE)) {
                const io_uring_cqe* cqe = &cqes[head & cq_mask];
                *res = cqe->res;
                __atomic_store_n(cq_head, head + 1, __ATOMIC_RELEASE);
                return true;
            }
            if (::syscall(__NR_io_uring_enter, fd, 0, 1, IORING_ENTER_GETEVENTS, nullptr, 0) < 0) return false;
        }
    }
};
#else
struct PackReader::Ring {
    bool init(uint32_t) { return false; }
    bool read_once(int, void*, size_t, uint64_t, int*) { return false; }
};
#endif

// ------------------------------------------------------------------------------------------------- reader
PackReader::PackReader() = default;
PackReader::~PackReader() { close(); }

void PackReader::close() {
    ring_.reset();
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
    std::free(scratch_);
    scratch_ = nullptr;
    scratch_bytes_ = 0;
}

bool PackReader::ensure_scratch(size_t n) {
    if (scratch_bytes_ >= n) return true;
    std::free(scratch_);
    scratch_ = nullptr;
    scratch_bytes_ = 0;
    void* p = nullptr;
    if (::posix_memalign(&p, kAlign, n) != 0 || !p) return false;
    scratch_ = static_cast<uint8_t*>(p);
    scratch_bytes_ = n;
    return true;
}

bool PackReader::open(const std::string& path, std::string* err) {
    PackOptions opt;
    opt.use_uring = uring_enabled_by_env();
    return open(path, opt, err);
}

bool PackReader::open(const std::string& path, const PackOptions& opt, std::string* err) {
    auto fail = [&](const std::string& s) {
        if (err) *err = s;
        close();
        return false;
    };
    close();
    path_ = path;
    direct_ = opt.use_direct;
    if (!reopen(direct_)) return fail(std::string("open: ") + std::strerror(errno));
    struct stat sb {};
    if (::fstat(fd_, &sb) != 0) return fail(std::string("fstat: ") + std::strerror(errno));
    total_bytes_ = static_cast<uint64_t>(sb.st_size);

    std::vector<uint8_t> hdr(kHeaderBytes, 0);
    std::string rerr;
    if (!read_at(0, hdr.data(), hdr.size(), &rerr)) return fail("read header: " + rerr);
    if (!parse_header(hdr, err)) return fail(err ? *err : "bad header");
    if (!load_index(&rerr)) return fail("read index: " + rerr);
    // The scratch window must hold a whole record (and a padded index read).
    if (!ensure_scratch(size_t(round_up(record_bytes_, kAlign)))) return fail("cannot allocate the aligned scratch buffer");

    try_uring_ = opt.use_uring && uring_enabled_by_env();
    if (try_uring_) {
        auto r = std::make_unique<Ring>();
        if (r->init(opt.ring_entries ? opt.ring_entries : 8)) ring_ = std::move(r);
        try_uring_ = false;  // one attempt per open
        if (ring_ && opt.simulate_ring_failure) {  // tests: force the automatic fallback
            ::close(ring_->fd);
            ring_->fd = -1;
        }
    }
    return true;
}

bool PackReader::reopen(bool direct) {
    if (fd_ >= 0) ::close(fd_);
    fd_ = ::open(path_.c_str(), O_RDONLY | (direct ? O_DIRECT : 0) | O_CLOEXEC);
    if (fd_ < 0 && direct) {
        fd_ = ::open(path_.c_str(), O_RDONLY | O_CLOEXEC);
        direct = false;
    }
    direct_ = direct;
    if (fd_ < 0) return false;
    return true;
}

bool PackReader::parse_header(const std::vector<uint8_t>& h, std::string* err) {
    auto fail = [&](const char* s) {
        if (err) *err = s;
        return false;
    };
    if (std::memcmp(h.data(), kMagic, 8) != 0) return fail("not a VJEXPACK file");
    if (get_u32(h.data(), 8) != kVersion) return fail("unsupported pack version");
    if (get_u32(h.data(), 12) != kHeaderBytes) return fail("bad header size");
    desc_.n_layers = get_u32(h.data(), 16);
    desc_.n_experts = get_u32(h.data(), 20);
    desc_.gate_elems = get_u32(h.data(), 24);
    desc_.up_elems = get_u32(h.data(), 28);
    desc_.down_elems = get_u32(h.data(), 32);
    if (!desc_.n_layers || !desc_.n_experts || !desc_.gate_elems || !desc_.up_elems || !desc_.down_elems)
        return fail("empty pack description");
    if (get_u32(h.data(), 36) != kAlign) return fail("unsupported alignment");
    if (get_u32(h.data(), 40) != 0) return fail("unsupported dtype");
    record_bytes_ = get_u64(h.data(), 48);
    const uint64_t idx_off = get_u64(h.data(), 56), idx_n = get_u64(h.data(), 64);
    data_offset_ = get_u64(h.data(), 72);
    const uint64_t total = get_u64(h.data(), 80);
    if (record_bytes_ != record_bytes(desc_)) return fail("record size does not match the description");
    if (idx_n != uint64_t(desc_.n_layers) * desc_.n_experts) return fail("index entry count mismatch");
    if (idx_off % kAlign || data_offset_ % kAlign) return fail("misaligned index/data offset");
    if (data_offset_ < idx_off + idx_n * kIndexEntryBytes) return fail("data overlaps the index");
    if (total != data_offset_ + record_bytes_ * idx_n) return fail("total size mismatch");
    if (total != total_bytes_) return fail("file is truncated or has trailing bytes");
    return true;
}

bool PackReader::load_index(std::string* err) {
    const uint64_t n = index_bytes(desc_);
    index_.assign(size_t(n), 0);
    const size_t padded = size_t(round_up(n, kAlign));
    if (!ensure_scratch(padded)) {
        if (err) *err = "cannot allocate the aligned scratch buffer";
        return false;
    }
    // The index is followed by zero padding up to the 4 KiB-aligned data offset.
    if (!read_irect(kHeaderBytes, scratch_, padded, err)) return false;
    std::memcpy(index_.data(), scratch_, size_t(n));
    return true;
}

uint64_t PackReader::expert_offset(uint32_t layer, uint32_t expert) const {
    const uint64_t i = uint64_t(layer) * desc_.n_experts + expert;
    uint64_t off = 0;
    if ((i + 1) * kIndexEntryBytes <= index_.size()) std::memcpy(&off, index_.data() + size_t(i) * kIndexEntryBytes, 8);
    return off;
}

bool PackReader::read_irect(uint64_t off, void* buf, size_t n, std::string* err) {
    for (int attempt = 0; attempt < 2; ++attempt) {
        std::string why;
        int e = 0;
        bool ok = false;
        if (ring_) {
            int res = 0;
            if (!ring_->read_once(fd_, buf, n, off, &res)) {
                ring_.reset();  // unusable ring: pread for the rest of the session
                ++st_.uring_failures;
                why = "io_uring_enter failed";
            } else if (res < 0) {
                e = -res;
                why = std::strerror(e);
            } else if (static_cast<size_t>(res) != n) {
                why = "short io_uring read";
            } else {
                ok = true;
                ++st_.uring_reads;
            }
        }
        if (!ok) {
            const ssize_t r = ::pread(fd_, buf, n, static_cast<off_t>(off));
            if (r == static_cast<ssize_t>(n)) {
                ok = true;
                ++st_.pread_reads;
            } else if (r < 0) {
                e = errno;
                why = std::strerror(errno);
            } else {
                why = "short read";
            }
        }
        if (ok) {
            ++st_.reads;
            if (direct_) ++st_.direct;
            return true;
        }
        if (direct_ && e == EINVAL && attempt == 0) {  // filesystem rejects O_DIRECT: page-cache path
            if (!reopen(false)) {
                if (err) *err = std::string("reopen without O_DIRECT: ") + std::strerror(errno);
                return false;
            }
            ++st_.retries;
            continue;
        }
        if (err) *err = why;
        return false;
    }
    if (err) *err = "read failed";
    return false;
}

bool PackReader::read_at(uint64_t off, void* dst, size_t n, std::string* err) {
    if (aligned_ptr(dst) && (n % kAlign) == 0 && (off % kAlign) == 0) return read_irect(off, dst, n, err);
    const uint64_t start = off & ~(kAlign - 1);
    const size_t pad = size_t(off - start);
    const size_t len = size_t(round_up(pad + n, kAlign));
    if (!ensure_scratch(len)) {
        if (err) *err = "cannot allocate the aligned scratch buffer";
        return false;
    }
    if (!read_irect(start, scratch_, len, err)) return false;
    std::memcpy(dst, scratch_ + pad, n);
    ++st_.bounced;
    return true;
}

bool PackReader::read_raw(uint32_t layer, uint32_t expert, void* dst, size_t dst_bytes, std::string* err) {
    auto fail = [&](const std::string& s) {
        if (err) *err = s;
        return false;
    };
    if (fd_ < 0) return fail("reader is not open");
    if (layer >= desc_.n_layers || expert >= desc_.n_experts) return fail("expert id out of range");
    if (dst_bytes < record_bytes_) return fail("destination too small");
    const uint64_t off = expert_offset(layer, expert);
    if (off < data_offset_ || off + record_bytes_ > total_bytes_ || off % kAlign) return fail("index entry out of range");
    uint32_t bytes = 0;
    std::memcpy(&bytes, index_.data() + (size_t(layer) * desc_.n_experts + expert) * kIndexEntryBytes + 8, 4);
    if (bytes != payload_bytes(desc_)) return fail("index entry size mismatch");
    return read_at(off, dst, size_t(record_bytes_), err);
}

bool PackReader::read_expert(uint32_t layer, uint32_t expert, float* gate, float* up, float* down,
                             std::string* err) {
    if (!gate || !up || !down) {
        if (err) *err = "null expert buffer";
        return false;
    }
    if (!ensure_scratch(size_t(record_bytes_))) {
        if (err) *err = "cannot allocate the aligned scratch buffer";
        return false;
    }
    // read_raw validates the ids and the index entry as well as the alignment.
    if (!read_raw(layer, expert, scratch_, size_t(record_bytes_), err)) return false;
    const uint8_t* p = scratch_;
    std::memcpy(gate, p, size_t(desc_.gate_elems) * 4);
    p += size_t(desc_.gate_elems) * 4;
    std::memcpy(up, p, size_t(desc_.up_elems) * 4);
    p += size_t(desc_.up_elems) * 4;
    std::memcpy(down, p, size_t(desc_.down_elems) * 4);
    return true;
}

const char* PackReader::io_mode() const {
    const bool u = ring_ != nullptr || st_.uring_reads > 0;
    if (u && direct_) return "io_uring+O_DIRECT";
    if (u) return "io_uring";
    if (direct_) return "pread+O_DIRECT";
    return "pread";
}

}  // namespace vegajet::expertpack

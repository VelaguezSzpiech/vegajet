// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// core/async_table_prefetch against a temporary table file: FP8 / F16 / Q4 row decoding (E4M3 exhaustively vs an
// independent formula), dedup + run coalescing into vectored reads, LRU eviction, cache hits, short reads beyond
// EOF, out-of-range rows, slot exhaustion, the sync / thread-pool / io_uring backends giving identical bytes (io_uring
// also with every read split into partial SQEs), a mixed prefetch/fetch stress loop, the n-gram bucket hash, and the
// io_uring backend directly: >IOV_MAX scatter lists, backlog beyond the ring size, EOF / EBADF completions, 4
// submitter + 2 reaper threads on one ring, destruction with reads in flight, and the setup / runtime / environment
// fallbacks to preadv.
#include "core/async_table_prefetch.hpp"
#include "tests/test_common.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <map>
#include <mutex>
#include <random>
#include <set>
#include <thread>

using namespace vegajet::prefetch;

namespace {

constexpr uint64_t kHeader = 4096;  // the table starts after an unrelated header, like a tensor inside a GGUF

uint8_t row_byte(uint32_t row, uint32_t i) { return static_cast<uint8_t>((row * 2654435761u + i * 40503u) >> 13); }

struct TempTable {
    std::string path;
    int fd = -1;
    TempTable(uint32_t rows, uint32_t row_bytes) {
        const char* dir = std::getenv("TMPDIR");
        path = std::string(dir && *dir ? dir : "/tmp") + "/vj-prefetch-XXXXXX";
        fd = mkstemp(path.data());
        VJ_CHECK(fd >= 0);
        std::vector<uint8_t> buf(kHeader, 0xEE);
        for (uint32_t r = 0; r < rows; ++r)
            for (uint32_t i = 0; i < row_bytes; ++i) buf.push_back(row_byte(r, i));
        VJ_CHECK(write(fd, buf.data(), buf.size()) == static_cast<ssize_t>(buf.size()));
    }
    ~TempTable() {
        close(fd);
        unlink(path.c_str());
    }
};

float e4m3_ref(uint8_t b) {  // written from the OCP FP8 table: bias 7, subnormals at e = 0, only S.1111.111 is NaN
    const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
    if (e == 15 && m == 7) return NAN;
    const double v = e ? (1.0 + m / 8.0) * std::pow(2.0, e - 7) : (m / 8.0) * std::pow(2.0, -6);
    return static_cast<float>(s ? -v : v);
}

bool same_bytes(RowPrefetcher& p, uint32_t row, uint32_t rb) {
    std::vector<uint8_t> got(rb);
    std::string err;
    if (!p.fetch(row, got.data(), &err)) return false;
    for (uint32_t i = 0; i < rb; ++i)
        if (got[i] != row_byte(row, i)) return false;
    return true;
}

// Submits `ops`, reaps until every one completed exactly once; returns user_data -> result.
std::map<uint64_t, int64_t> run_all(IoBackend& io, std::vector<ReadOp> ops) {
    const size_t n = ops.size();
    io.submit(std::move(ops));
    std::map<uint64_t, int64_t> res;
    std::vector<Completion> c;
    for (int spins = 0; res.size() < n && spins < 4000; ++spins) {
        c.clear();
        io.reap(c, 50);
        for (const auto& x : c) VJ_CHECK(res.emplace(x.user_data, x.result).second);  // exactly once
    }
    VJ_CHECK(res.size() == n);
    return res;
}

// Random scatter reads over the whole file (header + rows), with their expected byte counts.
struct OpSet {
    std::vector<std::vector<uint8_t>> bufs;  // one per iovec, alive until the ops completed
    std::vector<ReadOp> ops;
    std::vector<uint64_t> want;              // expected result per op (bytes)
};

OpSet random_ops(int fd, uint64_t file_size, uint32_t n, uint32_t seed, uint64_t tag_base) {
    OpSet s;
    std::mt19937 rng(seed);
    s.bufs.reserve(n * 48 + 1600);
    for (uint32_t i = 0; i < n; ++i) {
        ReadOp op;
        op.fd = fd;
        op.user_data = tag_base + i;
        op.offset = rng() % 7 == 0 ? file_size - rng() % 600 : rng() % file_size;  // some ops cross EOF
        const uint32_t k = i == 0 ? 1500 : 1 + rng() % 40;                         // op 0: more than IOV_MAX iovecs
        uint64_t total = 0;
        for (uint32_t j = 0; j < k; ++j) {
            s.bufs.emplace_back(i == 0 ? 1 + rng() % 9 : 1 + rng() % 300, 0x5A);
            op.iov.push_back({s.bufs.back().data(), s.bufs.back().size()});
            total += s.bufs.back().size();
        }
        s.want.push_back(std::min<uint64_t>(total, file_size - std::min(file_size, op.offset)));
        s.ops.push_back(std::move(op));
    }
    return s;
}

bool check_op(const ReadOp& op, int64_t result, uint64_t want, uint32_t rb) {
    if (result != static_cast<int64_t>(want)) return false;
    uint64_t pos = op.offset, left = want;
    for (const auto& v : op.iov) {
        const auto* b = static_cast<const uint8_t*>(v.iov_base);
        for (size_t i = 0; i < v.iov_len && left; ++i, ++pos, --left) {
            const uint8_t f = pos < kHeader ? 0xEE : row_byte(static_cast<uint32_t>((pos - kHeader) / rb),
                                                              static_cast<uint32_t>((pos - kHeader) % rb));
            if (b[i] != f) return false;
        }
    }
    return true;
}

void check_opset(IoBackend& io, OpSet s, uint32_t rb) {
    const std::vector<ReadOp> copy = s.ops;  // iovecs still point at s.bufs
    const auto res = run_all(io, std::move(s.ops));
    for (size_t i = 0; i < copy.size(); ++i) VJ_CHECK(check_op(copy[i], res.at(copy[i].user_data), s.want[i], rb));
}

} // namespace

int main() {
    // ---- formats
    for (int b = 0; b < 256; ++b) {
        const float a = e4m3_to_f32(static_cast<uint8_t>(b)), r = e4m3_ref(static_cast<uint8_t>(b));
        VJ_CHECK((std::isnan(a) && std::isnan(r)) || a == r);
    }
    VJ_CHECK(e4m3_to_f32(0x7E) == 448.0f && e4m3_to_f32(0x01) == std::ldexp(1.0f, -9));
    VJ_CHECK(f16_to_f32(0x3C00) == 1.0f && f16_to_f32(0xC000) == -2.0f && f16_to_f32(0x0001) == std::ldexp(1.0f, -24));
    VJ_CHECK(row_bytes_for(RowFormat::FP8, 256) == 258 && row_bytes_for(RowFormat::F16, 256) == 512 &&
             row_bytes_for(RowFormat::Q4, 256) == 144 && row_bytes_for(RowFormat::Q4, 100) == 0);
    {
        uint8_t q4[18] = {0x00, 0x38};  // scale 0.5
        for (int j = 0; j < 16; ++j) q4[2 + j] = static_cast<uint8_t>((j & 15) | (((15 - j) & 15) << 4));
        float out[32];
        VJ_CHECK(decode_row(RowFormat::Q4, q4, 32, out));
        for (int j = 0; j < 16; ++j) {  // ggml Q4_0: d * (nibble - 8)
            VJ_CHECK(out[j] == 0.5f * (j - 8));
            VJ_CHECK(out[16 + j] == 0.5f * ((15 - j) - 8));
        }
        uint8_t fp8[2 + 4] = {0x00, 0x40, 0x38, 0xB8, 0x00, 0x7E};  // scale 2: 1, -1, 0, 448
        VJ_CHECK(decode_row(RowFormat::FP8, fp8, 4, out) && out[0] == 2 && out[1] == -2 && out[2] == 0 && out[3] == 896);
    }

    const uint32_t kRows = 5000, kElems = 256;
    const uint32_t rb = row_bytes_for(RowFormat::FP8, kElems);
    TempTable file(kRows, rb);
    TableDesc t;
    t.fd = file.fd;
    t.file_offset = kHeader;
    t.n_rows = kRows;
    t.row_elems = kElems;
    t.fmt = RowFormat::FP8;

    std::string why;
    const bool have_uring = IoUringBackend().ok(&why);
    if (!have_uring) std::printf("  NOTE: io_uring unavailable here (%s): io_uring cases run on the fallback\n", why.c_str());
    for (int backend = 0; backend < 4; ++backend) {
        auto make_io = [&](uint32_t threads) -> std::unique_ptr<IoBackend> {
            if (backend == 0) return std::make_unique<SyncBackend>();
            if (backend == 1) return std::make_unique<PreadvBackend>(threads);
            IoUringOptions o;
            o.entries = 32;
            if (backend == 3) o.max_bytes_per_sqe = 100;  // every 258-byte row takes 3 SQEs: partial-read path
            return make_io_backend(BackendKind::IoUring, threads, nullptr, o);
        };
        auto make = [&](PrefetchConfig c) { return std::make_unique<RowPrefetcher>(t, c, make_io(4)); };
        // ---- coalescing: 256 adjacent rows -> 4 vectored reads of 64; duplicates and resident rows dropped.
        {
            auto p = make(PrefetchConfig{1024, 64});
            VJ_CHECK(p->ok());
            std::vector<uint32_t> rows;
            for (uint32_t r = 1000; r < 1256; ++r) { rows.push_back(r); if (r % 3 == 0) rows.push_back(r); }
            VJ_CHECK(p->prefetch(rows.data(), rows.size()) == 256);
            VJ_CHECK(p->stats().read_ops == 4 && p->stats().deduped == rows.size() - 256);
            VJ_CHECK(p->prefetch(rows.data(), 10) == 0);  // already in flight or resident
            for (uint32_t r = 1000; r < 1256; ++r) VJ_CHECK(same_bytes(*p, r, rb));
            VJ_CHECK(p->in_flight() == 0 && p->stats().read_ops == 4);
            const uint64_t hits = p->stats().hits;
            for (uint32_t r = 1000; r < 1010; ++r) VJ_CHECK(same_bytes(*p, r, rb));
            VJ_CHECK(p->stats().hits == hits + 10 && p->stats().read_ops == 4);
            // Scattered rows: one read per run of adjacent ids.
            const uint32_t scattered[] = {7, 8, 9, 400, 402, 403, 404, 4999};
            p->prefetch(scattered, 8);
            VJ_CHECK(p->stats().read_ops == 4 + 4);
            std::vector<float> f(kElems);
            VJ_CHECK(p->fetch_f32(403, f.data()));
            std::vector<uint8_t> raw(rb);
            for (uint32_t i = 0; i < rb; ++i) raw[i] = row_byte(403, i);
            std::vector<float> want(kElems);
            decode_row(RowFormat::FP8, raw.data(), kElems, want.data());
            for (uint32_t i = 0; i < kElems; ++i) VJ_CHECK((std::isnan(f[i]) && std::isnan(want[i])) || f[i] == want[i]);
        }
        // ---- LRU eviction with a small cache; rows prefetched again are touched, not re-read.
        {
            auto p = make(PrefetchConfig{64, 16});
            std::vector<uint32_t> rows(48);
            for (uint32_t i = 0; i < 48; ++i) rows[i] = 2000 + i;
            p->prefetch(rows.data(), rows.size());
            for (uint32_t r : rows) VJ_CHECK(same_bytes(*p, r, rb));
            const uint32_t keep[] = {2000, 2001};
            p->prefetch(keep, 2);  // touch: now most recent
            std::vector<uint32_t> more(40);
            for (uint32_t i = 0; i < 40; ++i) more[i] = 3000 + i;
            p->prefetch(more.data(), more.size());
            for (uint32_t r : more) VJ_CHECK(same_bytes(*p, r, rb));
            VJ_CHECK(p->stats().evictions == 24);
            const uint64_t ops = p->stats().read_ops;
            VJ_CHECK(same_bytes(*p, 2000, rb) && same_bytes(*p, 2001, rb) && p->stats().read_ops == ops);  // survived
            VJ_CHECK(same_bytes(*p, 2002, rb) && p->stats().read_ops == ops + 1);                          // evicted
        }
        // ---- errors: rows beyond EOF (table claims more rows than the file has), out-of-range ids.
        {
            TableDesc big = t;
            big.n_rows = kRows + 10;
            RowPrefetcher p(big, PrefetchConfig{128, 64}, make_io(2));
            const uint32_t tail[] = {kRows - 2, kRows - 1, kRows, kRows + 1};
            p.prefetch(tail, 4);  // one vectored read, partially satisfied
            VJ_CHECK(same_bytes(p, kRows - 2, rb) && same_bytes(p, kRows - 1, rb));
            std::vector<uint8_t> buf(rb);
            std::string err;
            VJ_CHECK(!p.fetch(kRows, buf.data(), &err) && err.find("short read") != std::string::npos);
            VJ_CHECK(!p.fetch(kRows + 1, buf.data(), &err));
            VJ_CHECK(!p.fetch(kRows + 10, buf.data(), &err) && err.find("out of range") != std::string::npos);
            VJ_CHECK(p.stats().errors == 3);
            VJ_CHECK(!p.fetch(kRows, buf.data(), &err));  // retried after the error slot was freed, still short
        }
        // ---- slot exhaustion: more rows than slots in one call -> deferred, fetch still works (direct read).
        {
            auto p = make(PrefetchConfig{8, 64});
            std::vector<uint32_t> rows(20);
            for (uint32_t i = 0; i < 20; ++i) rows[i] = 100 + 2 * i;
            VJ_CHECK(p->prefetch(rows.data(), rows.size()) == 8 && p->stats().deferred == 12);
            for (uint32_t r : rows) VJ_CHECK(same_bytes(*p, r, rb));
        }
        // ---- stress: random prefetch / fetch interleaving (dedup, eviction, in-flight waits).
        {
            auto p = make(PrefetchConfig{256, 32});
            std::mt19937 rng(static_cast<uint32_t>(backend + 1));
            uint32_t checked = 0;
            for (int it = 0; it < 400; ++it) {
                std::vector<uint32_t> rows;
                const uint32_t base = rng() % kRows;
                for (int k = 0; k < 24; ++k) rows.push_back((base + (rng() % 3 ? k : rng() % 500)) % kRows);
                p->prefetch(rows.data(), rows.size());
                for (int k = 0; k < 6; ++k) { VJ_CHECK(same_bytes(*p, rows[rng() % rows.size()], rb)); ++checked; }
            }
            const auto& s = p->stats();
            std::printf("  %s: %u fetches checked, %llu rows read in %llu vectored reads (%.1f rows/read), %llu deduped, "
                        "%llu evictions, hit rate %.2f\n", p->backend_name(), checked, (unsigned long long)s.issued,
                        (unsigned long long)s.read_ops, double(s.issued) / std::max<uint64_t>(1, s.read_ops),
                        (unsigned long long)s.deduped, (unsigned long long)s.evictions, double(s.hits) / (s.hits + s.misses));
            VJ_CHECK(s.read_ops < s.issued && s.errors == 0);
        }
    }
    // ---- io_uring backend directly
    {
        const uint64_t fsize = kHeader + uint64_t(kRows) * rb;
        const char* env_before = std::getenv("VJ_IO_URING");
        VJ_CHECK(env_before == nullptr || std::string(env_before) != "0");
        if (have_uring) {
            IoUringOptions o;
            o.entries = 8;  // 300 ops through an 8-entry ring: backlog + CQ cap
            IoUringBackend io(o);
            VJ_CHECK(io.ok() && std::string(io.name()) == "io_uring" && io.sq_entries() == 8);
            check_opset(io, random_ops(t.fd, fsize, 300, 11, 0), rb);
            IoUringStats st = io.stats();
            VJ_CHECK(st.ops == 300 && st.max_in_flight <= 8 && st.sqes >= 302 && !st.degraded && st.inline_ops == 0);
            std::printf("  io_uring: 300 ops (one of 1500 iovecs), %llu SQEs, %llu resubmits, max in flight %u, %llu enters\n",
                        (unsigned long long)st.sqes, (unsigned long long)st.resubmits, st.max_in_flight,
                        (unsigned long long)st.enter_calls);
            // Every read split into <= 77-byte SQEs: the short-read path carries every op to completion.
            IoUringOptions sp;
            sp.entries = 16;
            sp.max_bytes_per_sqe = 77;
            IoUringBackend split(sp);
            check_opset(split, random_ops(t.fd, fsize, 120, 12, 1000), rb);
            VJ_CHECK(split.stats().resubmits > 500);
            // Error completions: bad fd, offset past EOF, empty scatter list.
            std::vector<uint8_t> b(64);
            std::vector<ReadOp> odd(3);
            odd[0].fd = -1; odd[0].iov.push_back({b.data(), b.size()}); odd[0].user_data = 1;
            odd[1].fd = t.fd; odd[1].offset = fsize + 4096; odd[1].iov.push_back({b.data(), b.size()}); odd[1].user_data = 2;
            odd[2].fd = t.fd; odd[2].user_data = 3;
            const auto r = run_all(io, odd);
            VJ_CHECK(r.at(1) == -EBADF && r.at(2) == 0 && r.at(3) == 0);
        }
        // ---- 4 submitter + 2 reaper threads sharing one ring (or its fallback)
        {
            IoUringOptions o;
            o.entries = 16;
            o.max_bytes_per_sqe = 4096;
            std::unique_ptr<IoBackend> io = make_io_backend(BackendKind::IoUring, 4, nullptr, o);
            std::vector<OpSet> sets;
            for (uint32_t th = 0; th < 4; ++th) sets.push_back(random_ops(t.fd, fsize, 250, 100 + th, uint64_t(th) << 32));
            std::vector<std::vector<ReadOp>> copies;
            for (const auto& s : sets) copies.push_back(s.ops);
            std::mutex mu;
            std::map<uint64_t, int64_t> got;
            std::atomic<bool> dup{false};
            std::vector<std::thread> ts;
            for (uint32_t th = 0; th < 4; ++th)
                ts.emplace_back([&, th] {
                    std::mt19937 rng(th);
                    auto& ops = sets[th].ops;
                    for (size_t i = 0; i < ops.size();) {
                        const size_t k = std::min<size_t>(ops.size() - i, 1 + rng() % 20);
                        std::vector<ReadOp> batch(std::make_move_iterator(ops.begin() + static_cast<std::ptrdiff_t>(i)),
                                                  std::make_move_iterator(ops.begin() + static_cast<std::ptrdiff_t>(i + k)));
                        io->submit(std::move(batch));
                        i += k;
                    }
                });
            for (int rp = 0; rp < 2; ++rp)
                ts.emplace_back([&] {
                    std::vector<Completion> c;
                    for (int spins = 0; spins < 20000; ++spins) {
                        {
                            std::lock_guard<std::mutex> lk(mu);
                            if (got.size() == 1000) return;
                        }
                        c.clear();
                        io->reap(c, 5);
                        std::lock_guard<std::mutex> lk(mu);
                        for (const auto& x : c) if (!got.emplace(x.user_data, x.result).second) dup = true;
                    }
                });
            for (auto& th : ts) th.join();
            VJ_CHECK(!dup && got.size() == 1000);
            for (uint32_t th = 0; th < 4; ++th)
                for (size_t i = 0; i < copies[th].size(); ++i)
                    VJ_CHECK(check_op(copies[th][i], got.at(copies[th][i].user_data), sets[th].want[i], rb));
            std::printf("  %s: 4 submitters + 2 reapers, 1000 ops, every completion exactly once and byte-exact\n", io->name());
        }
        // ---- destruction with reads in flight waits for every CQE (buffers outlive the backend)
        if (have_uring) {
            std::vector<std::vector<uint8_t>> big(64, std::vector<uint8_t>(fsize));
            {
                IoUringBackend io(IoUringOptions{16, 0, false, 0});
                std::vector<ReadOp> ops(64);
                for (size_t i = 0; i < ops.size(); ++i) {
                    ops[i].fd = t.fd;
                    ops[i].user_data = i;
                    ops[i].iov.push_back({big[i].data(), big[i].size()});
                }
                io.submit(std::move(ops));
            }  // ~IoUringBackend: must not return before the kernel is done with `big`
            for (size_t i = 0; i < 16; ++i) VJ_CHECK(big[i][kHeader] == row_byte(0, 0) && big[i][fsize - 1] == row_byte(kRows - 1, rb - 1));
        }
        // ---- fallbacks: setup failure, runtime io_uring_enter failure, VJ_IO_URING=0
        {
            IoUringOptions o;
            o.simulate_unavailable = true;
            std::string w;
            auto fb = make_io_backend(BackendKind::IoUring, 2, &w, o);
            VJ_CHECK(std::string(fb->name()) == "threadpool-preadv" && w.find("io_uring unavailable") != std::string::npos);
            check_opset(*fb, random_ops(t.fd, fsize, 60, 21, 0), rb);
            IoUringBackend dead(o);
            VJ_CHECK(!dead.ok(&w) && w.find("simulated") != std::string::npos);
            check_opset(dead, random_ops(t.fd, fsize, 40, 22, 0), rb);  // inline preadv
            VJ_CHECK(dead.stats().inline_ops == 40);
            if (have_uring) {
                IoUringOptions e;
                e.entries = 8;
                e.simulate_enter_errno = EPERM;  // e.g. a seccomp profile that allows setup but not enter
                IoUringBackend deg(e);
                VJ_CHECK(deg.ok());
                check_opset(deg, random_ops(t.fd, fsize, 50, 23, 0), rb);
                VJ_CHECK(deg.stats().degraded && deg.stats().inline_ops == 50 && !deg.ok(&w));
                VJ_CHECK(std::string(deg.name()) == "io_uring-degraded-preadv" && w.find("degraded") != std::string::npos);
                // Degraded backend inside a prefetcher still serves byte-exact rows.
                IoUringOptions e2 = e;
                RowPrefetcher p(t, PrefetchConfig{64, 16}, std::make_unique<IoUringBackend>(e2));
                std::vector<uint32_t> rows(40);
                for (uint32_t i = 0; i < 40; ++i) rows[i] = 700 + i * 3 / 2;
                p.prefetch(rows.data(), rows.size());
                for (uint32_t row : rows) VJ_CHECK(same_bytes(p, row, rb));
            }
            setenv("VJ_IO_URING", "0", 1);
            VJ_CHECK(std::string(make_io_backend(BackendKind::Auto, 1, &w)->name()) == "threadpool-preadv" &&
                     w.find("VJ_IO_URING=0") != std::string::npos);
            unsetenv("VJ_IO_URING");
            VJ_CHECK(std::string(make_io_backend(BackendKind::Auto, 1, &w)->name()) ==
                     (have_uring ? "io_uring" : "threadpool-preadv"));
            std::printf("  auto backend: %s\n", w.c_str());
        }
    }
    // ---- invalid configurations
    {
        TableDesc bad = t;
        bad.row_elems = 100;
        bad.fmt = RowFormat::Q4;
        std::string err;
        VJ_CHECK(!RowPrefetcher(bad, {}, std::make_unique<SyncBackend>()).ok(&err) && err.find("row_elems") != std::string::npos);
        bad = t;
        bad.fd = -1;
        VJ_CHECK(!RowPrefetcher(bad, {}, std::make_unique<SyncBackend>()).ok(&err));
        VJ_CHECK(!RowPrefetcher(t, PrefetchConfig{0, 8}, std::make_unique<SyncBackend>()).ok(&err));
        VJ_CHECK(!RowPrefetcher(t, {}, nullptr).ok(&err));
    }
    // ---- n-gram buckets: deterministic, in range, order-sensitive, roughly uniform.
    {
        const int32_t a[] = {11, 22, 33}, b[] = {22, 11, 33};
        VJ_CHECK(ngram_row(a, 3, 1, 1000003) == ngram_row(a, 3, 1, 1000003));
        VJ_CHECK(ngram_row(a, 3, 1, 1000003) != ngram_row(b, 3, 1, 1000003));
        VJ_CHECK(ngram_row(a, 3, 1, 1000003) != ngram_row(a, 3, 2, 1000003));
        VJ_CHECK(ngram_row(a, 2, 1, 1000003) != ngram_row(a, 3, 1, 1000003));
        std::vector<uint32_t> hist(64, 0);
        for (int32_t i = 0; i < 64000; ++i) {
            const int32_t g[2] = {i / 300, i % 300};
            const uint64_t r = ngram_row(g, 2, 7, 64);
            VJ_CHECK(r < 64);
            ++hist[r];
        }
        double chi = 0;
        for (uint32_t h : hist) chi += (h - 1000.0) * (h - 1000.0) / 1000.0;
        VJ_CHECK(chi < 120.0);  // 63 degrees of freedom: p(chi2 > 120) < 1e-4
        VJ_CHECK(ngram_row(a, 3, 1, 0) == 0);
    }
    std::printf("test_async_table_prefetch: formats, coalescing, LRU, EOF / range errors, exhaustion, 4 backends, "
                "stress, n-gram hash, io_uring (partial reads, threads, fallbacks) OK\nPASS\n");
    return 0;
}

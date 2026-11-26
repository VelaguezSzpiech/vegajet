# AUDIT — model-file loading paths (GGUF reader, tokenizer, chat template, repackers, mmap)

Scope: `core/gguf_reader.*`, `core/tokenizer.*`, `core/bpe_tokenizer.*`, `core/pretok_families.*`,
`core/chat_template.*`, `core/quant_repacker.*`, `core/sharder.*`, `models/ornith/ornith_loader.cpp`,
`models/ornith/ornith_gpu.cpp` (tensor ingest only). Branch `release/v1.2`.

Method: source review plus hand-crafted malformed GGUF files driven through the real parsers.
Two trees were built from the untouched worktree:

* `build-audit-parse` — `-DVEGAJET_CPU_ONLY=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo`
* `build-audit-asan`  — same plus `-fsanitize=address,undefined -fno-sanitize-recover=undefined`

Probe: `build-audit-parse/probe_parsers.cpp` (untracked, never added to CMake) emits the malicious
files and calls the production entry points (`GgufFile::open/tensor_data`, `BpeTokenizer::load_from_gguf`,
`ornith::OrnithModel::load`, `chat::Template::parse/render`). Run `probe_parsers <case>`; sanitized runs
use `build-audit-asan/probe_asan <case>`. No tracked file was modified.

Every finding below is either confirmed by a crash/sanitizer report from those runs or marked
**static-only** (not reachable in the CPU-only build).

---

## B1 — Tensor-offset bounds check overflows: pointer handed out is outside the mmap  [BLOCKER]

`core/gguf_reader.cpp:236` (check) and `:294` (use)

```cpp
        if (data_offset_ + ti.offset + nb > file_size_)          // :236
            return fail("tensor " + ti.name + " extends past end of file");
...
    return static_cast<const uint8_t*>(map_) + data_offset_ + t.offset;   // :294
```

`ti.offset` is an arbitrary `uint64` taken from the file, `data_offset_` is file-controlled through
`general.alignment`; the sum wraps, the check passes, and `tensor_data()` returns `map_ - 8`
(sum `2^64-8`) for a file that is 80 bytes long. Nothing re-validates the pointer against
`file_size_` at `:294`.

Trigger: 80-byte GGUF v3, one tensor `a` of type F32 with `ne = {2}` and `offset = 2^64-72`
(probe case `offset`). Any consumer (`ornith_loader`, `ornith_gpu`, `inspect`, `dflash_ref`) then
reads model weights at an address outside the mapping.

Evidence:

```
open=1 err='' data_offset=64 file_size=80
n_elems=2 n_bytes=8 offset=18446744073709551544 (sum=18446744073709551608)
tensor_data=0x71b0fd707ff8 -- reading 8 bytes at that address now
Segmentation fault (exit 139)
```

Fix: do the arithmetic in a form that cannot wrap, and re-check in `tensor_data()`:

```cpp
if (data_offset_ > file_size_ || ti.offset > file_size_ - data_offset_ ||
    nb > file_size_ - data_offset_ - ti.offset)
    return fail("tensor " + ti.name + " extends past end of file");
```
Store the validated size in `GgufTensorInfo` and assert `data_offset_ + offset + nb <= file_size_`
inside `tensor_data()` as well.

---

## B2 — `nextn_predict_layers > block_count` underflows `n_trunk`: heap OOB write  [BLOCKER]

`models/ornith/ornith_loader.cpp:61`, `:73`, `:77`, `:79`

```cpp
    h.n_trunk = h.n_block - h.n_nextn;                                            // :61  uint32 underflow
...
    h.recurrent.assign(h.n_block, false);                                         // :73
...
        for (uint32_t i = 0; i < h.n_trunk; ++i) h.recurrent[i] = (i + 1) % h.full_attn_interval != 0;  // :77
    }
    for (uint32_t i = 0; i < h.n_trunk; ++i) (h.recurrent[i] ? h.n_gdn : h.n_attn)++;                    // :79
```

`n_block` and `n_nextn` both come straight from the file (`need("block_count", …)`,
`get_arch_u64("nextn_predict_layers", …)`), both truncated to `uint32`. With `block_count=1` and
`nextn_predict_layers=2`, `n_trunk` becomes `0xFFFFFFFF` while `recurrent` holds one element; the
loop writes (and later reads) far past the `vector<bool>` storage. `recurrent_layers` being absent is
the common case, so the `else` branch is taken.

Trigger: GGUF with `general.architecture=qwen35moe`, `block_count=1`, `nextn_predict_layers=2` and the
`can_load` signature tensors (probe case `ornith_nextn`); reached by `vegajet bench <file>` →
`ModelRegistry::create_for` → `load()` → `parse_hparams()`.

Evidence (ASan):

```
==ERROR: AddressSanitizer: heap-buffer-overflow ... READ of size 8
    #0 std::_Bit_reference::operator=(bool)
    #1 vegajet::ornith::OrnithModel::parse_hparams(...) models/ornith/ornith_loader.cpp:77
0x502000000058 is located 0 bytes after 8-byte region  (allocated at ornith_loader.cpp:73)
```

Fix: validate hparams before any arithmetic — `if (h.n_block == 0 || h.n_block > 512) fail`,
`if (h.n_nextn > h.n_block) fail("nextn_predict_layers > block_count")`, and compute
`n_trunk` in `uint64`/`size_t`. Ideally also require `n_trunk > 0`.

---

## B3 — Metadata-driven integer division by zero (two sites)  [BLOCKER]

`models/ornith/ornith_loader.cpp:62`

```cpp
    h.head_dim_k = f.get_arch_u64("attention.key_length", u) ? static_cast<uint32_t>(u) : h.n_embd / h.n_head;
```

`models/ornith/ornith_loader.cpp:77`

```cpp
        for (uint32_t i = 0; i < h.n_trunk; ++i) h.recurrent[i] = (i + 1) % h.full_attn_interval != 0;
```

`attention.head_count = 0` and `full_attention_interval = 0` are both accepted by `need()` /
`get_arch_u64()`; the first divides `n_embd / 0`, the second `(i+1) % 0`. SIGFPE kills the process
before any `LoadReport::error` can be produced.

Trigger: same shape as B2 with `attention.head_count=0` (probe `ornith_div0`) or
`full_attention_interval=0` (probe `ornith_iv0`) plus the can-load tensors.

Evidence:

```
models/ornith/ornith_loader.cpp:62:100: runtime error: division by zero
AddressSanitizer: FPE ... SUMMARY: AddressSanitizer: FPE ornith_loader.cpp:62
exit=136   (SIGFPE, also on the plain RelWithDebInfo build)
```

Fix: after `need()` returns, range-check every numeric hparam (`n_head != 0`, `n_head_kv != 0`,
`n_embd % n_head == 0`, `full_attn_interval != 0`, `n_ff_exp != 0`, …) and fail with a clear message.
A shared `validate_hparams()` that also caps `block_count`, `n_embd`, `vocab` would cover B2/B3 and the
`uint32` truncations of `get_arch_u64`.

---

## B4 — Tokenizer loader reserves file-controlled counts; uncaught `std::length_error` aborts  [BLOCKER]

`core/tokenizer.cpp:302`, `:311`, `:319`

```cpp
            if (et == 8u) {
                tokens.reserve(static_cast<size_t>(cnt));                     // :302  cnt is uint64 from file
                for (uint64_t i = 0; i < cnt && rd.ok; ++i) tokens.push_back(rd.read_str());
...
                merges.reserve(static_cast<size_t>(cnt));                     // :311
...
            token_types.reserve(static_cast<size_t>(cnt));                    // :319
```

`cnt` is read from the file and used without comparing it to the bytes actually left (the sibling
reader `core/bpe_tokenizer.cpp:280` does exactly that check). `vector::reserve` throws `length_error`
or `bad_alloc`, neither is caught in `BpeTokenizer::load_from_gguf` (`core/tokenizer.cpp:234`), so a
40-byte file terminates the process instead of returning `err_out`.

Trigger: GGUF header + one KV `tokenizer.ggml.tokens`, type ARRAY, element type STRING, count `2^60`
(probe case `tok_reserve`). Reached by `vegajet bench` (`cli/bench.cpp:218`) and by
`ornith_gpu.cpp:841`, i.e. the model path, not only tests.

Evidence:

```
calling load_from_gguf (cnt=2^60, no payload)
terminate called after throwing an instance of 'std::length_error'
  what():  vector::reserve
exit=134  (SIGABRT)
```

Fix: `if (cnt > (rd.remaining() / 8)) return false;` before each reserve (a string element occupies at
least 8 bytes for its length prefix), cap `cnt` at a plausible vocabulary size (e.g. `1<<24`), and wrap
the whole body in `try { … } catch (const std::exception& e) { *err_out = e.what(); return false; }`.

---

## B5 — Unbounded recursion on nested GGUF arrays → stack overflow  [BLOCKER]

`core/tokenizer.cpp:129-146`

```cpp
            case 9: {
                uint32_t et = read<uint32_t>();
                uint64_t cnt = read<uint64_t>();
                if (!ok) return;
                uint64_t elem_sz = 0;
                ...
                if (elem_sz > 0) { ... } else {
                    for (uint64_t i = 0; i < cnt && ok; ++i) skip_val(et);   // :143  et == 9 recurses
                }
```

Element type `9` (ARRAY) is not in the fixed-width table, so `skip_val` calls itself with no depth
limit. Each nesting level costs 12 file bytes, so a 4.8 MB metadata section buys ~400 000 frames.

Trigger: one unknown KV whose value is ARRAY-of-ARRAY-of-… (probe case `tok_nested`, 400 000 levels).
Any unknown metadata key in an untrusted file takes this path.

Evidence:

```
==ERROR: AddressSanitizer: stack-overflow
    #0 skip_val core/tokenizer.cpp:130
    #1 skip_val core/tokenizer.cpp:143
    #2 skip_val core/tokenizer.cpp:143
    ... (hundreds of frames)
exit=141
```

Fix: add a depth parameter (`skip_val(vt, depth)` with `depth > 4 → ok = false`), as
`core/bpe_tokenizer.cpp:251` already does, or skip arrays iteratively with an explicit stack. Also
reject array-of-array element types outright: GGUF has no nested arrays in practice.

---

## M1 — `n_elems()` / `n_bytes()` overflow and "size 0 means unchecked" let impossible tensors through  [MAJOR]

`core/gguf_reader.hpp:49`, `core/ggml_types.hpp:74-77`, `core/gguf_reader.cpp:233-236`

```cpp
    uint64_t n_elems() const { return ne[0] * ne[1] * ne[2] * ne[3]; }      // hpp:49  unchecked product
...
    if (tr.block_elems == 0 || n_elems % tr.block_elems != 0) return 0;     // ggml_types.hpp:76
...
        const uint64_t nb = ti.n_bytes();
        if (nb == 0 && ggml_type_traits(ti.type).name == nullptr)           // only *unknown* types fail
            return fail(...);
        if (data_offset_ + ti.offset + nb > file_size_)
```

Two independent ways to get `nb == 0` from a known type, which the reader then treats as "nothing to
check":

* product overflow — `ne = {2^32, 2^32}` gives `n_elems == 0`;
* block misalignment — `type = Q4_0` with `ne[0] = 33` gives `ggml_row_bytes == 0`.

Both are accepted by `open()` with a payload of a few bytes. Consumers that trust `ne[]` (e.g.
`ornith_loader.cpp:251` `p.in * p.out * p.experts * 4`, `ornith_gpu.cpp:1170`
`eqs(vocab*D)`, `inspect.cpp:77-87`) then either read far past the tensor or size a buffer from a
number that does not describe the file.

Evidence:

```
$ probe_parsers ne
open=1 err=''
dims=4294967296 x 4294967296 but n_elems=0 n_bytes=0

$ probe_parsers misaligned
open=1 err='' (file holds 8 payload bytes)
Q4_0 dims=33x2 n_elems=66 n_bytes=0
```

Fix: `n_elems()` should be an overflow-checked multiply (`__builtin_mul_overflow`) and `open()` should
fail when the multiply overflows, when any `ne[i] == 0` for `i < n_dims`, when `ne[i] > 2^40`, or when
`n_bytes() == 0` for a *known* type with `n_elems > 0` (that combination is always malformed). Keep the
"0 means unknown type" convention out of the size check entirely.

---

## M2 — `tensors_.reserve(n_tensors)` allocates 1.4 GiB from a 24-byte file  [MAJOR]

`core/gguf_reader.cpp:220`

```cpp
    tensors_.reserve(n_tensors);
```

`n_tensors` is capped at `1<<24` (`:169`) but nothing checks that the file can hold that many tensor
records before reserving `1<<24 * sizeof(GgufTensorInfo)` = 1 476 395 008 bytes.

Trigger: 24-byte header with `n_tensors = 1<<24`, no KV, no tensor infos (probe case `bigtensors`).

Evidence:

```
$ probe_parsers bigtensors            # no limit: 1.4 GiB virtual, then clean error
open=0 err='...: bad tensor info header'
$ ( ulimit -v 500000; ./probe_parsers bigtensors )
terminate called after throwing an instance of 'std::bad_alloc'
Aborted (core dumped)  exit=134
```

Anything with a memory cap (container, systemd `MemoryMax`, `ulimit -v`) aborts instead of reporting
`err_`. Fix: reserve only after the count is plausible against the remaining file size
(`n_tensors > (file_size_ - r.pos()) / kMinTensorInfoBytes → fail`), or `reserve(std::min<uint64_t>(n_tensors, 4096))`
and let the vector grow (it does so anyway), and catch `bad_alloc` in `open()`.

---

## M3 — Expert tensors are consumed at a hard-coded row layout with no shape/type validation (static-only)  [MAJOR]

`models/ornith/ornith_gpu.cpp:1327-1333`, `:1383`, `:1422`

```cpp
            const auto* tig = f.tensor(p + "ffn_gate_exps.weight");
            const auto* tiu = f.tensor(p + "ffn_up_exps.weight");
            const uint8_t* rg = f.tensor_data(*tig);
            ...
            const size_t raw_rb = 12u * 82u;                 // IQ2_S row: 984 bytes
            const size_t n_rows_rank = 256ULL * local_ff;
...
                                rg + src_r0 * raw_rb, ru + src_r0 * raw_rb, raw_rb, ...
```

Only `token_embd.weight` is shape-validated (`:1155`). For the MoE tensors the loader assumes
`IQ2_S`, 984-byte rows and `256 * n_ff_exp` rows; `n_ff_exp` comes from metadata and the tensor's own
`ne[]/type` are never compared to it. `GgufFile::open` only proves the *declared* `n_bytes()` fits, so a
file that declares `ffn_*_exps.weight` as a 1 KB tensor while claiming `expert_feed_forward_length`
= 512 makes this loop read 256·512·984 ≈ 129 MB from a 1 KB region — past the tensor, past the
following tensors, and past EOF on the last block (SIGSEGV, or garbage weights).

Not exercised here: this path needs the MI50 runtime, so it is static-only. Fix: next to the
`token_embd` check, require for every `*_exps.weight` that `type == IQ2_S`, `ne[0] == 984*...`
(the exact block geometry), `ne[1] == 256 * n_ff_exp`, `ne[2] == n_expert`, and
`n_bytes() >= rows * raw_rb`; reject otherwise with the tensor name.

---

## M4 — Chat-template parser recurses without a depth limit → stack overflow on a crafted template  [MAJOR]

`core/chat_template.cpp:326-540` (`ExprParser::expr → cond → or_ → and_ → not_ → compare → concat →
additive → mult → unary → power → postfix → primary`) and `:646-674` (`StmtParser::body → parse_if →
body`). `Template::parse` (`:1360`) catches only `chat::Error`, never stack exhaustion.

```
==ERROR: AddressSanitizer: stack-overflow
    #0 primary core/chat_template.cpp:472
    #1 postfix :433   #2 power :413   #3 unary :410 ... #13 parse_full :322
exit=141                       (400 KB template of 200 000 '(' … ')')
```

and for 100 000 nested `{% if 1 %}…{% endif %}` (1.7 MB):

```
    #14 parse_if core/chat_template.cpp:687
    #15 body     core/chat_template.cpp:674
    #16 parse_if core/chat_template.cpp:690
exit=141
```

Trigger: the `tokenizer.chat_template` string inside the GGUF (probe cases `tpl_parens`, `tpl_if`).
Reachability today: `vegajet serve` only calls `bpe::detect_chat_format(vocab.chat_template, …)`
(`cli/serve_cmd.cpp:176`) and renders with `bpe::apply_chat_template`, so the production server does
not hit this; every `chat::Template` caller (currently the tests, and any tool that renders the file's
template) does. Severity stays MAJOR because the input is untrusted file content.

Fix: thread a `depth` counter through `ExprParser` (`fail` above e.g. 64) and through
`StmtParser::body` (nesting limit, e.g. 32), matching the `kMaxDepth = 64` already used for macro
calls at `:1309`.

---

## M5 — Jinja `for` loops have no iteration budget: a 90-byte template runs for hours  [MAJOR]

`core/chat_template.cpp:856-863`

```cpp
        const int64_t len = static_cast<int64_t>(items.size());
        for (int64_t k = 0; k < len; ++k) {
            scopes_.emplace_back();
            ...
            render(n->branches[0].second);
            scopes_.pop_back();
        }
```

`range()` is capped at 100 000 elements per call (`:1294`), but nested loops multiply, and the only
global guard is the 32 MiB output cap (`:826`), which a body that writes nothing never reaches.

Trigger: `{% for a in range(100000) %}{% for b in range(100000) %}{% set x = 1 %}{% endfor %}{% endfor %}`
(90 bytes of template). Measured rate on this host: 2 000×2 000 = 4·10⁶ iterations in 3.13 s
(≈1.3 M iter/s) ⇒ 10¹⁰ iterations ≈ 2.2 hours of one core, with memory flat. `timeout 10` on the
10¹⁰ case does not return (`exit=124`).

Fix: a per-render loop-iteration budget (e.g. 10⁶ total, not per call) decremented in `render_for`,
failing via `fail("template loop budget exceeded")`, plus a nesting limit for `render()`.

---

## m1 — Signed integer overflow / out-of-range float→int in template arithmetic (UB, silent wrong output)  [MINOR]

`core/chat_template.cpp:983-986`

```cpp
        if (op == "+") return ints ? V::integer(static_cast<int64_t>(a) + static_cast<int64_t>(b)) : V::real(a + b);
        if (op == "*") return ints ? V::integer(static_cast<int64_t>(a) * static_cast<int64_t>(b)) : V::real(a * b);
        if (op == "**") return ints && b >= 0 ? V::integer(static_cast<int64_t>(std::pow(a, b))) : V::real(std::pow(a, b));
```

Values are routed through `double` and cast back, so `{{ 9223372036854775807 + 1 }}` prints
`-9223372036854775807`, `{{ 10 ** 100 }}` prints `-9223372036854775808`, and `*` overflows a signed
`int64` (UBSan at `:985`). Trigger: the template string in the file (probe `tpl_ub`). Impact is wrong
prompt text, not memory unsafety.

Fix: keep integer operands as `int64` and use `__builtin_add_overflow/sub/mul_overflow`; for `**`
clamp the exponent (e.g. `b > 63 → fail`) and range-check before the cast.

---

## m2 — Array skip arithmetic overflows in the GGUF KV walker  [MINOR]

`core/gguf_reader.cpp:205`

```cpp
                if (!r.skip((v.arr_len - keep) * es)) return fail("truncated array " + key);
```

`arr_len` is a file-controlled `uint64`, `es` up to 8; the product wraps, so `skip()` can be handed a
tiny value (e.g. `arr_len = 2^61 + 4096, es = 8` → 0) and the following KV entries are parsed from the
middle of the array payload. `PReader::skip` is bounds-checked, so this is a misparse (wrong metadata,
usually a later "truncated" error), not an OOB read.

Fix: `if (v.arr_len - keep > (file_size_ - r.pos()) / es) return fail(...)` before multiplying.

---

## m3 — Lazy mmap is not synchronized; two loader threads can race or leak a mapping  [MINOR]

`core/gguf_reader.cpp:287-295`

```cpp
const uint8_t* GgufFile::tensor_data(const GgufTensorInfo& t) {
    if (fd_ < 0) return nullptr;
    if (!map_) {
        void* m = ::mmap(...);
        ...
        map_ = m;
    }
    return static_cast<const uint8_t*>(map_) + data_offset_ + t.offset;
```

`map_` is read and written without a lock: concurrent first calls are a data race (UB) and the loser's
mapping is never `munmap`ed. Today the loaders map single-threaded, but `tensor_data` is a public
method and the header advertises lazy mapping.

Fix: map once in `open()` (or guard with `std::once_flag`/mutex), and make `close()` idempotent.

---

## m4 — `release()` trusts the caller's pointer/length  [MINOR]

`core/gguf_reader.cpp:297-303`

```cpp
    const uintptr_t hi = reinterpret_cast<uintptr_t>(ptr) + bytes;
    ::madvise(reinterpret_cast<void*>(lo), hi - lo, MADV_DONTNEED);
```

`ptr + bytes` can overflow; a stale/foreign pointer turns this into `madvise` over an arbitrary range
(harmless `ENOMEM` today, but it also silently accepts pointers that never came from `tensor_data`).
Fix: `if (ptr < map_ || ptr >= map_+file_size_) return;` and compute `hi` with the same overflow-checked
form as in B1.

---

## m5 — Tokenizer vocab read: 4× memory amplification on a sparse/huge file  [MINOR]

`core/bpe_tokenizer.cpp:280-282`

```cpp
            if (!s.pod(et) || !s.pod(n) || et != 8 || n > s.remaining() / 8) return false;
            out.resize(n);
```

The `remaining()/8` bound is correct for reaching EOF safely, but `n` strings of 32 bytes each are
allocated up front, i.e. 4× the file size. A 1 GiB sparse GGUF (zero-filled) is enough to make a 4 GiB
allocation and get the process OOM-killed rather than told "malformed". Fix: cap `n` at a plausible
vocabulary size (e.g. `1<<22`) and/or grow the vector while reading instead of `resize`.

---

## Checked and found sound

* `PReader::read/skip/str` (`core/gguf_reader.cpp:23-56`) — every access is bounded by
  `size_ - min(pos_, size_)`; the buffer refill clamps `want` to the remaining file bytes.
* `GgufReader` in `core/tokenizer.cpp:70-150` — all reads bounds-checked; the array fast path
  validates `cnt` against `end - p`.
* `core/bpe_tokenizer.cpp` KV walker — `skip_value` already depth-limits nested arrays
  (`depth < 4`, `:251`) and `str_array` bounds `n` by `remaining()/8` (`:280`).
* Truncated-header and bad-magic rejection (`tests/test_gguf_reader.cpp:59-66`) still pass in both trees.
* `quant_repacker.cpp` decoders — fixed 32/64/256-element blocks, grid indices bounded by the static
  tables, `n % block_elems` handled by the caller; the `VJ_CHECK` in `repack_to_vj_iq` prevents
  misaligned rows. The repacker trusts its `(src, rows, cols)` contract, so its safety depends on M1/M3.
* `core/sharder.cpp` `apportion/to_ranges` — `min(pos, limit)` clamps every range; `k % n` guarded by
  the `n == 0` early return.

---

## BLOCKERS + MAJORS

| # | Sev | Location | One-line |
|---|-----|----------|----------|
| B1 | BLOCKER | `core/gguf_reader.cpp:236`, `:294` | tensor offset check overflows `uint64`; `tensor_data()` returns a pointer outside the mmap → SIGSEGV |
| B2 | BLOCKER | `models/ornith/ornith_loader.cpp:61,73,77,79` | `nextn_predict_layers > block_count` underflows `n_trunk`; heap OOB write on `vector<bool>` (ASan) |
| B3 | BLOCKER | `models/ornith/ornith_loader.cpp:62`, `:77` | `head_count == 0` / `full_attention_interval == 0` divide by zero → SIGFPE |
| B4 | BLOCKER | `core/tokenizer.cpp:302,311,319` | `reserve()` of a file-declared count → uncaught `length_error`/`bad_alloc` → SIGABRT |
| B5 | BLOCKER | `core/tokenizer.cpp:143` | unbounded recursion on nested GGUF arrays → stack overflow |
| M1 | MAJOR | `core/gguf_reader.hpp:49`, `ggml_types.hpp:76`, `gguf_reader.cpp:233` | `n_elems`/`n_bytes` overflow and "0 = unvalidated" accept impossible tensors |
| M2 | MAJOR | `core/gguf_reader.cpp:220` | `reserve(1<<24)` = 1.4 GiB from a 24-byte file → `bad_alloc` abort under a memory cap |
| M3 | MAJOR | `models/ornith/ornith_gpu.cpp:1331-1333`, `:1383`, `:1422` | `*_exps.weight` read at a hard-coded 984-byte row layout with no shape/type check → OOB read past the tensor (static-only) |
| M4 | MAJOR | `core/chat_template.cpp:326-540`, `646-674` | unbounded parser recursion → stack overflow on a crafted `tokenizer.chat_template` |
| M5 | MAJOR | `core/chat_template.cpp:856-863` | no loop-iteration budget: nested `range()` loops run ~2.2 h from a 90-byte template |

Minor: m1 template arithmetic UB (`chat_template.cpp:983-986`), m2 array-skip overflow
(`gguf_reader.cpp:205`), m3 unsynchronized lazy mmap (`gguf_reader.cpp:287`), m4 unchecked
`release()` (`gguf_reader.cpp:297`), m5 vocab memory amplification (`bpe_tokenizer.cpp:280`).

### Suggested order of work

1. **B1 + M1** in `gguf_reader.cpp` — one commit: overflow-safe geometry (`n_elems`, `data_offset_`,
   `offset + nb`), reject `n_bytes()==0` for known types, re-validate in `tensor_data()`.
2. **B3 + B2** — a `validate_hparams()` gate right after `need()` in `parse_hparams` (zero/underflow
   guards, caps), which also protects the rest of `load()`.
3. **B4 + B5** — bound `cnt` before `reserve`, catch `std::exception`, depth-limit `skip_val`
   (or drop nested-array support in this reader, as `gguf_reader.cpp:197` already does).
4. **M2** — plausible-count check before `reserve`.
5. **M3** — per-tensor shape/type validation for the expert tensors next to the `token_embd` check.
6. **M4 + M5 + m1** — parser/render budgets in the Jinja engine.

---

## Appendix — reproduction

```
# trees (from the untouched worktree on release/v1.2)
cmake -S . -B build-audit-parse -DVEGAJET_CPU_ONLY=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-audit-parse -j2
cmake -S . -B build-audit-asan  -DVEGAJET_CPU_ONLY=ON -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g -O1" \
      -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
cmake --build build-audit-asan -j2 --target vegajet_core vegajet_models

# probe (untracked; not part of CMakeLists)
g++ -std=c++17 -O1 -g -I. build-audit-parse/probe_parsers.cpp -o build-audit-parse/probe_parsers \
    build-audit-parse/libvegajet_models.a build-audit-parse/libvegajet_core.a -lgomp -lpthread
./build-audit-parse/probe_parsers {offset|ne|misaligned|bigtensors|tok_reserve|tok_nested|
                                   ornith_div0|ornith_iv0|ornith_nextn|tpl_parens|tpl_if|tpl_ub|tpl_hang [n]}

# sanitized
./build-audit-asan/probe_asan ornith_nextn     # heap-buffer-overflow, ornith_loader.cpp:77
./build-audit-asan/probe_asan tok_nested       # stack-overflow, tokenizer.cpp:143
./build-audit-asan/probe_asan tpl_parens       # stack-overflow, chat_template.cpp:472
```

Malformed files are written to `/home/hermes/.hermes/cache/scratch/probe/`.
Baseline suite on the same tree (`ctest -R "gguf|bpe|tokenizer|quant|template|grammar"`):
11/11 pass, so none of the above is covered by an existing test.

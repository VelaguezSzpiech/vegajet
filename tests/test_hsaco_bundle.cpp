// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
// CPU-only qualification tests, synthesized in memory (no GPU, no ROCm, no code-object toolchain):
//  * core/hsaco_bundle: SHA-256 vectors, fail-closed manifest parsing/path sanitization, mirror digest+size
//    verification against a temp bundle root, strict AMDGPU ELF note (MessagePack) parsing and the gfx906
//    spill/VGPR/LDS/wavefront/target gates including malformed, truncated and conflicting metadata;
//  * core/gguf_arch_detect: the four GGUF families (dense GQA, pure MoE, hybrid GDN+MoE, MLA+MoE) from keys
//    and tensor patterns, plus negative cases that must be rejected rather than guessed.
#include "core/gguf_arch_detect.hpp"
#include "core/hsaco_bundle.hpp"
#include "tests/test_common.hpp"

#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <string>
#include <vector>

using namespace vegajet;

namespace {

struct MpW {
    std::vector<uint8_t> b;
    void map(size_t n) {
        if (n < 16) {
            b.push_back(static_cast<uint8_t>(0x80u | n));
        } else {
            b.push_back(0xde);
            b.push_back(static_cast<uint8_t>(n >> 8));
            b.push_back(static_cast<uint8_t>(n));
        }
    }
    void arr(size_t n) {
        if (n < 16) {
            b.push_back(static_cast<uint8_t>(0x90u | n));
        } else {
            b.push_back(0xdc);
            b.push_back(static_cast<uint8_t>(n >> 8));
            b.push_back(static_cast<uint8_t>(n));
        }
    }
    void str(const std::string& s) {
        if (s.size() < 32) {
            b.push_back(static_cast<uint8_t>(0xa0u | s.size()));
        } else if (s.size() < 256) {
            b.push_back(0xd9);
            b.push_back(static_cast<uint8_t>(s.size()));
        } else {
            b.push_back(0xda);
            b.push_back(static_cast<uint8_t>(s.size() >> 8));
            b.push_back(static_cast<uint8_t>(s.size()));
        }
        b.insert(b.end(), s.begin(), s.end());
    }
    void u(uint64_t v) {
        if (v < 128) {
            b.push_back(static_cast<uint8_t>(v));
        } else if (v < 256) {
            b.push_back(0xcc);
            b.push_back(static_cast<uint8_t>(v));
        } else if (v < 65536) {
            b.push_back(0xcd);
            b.push_back(static_cast<uint8_t>(v >> 8));
            b.push_back(static_cast<uint8_t>(v));
        } else if (v <= 0xffffffffull) {
            b.push_back(0xce);
            for (int s = 24; s >= 0; s -= 8) b.push_back(static_cast<uint8_t>(v >> s));
        } else {
            b.push_back(0xcf);
            for (int s = 56; s >= 0; s -= 8) b.push_back(static_cast<uint8_t>(v >> s));
        }
    }
};

struct KernSpec {
    std::string name = "vj_kernel";
    bool emit_name = true;
    bool emit_symbol = true;
    bool emit_vgpr = true;
    bool duplicate_vgpr = false;
    std::string symbol{};  // default: name + ".kd"
    uint64_t vgpr = 32, sgpr = 24, lds = 0, priv = 0, kernarg = 272, wavefront = 64, max_wg = 256;
};

std::vector<uint8_t> metadata(const std::vector<KernSpec>& kernels,
                              const std::string& target = "amdgcn-amd-amdhsa--gfx906", bool extra_key = true,
                              bool trailing_garbage = false) {
    MpW m;
    m.map(extra_key ? 3 : 2);
    m.str("amdhsa.target");
    m.str(target);
    m.str("amdhsa.kernels");
    m.arr(kernels.size());
    for (const auto& k : kernels) {
        const std::string symbol = k.symbol.empty() ? k.name + ".kd" : k.symbol;
        size_t fields = 6;  // sgpr, group, private, kernarg, wavefront, max_flat_workgroup_size
        if (k.emit_name) ++fields;
        if (k.emit_symbol) ++fields;
        if (k.emit_vgpr) fields += k.duplicate_vgpr ? 2 : 1;
        m.map(fields);
        if (k.emit_name) {
            m.str(".name");
            m.str(k.name);
        }
        if (k.emit_symbol) {
            m.str(".symbol");
            m.str(symbol);
        }
        if (k.emit_vgpr) {
            m.str(".vgpr_count");
            m.u(k.vgpr);
            if (k.duplicate_vgpr) {
                m.str(".vgpr_count");
                m.u(k.vgpr);
            }
        }
        m.str(".sgpr_count");
        m.u(k.sgpr);
        m.str(".group_segment_fixed_size");
        m.u(k.lds);
        m.str(".private_segment_fixed_size");
        m.u(k.priv);
        m.str(".kernarg_segment_size");
        m.u(k.kernarg);
        m.str(".wavefront_size");
        m.u(k.wavefront);
        m.str(".max_flat_workgroup_size");
        m.u(k.max_wg);
    }
    if (extra_key) {
        m.str("amdhsa.version");
        m.arr(2);
        m.u(1);
        m.u(2);
    }
    if (trailing_garbage) m.b.push_back(0xc0);  // nil after the top-level map
    return m.b;
}

std::vector<uint8_t> make_note(const std::vector<uint8_t>& desc, uint32_t type = 32,
                               const std::string& owner = "AMDGPU") {
    std::vector<uint8_t> note;
    auto u32 = [&](uint32_t v) {
        for (int i = 0; i < 4; ++i) note.push_back(static_cast<uint8_t>(v >> (8 * i)));
    };
    const uint32_t namesz = static_cast<uint32_t>(owner.size() + 1);
    u32(namesz);
    u32(static_cast<uint32_t>(desc.size()));
    u32(type);
    note.insert(note.end(), owner.begin(), owner.end());
    note.push_back(0);
    while (note.size() % 4) note.push_back(0);
    note.insert(note.end(), desc.begin(), desc.end());
    while (note.size() % 4) note.push_back(0);
    return note;
}

struct BuiltElf {
    std::vector<uint8_t> bytes;
    std::vector<uint64_t> note_offsets;
};

BuiltElf make_elf(const std::vector<std::vector<uint8_t>>& notes, uint16_t machine = 224,
                  uint32_t eflags = 0x2fu) {
    BuiltElf elf;
    std::vector<uint8_t>& out = elf.bytes;
    out.assign(64, 0);
    const uint8_t ident[16] = {0x7f, 'E', 'L', 'F', 2, 1, 1};
    std::memcpy(out.data(), ident, sizeof(ident));
    auto w16 = [&](size_t at, uint16_t v) { std::memcpy(out.data() + at, &v, 2); };
    auto w32 = [&](size_t at, uint32_t v) { std::memcpy(out.data() + at, &v, 4); };
    auto w64 = [&](size_t at, uint64_t v) { std::memcpy(out.data() + at, &v, 8); };
    w16(16, 3);  // ET_DYN
    w16(18, machine);
    w32(20, 1);
    w32(48, eflags);
    w16(52, 64);
    w16(58, 64);
    w16(62, 0);
    for (const auto& note : notes) {
        while (out.size() % 4) out.push_back(0);
        elf.note_offsets.push_back(out.size());
        out.insert(out.end(), note.begin(), note.end());
    }
    while (out.size() % 8) out.push_back(0);
    const uint64_t shoff = out.size();
    const uint64_t shnum = 1 + notes.size();
    std::vector<uint8_t> sh(static_cast<size_t>(shnum) * 64, 0);
    for (size_t i = 0; i < notes.size(); ++i) {
        uint8_t* entry = sh.data() + (i + 1) * 64;
        const uint32_t type = 7;  // SHT_NOTE
        std::memcpy(entry + 4, &type, 4);
        const uint64_t off = elf.note_offsets[i];
        const uint64_t len = notes[i].size();
        std::memcpy(entry + 24, &off, 8);
        std::memcpy(entry + 32, &len, 8);
    }
    out.insert(out.end(), sh.begin(), sh.end());
    w64(40, shoff);
    w16(58, 64);
    w16(60, static_cast<uint16_t>(shnum));
    return elf;
}

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

bool any_problem_contains(const std::vector<std::string>& problems, const std::string& needle) {
    for (const auto& p : problems)
        if (contains(p, needle)) return true;
    return false;
}

std::string sha_hex(const std::string& s) {
    return bundle::to_hex(bundle::sha256(reinterpret_cast<const uint8_t*>(s.data()), s.size()));
}

std::string base_name(const std::string& path) {
    const size_t slash = path.rfind('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

void add_tensor(vjtest::GgufWriter& w, const std::string& name) {
    w.tensor(name, {1}, 0, {0, 0, 0, 0});
}

void add_tensors(vjtest::GgufWriter& w, const std::string& prefix, std::initializer_list<const char*> suffixes) {
    for (const char* suffix : suffixes) add_tensor(w, prefix + suffix);
}

std::vector<uint8_t> test_elf_ok() {
    const std::vector<KernSpec> kernels = {KernSpec{"vj_gemv_a", true, true, true, false, "", 32, 24, 4096, 0, 272, 64, 256},
                                           KernSpec{"vj_gemv_b", true, true, true, false, "", 64, 24, 65536, 0, 272, 64, 256}};
    return make_elf({make_note(metadata(kernels))}).bytes;
}

// ---- synthetic GGUFs
std::vector<uint8_t> gguf_dense_gqa() {
    vjtest::GgufWriter w;
    w.kv_str("general.architecture", "densegqa");
    w.kv_u32("densegqa.block_count", 2);
    w.kv_u32("densegqa.attention.head_count", 32);
    w.kv_u32("densegqa.attention.head_count_kv", 8);
    w.kv_u32("densegqa.feed_forward_length", 64);
    add_tensor(w, "token_embd.weight");
    for (int L = 0; L < 2; ++L) {
        const std::string p = "blk." + std::to_string(L) + ".";
        add_tensors(w, p, {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                           "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"});
    }
    return w.build();
}

std::vector<uint8_t> gguf_pure_moe() {
    vjtest::GgufWriter w;
    w.kv_str("general.architecture", "moeonly");
    w.kv_u32("moeonly.block_count", 2);
    w.kv_u32("moeonly.attention.head_count", 32);
    w.kv_u32("moeonly.attention.head_count_kv", 8);
    w.kv_u32("moeonly.expert_count", 8);
    w.kv_u32("moeonly.expert_used_count", 2);
    w.kv_u32("moeonly.expert_feed_forward_length", 32);
    add_tensor(w, "token_embd.weight");
    for (int L = 0; L < 2; ++L) {
        const std::string p = "blk." + std::to_string(L) + ".";
        add_tensors(w, p, {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                           "ffn_gate_inp.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"});
    }
    return w.build();
}

std::vector<uint8_t> gguf_hybrid_gdn_moe() {
    vjtest::GgufWriter w;
    w.kv_str("general.architecture", "hybrid");
    w.kv_u32("hybrid.block_count", 4);
    w.kv_u32("hybrid.attention.head_count", 32);
    w.kv_u32("hybrid.attention.head_count_kv", 8);
    w.kv_u32("hybrid.ssm.group_count", 4);
    w.kv_u32("hybrid.ssm.time_step_rank", 8);
    w.kv_u32("hybrid.ssm.state_size", 16);
    w.kv_u32("hybrid.ssm.inner_size", 64);
    w.kv_u32("hybrid.ssm.conv_kernel", 4);
    w.kv_u32("hybrid.expert_count", 8);
    w.kv_u32("hybrid.expert_used_count", 2);
    w.kv_u32("hybrid.expert_feed_forward_length", 32);
    w.kv_arr_i32("hybrid.attention.recurrent_layers", {1, 1, 1, 0});
    add_tensor(w, "token_embd.weight");
    const std::vector<std::string> moe = {"ffn_gate_inp.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight",
                                          "ffn_down_exps.weight"};
    for (int L = 0; L < 4; ++L) {
        const std::string p = "blk." + std::to_string(L) + ".";
        if (L < 3) {
            add_tensors(w, p, {"attn_qkv.weight", "attn_gate.weight", "ssm_out.weight", "ssm_a", "ssm_conv1d.weight"});
        } else {
            add_tensors(w, p, {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"});
        }
        for (const auto& name : moe) add_tensor(w, p + name);
    }
    return w.build();
}

std::vector<uint8_t> gguf_mla_moe() {
    vjtest::GgufWriter w;
    w.kv_str("general.architecture", "mlamoe");
    w.kv_u32("mlamoe.block_count", 2);
    w.kv_u32("mlamoe.attention.head_count", 32);
    w.kv_u32("mlamoe.attention.head_count_kv", 8);
    w.kv_u32("mlamoe.attention.kv_lora_rank", 64);
    w.kv_u32("mlamoe.rope.dimension_count", 16);
    w.kv_u32("mlamoe.attention.key_length", 32);
    w.kv_u32("mlamoe.attention.value_length", 16);
    w.kv_u32("mlamoe.attention.sliding_window", 128);
    w.kv_u32("mlamoe.expert_count", 8);
    w.kv_u32("mlamoe.expert_used_count", 2);
    w.kv_u32("mlamoe.expert_feed_forward_length", 32);
    w.kv_u32("mlamoe.moe_latent_size", 32);
    w.kv_arr_i32("mlamoe.attention.sliding_window_pattern", {0, 1});
    add_tensor(w, "token_embd.weight");
    const std::vector<std::string> moe = {"ffn_gate_inp.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight",
                                          "ffn_down_exps.weight", "ffn_latent_down.weight", "ffn_latent_up.weight"};
    add_tensors(w, "blk.0.", {"attn_q.weight", "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight", "attn_kv_b.weight",
                              "attn_output.weight"});
    add_tensors(w, "blk.1.", {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"});
    for (int L = 0; L < 2; ++L) {
        const std::string p = "blk." + std::to_string(L) + ".";
        for (const auto& name : moe) add_tensor(w, p + name);
    }
    return w.build();
}

arch::Detection detect_bytes(const std::vector<uint8_t>& bytes, const std::string& tag) {
    GgufFile f;
    VJ_CHECK(f.open(vjtest::write_temp(bytes, tag)));
    arch::Detection d;
    std::string err;
    const bool ok = arch::detect(f, d, &err);
    if (!ok) VJ_CHECK(!err.empty() && !d.ok() && !d.problems.empty());
    return d;
}

void expect_reject(const std::vector<uint8_t>& bytes, const std::string& tag) {
    GgufFile f;
    VJ_CHECK(f.open(vjtest::write_temp(bytes, tag)));
    arch::Detection d;
    std::string err;
    VJ_CHECK(!arch::detect(f, d, &err));
    VJ_CHECK(d.family == arch::Family::Unknown && !d.ok() && !d.problems.empty() && !err.empty());
}

} // namespace

int main() {
    // ---- SHA-256 known-answer vectors -------------------------------------------------------------
    VJ_CHECK(sha_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    VJ_CHECK(sha_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    VJ_CHECK(sha_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
             "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    const std::string million(1000000, 'a');
    VJ_CHECK(sha_hex(million) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
    bundle::Digest round{};
    VJ_CHECK(bundle::digest_from_hex("BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD", round));
    VJ_CHECK(bundle::to_hex(round) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    VJ_CHECK(!bundle::digest_from_hex("abc", round));

    // ---- manifest parsing and path sanitization ----------------------------------------------------
    const std::string hex = sha_hex("payload");
    bundle::Manifest manifest;
    std::string err;
    VJ_CHECK(bundle::parse_manifest("# comment\r\n\r\n" + hex + " 7 kernels/a.hsaco\r\n", manifest, &err));
    VJ_CHECK(manifest.entries.size() == 1 && manifest.entries[0].path == "kernels/a.hsaco" &&
             manifest.entries[0].size == 7 && manifest.entries[0].digest_hex == hex);
    VJ_CHECK(!bundle::parse_manifest(hex + " 7 /abs/a.hsaco\n", manifest, &err));
    VJ_CHECK(!bundle::parse_manifest(hex + " 7 ../a.hsaco\n", manifest, &err));
    VJ_CHECK(!bundle::parse_manifest(hex + " 7 a/./b.hsaco\n", manifest, &err));
    VJ_CHECK(!bundle::parse_manifest(hex + " 7 a//b.hsaco\n", manifest, &err));
    VJ_CHECK(!bundle::parse_manifest(hex + " 7 a\\b.hsaco\n", manifest, &err));
    VJ_CHECK(!bundle::parse_manifest(hex + " x a.hsaco\n", manifest, &err));
    VJ_CHECK(!bundle::parse_manifest(hex + " 7 a.hsaco extra\n", manifest, &err));
    VJ_CHECK(!bundle::parse_manifest(hex + " 7 a.hsaco\n" + hex + " 7 a.hsaco\n", manifest, &err));
    VJ_CHECK(!bundle::parse_manifest("not-a-digest 7 a.hsaco\n", manifest, &err));
    VJ_CHECK(!bundle::parse_manifest("\n# only comments\n", manifest, &err));
    std::string sanitized;
    std::string why;
    VJ_CHECK(bundle::sanitize_relative_path("a/b/c.hsaco", sanitized, &why) && sanitized == "a/b/c.hsaco");
    VJ_CHECK(!bundle::sanitize_relative_path("a b.hsaco", sanitized, &why));

    // ---- positive code object ----------------------------------------------------------------------
    {
        const std::vector<KernSpec> kernels = {
            KernSpec{"vj_gemv_a", true, true, true, false, "", 32, 24, 4096, 0, 272, 64, 256},
            KernSpec{"vj_gemv_b", true, true, true, false, "", 64, 24, 65536, 0, 272, 64, 256}};
        const auto elf = make_elf({make_note(metadata(kernels))});
        const auto rep = bundle::inspect_code_object(elf.bytes.data(), elf.bytes.size());
        VJ_CHECK(rep.ok && rep.target == "amdgcn-amd-amdhsa--gfx906" && rep.kernels.size() == 2);
        VJ_CHECK(rep.kernels[0].name == "vj_gemv_a" && rep.kernels[0].vgpr_count == 32 &&
                 rep.kernels[0].group_segment_fixed_size == 4096 && rep.kernels[0].private_segment_fixed_size == 0);
        VJ_CHECK(rep.kernels[1].within_budget && rep.kernels[1].group_segment_fixed_size == 65536);
        // gfx906 target ids may carry feature suffixes; .symbol alone is also accepted.
        KernSpec symbol_only{"vj_symbol_only"};
        symbol_only.emit_name = false;
        const auto elf_sym = make_elf(
            {make_note(metadata({symbol_only}, "amdgcn-amd-amdhsa--gfx906:xnack-:sramecc+"))});
        const auto rep_sym = bundle::inspect_code_object(elf_sym.bytes.data(), elf_sym.bytes.size());
        VJ_CHECK(rep_sym.ok && rep_sym.kernels.size() == 1 && rep_sym.kernels[0].name == "vj_symbol_only");
    }

    // ---- gfx906 budget gates -----------------------------------------------------------------------
    {
        KernSpec spill{"vj_spill"};
        spill.priv = 16;
        auto elf = make_elf({make_note(metadata({spill}))});
        auto rep = bundle::inspect_code_object(elf.bytes.data(), elf.bytes.size());
        VJ_CHECK(!rep.ok && rep.error.empty() && any_problem_contains(rep.problems, "private_segment_fixed_size=16"));

        KernSpec vgpr{"vj_vgpr"};
        vgpr.vgpr = 65;
        elf = make_elf({make_note(metadata({vgpr}))});
        rep = bundle::inspect_code_object(elf.bytes.data(), elf.bytes.size());
        VJ_CHECK(!rep.ok && any_problem_contains(rep.problems, "vgpr_count=65"));

        KernSpec lds{"vj_lds"};
        lds.lds = 65537;
        elf = make_elf({make_note(metadata({lds}))});
        rep = bundle::inspect_code_object(elf.bytes.data(), elf.bytes.size());
        VJ_CHECK(!rep.ok && any_problem_contains(rep.problems, "LDS budget"));

        KernSpec wave{"vj_wave"};
        wave.wavefront = 32;
        elf = make_elf({make_note(metadata({wave}))});
        rep = bundle::inspect_code_object(elf.bytes.data(), elf.bytes.size());
        VJ_CHECK(!rep.ok && any_problem_contains(rep.problems, "wavefront_size=32"));

        // A tightened budget is honoured (e.g. the 48-VGPR stream class).
        bundle::Gfx906Budget stream;
        stream.max_vgpr = 48;
        KernSpec sixtyfour{"vj_64"};
        sixtyfour.vgpr = 64;
        elf = make_elf({make_note(metadata({sixtyfour}))});
        rep = bundle::inspect_code_object(elf.bytes.data(), elf.bytes.size(), stream);
        VJ_CHECK(!rep.ok && any_problem_contains(rep.problems, "budget 48"));
    }

    // ---- malformed / truncated / conflicting ELF and metadata --------------------------------------
    {
        const uint8_t junk[80] = {1, 2, 3, 4};
        VJ_CHECK(!bundle::inspect_code_object(junk, sizeof(junk)).ok);
        VJ_CHECK(!bundle::inspect_code_object(nullptr, 0).ok);

        auto elf = make_elf({make_note(metadata({KernSpec{}}))});
        auto truncated = elf;
        truncated.bytes.resize(truncated.bytes.size() / 2);
        VJ_CHECK(!bundle::inspect_code_object(truncated.bytes.data(), truncated.bytes.size()).ok);

        elf = make_elf({make_note(metadata({KernSpec{}}))}, 62);
        VJ_CHECK(!bundle::inspect_code_object(elf.bytes.data(), elf.bytes.size()).ok);

        elf = make_elf({make_note(metadata({KernSpec{}}))}, 224, 0x30u);
        VJ_CHECK(!bundle::inspect_code_object(elf.bytes.data(), elf.bytes.size()).ok);

        elf = make_elf({make_note(metadata({KernSpec{}}, "amdgcn-amd-amdhsa--gfx908"))});
        VJ_CHECK(!bundle::inspect_code_object(elf.bytes.data(), elf.bytes.size()).ok);

        auto meta = metadata({KernSpec{}});
        meta.resize(meta.size() / 2);
        elf = make_elf({make_note(meta)});
        VJ_CHECK(!bundle::inspect_code_object(elf.bytes.data(), elf.bytes.size()).ok);

        // Note descriptor size pointing past the note section.
        elf = make_elf({make_note(metadata({KernSpec{}}))});
        const uint64_t note_off = elf.note_offsets[0];
        const uint32_t bogus = 0xffffffu;
        std::memcpy(elf.bytes.data() + note_off + 4, &bogus, 4);
        VJ_CHECK(!bundle::inspect_code_object(elf.bytes.data(), elf.bytes.size()).ok);

        // Duplicate metadata note.
        const auto note = make_note(metadata({KernSpec{}}));
        elf = make_elf({note, note});
        VJ_CHECK(!bundle::inspect_code_object(elf.bytes.data(), elf.bytes.size()).ok);

        // Wrong note owner name for NT_AMDGPU_METADATA.
        elf = make_elf({make_note(metadata({KernSpec{}}), 32, "AMDGPX")});
        VJ_CHECK(!bundle::inspect_code_object(elf.bytes.data(), elf.bytes.size()).ok);

        // Duplicate kernel names.
        const KernSpec dup{"vj_dup"};
        elf = make_elf({make_note(metadata({dup, dup}))});
        VJ_CHECK(!bundle::inspect_code_object(elf.bytes.data(), elf.bytes.size()).ok);

        // .name/.symbol disagreement.
        KernSpec mismatched{"vj_name"};
        mismatched.symbol = "vj_other.kd";
        elf = make_elf({make_note(metadata({mismatched}))});
        VJ_CHECK(!bundle::inspect_code_object(elf.bytes.data(), elf.bytes.size()).ok);

        // Missing required field and duplicate field.
        KernSpec no_vgpr{"vj_no_vgpr"};
        no_vgpr.emit_vgpr = false;
        elf = make_elf({make_note(metadata({no_vgpr}))});
        VJ_CHECK(!bundle::inspect_code_object(elf.bytes.data(), elf.bytes.size()).ok);
        KernSpec dup_field{"vj_dup_field"};
        dup_field.duplicate_vgpr = true;
        elf = make_elf({make_note(metadata({dup_field}))});
        VJ_CHECK(!bundle::inspect_code_object(elf.bytes.data(), elf.bytes.size()).ok);

        // Trailing bytes after the metadata map.
        elf = make_elf({make_note(metadata({KernSpec{}}, "amdgcn-amd-amdhsa--gfx906", true, true))});
        VJ_CHECK(!bundle::inspect_code_object(elf.bytes.data(), elf.bytes.size()).ok);
    }

    // ---- bundle verification against a temp root ---------------------------------------------------
    {
        const auto elf = test_elf_ok();
        const auto digest = bundle::sha256(elf.data(), elf.size());
        const std::string hex = bundle::to_hex(digest);
        const std::string path = vjtest::write_temp(elf, "hsaco_bundle_ok");
        const std::string name = base_name(path);
        const std::string root = "/tmp";
        const std::string manifest_text =
            "# release bundle\n" + hex + " " + std::to_string(elf.size()) + " " + name + "\n";
        bundle::Manifest parsed;
        std::string manifest_err;
        VJ_CHECK(bundle::parse_manifest(manifest_text, parsed, &manifest_err));
        bundle::BundleReport report;
        std::string verify_err;
        VJ_CHECK(bundle::verify_bundle(parsed, root, report, &verify_err));
        VJ_CHECK(report.ok && report.files.size() == 1 && report.files[0].ok && report.files[0].digest_ok &&
                 report.files[0].size_ok && report.files[0].code.ok);
        VJ_CHECK(report.files[0].digest_hex == hex);

        // The file APIs load the same content from disk.
        const std::vector<uint8_t> manifest_bytes(manifest_text.begin(), manifest_text.end());
        const std::string manifest_path = vjtest::write_temp(manifest_bytes, "hsaco_bundle_manifest");
        bundle::Manifest loaded;
        VJ_CHECK(bundle::load_manifest(manifest_path, loaded, &manifest_err));
        bundle::BundleReport loaded_report;
        VJ_CHECK(bundle::verify_bundle(loaded, root, loaded_report, &verify_err) && loaded_report.ok);
        const auto file_rep = bundle::inspect_code_object_file(path);
        VJ_CHECK(file_rep.ok && file_rep.kernels.size() == 2);

        // Manifest digests are case-insensitive on input and normalized to lowercase.
        std::string upper = hex;
        for (char& c : upper)
            if (c >= 'a' && c <= 'f') c = static_cast<char>(c - 'a' + 'A');
        bundle::Manifest upper_manifest;
        VJ_CHECK(bundle::parse_manifest(upper + " " + std::to_string(elf.size()) + " " + name + "\n", upper_manifest,
                                        &manifest_err));
        VJ_CHECK(upper_manifest.entries.size() == 1 && upper_manifest.entries[0].digest_hex == hex);

        // Digest mismatch: one changed byte in a copy of the same code object.
        std::vector<uint8_t> corrupt = elf;
        corrupt[corrupt.size() / 2] ^= 0x5au;
        const std::string corrupt_path = vjtest::write_temp(corrupt, "hsaco_bundle_corrupt");
        bundle::Manifest bad_digest;
        VJ_CHECK(bundle::parse_manifest(hex + " " + std::to_string(corrupt.size()) + " " +
                                            base_name(corrupt_path) + "\n",
                                        bad_digest, &manifest_err));
        report = bundle::BundleReport{};
        VJ_CHECK(!bundle::verify_bundle(bad_digest, root, report, &verify_err));
        VJ_CHECK(!report.ok && report.files.size() == 1 && !report.files[0].digest_ok);

        // Size mismatch with a matching digest line for the real file.
        bundle::Manifest bad_size;
        VJ_CHECK(bundle::parse_manifest(hex + " " + std::to_string(elf.size() + 1) + " " + name + "\n", bad_size,
                                        &manifest_err));
        report = bundle::BundleReport{};
        VJ_CHECK(!bundle::verify_bundle(bad_size, root, report, &verify_err));
        VJ_CHECK(!report.ok && !report.files[0].size_ok);

        // Missing file.
        bundle::Manifest missing;
        VJ_CHECK(bundle::parse_manifest(hex + " 7 vegajet_bundle_missing_file.hsaco\n", missing, &manifest_err));
        report = bundle::BundleReport{};
        VJ_CHECK(!bundle::verify_bundle(missing, "/tmp", report, &verify_err));
        VJ_CHECK(!report.ok && report.files.size() == 1 && !report.files[0].present);
    }

    // ---- GGUF families: four positives -------------------------------------------------------------
    {
        const auto dense = detect_bytes(gguf_dense_gqa(), "gguf_dense_gqa");
        VJ_CHECK(dense.ok() && dense.family == arch::Family::DenseGqa);
        VJ_CHECK(dense.gqa_layers == 2 && dense.mla_layers == 0 && dense.moe_ffn_layers == 0 &&
                 dense.dense_ffn_layers == 2);
        VJ_CHECK(std::string(arch::family_name(dense.family)) == "dense-gqa");

        const auto moe = detect_bytes(gguf_pure_moe(), "gguf_pure_moe");
        VJ_CHECK(moe.ok() && moe.family == arch::Family::PureMoe);
        VJ_CHECK(moe.moe_ffn_layers == 2 && moe.n_experts == 8 && moe.top_k == 2);

        const auto hybrid = detect_bytes(gguf_hybrid_gdn_moe(), "gguf_hybrid_gdn_moe");
        VJ_CHECK(hybrid.ok() && hybrid.family == arch::Family::HybridGdnMoe);
        VJ_CHECK(hybrid.gdn_layers == 3 && hybrid.gqa_layers == 1 && hybrid.moe_ffn_layers == 4);

        const auto mla = detect_bytes(gguf_mla_moe(), "gguf_mla_moe");
        VJ_CHECK(mla.ok() && mla.family == arch::Family::MlaMoe);
        VJ_CHECK(mla.mla_layers == 1 && mla.swa_layers == 1 && mla.moe_ffn_layers == 2);

        // detect_file() opens the GGUF itself.
        const std::string path = vjtest::write_temp(gguf_dense_gqa(), "gguf_dense_file");
        arch::Detection from_file;
        std::string file_err;
        VJ_CHECK(arch::detect_file(path, from_file, &file_err) && from_file.family == arch::Family::DenseGqa);
    }

    // ---- GGUF families: negatives ------------------------------------------------------------------
    {
        // No general.architecture at all.
        {
            vjtest::GgufWriter w;
            add_tensor(w, "token_embd.weight");
            expect_reject(w.build(), "gguf_no_arch");
        }
        // expert_count present but no expert tensors.
        {
            vjtest::GgufWriter w;
            w.kv_str("general.architecture", "bad");
            w.kv_u32("bad.block_count", 1);
            w.kv_u32("bad.attention.head_count", 8);
            w.kv_u32("bad.attention.head_count_kv", 2);
            w.kv_u32("bad.expert_count", 4);
            w.kv_u32("bad.expert_used_count", 2);
            add_tensor(w, "token_embd.weight");
            add_tensors(w, "blk.0.", {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                                      "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"});
            expect_reject(w.build(), "gguf_expert_meta_only");
        }
        // Expert tensors without expert_count.
        {
            vjtest::GgufWriter w;
            w.kv_str("general.architecture", "bad");
            w.kv_u32("bad.block_count", 1);
            w.kv_u32("bad.attention.head_count", 8);
            w.kv_u32("bad.attention.head_count_kv", 2);
            add_tensor(w, "token_embd.weight");
            add_tensors(w, "blk.0.", {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                                      "ffn_gate_inp.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight",
                                      "ffn_down_exps.weight"});
            expect_reject(w.build(), "gguf_expert_tensors_only");
        }
        // GDN tensors without ssm.* metadata.
        {
            vjtest::GgufWriter w;
            w.kv_str("general.architecture", "bad");
            w.kv_u32("bad.block_count", 1);
            w.kv_u32("bad.attention.head_count", 8);
            w.kv_u32("bad.attention.head_count_kv", 2);
            w.kv_u32("bad.expert_count", 4);
            w.kv_u32("bad.expert_used_count", 2);
            add_tensor(w, "token_embd.weight");
            add_tensors(w, "blk.0.", {"attn_qkv.weight", "attn_gate.weight", "ssm_out.weight", "ssm_a",
                                      "ffn_gate_inp.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight",
                                      "ffn_down_exps.weight"});
            expect_reject(w.build(), "gguf_gdn_no_ssm_keys");
        }
        // attention.kv_lora_rank without MLA tensors.
        {
            vjtest::GgufWriter w;
            w.kv_str("general.architecture", "bad");
            w.kv_u32("bad.block_count", 1);
            w.kv_u32("bad.attention.head_count", 8);
            w.kv_u32("bad.attention.head_count_kv", 2);
            w.kv_u32("bad.attention.kv_lora_rank", 16);
            w.kv_u32("bad.rope.dimension_count", 8);
            add_tensor(w, "token_embd.weight");
            add_tensors(w, "blk.0.", {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                                      "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"});
            expect_reject(w.build(), "gguf_kv_lora_without_mla");
        }
        // MLA tensors without attention.kv_lora_rank.
        {
            vjtest::GgufWriter w;
            w.kv_str("general.architecture", "bad");
            w.kv_u32("bad.block_count", 1);
            w.kv_u32("bad.attention.head_count", 8);
            w.kv_u32("bad.attention.head_count_kv", 2);
            w.kv_u32("bad.expert_count", 4);
            w.kv_u32("bad.expert_used_count", 2);
            add_tensor(w, "token_embd.weight");
            add_tensors(w, "blk.0.", {"attn_q.weight", "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight",
                                      "attn_kv_b.weight", "attn_output.weight", "ffn_gate_inp.weight",
                                      "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"});
            expect_reject(w.build(), "gguf_mla_without_kv_lora");
        }
        // recurrent_layers disagrees with the tensor layout.
        {
            vjtest::GgufWriter w;
            w.kv_str("general.architecture", "bad");
            w.kv_u32("bad.block_count", 2);
            w.kv_u32("bad.attention.head_count", 8);
            w.kv_u32("bad.attention.head_count_kv", 2);
            w.kv_arr_i32("bad.attention.recurrent_layers", {1, 0});
            add_tensor(w, "token_embd.weight");
            for (int L = 0; L < 2; ++L) {
                const std::string p = "blk." + std::to_string(L) + ".";
                add_tensors(w, p, {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                                   "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"});
            }
            expect_reject(w.build(), "gguf_recurrent_mismatch");
        }
        // Dense and expert FFN tensors in one block.
        {
            vjtest::GgufWriter w;
            w.kv_str("general.architecture", "bad");
            w.kv_u32("bad.block_count", 1);
            w.kv_u32("bad.attention.head_count", 8);
            w.kv_u32("bad.attention.head_count_kv", 2);
            w.kv_u32("bad.expert_count", 4);
            w.kv_u32("bad.expert_used_count", 2);
            add_tensor(w, "token_embd.weight");
            add_tensors(w, "blk.0.", {"attn_q.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
                                      "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"});
            expect_reject(w.build(), "gguf_mixed_ffn");
        }
        // GDN with dense FFN only (no routed experts) is not one of the four families.
        {
            vjtest::GgufWriter w;
            w.kv_str("general.architecture", "bad");
            w.kv_u32("bad.block_count", 2);
            w.kv_u32("bad.attention.head_count", 8);
            w.kv_u32("bad.attention.head_count_kv", 2);
            w.kv_u32("bad.ssm.group_count", 2);
            w.kv_u32("bad.ssm.time_step_rank", 4);
            w.kv_u32("bad.ssm.state_size", 8);
            w.kv_u32("bad.ssm.inner_size", 16);
            w.kv_u32("bad.ssm.conv_kernel", 4);
            w.kv_arr_i32("bad.attention.recurrent_layers", {1, 1});
            add_tensor(w, "token_embd.weight");
            for (int L = 0; L < 2; ++L) {
                const std::string p = "blk." + std::to_string(L) + ".";
                add_tensors(w, p, {"attn_qkv.weight", "attn_gate.weight", "ssm_out.weight", "ssm_a",
                                   "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"});
            }
            expect_reject(w.build(), "gguf_gdn_dense_only");
        }
        // GDN and MLA tensors in the same block.
        {
            vjtest::GgufWriter w;
            w.kv_str("general.architecture", "bad");
            w.kv_u32("bad.block_count", 1);
            w.kv_u32("bad.attention.head_count", 8);
            w.kv_u32("bad.attention.head_count_kv", 2);
            w.kv_u32("bad.ssm.group_count", 2);
            w.kv_u32("bad.attention.kv_lora_rank", 16);
            w.kv_u32("bad.rope.dimension_count", 8);
            w.kv_u32("bad.expert_count", 4);
            w.kv_u32("bad.expert_used_count", 2);
            add_tensor(w, "token_embd.weight");
            add_tensors(w, "blk.0.", {"attn_qkv.weight", "ssm_a", "attn_kv_a_mqa.weight",
                                      "ffn_gate_inp.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight",
                                      "ffn_down_exps.weight"});
            expect_reject(w.build(), "gguf_gdn_mla_mix");
        }
        // Missing block_count.
        {
            vjtest::GgufWriter w;
            w.kv_str("general.architecture", "bad");
            w.kv_u32("bad.attention.head_count", 8);
            add_tensor(w, "token_embd.weight");
            expect_reject(w.build(), "gguf_no_blocks");
        }
    }

    std::printf("[PASS] hsaco bundle digest/ELF gates + four GGUF families with malformed and conflicting "
                "metadata rejected\n");
    return 0;
}

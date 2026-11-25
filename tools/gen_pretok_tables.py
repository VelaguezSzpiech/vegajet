#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Generates the two data files behind core/pretok_families (developer tool, not run by the build):

  core/unicode_class_table.inc  general-category ranges (L, M, N, P, S, Z, C) from Python's unicodedata (the Unicode
                                Character Database; Unicode License v3) plus the DeepSeek-LLM letter class;
  tests/pretok_goldens.inc      golden pre-token splits of a mixed-script corpus, produced by the published
                                llama.cpp pre-tokenizer regexes run through the `regex` module (pip install regex),
                                applied the way llama.cpp applies a list of regexes: each one refines the chunks the
                                previous one produced (matches and the gaps between them both become chunks).

Usage: tools/gen_pretok_tables.py <repo-root>"""
import sys
import unicodedata
from pathlib import Path

LLAMA3 = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
QWEN2 = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
GPT2 = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
# Code points spelled as escapes: several are canonical singletons (U+1FBE, U+1FD3, U+2126, U+212A, ...) that an
# editor's NFC normalization would silently replace.
DS_RANGES = [(0x41, 0x5A), (0x61, 0x7A), (0xB5, 0xB5), (0xC0, 0xD6), (0xD8, 0xF6), (0xF8, 0x1BA), (0x1BC, 0x1BF),
             (0x1C4, 0x293), (0x295, 0x2AF), (0x370, 0x373), (0x376, 0x377), (0x37B, 0x37D), (0x37F, 0x37F),
             (0x386, 0x386), (0x388, 0x38A), (0x38C, 0x38C), (0x38E, 0x3A1), (0x3A3, 0x3F5), (0x3F7, 0x481),
             (0x48A, 0x52F), (0x531, 0x556), (0x10A0, 0x10C5), (0x13A0, 0x13F5), (0x13F8, 0x13FD), (0x1C90, 0x1CBA),
             (0x1CBD, 0x1CBF), (0x1D00, 0x1D2B), (0x1D6B, 0x1D77), (0x1D79, 0x1D9A), (0x1E00, 0x1F15),
             (0x1F18, 0x1F1D), (0x1F20, 0x1F45), (0x1F48, 0x1F4D), (0x1F50, 0x1F57), (0x1F59, 0x1F59),
             (0x1F5B, 0x1F5B), (0x1F5D, 0x1F5D), (0x1F5F, 0x1F7D), (0x1F80, 0x1FB4), (0x1FB6, 0x1FBC),
             (0x1FBE, 0x1FBE), (0x1FC2, 0x1FC4), (0x1FC6, 0x1FCC), (0x1FD0, 0x1FD3), (0x1FD6, 0x1FDB),
             (0x1FE0, 0x1FEC), (0x1FF2, 0x1FF4), (0x1FF6, 0x1FFC), (0x2102, 0x2102), (0x2107, 0x2107),
             (0x210A, 0x2113), (0x2115, 0x2115), (0x2119, 0x211D), (0x2124, 0x2124), (0x2126, 0x2126),
             (0x2128, 0x2128), (0x212A, 0x212D), (0x212F, 0x2134), (0x2139, 0x2139), (0x213C, 0x213F),
             (0x2145, 0x2149), (0x214E, 0x214E), (0x2183, 0x2184), (0x2C00, 0x2C7B), (0x2C7E, 0x2CE4),
             (0x2CEB, 0x2CEE), (0x2CF2, 0x2CF3), (0xA640, 0xA66D), (0xA680, 0xA69B), (0xA722, 0xA76F),
             (0xA771, 0xA787), (0xA78B, 0xA78E), (0xAB70, 0xABBF), (0xFB00, 0xFB06), (0xFB13, 0xFB17),
             (0xFF21, 0xFF3A), (0xFF41, 0xFF5A), (0x10400, 0x1044F), (0x104B0, 0x104D3), (0x104D8, 0x104FB),
             (0x10C80, 0x10CB2), (0x10CC0, 0x10CF2), (0x118A0, 0x118DF), (0x1E900, 0x1E943)]


def cls(rs):
    return "".join("\\U%08X" % a if a == b else "\\U%08X-\\U%08X" % (a, b) for a, b in rs)


DS_LETTERS = cls(DS_RANGES)
DS_PUNCT = cls([(0x21, 0x2F), (0x3A, 0x7E), (0xFF01, 0xFF0F), (0xFF1A, 0xFF5E), (0x2018, 0x201F), (0x3000, 0x3002)])
CJK_HANGUL = cls([(0x4E00, 0x9FA5), (0x800, 0x4E00), (0xAC00, 0xD7FF)])
CJK_KANA = cls([(0x4E00, 0x9FA5), (0x3040, 0x309F), (0x30A0, 0x30FF)])
DEEPSEEK_LLM = [r"[\r\n]", r"\s?[" + DS_LETTERS + r"]+", r"\s?[" + DS_PUNCT + r"]+", r"\s+\Z",
                r"[" + CJK_HANGUL + r"]+", r"\p{N}+"]
DEEPSEEK_CODER = [r"[\r\n]", r"\s?\p{L}+", r"\s?\p{P}+", r"[" + CJK_HANGUL + r"]+", r"\p{N}"]
DEEPSEEK_V3 = [r"\p{N}{1,3}", r"[" + CJK_KANA + r"]+",
               r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"]
TEKKEN = r"[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))*((?=[\p{L}])([^A-Z]))+|[^\r\n\p{L}\p{N}]?((?=[\p{L}])([^a-z]))+((?=[\p{L}])([^A-Z]))*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"

FAMILIES = [("Gpt2", [GPT2]), ("Llama3", [LLAMA3]), ("Qwen2", [QWEN2]), ("DeepSeekLlm", DEEPSEEK_LLM),
            ("DeepSeekCoder", DEEPSEEK_CODER), ("DeepSeekV3", DEEPSEEK_V3), ("Tekken", [TEKKEN])]

CORPUS = [
    "Hello world", "I'm 12345 don't YOU'LL we've", "  \n\n  x\r\n\ty  ",
    "x = foo(bar);  // comment\n\treturn a+b*3.14159;", "HelloWorld XMLHttpRequest iPhone getHTTPResponseCode",
    "café naïve Ångström Müller façade", "Привет, мир! ПРИВЕТ мир", "Γειά σου Κόσμε αβγ ΑΒΓ",
    "你好，世界！今天天气很好。", "日本語のテキスト、カタカナとひらがな。", "한국어 텍스트입니다.", "ok 🙂👍! emoji🚀test",
    "price: $1,234.56 (approx) 100% <tag> a|b ^ ~", "def f(x):\n    return x**2  # square\n",
    "line1\n\n\nline2   \n   ", "２０２６年１１月 ＡＢＣ ａｂｃ ！？", "»quoted« «guillemets» ¿Qué? ¡Sí!",
    "a\u00a0b\u3000c d", "under_score snake_case CONSTANT_VALUE camelCaseX", "https://example.com/path?q=1&r=2#frag",
    "1234567890 12 3 4567", "'s 't 're 've 'm 'll 'd 'S 'RE", "mixed123abc456DEF", "\t\ttabs\tand spaces  \n",
    "...!!!???---***", "x²+y³=z¹ ½ ¼", "e\u0301 combining acute, n\u0303o", "ÀÉÎÕÜ àéîõü", "Ŝtrange Ĳ ǅ ǈ",
    "end with spaces   ", "a/b/c\n/d", "\"quoted\" 'single' `tick`", "C++ & C# > Java", "ünïcödé ÜNÏCÖDÉ",
    "word.\n\nNext paragraph.", "αβγ123δεζ", "#include <vector>\nint main(){}",
]


def split(text, patterns):
    import regex
    chunks = [text]
    for p in patterns:
        rx = regex.compile(p)
        out = []
        for c in chunks:
            pos = 0
            for m in rx.finditer(c):
                if m.start() > pos:
                    out.append(c[pos:m.start()])
                if m.end() > m.start():
                    out.append(m.group(0))
                pos = m.end()
            if pos < len(c):
                out.append(c[pos:])
        chunks = out
    return chunks


def cstr(s):
    out = []
    for b in s.encode("utf-8"):
        if 32 <= b < 127 and b not in (34, 92, 63):
            out.append(chr(b))
        else:
            out.append("\\x%02X\"\"" % b)
    return '"' + "".join(out) + '"'


def major(cp):
    c = unicodedata.category(chr(cp))
    return {"L": 1, "M": 2, "N": 3, "P": 4, "S": 5, "Z": 6}.get(c[0], 0)


def ranges(pred, lo=0, hi=0x10FFFF):
    out, start, cur = [], None, None
    for cp in range(lo, hi + 1):
        v = pred(cp)
        if v != cur:
            if cur:
                out.append((start, cp - 1, cur))
            start, cur = cp, v
    if cur:
        out.append((start, hi, cur))
    return out


def main():
    root = Path(sys.argv[1])
    import regex
    cat = ranges(major)
    ds = regex.compile("[" + DS_LETTERS + "]")
    ds_r = ranges(lambda cp: 1 if ds.match(chr(cp)) else 0)
    hdr = ("// SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0\n// GENERATED by tools/gen_pretok_tables.py from "
           "the Unicode Character Database %s (Unicode License v3). Do not edit.\n" % unicodedata.unidata_version)
    t = [hdr, "// {first, last, class}: 1 L, 2 M, 3 N, 4 P, 5 S, 6 Z; code points not listed are C (other).\n",
         "static const UcRange kUcClass[] = {\n"]
    t += ["    {0x%X, 0x%X, %d},\n" % r for r in cat]
    t += ["};\n// DeepSeek-LLM pre-tokenizer letter class (its regex spells it out explicitly).\n",
          "static const UcRange kDeepSeekLetters[] = {\n"]
    t += ["    {0x%X, 0x%X, 1},\n" % r[:2] for r in ds_r]
    t.append("};\n")
    (root / "core/unicode_class_table.inc").write_text("".join(t))
    g = [hdr.replace("Unicode Character Database", "llama.cpp pre-tokenizer regexes (regex %s) over the" % regex.__version__),
         "// {family, input, {chunks...}}\n"]
    for fam, pats in FAMILIES:
        for s in CORPUS:
            parts = split(s, pats)
            assert "".join(parts) == s
            g.append("{PreTok::%s, %s, {%s}},\n" % (fam, cstr(s), ", ".join(cstr(p) for p in parts)))
    (root / "tests/pretok_goldens.inc").write_text("".join(g))
    print("ranges %d, deepseek letter ranges %d, goldens %d" % (len(cat), len(ds_r), len(FAMILIES) * len(CORPUS)))


if __name__ == "__main__":
    main()

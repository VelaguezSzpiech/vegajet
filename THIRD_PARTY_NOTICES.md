# Third-Party Notices

VegaJet is licensed under the PolyForm Noncommercial License 1.0.0 (see `LICENSE`, including its `Required Notice:` line).
The third-party material listed below remains under its own license; nothing in this file relicenses it, and the
PolyForm terms do not apply to it.

Every item below was checked against the code that ships in this repository (see `evidence/legal.md` in the release
staging area for the comparison commands, source hashes and the remaining open questions).

## ggml (MIT License)

Copyright (c) 2023-2026 The ggml authors

- `core/ggml_iq_tables.hpp` — the IQ/quant lookup tables `kmask_iq2xs`, `ksigns_iq2xs`, `iq2xxs_grid`, `iq2xs_grid`,
  `iq2s_grid`, `iq3xxs_grid`, `iq3s_grid` and `kvalues_iq4nl` are copied value-for-value from
  `ggml/src/ggml-common.h`; every entry was compared against ggml and matches. The same file's `iq2s_grid32[1024]` is
  VegaJet's own 32-bit packing of `iq2s_grid` (each grid byte is one of {8, 25, 43}; eight 2-bit selectors per entry,
  decoded with `v_perm_b32`), verified lossless for all 1024 entries, so it is a derivative of the ggml table and
  carries the same MIT terms.
- `core/ggml_types.hpp` — the GGUF/ggml tensor type ids and block geometry (element count and byte size per block) as
  published by ggml. Ids and sizes were verified against `ggml/include/ggml.h` and the `static_assert(sizeof(block_*))`
  lines in `ggml/src/ggml-common.h`.
- `core/quant_repacker.cpp`, `core/quant_unpackers.cpp`, `core/kv_quant_oracle.cpp`, `core/q4_dot8_repack.cpp` — the
  dequantization routines for `Q4_0` … `Q6_K` and the IQ types are adapted from ggml's reference `dequantize_row_*`
  functions (`ggml/src/ggml-quants.c`) and deliberately keep the reference arithmetic order so the decoded FP32 values
  match bit-for-bit. They are not independent reimplementations of the layouts; the MIT terms cover them.
  `tests/test_quant_repacker.cpp` pins golden FNV-1a hashes of decoded output; those hashes were cross-checked against
  the ggml reference dequantizers when the fixtures were created.
- The GGUF container parser (`core/gguf_reader.cpp`) and the tensor type list follow the public GGUF format as
  implemented by ggml/llama.cpp.

Source: https://github.com/ggml-org/ggml (also vendored in https://github.com/ggml-org/llama.cpp).

## z-lab DFlash (MIT License)

Copyright (c) 2026 Z Lab

- `tools/dflash_ref/dflash_ref.cpp` is a CPU reference implementation of the z-lab DFlash block-diffusion drafter
  (arXiv 2602.06036; https://github.com/z-lab/dflash, `dflash/model.py`) written in C++ for this repository from the
  published semantics: target-layer context taps, `hidden_norm(fc(·))` order, block mask, and the block slot / logit
  convention. `tools/dflash_parity_diag.py` and the `vj_dflash2` device path exercise the same math.
- The drafter's model weights (HF `DFlashDraftModel`) are not part of this repository and are not covered by this
  notice.

Source: https://github.com/z-lab/dflash.

## OpenAI GPT-2 encoder conventions (Modified MIT License)

Copyright (c) 2019 OpenAI

- `core/tokenizer.cpp` builds the GPT-2 byte-to-unicode bijection algorithmically (printable ASCII `!`..`~`, Latin-1
  `¡`..`¬` and `®`..`ÿ` map to themselves, the remaining 68 bytes map to U+0100..U+0143) instead of embedding a table.
  The result equals `bytes_to_unicode()` from https://github.com/openai/gpt-2 `src/encoder.py` for all 256 byte values.
- `core/bpe_tokenizer.cpp` implements the pre-tokenizer patterns published with the `gpt2`, `qwen2`/`qwen3` and `llama3`
  tokenizer families as hand-written matchers over compact Unicode class ranges (letter / number / whitespace / other)
  and contraction prefixes, rather than by running the published regular expressions.

The GPT-2 repository is under the Modified MIT License, reproduced below.

Source: https://github.com/openai/gpt-2.

## Public standards and algorithms (no third-party source copied)

- SHA-256 (`core/hsaco_bundle.cpp`): implements FIPS 180-4. The 64 round constants and 8 initial hash values are the
  standard's constants (fractional parts of the cube and square roots of the first primes); they were re-derived from
  that definition and verified for this audit. The routine is contained in this repository; the algorithm and its
  constants are the public standard's.
- SplitMix64 (`core/sampler.cpp`): the published finalizer constants and mixing steps (Steele, Lea & Flood; Vigna),
  whose reference code is public domain.
- Online suffix automaton (`core/suffix_trie.cpp`): the textbook construction (Blumer et al., 1985).
- gfx9 instruction encodings (`core/hsaco_isa_scan.cpp`): encoding fields and opcode numbers taken from AMD's public
  gfx9/GCN ISA reference material; these are numeric facts and no vendor code is included.
- Unicode character properties (`core/bpe_tokenizer.cpp`): the class tables encode `White_Space` and general-category
  (letter / number) property values as compact ranges. The Unicode Character Database is distributed under the Unicode
  License v3 (https://www.unicode.org/license.txt); no UCD data file is included in this repository.
- Chat and tool-call template text (`core/tool_call_parser.cpp`): the model families' published special-token strings
  (`<|im_start|>`, `<|im_end|>`, `<tools>`, `<tool_call>`, …).
- Recorded reference token ids (`models/ornith/ornith_golden.hpp`) are outputs recorded from llama.cpp greedy runs on
  the reference GGUF; they are measurement data, not third-party code.

## License texts

### MIT License

Copyright (c) 2023-2026 The ggml authors
Copyright (c) 2026 Z Lab

Each copyright line above applies to the material of that holder listed in this file.

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Modified MIT License (GPT-2)

```
Modified MIT License

Software Copyright (c) 2019 OpenAI

We don’t claim ownership of the content you create with GPT-2, so it is yours to do with as you please.
We only ask that you use GPT-2 responsibly and clearly indicate your content was created using GPT-2.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.
The above copyright notice and this permission notice need not be included
with content created by the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.
```

## No copyleft code

This audit found no reciprocal copyleft-licensed code in the repository. First-party sources are licensed under
`PolyForm-Noncommercial-1.0.0`, and `core/ggml_iq_tables.hpp` carries an `MIT` SPDX header; build files, docs, CI
configuration and any remaining first-party sources without an explicit header are covered by the repository
`LICENSE` and this notice. The Python tooling imports only the standard library, and a token-level overlap scan
against the ggml/llama.cpp sources matched only the material described above. Build and test container images
(the `rocm/dev-ubuntu-24.04` and `ubuntu:24.04` bases in `Dockerfile`) are pulled at build time and are not
redistributed here.

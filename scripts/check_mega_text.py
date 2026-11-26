#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Conservative mega instruction footprint: kernel symbol + ALL shared device functions.

The normal hsaco gate checks kernel ELF symbol sizes. Wave 2 also uses out-of-line
helpers, so charge their union to each kernel rather than hiding code outside its
symbol. This is an upper bound on reachable instructions; no GPU/runtime access.
"""
import struct
import sys


def check(path):
    data = open(path, "rb").read()
    if data[:6] != b"\x7fELF\x02\x01":
        raise ValueError("expected little-endian ELF64")
    shoff = struct.unpack_from("<Q", data, 40)[0]
    shsize, shnum = struct.unpack_from("<HH", data, 58)
    if shsize != 64:
        raise ValueError("unexpected section header size")
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shsize) for i in range(shnum)]
    functions = {}
    for sh in sections:
        if sh[1] != 2:  # SHT_SYMTAB
            continue
        strings = sections[sh[6]]
        names = data[strings[4]:strings[4] + strings[5]]
        for offset in range(sh[4], sh[4] + sh[5], sh[9]):
            name, info, _, section, value, size = struct.unpack_from("<IBBHQQ", data, offset)
            if info & 15 != 2 or not size or section == 0:  # STT_FUNC
                continue
            end = names.find(b"\0", name)
            functions[names[name:end].decode()] = (value, value + size)
    kernels = ("ornith_m1_gdn_moe_loop_q3_kernel", "ornith_m1_gqa_moe_q3_kernel")
    ranges = sorted(value for name, value in functions.items() if name not in kernels)
    merged = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    helper_bytes = sum(end - start for start, end in merged)
    for name in kernels:
        start, end = functions[name]
        footprint = end - start + helper_bytes
        print(f"mega text gate: {name}: kernel={end-start} B helpers={helper_bytes} B total={footprint} B")
        if footprint >= 24576:
            raise ValueError(f"{name}: reachable upper bound {footprint} B must be <24576 B")


if __name__ == "__main__":
    try:
        check(sys.argv[1])
    except (IndexError, KeyError, OSError, ValueError, struct.error) as error:
        sys.exit(f"mega text gate FAIL: {error}")

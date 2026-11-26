#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Read-only gfx906 ISA audit; counts static instructions, never opens a GPU.

Example: audit_mega_isa.py --objdump /path/to/llvm-objdump old.hsaco new.hsaco
JSON includes exact ELF symbol bounds (excluding alignment padding), AMDHSA
resources, opcode categories and every direct backwards branch's address range.
Loop ranges are static backedge spans, not executed-path or timing estimates;
overlapping/nested spans must not be summed. Calls do not include callee bodies.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import struct
import subprocess
import sys


class MessagePack:
    """Minimal stdlib decoder for AMDHSA's integer/string/container metadata."""

    def __init__(self, data):
        self.data, self.pos = data, 0

    def take(self, size):
        value = self.data[self.pos:self.pos + size]
        if len(value) != size:
            raise ValueError("truncated AMDHSA metadata")
        self.pos += size
        return value

    def integer(self, size, signed=False):
        return int.from_bytes(self.take(size), "big", signed=signed)

    def get(self):
        tag = self.integer(1)
        if tag < 0x80:
            return tag
        if tag >= 0xe0:
            return tag - 256
        if 0x80 <= tag < 0x90:
            return {self.get(): self.get() for _ in range(tag & 15)}
        if 0x90 <= tag < 0xa0:
            return [self.get() for _ in range(tag & 15)]
        if 0xa0 <= tag < 0xc0:
            return self.take(tag & 31).decode()
        if tag in (0xc0, 0xc2, 0xc3):
            return {0xc0: None, 0xc2: False, 0xc3: True}[tag]
        if 0xcc <= tag <= 0xcf:
            return self.integer(1 << (tag - 0xcc))
        if 0xd0 <= tag <= 0xd3:
            return self.integer(1 << (tag - 0xd0), signed=True)
        if tag in (0xca, 0xcb):
            return struct.unpack(">f" if tag == 0xca else ">d", self.take(4 if tag == 0xca else 8))[0]
        if 0xd9 <= tag <= 0xdb:
            return self.take(self.integer(1 << (tag - 0xd9))).decode()
        if tag in (0xdc, 0xdd):
            return [self.get() for _ in range(self.integer(2 << (tag - 0xdc)))]
        if tag in (0xde, 0xdf):
            return {self.get(): self.get() for _ in range(self.integer(2 << (tag - 0xde)))}
        raise ValueError(f"unsupported AMDHSA metadata MessagePack tag {tag:#x}")


def elf_info(path):
    data = Path(path).read_bytes()
    if data[:6] != b"\x7fELF\x02\x01":
        raise ValueError("expected little-endian ELF64")
    shoff = struct.unpack_from("<Q", data, 40)[0]
    shsize, shnum = struct.unpack_from("<HH", data, 58)
    if shsize != 64:
        raise ValueError("unexpected ELF section header size")
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shsize) for i in range(shnum)]
    functions, resources = {}, {}
    for section in sections:
        if section[1] == 2:  # SHT_SYMTAB
            strings = sections[section[6]]
            names = data[strings[4]:strings[4] + strings[5]]
            for offset in range(section[4], section[4] + section[5], section[9]):
                name, info, _, shndx, address, size = struct.unpack_from("<IBBHQQ", data, offset)
                if info & 15 == 2 and size and shndx:  # STT_FUNC
                    name = names[name:names.index(b"\0", name)].decode()
                    functions[name] = (address, size)
        if section[1] == 7:  # SHT_NOTE
            offset, end = section[4], section[4] + section[5]
            while offset + 12 <= end:
                namesz, descsz, kind = struct.unpack_from("<III", data, offset)
                offset += 12
                owner = data[offset:offset + namesz].rstrip(b"\0")
                offset += (namesz + 3) & ~3
                desc = data[offset:offset + descsz]
                offset += (descsz + 3) & ~3
                if owner == b"AMDGPU" and kind == 32:  # NT_AMDGPU_METADATA
                    for kernel in MessagePack(desc).get().get("amdhsa.kernels", []):
                        resources[kernel[".name"]] = {
                            key.lstrip("."): value for key, value in kernel.items()
                            if key in (".vgpr_count", ".sgpr_count", ".group_segment_fixed_size",
                                       ".private_segment_fixed_size", ".wavefront_size",
                                       ".sgpr_spill_count", ".vgpr_spill_count", ".uses_dynamic_stack")}
    return functions, resources


def counts(instructions):
    opcodes = Counter(ins[1] for ins in instructions)
    prefixes = {
        "scalar_load": "s_load", "global_load": "global_load", "flat_load": "flat_load",
        "buffer_load": "buffer_load", "global_atomic": "global_atomic", "lds_read": "ds_read",
        "lds_write": "ds_write", "dot4": "v_dot4", "scalar_compare": "s_cmp",
        "vector_compare": "v_cmp", "conditional_branch": "s_cbranch"}
    categories = {name: sum(n for op, n in opcodes.items() if op.startswith(prefix))
                  for name, prefix in prefixes.items()}
    categories.update({"call": opcodes["s_swappc_b64"], "return": opcodes["s_setpc_b64"],
                       "sleep": opcodes["s_sleep"], "wg_barrier": opcodes["s_barrier"],
                       "waitcnt": opcodes["s_waitcnt"], "readfirstlane": opcodes["v_readfirstlane_b32"],
                       "unconditional_branch": opcodes["s_branch"]})
    categories["full_waitcnt"] = sum(
        ins[1] == "s_waitcnt" and all(counter in ins[3] for counter in
                                      ("vmcnt(0)", "expcnt(0)", "lgkmcnt(0)"))
        for ins in instructions)
    return {"instructions": len(instructions), "categories": categories, "opcodes": dict(sorted(opcodes.items()))}


def audit(path, objdump):
    functions, resources = elf_info(path)
    isa = subprocess.check_output([objdump, "-d", str(path)], text=True)
    parsed, current = {}, None
    for line in isa.splitlines():
        header = re.match(r"^([0-9a-fA-F]+) <(.+)>:$", line)
        if header:
            current = header[2] if header[2] in functions else None
            if current is not None:
                parsed[current] = []
            continue
        if current is None or "//" not in line:
            continue
        assembly, encoded = line.split("//", 1)
        match = re.match(r"\s*([0-9a-fA-F]+):\s*((?:[0-9a-fA-F]{8}(?:\s+|$))+)", encoded)
        if not match or not assembly.strip():
            continue
        address = int(match[1], 16)
        start, size = functions[current]
        if start <= address < start + size:
            opcode = assembly.split()[0]
            width = 4 * len(match[2].split())
            parsed[current].append((address, opcode, encoded, assembly.strip(), width))
    records = []
    for name, (start, size) in sorted(functions.items(), key=lambda item: item[1][0]):
        instructions = parsed.get(name, [])
        if not instructions:
            raise ValueError(f"objdump produced no instructions for {name}")
        record = {"name": name, "start": hex(start), "symbol_bytes": size,
                  "kind": "kernel" if name in resources else "helper",
                  "first_instruction": instructions[0][3], **counts(instructions)}
        if name in resources:
            record["resources"] = resources[name]
        record["backedges"] = []
        for address, opcode, encoded, assembly, width in instructions:
            if opcode != "s_branch" and not opcode.startswith("s_cbranch"):
                continue
            # gfx906 SOPP branches have a signed 16-bit displacement in dwords.
            word = int(encoded.split(":", 1)[1].split()[0], 16)
            displacement = word & 0xffff
            if displacement >= 0x8000:
                displacement -= 0x10000
            target = address + 4 + 4 * displacement
            if not start <= target <= address:
                continue
            span = [ins for ins in instructions if target <= ins[0] <= address]
            record["backedges"].append({"start": hex(target), "branch": hex(address),
                                         "span_bytes": address + width - target,
                                         "branch_assembly": assembly, **counts(span)})
        records.append(record)
    return {"path": str(path), "functions": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objdump", required=True, help="explicit llvm-objdump executable")
    parser.add_argument("hsaco", nargs="+", type=Path)
    args = parser.parse_args()
    report = {"note": "Static symbol-bounded counts; backedge spans can overlap; no GPU execution or timing.",
              "objects": [audit(path, args.objdump) for path in args.hsaco]}
    json.dump(report, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, struct.error, subprocess.CalledProcessError) as error:
        sys.exit(f"mega ISA audit failed: {error}")

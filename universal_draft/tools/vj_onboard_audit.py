#!/usr/bin/env python3
"""Audit gfx906 onboarding binaries offline, using the baseline HSACO reader."""
import argparse
import importlib.util
from pathlib import Path
import shutil
import sys

NAMES = {"vj_vm_attn_block", "vj_vm_ffn_block", "vj_vm_aux_block"}


def check(rows):
    if len(rows) != len(NAMES) or {r["name"] for r in rows} != NAMES:
        raise ValueError("missing, duplicate or unexpected VM symbol")
    for r in rows:
        if not (0 < r["vgpr_count"] <= 64 and 0 < r["text_bytes"] < 24576 and
                r["wavefront_size"] == 64 and r["private_segment_fixed_size"] == 0 and
                r["lds_bad"] == 0 and (r["name"] == "vj_vm_aux_block" or r["flags"].get((1, 1), 0) > 0) and
                r["flags"].get((0, 0), 0) > 0):
            raise ValueError("resource/cache gate: " + r["name"])


def self_test():
    rows = [dict(name=name, vgpr_count=64, text_bytes=24575, wavefront_size=64,
                 private_segment_fixed_size=0, lds_bad=0, flags={(1, 1): 1, (0, 0): 1})
            for name in sorted(NAMES)]
    check(rows)
    controls = [("name", "unknown"), ("vgpr_count", 65), ("text_bytes", 24576),
                ("wavefront_size", 32), ("private_segment_fixed_size", 4), ("lds_bad", 1),
                ("flags", {(1, 1): 1})]
    rejected = 0
    for index, row in enumerate(rows):
        cases = controls + ([] if row["name"] == "vj_vm_aux_block" else [("flags", {(0, 0): 1})])
        for key, bad in cases:
            trial = rows.copy()
            trial[index] = dict(row, **{key: bad})
            try:
                check(trial)
            except ValueError:
                rejected += 1
                continue
            raise AssertionError((row["name"], key))
    print(f"audit self-test: three symbols, {rejected} rejection controls PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--hsaco", type=Path)
    parser.add_argument("--reader", type=Path, default=Path("tools/hsaco_audit.py"))
    args = parser.parse_args()
    if args.self_test:
        self_test()
    if args.hsaco:
        spec = importlib.util.spec_from_file_location("hsaco_reader", args.reader)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        objdump = shutil.which("llvm-objdump") or shutil.which("llvm-objdump-18") or shutil.which("llvm-objdump-19")
        readobj = (shutil.which("llvm-readobj") or shutil.which("llvm-readobj-18") or
                   shutil.which("llvm-readobj-19") or shutil.which("llvm-readelf"))
        if not objdump or not readobj:
            raise RuntimeError("llvm-objdump and llvm-readobj must be on PATH")
        rows = mod.audit(args.hsaco, objdump, readobj, {}, "super")
        print(mod.table(rows))
        check(rows)
        print("VM binary resource and cache-mode presence gates PASS (not operand provenance or device parity)")
    if not args.self_test and not args.hsaco:
        parser.error("provide --hsaco or --self-test")


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    main()

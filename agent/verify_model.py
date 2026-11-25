#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""VegaJet model verification harness (standard library only).

Gates:
  1. CPU unit tests: ctest in the build directory (model files are passed to tests through --env).
  2. Model header: `vegajet inspect` succeeds, all tensor types have CPU decoders; reports decode bytes/token.
  3. HBM2 utilization (optional): measured decode tokens/s x bytes read per token vs. the measured per-GPU
     streaming ceilings (`vegajet soak`), never the datasheet peak.
  4. Logit parity (optional): candidate vs reference raw float32 logits (cosine, max abs diff, argmax agreement).
  5. ECC (optional, Linux sysfs, read-only): RAS uncorrectable counters must not increase during the run.
  6. Code objects (optional): vj_hsaco_gate (metadata) and vj_isa_scan (no wide LDS-direct loads) per --hsaco file.

Model paths are only ever taken from the command line.
"""
import argparse
import array
import glob
import math
import os
import re
import subprocess
import sys


def run(cmd, env=None):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    return p.returncode, p.stdout


def ras_counts():
    """Sums ue/ce over every AMD PCI device exposing ras/*_err_count (read-only)."""
    totals = {"ue": 0, "ce": 0, "files": 0}
    root = os.environ.get("VJ_SYSFS_ROOT", "/")
    for f in glob.glob(os.path.join(root, "sys/bus/pci/devices/*/ras/*_err_count")):
        dev = f.split("/ras/")[0]
        try:
            if open(os.path.join(dev, "vendor")).read().strip() != "0x1002":
                continue
            for line in open(f):
                k, _, v = line.partition(":")
                if k.strip() in ("ue", "ce"):
                    totals[k.strip()] += int(v)
            totals["files"] += 1
        except (OSError, ValueError):
            continue
    return totals


def parse_ceilings(spec, legacy_peak, gpus):
    """Per-GPU measured ceilings in GB/s. A single legacy value is replicated over --gpus."""
    if spec:
        vals = [float(x) for x in spec.split(",") if x.strip()]
        if any(v <= 0 for v in vals):
            raise SystemExit("--hbm-ceiling-gbs values must be > 0")
        return vals
    if legacy_peak:
        return [legacy_peak] * max(1, gpus)
    return []


def hbm_utilization(tok_s, bytes_per_token, ceilings):
    """Aggregate achieved GB/s, aggregate ceiling, percent. bytes_per_token counts all GPUs."""
    achieved = tok_s * bytes_per_token / 1e9
    total = sum(ceilings)
    return achieved, total, 100.0 * achieved / total


def load_f32(path):
    a = array.array("f")
    with open(path, "rb") as fh:
        a.frombytes(fh.read())
    return a


def logit_parity(ref_path, cand_path, vocab):
    ref, cand = load_f32(ref_path), load_f32(cand_path)
    if len(ref) != len(cand) or len(ref) == 0 or (vocab and len(ref) % vocab):
        return False, "length mismatch (%d vs %d)" % (len(ref), len(cand))
    rows = len(ref) // vocab if vocab else 1
    width = vocab if vocab else len(ref)
    worst_cos, max_abs, agree = 1.0, 0.0, 0
    for r in range(rows):
        a, b = ref[r * width:(r + 1) * width], cand[r * width:(r + 1) * width]
        ab = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        cos = ab / (na * nb) if na and nb else float(na == nb)
        worst_cos = min(worst_cos, cos)
        max_abs = max(max_abs, max(abs(x - y) for x, y in zip(a, b)))
        agree += max(range(width), key=a.__getitem__) == max(range(width), key=b.__getitem__)
    ok = worst_cos >= 0.999 and agree == rows
    return ok, "rows=%d worst_cos=%.6f max_abs=%.4g argmax_agree=%d/%d" % (rows, worst_cos, max_abs, agree, rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", default="build-cpu", help="CMake build directory")
    ap.add_argument("--model", help="target model file (passed through, never stored)")
    ap.add_argument("--env", action="append", default=[], metavar="NAME=VALUE",
                    help="extra environment for ctest, e.g. VEGAJET_ORNITH_GGUF=<path>")
    ap.add_argument("--decode-tok-s", type=float, help="measured decode tokens/s (aggregate)")
    ap.add_argument("--hbm-ceiling-gbs", default=None,
                    help="measured streaming ceiling per GPU from `vegajet soak`, comma separated (e.g. 840,972); "
                         "the datasheet peak is never used as the denominator")
    ap.add_argument("--hbm-peak-gbs", type=float, default=None, help=argparse.SUPPRESS)  # legacy alias (one value)
    ap.add_argument("--gpus", type=int, default=1, help="GPUs sharing the decode read volume")
    ap.add_argument("--bytes-per-token", type=float, default=None,
                    help="bytes the kernels actually read per token, all GPUs (the engine's real_read figure); "
                         "overrides the disk-format estimate from `vegajet inspect`")
    ap.add_argument("--hsaco", nargs="*", default=[], help="code objects to gate (vj_hsaco_gate + vj_isa_scan)")
    ap.add_argument("--reference-logits", help="raw float32 logits from the reference engine")
    ap.add_argument("--candidate-logits", help="raw float32 logits from VegaJet")
    ap.add_argument("--vocab", type=int, default=0, help="logit row width (vocab size)")
    ap.add_argument("--ecc", action="store_true", help="require no new RAS uncorrectable errors during the run")
    args = ap.parse_args()

    failures = []
    ras0 = ras_counts() if args.ecc else None

    env = dict(os.environ)
    for kv in args.env:
        k, _, v = kv.partition("=")
        env[k] = v
    if args.model:
        env.setdefault("VEGAJET_MODEL_GGUF", args.model)
    rc, out = run(["ctest", "--test-dir", args.build, "--output-on-failure"], env)
    summary = [l for l in out.splitlines() if "tests passed" in l or "tests failed" in l]
    print("[ctest] %s" % (summary[-1].strip() if summary else "no summary"))
    if rc != 0:
        failures.append("ctest")
        print(out)

    bytes_per_token = None
    if args.model:
        rc, out = run([os.path.join(args.build, "vegajet"), "inspect", args.model])
        if rc != 0:
            failures.append("inspect")
            print(out)
        else:
            arch = re.search(r"^general\.architecture = (\S+)", out, re.M)
            m = re.search(r"decode bytes/token \(disk formats\): ([0-9.]+) GiB", out)
            print("[inspect] architecture=%s" % (arch.group(1) if arch else "?"))
            if "types without a CPU decoder" in out:
                failures.append("unsupported tensor types")
            if m:
                bytes_per_token = float(m.group(1)) * (1 << 30)
                print("[inspect] decode read volume: %.3f GiB/token" % float(m.group(1)))

    if args.bytes_per_token:
        bytes_per_token = args.bytes_per_token
    ceilings = parse_ceilings(args.hbm_ceiling_gbs, args.hbm_peak_gbs, args.gpus)
    if args.decode_tok_s and bytes_per_token:
        if not ceilings:
            print("[hbm2] skipped: pass --hbm-ceiling-gbs with the measured `vegajet soak` ceiling per GPU")
        else:
            achieved, total, pct = hbm_utilization(args.decode_tok_s, bytes_per_token, ceilings)
            print("[hbm2] %.1f GB/s of %.1f GB/s measured ceiling = %.1f%% (%.2f tok/s, %d GPU)"
                  % (achieved, total, pct, args.decode_tok_s, len(ceilings)))

    for path in args.hsaco:
        for tool in ("vj_hsaco_gate", "vj_isa_scan"):
            exe = os.path.join(args.build, tool)
            if not os.path.exists(exe):
                print("[%s] not built; skipped" % tool)
                continue
            rc, out = run([exe, path])
            print("[%s] %s %s" % (tool, "PASS" if rc == 0 else "FAIL", os.path.basename(path)))
            if rc != 0:
                failures.append("%s %s" % (tool, os.path.basename(path)))
                print(out)

    if args.reference_logits and args.candidate_logits:
        ok, msg = logit_parity(args.reference_logits, args.candidate_logits, args.vocab)
        print("[parity] %s %s" % ("PASS" if ok else "FAIL", msg))
        if not ok:
            failures.append("logit parity")

    if args.ecc:
        ras1 = ras_counts()
        if ras0["files"] == 0:
            print("[ecc] no RAS counters exposed; skipped")
        else:
            d_ue, d_ce = ras1["ue"] - ras0["ue"], ras1["ce"] - ras0["ce"]
            print("[ecc] new UE=%d CE=%d (UE total %d)" % (d_ue, d_ce, ras1["ue"]))
            if d_ue > 0:
                failures.append("ECC uncorrectable errors")

    print("VERIFY: %s" % ("PASS" if not failures else "FAIL (" + ", ".join(failures) + ")"))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())

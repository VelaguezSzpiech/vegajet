#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""VegaJet regression runner for CI (stdlib only).

  ci_regression.py --cpu-only [--stages patches,opt,san,selftests,privacy] [--patch P ...] [--jobs N]
                   [--build-root DIR] [--report FILE] [--dry-run]
  ci_regression.py --gpu-matrix 1,2,3 --model FILE [--vegajet BIN] [--out-dir DIR] [--baseline-dir DIR]
                   [--from-json DIR] [--preflight] [--runner CMD] [--bench-arg ARG ...] [--max-drop PCT]
                   [--report FILE] [--dry-run]
  ci_regression.py --self-test

--cpu-only (GitHub Actions, any x86-64 runner, never opens a device node):
  patches    `git apply --check` of every --patch, in order, each on top of the previous, on a fresh `git archive
             HEAD` export (the stacked-patch audit; skipped without --patch)
  opt        -DVEGAJET_CPU_ONLY=ON Release (-O3) with -Werror, then ctest (vegajet_doctor excluded: it reads the
             live /sys)
  san        the same with ASan + UBSan (-fno-sanitize-recover=all); test_prefill_dot8_kld is excluded (known UB in
             its reference path, tracked separately)
  selftests  the stdlib tools' self-tests (regression_gate, bench_diff, hsaco_audit, this script)
  privacy    tools/privacy_audit.sh over the checkout and its history
  Known warnings of files outside a change are allowed one file / one warning at a time (KNOWN_WARNINGS below) via
  a generated CMake project include, so -Werror stays on everywhere else.

--gpu-matrix N[,N...] (nightly, self-hosted GPU runner): for every TP size N, `vegajet bench MODEL --tp N --gpu
  0-(N-1) --json-out DIR/tpN.json`, then the gates on that scorecard:
    spill      every spill / private_segment metric is 0 B, and at least one is reported
    ue / de    uncorrectable and deferred RAS counts are 0, and both are reported
    parity     a 32/32 code parity and a 129/129 speculative parity are reported, and every parity is complete
    baseline   with --baseline-dir DIR/tpN.json: tools/regression_gate.py (throughput drop <= --max-drop %, KLD,
               TMem) against it
  --preflight runs `vegajet doctor --json` once first (read-only) and stops on a failing check. --runner prefixes
  every GPU command (e.g. a lock / queue wrapper). --from-json DIR gates existing scorecards without running
  anything; --dry-run prints the commands only.

Exit codes: 0 every stage / gate passed, 1 something failed, 2 usage or environment error.
"""
import argparse
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
CPU_STAGES = ("patches", "opt", "san", "selftests", "privacy")
KNOWN_WARNINGS = [  # (source file, warning) pre-existing in the base; downgraded from error, still printed
    ("tests/vegajet_layer_test.cpp", "unused-but-set-variable"),
]
SAN_FLAGS = "-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer"
CTEST_EXCLUDE = {"opt": "vegajet_doctor", "san": "vegajet_doctor|test_prefill_dot8_kld"}
SELFTESTS = [
    ["tools/regression_gate.py", "--self-test"],
    ["tools/bench_diff.py", "--self-test"],
    ["tools/hsaco_audit.py", "--self-test"],
    ["tools/ci_regression.py", "--self-test"],
]


# ------------------------------------------------------------------------------------------------ helpers
class Report:
    def __init__(self, mode):
        self.doc = {"schema": "vegajet.ci.v1", "mode": mode, "stages": [], "ok": True}

    def add(self, name, ok, seconds=0.0, detail="", **extra):
        self.doc["stages"].append(dict({"name": name, "ok": bool(ok), "seconds": round(seconds, 2), "detail": detail}, **extra))
        self.doc["ok"] = self.doc["ok"] and bool(ok)
        print("%-4s %-24s %6.1fs  %s" % ("PASS" if ok else "FAIL", name, seconds, detail), flush=True)

    def write(self, path):
        if path:
            Path(path).write_text(json.dumps(self.doc, indent=2) + "\n")


def run(cmd, cwd=None, log=None, env=None, timeout=None, dry=False):
    """Runs cmd (list); returns (rc, seconds, tail of output)."""
    if dry:
        print("  $ " + " ".join(shlex.quote(str(c)) for c in cmd))
        return 0, 0.0, ""
    t0 = time.monotonic()
    try:
        p = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, timeout=timeout, capture_output=True, text=True)
        out, rc = p.stdout + p.stderr, p.returncode
    except (OSError, subprocess.TimeoutExpired) as e:
        out, rc = str(e), 127 if isinstance(e, OSError) else 124
    if log:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as f:
            f.write("$ %s\n%s\n" % (" ".join(map(str, cmd)), out))
    return rc, time.monotonic() - t0, out[-4000:]


def load_gate_module():
    spec = importlib.util.spec_from_file_location("regression_gate", TOOLS / "regression_gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def known_warnings_include(build_dir, source_root):
    build_dir.mkdir(parents=True, exist_ok=True)
    inc = build_dir / "ci_known_warnings.cmake"
    lines = ["# generated by tools/ci_regression.py: pre-existing warnings stay warnings under -Werror"]
    for f, w in KNOWN_WARNINGS:
        lines.append('set_source_files_properties("${CMAKE_SOURCE_DIR}/%s" PROPERTIES COMPILE_OPTIONS "-Wno-error=%s")' % (f, w))
    inc.write_text("\n".join(lines) + "\n")
    return inc


# ------------------------------------------------------------------------------------------------ CPU stages
def stage_patches(a, rep):
    if not a.patch:
        rep.add("patches", True, 0.0, "no --patch given (skipped)")
        return
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="vj_ci_stack_") as tmp:
        if a.dry_run:
            print("  $ git archive HEAD | tar -x -C %s" % tmp)
        else:
            arch = subprocess.run(["git", "-C", str(a.source), "archive", "HEAD"], capture_output=True)
            if arch.returncode != 0:
                rep.add("patches", False, time.monotonic() - t0, "git archive failed: %s" % arch.stderr.decode()[-300:])
                return
            subprocess.run(["tar", "-x", "-C", tmp], input=arch.stdout, check=True)
        results = []
        for p in a.patch:
            p = Path(p).resolve()
            rc, _, out = run(["git", "apply", "--check", str(p)], cwd=tmp, dry=a.dry_run)
            if rc == 0:
                rc2, _, out2 = run(["git", "apply", str(p)], cwd=tmp, dry=a.dry_run)
                rc, out = rc2, out2
            results.append({"patch": p.name, "ok": rc == 0, "error": "" if rc == 0 else out.strip()[-500:]})
            print("     %s %s" % ("ok  " if rc == 0 else "FAIL", p.name), flush=True)
            if rc != 0:
                break
    ok = all(r["ok"] for r in results) and len(results) == len(a.patch)
    rep.add("patches", ok, time.monotonic() - t0, "%d/%d applied in order on a fresh export" %
            (sum(r["ok"] for r in results), len(a.patch)), patches=results)


def stage_build(a, rep, mode):
    bdir = Path(a.build_root) / ("ci-" + mode)
    log = bdir / "ci.log"
    if not a.dry_run and bdir.exists():
        shutil.rmtree(bdir)
    inc = known_warnings_include(bdir, a.source) if not a.dry_run else bdir / "ci_known_warnings.cmake"
    flags = "-Werror -Wno-error=stringop-overflow" + (" " + SAN_FLAGS if mode == "san" else "")
    btype = "Release" if mode == "opt" else "RelWithDebInfo"
    cfg = ["cmake", "-S", str(a.source), "-B", str(bdir), "-DVEGAJET_CPU_ONLY=ON", "-DCMAKE_BUILD_TYPE=" + btype,
           "-DCMAKE_CXX_FLAGS=" + flags, "-DCMAKE_PROJECT_INCLUDE=" + str(inc)]
    if mode == "san":
        cfg.append("-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined")
    t0 = time.monotonic()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for name, cmd in (("configure", cfg), ("build", ["cmake", "--build", str(bdir), "-j", str(a.jobs)])):
        rc, _, out = run(cmd, log=None if a.dry_run else log, env=env, dry=a.dry_run)
        if rc != 0:
            errs = "\n".join(l for l in out.splitlines() if "error" in l.lower())[-1500:]
            rep.add(mode, False, time.monotonic() - t0, "%s failed (log %s)" % (name, log), errors=errs)
            return
    warnings = 0
    if not a.dry_run and log.exists():
        warnings = sum(1 for l in log.read_text(errors="replace").splitlines() if "warning:" in l)
    rc, _, out = run(["ctest", "--test-dir", str(bdir), "-j", str(a.jobs), "--output-on-failure", "-E", CTEST_EXCLUDE[mode]],
                     log=None if a.dry_run else log, env=env, dry=a.dry_run)
    summary = next((l.strip() for l in reversed(out.splitlines()) if "tests passed" in l), "dry run" if a.dry_run else "no ctest summary")
    failed = re.findall(r"\d+ - (\S+) \(", out)
    rep.add(mode, rc == 0, time.monotonic() - t0, "%s; %d warnings (known ones allowed)" % (summary, warnings),
            failed=failed)


def stage_selftests(a, rep):
    t0 = time.monotonic()
    bad = []
    for cmd in SELFTESTS:
        if a.self_test_inner and cmd[0].endswith("ci_regression.py"):
            continue
        if not (a.source / cmd[0]).exists():
            continue
        rc, _, out = run([sys.executable, str(a.source / cmd[0])] + cmd[1:], cwd=a.source, dry=a.dry_run,
                         env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
        if rc != 0:
            bad.append("%s: %s" % (cmd[0], out.strip()[-300:]))
    rep.add("selftests", not bad, time.monotonic() - t0, "; ".join(bad) or "%d tool self-tests" % len(SELFTESTS))


def stage_privacy(a, rep):
    t0 = time.monotonic()
    script = a.source / "tools/privacy_audit.sh"
    if not script.exists():
        rep.add("privacy", False, 0.0, "tools/privacy_audit.sh missing")
        return
    rc, _, out = run(["bash", str(script)], cwd=a.source, dry=a.dry_run)
    last = next((l for l in reversed(out.splitlines()) if l.strip()), "")
    rep.add("privacy", rc == 0, time.monotonic() - t0, last[:200] or ("dry run" if a.dry_run else ""))


def cpu_only(a):
    rep = Report("cpu-only")
    stages = [s.strip() for s in a.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in CPU_STAGES]
    if unknown:
        print("ci_regression: unknown stage(s) %s (known: %s)" % (", ".join(unknown), ", ".join(CPU_STAGES)), file=sys.stderr)
        return None
    for s in stages:
        if s == "patches":
            stage_patches(a, rep)
        elif s in ("opt", "san"):
            stage_build(a, rep, s)
        elif s == "selftests":
            stage_selftests(a, rep)
        elif s == "privacy":
            stage_privacy(a, rep)
    return rep


# ------------------------------------------------------------------------------------------------ GPU matrix
def gpu_spec(n):
    return "0" if n == 1 else "0-%d" % (n - 1)


def evaluate(card_path, n, gate_mod, baseline=None, max_drop=2.0, parity_required=(32, 129)):
    """Gates of one scorecard -> list of {gate, ok, detail}."""
    out = []
    try:
        m = gate_mod.load(str(card_path))
    except (OSError, ValueError) as e:
        return [{"gate": "scorecard", "ok": False, "detail": "unreadable: %s" % e}]
    kinds = {}
    for name, vals in m.items():
        k = gate_mod.classify(name)
        if k:
            kinds.setdefault(k, []).append((name, vals))

    def numbers(kind):
        return [(nm, v) for nm, vs in kinds.get(kind, []) for v in vs if isinstance(v, (int, float))]

    sp = numbers("spill")
    out.append({"gate": "spill", "ok": bool(sp) and all(v == 0 for _, v in sp),
                "detail": "not reported" if not sp else ", ".join("%s=%g B" % (nm, v) for nm, v in sp)})
    for k in ("ue", "de"):
        xs = numbers(k)
        out.append({"gate": k, "ok": bool(xs) and all(v == 0 for _, v in xs),
                    "detail": "not reported" if not xs else ", ".join("%s=%g" % (nm, v) for nm, v in xs)})
    par = []
    for nm, vs in kinds.get("parity", []):
        for v in vs:
            p = gate_mod.parse_parity(v)
            if isinstance(p, tuple):
                par.append((nm, p))
    complete = all(a_ == b_ for _, (a_, b_) in par)
    have = {t for _, (a_, t) in par if a_ == t}
    missing = [t for t in parity_required if t not in have]
    out.append({"gate": "parity", "ok": bool(par) and complete and not missing,
                "detail": (", ".join("%s=%d/%d" % (nm, a_, b_) for nm, (a_, b_) in par) or "not reported") +
                          ("; missing complete %s" % ", ".join("%d/%d" % (t, t) for t in missing) if missing else "")})
    if baseline is not None:
        if not Path(baseline).exists():
            out.append({"gate": "baseline", "ok": False, "detail": "baseline %s missing" % Path(baseline).name})
        else:
            checks, failed = gate_mod.gate(gate_mod.load(str(baseline)), m, max_drop)
            bad = [c for c in checks if c["status"] == "FAIL"]
            thr = [c for c in checks if c["kind"] == "throughput"]
            out.append({"gate": "baseline", "ok": not failed and bool(thr),
                        "detail": ("; ".join("%s %s" % (c["metric"], c["detail"]) for c in bad) or
                                   "%d throughput metrics within %.1f %%" % (len(thr), max_drop)) +
                                  ("" if thr else "; no throughput metric to compare")})
    for g in out:
        g["tp"] = n
    return out


def gpu_matrix(a):
    rep = Report("gpu-matrix")
    try:
        sizes = [int(x) for x in a.gpu_matrix.split(",") if x.strip()]
    except ValueError:
        sizes = []
    if not sizes or any(n < 1 or n > 8 for n in sizes):
        print("ci_regression: --gpu-matrix takes TP sizes 1..8, e.g. 1,2,3", file=sys.stderr)
        return None
    if not a.from_json and not a.model:
        print("ci_regression: --gpu-matrix needs --model (or --from-json DIR)", file=sys.stderr)
        return None
    gate_mod = load_gate_module()
    out_dir = Path(a.from_json or a.out_dir)
    if not a.from_json and not a.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    runner = shlex.split(a.runner) if a.runner else []
    if a.preflight and not a.from_json:
        t0 = time.monotonic()
        rc, _, out = run(runner + [a.vegajet, "doctor", "--json"], dry=a.dry_run)
        rep.add("preflight", rc == 0, time.monotonic() - t0, "dry run" if a.dry_run else "vegajet doctor --json exit %d" % rc)
        if rc != 0:
            return rep
    for n in sizes:
        card = out_dir / ("tp%d.json" % n)
        if not a.from_json:
            cmd = runner + [a.vegajet, "bench", a.model, "--tp", str(n), "--gpu", gpu_spec(n), "--json-out", str(card)] + a.bench_arg
            t0 = time.monotonic()
            rc, _, out = run(cmd, log=None if a.dry_run else out_dir / ("tp%d.log" % n), dry=a.dry_run)
            rep.add("bench tp%d" % n, rc == 0, time.monotonic() - t0, "dry run" if a.dry_run else "exit %d" % rc)
            if a.dry_run:
                continue
            if rc != 0:
                continue
        base = (Path(a.baseline_dir) / ("tp%d.json" % n)) if a.baseline_dir else None
        gates = evaluate(card, n, gate_mod, base, a.max_drop)
        for g in gates:
            rep.add("tp%d %s" % (n, g["gate"]), g["ok"], 0.0, g["detail"])
    return rep


# ------------------------------------------------------------------------------------------------ self-test
def self_test():
    gate_mod = load_gate_module()
    good = {"parity": {"code": "32/32", "spec": {"match": 129, "total": 129}}, "spill_bytes": 0,
            "telemetry": {"ue": 0, "de": 0, "tmem_c": 55.0}, "throughput": {"decode_m1_tok_s": 50.0}, "kld": 0.001}
    with tempfile.TemporaryDirectory() as d:
        def card(name, doc):
            p = Path(d) / name
            p.write_text(json.dumps(doc))
            return p
        g = evaluate(card("g.json", good), 1, gate_mod, card("b.json", good))
        assert all(x["ok"] for x in g), g
        bad = dict(good, spill_bytes=4096)
        assert not next(x for x in evaluate(card("s.json", bad), 1, gate_mod) if x["gate"] == "spill")["ok"]
        bad = dict(good, parity={"code": "31/32", "spec": {"match": 129, "total": 129}})
        assert not next(x for x in evaluate(card("p.json", bad), 1, gate_mod) if x["gate"] == "parity")["ok"]
        bad = dict(good, telemetry={"ue": 0, "de": 2})
        assert not next(x for x in evaluate(card("d.json", bad), 1, gate_mod) if x["gate"] == "de")["ok"]
        slow = dict(good, throughput={"decode_m1_tok_s": 45.0})
        assert not next(x for x in evaluate(card("t.json", slow), 1, gate_mod, card("b2.json", good)) if x["gate"] == "baseline")["ok"]
    assert gpu_spec(1) == "0" and gpu_spec(3) == "0-2"
    print("ci_regression self-test: PASS")


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cpu-only", action="store_true", help="CPU stages (GitHub Actions)")
    mode.add_argument("--gpu-matrix", metavar="N[,N...]", help="TP sizes to bench and gate (nightly, GPU runner)")
    mode.add_argument("--self-test", action="store_true")
    ap.add_argument("--stages", default=",".join(CPU_STAGES), help="CPU stages to run (default: all)")
    ap.add_argument("--patch", action="append", default=[], help="patch for the stacked apply check (repeatable, in order)")
    ap.add_argument("--source", default=str(ROOT), help="source tree (default: this checkout)")
    ap.add_argument("--build-root", default=None, help="where the ci-opt / ci-san build trees go (default: <source>/build-ci)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--model", help="model file for --gpu-matrix")
    ap.add_argument("--vegajet", default=str(ROOT / "build/vegajet"), help="vegajet binary (GPU build) for --gpu-matrix")
    ap.add_argument("--out-dir", default="ci-gpu-matrix", help="scorecards and logs of --gpu-matrix")
    ap.add_argument("--baseline-dir", help="baseline scorecards tpN.json for the throughput / KLD / TMem gate")
    ap.add_argument("--from-json", metavar="DIR", help="gate existing tpN.json scorecards, run nothing")
    ap.add_argument("--preflight", action="store_true", help="vegajet doctor --json first (read-only)")
    ap.add_argument("--runner", default="", help="command prefix for every GPU command (lock / queue wrapper)")
    ap.add_argument("--bench-arg", action="append", default=[],
                    help="extra argument for vegajet bench (repeatable; write flags as --bench-arg=--flag)")
    ap.add_argument("--max-drop", type=float, default=2.0, help="allowed throughput drop vs baseline, percent")
    ap.add_argument("--report", help="write the JSON report here")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    a = ap.parse_args(argv)
    a.self_test_inner = os.environ.get("CI_REGRESSION_INNER") == "1"
    if a.self_test:
        try:
            self_test()
        except AssertionError as e:
            print("ci_regression self-test: FAIL %s" % e, file=sys.stderr)
            return 1
        return 0
    a.source = Path(a.source).resolve()
    if a.build_root is None:
        a.build_root = str(a.source / "build-ci")
    if a.jobs < 1:
        ap.error("--jobs must be >= 1")
    rep = cpu_only(a) if a.cpu_only else gpu_matrix(a)
    if rep is None:
        return 2
    rep.write(a.report)
    n_fail = sum(not s["ok"] for s in rep.doc["stages"])
    print("CI %s: %s (%d stage(s), %d failing)" % (rep.doc["mode"], "PASS" if rep.doc["ok"] else "FAIL",
                                                   len(rep.doc["stages"]), n_fail))
    return 0 if rep.doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Release regression gate for VegaJet benchmark scorecards (stdlib only).

  regression_gate.py BASELINE CANDIDATE [--max-drop 2.0] [--kld-max 0.045] [--tmem-max 65] [--json]
  regression_gate.py --release-check [FILE ...]      (default: README.md and release/docs/*.md)
  regression_gate.py --self-test

Exit codes: 0 pass, 1 gate failed, 2 unusable input (unreadable, or no recognized metrics).

Scorecards are JSON or text. Every metric is classified by its name:
  throughput  name or unit says tok/s (tok_s, tokens_per_s, tps)  candidate may not drop more than --max-drop %
                                                                   vs baseline; a baseline throughput metric missing
                                                                   from the candidate fails too
  kld         name contains "kld"                                  candidate value must be <= --kld-max
  parity      name contains "parity" (also "first_diff")           "a/b" must have a == b; first_diff must be -1
  spill       name contains "spill" or "private_segment"           must be 0
  tmem        name contains "tmem"                                 must be < --tmem-max (°C)
  ue / de     name is ue / de / uncorrectable / deferred (+_count) must be 0
JSON: nested objects are flattened to dotted names; lists of numbers are repeated observations (median for
throughput, max for KLD / TMem / spill / UE / DE). Text: "name: value" or "name = value" lines, VegaJet bench lines
"[BENCH tg128 raw M=1] 50.0 tok/s", "[PARITY code vs Tier-1 llama.cpp] 32/32", and inline telemetry tokens such as
"KLD 0.0009", "TMem=54.0C", "UE 0 DE 0", "spill 0 B". Only the candidate's absolute gates use candidate values;
throughput compares both files. Nothing is inferred: a metric that is absent is reported as absent.
"""
import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

MARKER = "[FILL FROM FINAL SCORECARD]"
NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def classify(name, unit=""):
    n = name.lower()
    u = unit.lower()
    if "kld" in n:
        return "kld"
    if "parity" in n or "first_diff" in n:
        return "parity"
    if "spill" in n or "private_segment" in n:
        return "spill"
    if "tmem" in n:
        return "tmem"
    base = re.sub(r"[^a-z]", "", n.split(".")[-1])
    if base in ("ue", "uecount", "uncorrectable", "uncorrectableerrors"):
        return "ue"
    if base in ("de", "decount", "deferred", "deferrederrors"):
        return "de"
    if "tok/s" in u or "tok/s" in n or re.search(r"tok(?:en)?s?_?(?:per_?)?s(?:ec)?\b|\btps\b|tok_s", n):
        return "throughput"
    return None


def with_unit(name, unit):
    """Keeps the name; appends ' tok/s' only when the unit (not the name) says it is a throughput."""
    if classify(name) is None and "tok/s" in (unit or "").lower():
        return name + " tok/s"
    return name


def add(metrics, name, value):
    metrics.setdefault(name.strip(), []).append(value)


def parse_parity(v):
    """'32/32' or {'match': 32, 'total': 32} or an int first_diff -> (match, total) or first_diff int."""
    if isinstance(v, str):
        m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*(?:\(.*\))?\s*", v)
        if m:
            return (int(m[1]), int(m[2]))
        try:
            return int(float(v))
        except ValueError:
            return None
    if isinstance(v, dict):
        a, b = v.get("match", v.get("matched")), v.get("total")
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return (int(a), int(b))
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return int(v)
    return None


def flatten(obj, prefix, metrics):
    if isinstance(obj, dict):
        if prefix and ("match" in obj or "matched" in obj) and "total" in obj:
            add(metrics, prefix, obj)
            return
        for k, v in obj.items():
            flatten(v, f"{prefix}.{k}" if prefix else str(k), metrics)
    elif isinstance(obj, list):
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in obj):
            for x in obj:
                add(metrics, prefix, float(x))
        else:
            for i, x in enumerate(obj):
                flatten(x, f"{prefix}[{i}]", metrics)
    elif isinstance(obj, bool):
        return
    elif isinstance(obj, (int, float)):
        add(metrics, prefix, float(obj))
    elif isinstance(obj, str):
        m = re.fullmatch(r"\s*(" + NUM + r")\s*([A-Za-z/°%]*)\s*", obj)
        if m:
            add(metrics, with_unit(prefix, m[2]), float(m[1]))
        else:
            add(metrics, prefix, obj)


def parse_text(text, metrics):
    for line in text.splitlines():
        b = re.search(r"\[BENCH ([^\]]+)\]\s*(" + NUM + r")\s*tok/s", line)
        if b:
            add(metrics, f"bench {b[1].strip()} tok/s", float(b[2]))
            continue
        p = re.search(r"\[PARITY ([^\]]*)\]\s*(\d+\s*/\s*\d+)", line)
        if p:
            add(metrics, f"parity {p[1].strip()}", p[2])
            continue
        kv = re.match(r"^\s*([A-Za-z][\w .:/()\[\]-]*?)\s*[:=]\s*(.+?)\s*$", line)
        if kv and not re.search(r"\b(KLD|TMem|UE|DE)\b\s*[=:]?\s*" + NUM + r".*\b(KLD|TMem|UE|DE|spill)\b", line):
            name, val = kv[1], kv[2]
            unit = re.match(r"(" + NUM + r")\s*(\S*)", val)
            if classify(name) == "parity":
                add(metrics, name, val)
            elif unit:
                add(metrics, with_unit(name, unit[2]), float(unit[1]))
            continue
        for key, rx in (("kld", r"\bKLD\b\s*[=:]?\s*(" + NUM + ")"), ("tmem", r"\bTMem\b\s*[=:]?\s*(" + NUM + ")"),
                        ("ue", r"\bUE\b\s*[=:]?\s*(\d+)"), ("de", r"\bDE\b\s*[=:]?\s*(\d+)"),
                        ("spill", r"\bspill\b\s*[=:]?\s*(\d+)")):
            for m in re.finditer(rx, line):
                add(metrics, key, float(m[1]))


def load(path):
    text = Path(path).read_text()
    metrics = {}
    stripped = text.lstrip()
    parsed = None
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = None  # e.g. a log whose first line is "[BENCH ...]"
    if parsed is not None:
        flatten(parsed, "", metrics)
    else:
        parse_text(text, metrics)
    return {k: v for k, v in metrics.items() if k}


def numeric(vals):
    xs = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
    return xs


def gate(base, cand, max_drop=2.0, kld_max=0.045, tmem_max=65.0):
    """Returns (checks, failed). Each check: dict(metric, kind, status, detail)."""
    checks = []

    def rec(metric, kind, ok, detail):
        checks.append({"metric": metric, "kind": kind, "status": "PASS" if ok else "FAIL", "detail": detail})

    for name in sorted(set(base) | set(cand)):
        kind = classify(name)
        if kind != "throughput":
            continue
        b, c = numeric(base.get(name, [])), numeric(cand.get(name, []))
        if b and not c:
            rec(name, kind, False, "missing from candidate")
            continue
        if not b:
            checks.append({"metric": name, "kind": kind, "status": "INFO", "detail": f"new metric {statistics.median(c):.6g}"})
            continue
        bm, cm = statistics.median(b), statistics.median(c)
        if bm <= 0:
            checks.append({"metric": name, "kind": kind, "status": "INFO", "detail": "baseline <= 0, not comparable"})
            continue
        delta = 100.0 * (cm - bm) / bm
        rec(name, kind, delta >= -max_drop, f"{bm:.6g} -> {cm:.6g} ({delta:+.2f}%, limit -{max_drop:g}%)")
    for name, vals in sorted(cand.items()):
        kind = classify(name)
        xs = numeric(vals)
        if kind == "kld" and xs:
            rec(name, kind, max(xs) <= kld_max, f"max {max(xs):.6g} (limit {kld_max:g})")
        elif kind == "spill" and xs:
            rec(name, kind, max(xs) == 0, f"max {max(xs):.6g} (must be 0)")
        elif kind == "tmem" and xs:
            rec(name, kind, max(xs) < tmem_max, f"max {max(xs):.6g} C (limit < {tmem_max:g} C)")
        elif kind in ("ue", "de") and xs:
            rec(name, kind, max(xs) == 0, f"max {max(xs):.6g} (must be 0)")
        elif kind == "parity":
            for v in vals:
                p = parse_parity(v)
                if isinstance(p, tuple):
                    rec(name, kind, p[0] == p[1] and p[1] > 0, f"{p[0]}/{p[1]} (must be exact)")
                elif isinstance(p, int) and "first_diff" in name.lower():
                    rec(name, kind, p == -1, f"first_diff {p} (must be -1)")
                elif p is None:
                    rec(name, kind, False, f"unparseable parity value {v!r}")
    failed = any(c["status"] == "FAIL" for c in checks)
    return checks, failed


def release_check(paths):
    """Returns (marker hits, unreadable files)."""
    hits, errors = [], []
    for p in paths:
        try:
            for i, line in enumerate(Path(p).read_text().splitlines(), 1):
                if MARKER in line:
                    hits.append(f"{p}:{i}: {line.strip()}")
        except OSError as e:
            errors.append(f"{p}: unreadable ({e.strerror})")
    return hits, errors


def default_release_files():
    files = [Path("README.md")] if Path("README.md").exists() else []
    files += sorted(Path("release/docs").glob("*.md")) if Path("release/docs").is_dir() else []
    return files


def self_test():
    import tempfile
    base = {"decode_tok_s": [50.0, 51.0, 52.0], "bench pp512 M=256 tok/s": [100.0], "kld": [0.001]}
    cand = {"decode_tok_s": [50.5], "bench pp512 M=256 tok/s": [99.0], "kld": [0.002], "tmem_c": [54.0],
            "parity.code": ["32/32"], "spill_bytes": [0.0], "ue": [0.0], "de": [0.0]}
    checks, failed = gate(base, cand)
    assert not failed, checks
    cand2 = dict(cand, decode_tok_s=[49.0])
    assert gate(base, cand2)[1]                              # -3.9 % vs median 51
    assert not gate(base, dict(cand, decode_tok_s=[50.0]))[1]   # -1.96 % is within the 2 % budget
    for k, v in (("kld", [0.0451]), ("tmem_c", [65.0]), ("parity.code", ["31/32"]), ("spill_bytes", [4.0]),
                 ("ue", [1.0]), ("de", [2.0])):
        assert gate(base, dict(cand, **{k: v}))[1], k
    assert gate(base, {k: v for k, v in cand.items() if k != "decode_tok_s"})[1]   # missing throughput
    assert classify("first_diff") == "parity" and classify("x.UE") == "ue" and classify("prefill", "tok/s") == "throughput" and classify("ms_per_tok") is None
    with tempfile.TemporaryDirectory() as d:
        t = Path(d) / "c.txt"
        t.write_text("[BENCH tg128 raw M=1] 50.0 tok/s (20.0 ms/tok)\n[PARITY code vs Tier-1 llama.cpp] 32/32\n"
                     "[TELEM] TMem=54.5C UE 0 DE 0 KLD 0.0009 spill 0\ndecode_tok_s: 51.2 tok/s\n")
        m = load(t)
        assert m["bench tg128 raw M=1 tok/s"] == [50.0] and m["tmem"] == [54.5] and m["kld"] == [0.0009], m
        assert m["parity code vs Tier-1 llama.cpp"] == ["32/32"] and m["ue"] == [0.0] and m["spill"] == [0.0], m
        r = Path(d) / "README.md"
        r.write_text("| a | " + MARKER + " |\nok\n")
        assert len(release_check([r])[0]) == 1 and release_check([Path(d) / "missing.md"])[1]
    print("regression_gate self-test: PASS")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline", nargs="?")
    ap.add_argument("candidate", nargs="?")
    ap.add_argument("--max-drop", type=float, default=2.0, help="allowed throughput drop in percent (default 2)")
    ap.add_argument("--kld-max", type=float, default=0.045)
    ap.add_argument("--tmem-max", type=float, default=65.0, help="TMem must stay below this (°C)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--release-check", nargs="*", metavar="FILE")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        self_test()
        return 0
    if a.release_check is not None:
        files = [Path(p) for p in a.release_check] or default_release_files()
        if not files:
            print("release check: no files to scan", file=sys.stderr)
            return 2
        hits, errors = release_check(files)
        if a.json:
            print(json.dumps({"files": [str(f) for f in files], "markers": hits, "errors": errors,
                              "ready": not hits and not errors}, indent=2))
        else:
            for h in hits + errors:
                print(h)
            print(f"RELEASE CHECK: {'NOT READY' if hits or errors else 'READY'} "
                  f"({len(hits)} unfilled marker(s) in {len(files)} file(s), {len(errors)} unreadable)")
        return 2 if errors else 1 if hits else 0
    if not a.baseline or not a.candidate:
        ap.error("BASELINE and CANDIDATE are required (or --release-check / --self-test)")
    try:
        base, cand = load(a.baseline), load(a.candidate)
    except (OSError, ValueError) as e:
        print(f"GATE ERROR: cannot read scorecard: {e}", file=sys.stderr)
        return 2
    for label, card in (("baseline", base), ("candidate", cand)):
        if not any(classify(k) for k in card):
            print(f"GATE ERROR: no recognized metrics in the {label} scorecard", file=sys.stderr)
            return 2
    checks, failed = gate(base, cand, a.max_drop, a.kld_max, a.tmem_max)
    if not [c for c in checks if c["status"] != "INFO"]:
        print("GATE ERROR: no gateable metrics recognized in the candidate", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps({"pass": not failed, "checks": checks}, indent=2))
    else:
        w = max(len(c["metric"]) for c in checks)
        for c in checks:
            print(f"{c['status']:<4}  {c['kind']:<10}  {c['metric']:<{w}}  {c['detail']}")
        print(f"GATE: {'FAIL' if failed else 'PASS'} ({sum(c['status'] == 'FAIL' for c in checks)} failing of {len(checks)} checks)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

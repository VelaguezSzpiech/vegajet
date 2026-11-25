#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Diff two VegaJet bench JSON scorecards and gate regressions (stdlib only).

  bench_compare.py BASELINE CANDIDATE [--max-drop PCT] [--kld-max X] [--tmem-max C] [--json]
  bench_compare.py --self-test

Inputs are JSON documents, typically written by `vegajet bench --json-out` (schema vegajet.bench.v1). Nested objects
are flattened to dotted names; a list of numbers is one observation set, reduced with the median for throughput and
the maximum for every other gated kind. Metric kinds are classified by name, like tools/regression_gate.py:

  throughput   name says tok/s (tok_s, tokens_per_s, tok/s, tps)   candidate may not drop more than --max-drop %
  kld          name contains "kld"                                 candidate must be <= --kld-max
  parity       name contains "parity", or first_diff               match must equal total; first_diff must be -1
  spill        name contains "spill" or "private_segment"          candidate must be 0
  tmem         name contains "tmem"                                candidate must be < --tmem-max
  ue / de      leaf is ue / de (+ _count, uncorrectable, deferred) candidate must be 0
  info         everything else                                     reported, never gated

Unlike regression_gate.py this tool never parses bench logs, README cells or release docs; it only diffs JSON, so it is
safe to run against two captured scorecards anywhere. A baseline throughput metric absent from the candidate is a
regression; the other gated kinds are absolute checks that only apply when the candidate measured them (a null or
missing value means "not measured", as in the vegajet.bench.v1 schema). Exit codes: 0 pass, 1 regression, 2 unusable
input.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

SCHEMA = "vegajet.bench_compare.v1"


def kind_of(name):
    n = name.lower()
    leaf = n.rsplit(".", 1)[-1]
    if "kld" in n:
        return "kld"
    if "first_diff" in n or "parity" in n:
        return "parity"
    if "spill" in n or "private_segment" in n:
        return "spill"
    if "tmem" in n:
        return "tmem"
    if leaf in ("ue", "ue_count", "uncorrectable", "uncorrectable_errors"):
        return "ue"
    if leaf in ("de", "de_count", "deferred", "deferred_errors"):
        return "de"
    if "tok_s" in n or "tok/s" in n or "tokens_per_s" in n or "tokens_per_sec" in n or leaf == "tps":
        return "throughput"
    return "info"


def flatten(obj, prefix, out):
    if isinstance(obj, dict):
        for key, value in obj.items():
            flatten(value, prefix + "." + str(key) if prefix else str(key), out)
    elif isinstance(obj, list):
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in obj) and obj:
            out[prefix] = list(obj)
        else:
            for index, value in enumerate(obj):
                flatten(value, prefix + "[" + str(index) + "]", out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = obj
    elif isinstance(obj, (str, bool)) or obj is None:
        out[prefix] = obj


def reduce_value(value, kind):
    """Observation lists: median for throughput, max elsewhere. Scalars pass through."""
    if isinstance(value, list):
        if kind == "throughput":
            return statistics.median(value)
        return max(value)
    return value


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def fmt(value):
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def compare_docs(base_doc, cand_doc, max_drop=2.0, kld_max=0.045, tmem_max=65.0):
    base, cand = {}, {}
    flatten(base_doc, "", base)
    flatten(cand_doc, "", cand)
    rows, fails = [], 0
    for name in sorted(set(base) | set(cand)):
        kind = kind_of(name)
        b, c = base.get(name), cand.get(name)
        bv, cv = reduce_value(b, kind) if name in base else None, reduce_value(c, kind) if name in cand else None
        row = {"metric": name, "kind": kind, "base": bv, "candidate": cv, "delta_pct": None,
               "status": "info", "detail": ""}
        if kind == "parity" and name.endswith(".match"):
            total_key = name[:-len("match")] + "total"
            total = reduce_value(cand[total_key], "parity") if total_key in cand else None
            if name not in cand or cv is None or total is None:
                row.update(status="info", detail="parity not measured in candidate")
            elif not numeric(cv) or not numeric(total):
                row.update(status="fail", detail="parity match/total not numeric")
            elif cv != total:
                row.update(status="fail", detail=f"match {fmt(cv)} != total {fmt(total)}")
            else:
                row.update(status="pass", detail=f"{fmt(cv)}/{fmt(total)}")
        elif kind == "parity" and name.endswith(".total"):
            row.update(status="info", detail="paired with .match")
        elif kind == "parity" and name.endswith("first_diff"):
            if name not in cand or cv is None:
                row.update(status="info", detail="first_diff not measured in candidate")
            elif not numeric(cv):
                row.update(status="fail", detail="first_diff is not numeric")
            elif cv != -1:
                row.update(status="fail", detail=f"first_diff {fmt(cv)} != -1")
            else:
                row.update(status="pass", detail="-1")
        elif kind in ("throughput", "kld", "spill", "tmem", "ue", "de"):
            if name not in cand or cv is None:
                # A scorecard writes null for a metric the run did not produce. A lost throughput baseline is a
                # regression; other gated kinds are absolute checks that only apply when measured (like regression_gate).
                if kind == "throughput":
                    row.update(status="fail", detail="gated throughput metric missing from candidate")
                else:
                    row.update(status="info", detail="not measured in candidate")
            elif not numeric(cv):
                row.update(status="fail", detail="candidate value is not numeric")
            elif kind == "throughput":
                if name not in base or not numeric(bv) or bv == 0:
                    row.update(status="info", detail="no baseline throughput to compare")
                else:
                    pct = 100.0 * (cv - bv) / bv
                    row["delta_pct"] = pct
                    row.update(status="pass" if pct >= -max_drop else "fail",
                               detail=f"{pct:+.2f}% vs baseline (floor -{max_drop:.2f}%)")
            elif kind == "kld":
                row.update(status="pass" if cv <= kld_max else "fail",
                           detail=f"<= {kld_max:.4f}" if cv <= kld_max else f"{fmt(cv)} > {kld_max:.4f}")
            elif kind == "tmem":
                row.update(status="pass" if cv < tmem_max else "fail",
                           detail=f"< {tmem_max:.1f} C" if cv < tmem_max else f"{fmt(cv)} C >= {tmem_max:.1f} C")
            elif kind == "spill":
                row.update(status="pass" if cv == 0 else "fail", detail="0 B" if cv == 0 else f"{fmt(cv)} B")
            else:  # ue / de
                row.update(status="pass" if cv == 0 else "fail", detail="0" if cv == 0 else fmt(cv))
            if name in base and numeric(bv) and numeric(cv) and bv != 0 and row["delta_pct"] is None:
                row["delta_pct"] = 100.0 * (cv - bv) / bv
        else:
            if numeric(bv) and numeric(cv) and bv != 0:
                row["delta_pct"] = 100.0 * (cv - bv) / bv
            row["detail"] = "descriptive"
        if row["status"] == "fail":
            fails += 1
        rows.append(row)
    return {"schema": SCHEMA, "ok": fails == 0, "checks": rows, "fails": fails,
            "thresholds": {"max_drop_pct": max_drop, "kld_max": kld_max, "tmem_max_c": tmem_max}}


def text_report(report):
    lines = ["| Metric | Kind | Base | Candidate | Delta | Verdict |", "|---|---|---:|---:|---:|---|"]
    for row in report["checks"]:
        delta = f"{row['delta_pct']:+.2f}%" if row["delta_pct"] is not None else "n/a"
        verdict = row["status"].upper()
        if row["status"] == "fail" and row["detail"]:
            verdict += " (" + row["detail"] + ")"
        lines.append(f"| {row['metric']} | {row['kind']} | {fmt(row['base'])} | {fmt(row['candidate'])} | "
                     f"{delta} | {verdict} |")
    lines.append("")
    lines.append(f"summary: {report['fails']} regression(s) -> {'PASS' if report['ok'] else 'FAIL'}")
    return "\n".join(lines)


def load_doc(path):
    with open(path, "r", encoding="utf-8") as handle:
        doc = json.load(handle)
    if not isinstance(doc, dict):
        raise ValueError("top-level JSON value is not an object")
    return doc


def self_test():
    def doc(**overrides):
        value = {
            "schema": "vegajet.bench.v1",
            "throughput": {"raw_m1_tok_s": 134.0, "pp512_w4a8_tok_s": 1000.0},
            "kld": {"dot8_prefill_kld": 0.004},
            "parity": {"code_vs_tier1": {"match": 32, "total": 32, "verdict": "exact"}},
            "spill": {"private_segment_bytes_max": 0},
            "telemetry": {"tmem_max_c": 61.0},
            "ras": {"gpu0": {"umc": {"ue": 0, "de": 0}}},
            "observations": {"raw_m1_tok_s": [133.0, 134.0, 135.0]},
        }
        value.update(overrides)
        return value

    healthy = compare_docs(doc(), doc())
    assert healthy["ok"], text_report(healthy)
    assert any(r["kind"] == "throughput" and r["status"] == "pass" for r in healthy["checks"])
    # observation lists: median for throughput
    assert next(r for r in healthy["checks"] if r["metric"] == "observations.raw_m1_tok_s")["candidate"] == 134.0

    slow = doc()
    slow["throughput"] = {"raw_m1_tok_s": 120.0, "pp512_w4a8_tok_s": 1000.0}
    r = compare_docs(doc(), slow)
    assert not r["ok"]
    failed = {row["metric"] for row in r["checks"] if row["status"] == "fail"}
    assert failed == {"throughput.raw_m1_tok_s"}, failed

    r = compare_docs(doc(), doc(kld={"dot8_prefill_kld": 0.06}))
    assert not r["ok"] and any(x["kind"] == "kld" and x["status"] == "fail" for x in r["checks"])

    r = compare_docs(doc(), doc(parity={"code_vs_tier1": {"match": 30, "total": 32}}))
    assert not r["ok"] and any(x["kind"] == "parity" and x["status"] == "fail" for x in r["checks"])

    r = compare_docs(doc(), doc(spill={"private_segment_bytes_max": 256}))
    assert not r["ok"] and any(x["kind"] == "spill" and x["status"] == "fail" for x in r["checks"])

    r = compare_docs(doc(), doc(telemetry={"tmem_max_c": 66.0}))
    assert not r["ok"] and any(x["kind"] == "tmem" and x["status"] == "fail" for x in r["checks"])

    r = compare_docs(doc(), doc(ras={"gpu0": {"umc": {"ue": 1, "de": 0}}}))
    assert not r["ok"] and any(x["kind"] == "ue" and x["status"] == "fail" for x in r["checks"])

    missing = doc()
    del missing["throughput"]["raw_m1_tok_s"]
    r = compare_docs(doc(), missing)
    assert not r["ok"] and any(x["metric"] == "throughput.raw_m1_tok_s" and x["status"] == "fail"
                               for x in r["checks"])

    # A scorecard writes null for a metric the run did not produce: gated metrics treat that as missing.
    nulled = doc()
    nulled["throughput"]["raw_m1_tok_s"] = None
    r = compare_docs(doc(), nulled)
    assert not r["ok"] and any(x["metric"] == "throughput.raw_m1_tok_s" and x["status"] == "fail"
                               for x in r["checks"])

    # A small drop inside the default floor passes; first_diff must be -1.
    tiny = doc()
    tiny["throughput"]["raw_m1_tok_s"] = 132.0  # -1.5 %
    assert compare_docs(doc(), tiny)["ok"]
    r = compare_docs(doc(), doc(parity={"code_vs_tier1": {"match": 32, "total": 32, "first_diff": 5}}))
    assert not r["ok"]

    # Info-only extras never fail and JSON round-trips.
    r = compare_docs(doc(), doc(notes="hello", extra={"count": 3}))
    assert r["ok"]
    assert json.loads(json.dumps(r))["schema"] == SCHEMA
    print("bench_compare self-test: PASS")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", nargs="?", type=Path)
    parser.add_argument("candidate", nargs="?", type=Path)
    parser.add_argument("--max-drop", type=float, default=2.0, help="allowed throughput drop in percent")
    parser.add_argument("--kld-max", type=float, default=0.045)
    parser.add_argument("--tmem-max", type=float, default=65.0)
    parser.add_argument("--json", action="store_true", help="print the machine-readable report")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    if not args.baseline or not args.candidate:
        parser.error("BASELINE and CANDIDATE are required (or --self-test)")
    try:
        report = compare_docs(load_doc(args.baseline), load_doc(args.candidate),
                              args.max_drop, args.kld_max, args.tmem_max)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2) if args.json else text_report(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

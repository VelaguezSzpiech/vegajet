#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Bench-log scorecard: parse `vegajet bench` output, account for host overhead, fill the README performance table.

Standard library only. Reads one or more log files (or stdin with "-"), never the GPU.

Lines understood (formats as printed by the engine at the time of writing):
  [BENCH tg128 raw M=1] 33.10 tok/s (30.21 ms/tok) | eff_HBM2=612.0 GB/s (71.2% of 860.0 GB/s calibrated) | ...
  [BENCH pp2048 M=8   ] 410.22 tok/s (WY_GDN=on, ...)
  [BENCH tg128 Suffix ] / [BENCH tg128 MTP K=2] / [BENCH tg128 Hybrid ] / [BENCH tg128 DFlash2 K=8 code ]
        ... X tok/s (Nx vs raw ...) | ... lossless=a/b ...
  [PARITY code  vs Tier-1 llama.cpp] 32/32 exact greedy tokens -> PASS
  [TELEM] power1_input: worst 10 s rolling avg=... -> PASS
  [PROFILE in-situ M=1] base 30.21 ms/tok | gdn_qkv=1.20 ... | sum=24.00 ms (rest: ...)
  [PROFILE M=1 GPU0 48L] gdn_qkv=1.10ms ... | sum=22.10ms

Host-overhead accounting:
  * From the in-situ profile: unattributed = base - sum (AR, conv, recurrence, embed, LM head and host time that the
    stage sweep does not isolate).
  * From device timestamps (proposed format; the engine does not emit it yet): kernels read s_memrealtime at entry
    and exit and the host prints one line per kernel and one wall line per token:
        [KSTAMP] gpu=0 tok=17 name=moe_gu start=123456789 end=123459000
        [KWALL]  gpu=0 tok=17 start=123450000 end=123790000
    Ticks are converted with --rtc-mhz (default 100; check the device's timestamp frequency before trusting
    absolute numbers). Per (gpu, token): host_gap = T_wall - sum(T_kernels), plus the overlap-aware
    idle = T_wall - |union of kernel intervals|. Reported as mean / p50 / p99 over tokens.

Bench JSON: `vegajet bench --json-out` scorecards (schema "vegajet.bench.v1", core/bench_json) are accepted wherever a log
is, and overlay what the logs gave; with both W4A8 and dot8 prefill numbers in one scorecard, the README prefill cell
reports W4A8 (the lossless default).

Fill mode replaces the `[FILL FROM FINAL SCORECARD]` cells of a README performance table, and adds a provenance line
(commit, flags, calibrated HBM2 ceiling) under the table, because the README requires every number to name its run.
Cells whose metric is missing in the log are left as placeholders and listed; --strict turns that into exit 1.
"""
import argparse
import json
import re
import statistics
import sys

PLACEHOLDER = "[FILL FROM FINAL SCORECARD]"

RE_RAW = re.compile(r"\[BENCH tg128 raw M=1\]\s+([\d.]+) tok/s \(([\d.]+) ms/tok\)"
                    r"(?:.*?eff_HBM2=([\d.]+) GB/s \(([\d.]+)% of ([\d.]+) GB/s calibrated\))?"
                    r"(?:.*?real_read=([\d.]+) GB/tok)?(?:.*?disp/tok=(\d+))?")
RE_PP = re.compile(r"\[BENCH pp(\d+)\s+M=\s*(\d+)\s*\]\s+([\d.]+) tok/s")
RE_SPEC = re.compile(r"\[BENCH tg(\d+) (Suffix|MTP K=\d+|Hybrid|DFlash2 K=\d+ \w+|DFlash2\+Hyb \w+)\s*\]\s+([\d.]+) tok/s"
                     r"(?:\s+\(([\d.]+)x vs raw)?")
RE_LOSSLESS = re.compile(r"lossless=(\d+)/(\d+)")
RE_PARITY = re.compile(r"\[PARITY code\s+vs Tier-1 llama\.cpp\]\s+(\d+)/(\d+) exact greedy tokens(?: -> (\w+))?")
RE_TELEM = re.compile(r"\[TELEM\].*?rolling avg=([\d.]+) W.*?peak=([\d.]+) W.*?max_TMem=([\d.]+) C.*?-> (\w+)")
RE_INSITU = re.compile(r"\[PROFILE in-situ M=1\] base ([\d.]+) ms/tok \|(.*?)\| sum=([\d.]+) ms")
RE_GPU0 = re.compile(r"\[PROFILE M=1 GPU0 48L\](.*?)\| sum=([\d.]+)ms")
RE_KV = re.compile(r"(\w[\w+]*)=([\d.]+)(?:ms)?")
RE_KSTAMP = re.compile(r"\[KSTAMP\]\s+gpu=(\d+)\s+tok=(\d+)\s+name=(\S+)\s+start=(\d+)\s+end=(\d+)")
RE_KWALL = re.compile(r"\[KWALL\]\s+gpu=(\d+)\s+tok=(\d+)\s+start=(\d+)\s+end=(\d+)")


def _pct(xs, p):
    s = sorted(xs)
    if not s:
        return 0.0
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _union(intervals):
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def parse(lines, rtc_mhz=100.0):
    r = {"raw": None, "prefill": {}, "spec": [], "parity": None, "telem": None, "insitu": None, "gpu0": None,
         "kstamp": None, "warnings": []}
    kern, wall = {}, {}
    for ln in lines:
        m = RE_RAW.search(ln)
        if m:
            r["raw"] = {"tps": float(m.group(1)), "ms_per_tok": float(m.group(2)),
                        "eff_hbm2_gbs": float(m.group(3)) if m.group(3) else None,
                        "hbm2_pct": float(m.group(4)) if m.group(4) else None,
                        "hbm2_ceiling_gbs": float(m.group(5)) if m.group(5) else None,
                        "real_read_gb_per_tok": float(m.group(6)) if m.group(6) else None,
                        "dispatches_per_tok": int(m.group(7)) if m.group(7) else None}
            continue
        m = RE_PP.search(ln)
        if m:
            r["prefill"][int(m.group(1))] = {"m": int(m.group(2)), "tps": float(m.group(3))}
            continue
        m = RE_SPEC.search(ln)
        if m:
            ll = RE_LOSSLESS.search(ln)
            r["spec"].append({"tokens": int(m.group(1)), "name": m.group(2), "tps": float(m.group(3)),
                              "speedup": float(m.group(4)) if m.group(4) else None,
                              "lossless": [int(ll.group(1)), int(ll.group(2))] if ll else None})
            continue
        m = RE_PARITY.search(ln)
        if m:
            r["parity"] = {"match": int(m.group(1)), "total": int(m.group(2)), "verdict": m.group(3)}
            continue
        m = RE_TELEM.search(ln)
        if m:
            r["telem"] = {"rolling_avg_w": float(m.group(1)), "peak_w": float(m.group(2)),
                          "max_tmem_c": float(m.group(3)), "verdict": m.group(4)}
            continue
        m = RE_INSITU.search(ln)
        if m:
            base, ssum = float(m.group(1)), float(m.group(3))
            stages = {k: float(v) for k, v in RE_KV.findall(m.group(2))}
            r["insitu"] = {"base_ms": base, "stages_ms": stages, "sum_ms": ssum, "unattributed_ms": round(base - ssum, 4),
                           "unattributed_pct": round(100.0 * (base - ssum) / base, 2) if base else None}
            continue
        m = RE_GPU0.search(ln)
        if m:
            r["gpu0"] = {"kernels_ms": {k: float(v) for k, v in RE_KV.findall(m.group(1))}, "sum_ms": float(m.group(2))}
            continue
        m = RE_KSTAMP.search(ln)
        if m:
            g, t, name, s, e = int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4)), int(m.group(5))
            if e < s:
                r["warnings"].append("KSTAMP end < start (gpu %d tok %d %s): counter wrap or bad stamp, skipped" % (g, t, name))
                continue
            kern.setdefault((g, t), []).append((s, e, name))
            continue
        m = RE_KWALL.search(ln)
        if m:
            g, t, s, e = (int(m.group(i)) for i in range(1, 5))
            wall[(g, t)] = (s, e)
    if kern or wall:
        r["kstamp"] = _account(kern, wall, rtc_mhz, r["warnings"])
    return r


def _account(kern, wall, rtc_mhz, warnings):
    us = 1.0 / rtc_mhz  # microseconds per tick
    per_gpu = {}
    by_kernel = {}
    for key in sorted(wall):
        g, t = key
        ws, we = wall[key]
        ks = kern.get(key, [])
        if not ks:
            warnings.append("KWALL without KSTAMP lines (gpu %d tok %d)" % key)
        outside = [k for k in ks if k[0] < ws or k[1] > we]
        if outside:
            warnings.append("%d kernel stamps outside the wall window (gpu %d tok %d)" % (len(outside), g, t))
        t_wall = (we - ws) * us
        t_sum = sum(e - s for s, e, _ in ks) * us
        t_union = _union([(s, e) for s, e, _ in ks]) * us
        for s, e, name in ks:
            by_kernel.setdefault((g, name), []).append((e - s) * us)
        d = per_gpu.setdefault(g, {"wall": [], "sum": [], "gap": [], "idle": []})
        d["wall"].append(t_wall); d["sum"].append(t_sum); d["gap"].append(t_wall - t_sum); d["idle"].append(t_wall - t_union)
    for key in kern:
        if key not in wall:
            warnings.append("KSTAMP lines without KWALL (gpu %d tok %d): not accounted" % key)
    out = {"rtc_mhz": rtc_mhz, "gpus": {}}
    for g, d in sorted(per_gpu.items()):
        n = len(d["wall"])
        summ = lambda xs: {"mean": round(statistics.fmean(xs), 3), "p50": round(_pct(xs, 0.5), 3), "p99": round(_pct(xs, 0.99), 3)}
        top = sorted(((name, sum(v) / n) for (gg, name), v in by_kernel.items() if gg == g), key=lambda x: -x[1])
        out["gpus"][str(g)] = {"tokens": n, "wall_us": summ(d["wall"]), "kernel_sum_us": summ(d["sum"]),
                               "host_gap_us": summ(d["gap"]), "idle_us": summ(d["idle"]),
                               "host_gap_pct": round(100.0 * sum(d["gap"]) / sum(d["wall"]), 2) if sum(d["wall"]) else None,
                               "top_kernels_us_per_tok": [[k, round(v, 3)] for k, v in top[:10]]}
    return out


BENCH_SCHEMA = "vegajet.bench.v1"


def from_bench_json(doc, r=None):
    """Overlays a vegajet.bench.v1 scorecard onto a parse() result (a fresh one when r is None)."""
    if doc.get("schema") != BENCH_SCHEMA:
        raise ValueError("not a %s scorecard" % BENCH_SCHEMA)
    r = r if r is not None else parse([])
    t, d = doc.get("throughput") or {}, doc.get("decode") or {}
    if t.get("raw_m1_tok_s") is not None:
        r["raw"] = {"tps": t["raw_m1_tok_s"], "ms_per_tok": d.get("raw_m1_ms_per_tok"),
                    "eff_hbm2_gbs": d.get("eff_hbm2_gbs"), "hbm2_pct": d.get("hbm2_pct_of_ceiling"),
                    "hbm2_ceiling_gbs": d.get("hbm2_ceiling_gbs"), "real_read_gb_per_tok": d.get("real_read_gb_per_tok"),
                    "dispatches_per_tok": d.get("dispatches_per_tok")}
    pm = doc.get("prefill_batch_m") or {}
    for n in (512, 2048):
        for path in ("w4a8", "dot8"):  # W4A8 first: the lossless default wins when both were measured
            v = t.get("pp%d_%s_tok_s" % (n, path))
            if v is not None and (path == "w4a8" or n not in r["prefill"] or r["prefill"][n].get("_from") != "json"):
                r["prefill"][n] = {"m": pm.get("pp%d" % n), "tps": v, "_from": "json"}
                break
    spec_detail, parity = doc.get("speculative") or {}, doc.get("parity") or {}
    specs = []
    for key, tps in (t.get("spec") or {}).items():
        if tps is None or not key.endswith("_tok_s"):
            continue
        k = key[:-len("_tok_s")]
        det = spec_detail.get(k) or {}
        ll = parity.get(k + "_lossless")
        specs.append({"tokens": int(det.get("tokens") or 128), "name": det.get("label") or k, "tps": tps,
                      "speedup": det.get("speedup_vs_raw"), "lossless": [ll["match"], ll["total"]] if ll else None})
    if specs:
        r["spec"] = specs
    code = parity.get("code_vs_tier1")
    if code:
        r["parity"] = {"match": code["match"], "total": code["total"], "verdict": code.get("verdict") or None}
    tel = doc.get("telemetry") or {}
    if tel.get("power_rolling_avg_w") is not None:
        r["telem"] = {"rolling_avg_w": tel["power_rolling_avg_w"], "peak_w": tel.get("power_peak_w"),
                      "max_tmem_c": tel.get("tmem_max_c"), "verdict": tel.get("verdict") or None}
    for v in r["prefill"].values():
        v.pop("_from", None)
    return r


def best_spec(r):
    """Fastest tg128 speculative mode that was fully lossless (a == b); None if none qualifies."""
    ok = [s for s in r["spec"] if s["tokens"] == 128 and s["lossless"] and s["lossless"][0] == s["lossless"][1]]
    return max(ok, key=lambda s: s["tps"]) if ok else None


def cells(r):
    """Metric-row label prefix -> cell text (None when the log lacks the metric)."""
    raw, spec, pp, par = r["raw"], best_spec(r), r["prefill"].get(2048), r["parity"]
    c = {}
    c["Decode tokens/s, M=1"] = "%.2f tok/s (%.2f ms/tok)" % (raw["tps"], raw["ms_per_tok"]) if raw else None
    if spec:
        extra = ", %.2fx vs raw" % spec["speedup"] if spec["speedup"] else ""
        c["Decode tokens/s with speculative decoding"] = "%.2f tok/s (%s%s, lossless %d/%d)" % (
            spec["tps"], spec["name"].strip(), extra, spec["lossless"][0], spec["lossless"][1])
    else:
        c["Decode tokens/s with speculative decoding"] = None
    c["Prefill tokens/s, 2048-token prompt"] = "%.2f tok/s (M=%d)" % (pp["tps"], pp["m"]) if pp else None
    if raw and raw["eff_hbm2_gbs"] is not None:
        c["HBM2 utilization, decode"] = "%.1f%% (%.1f of %.1f GB/s calibrated)" % (raw["hbm2_pct"], raw["eff_hbm2_gbs"], raw["hbm2_ceiling_gbs"])
    else:
        c["HBM2 utilization, decode"] = None
    c["Logit parity vs reference implementation"] = ("%d/%d exact greedy tokens%s" % (
        par["match"], par["total"], " (" + par["verdict"] + ")" if par["verdict"] else "")) if par else None
    return c


def fill_readme(text, r, commit=None, flags=None):
    """Returns (new_text, filled_labels, missing_labels)."""
    c = cells(r)
    filled, missing, out = [], [], []
    in_table, table_touched = False, False
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        if PLACEHOLDER in ln and ln.lstrip().startswith("|"):
            label = ln.strip().strip("|").split("|")[0].strip()
            key = next((k for k in c if label.startswith(k)), None)
            if key and c[key] is not None:
                ln = ln.replace(PLACEHOLDER, c[key], 1)
                filled.append(label)
            else:
                missing.append(label)
            in_table, table_touched = True, True
        out.append(ln)
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if in_table and not nxt.lstrip().startswith("|"):
            in_table = False
            if table_touched and filled:
                ceil = r["raw"]["hbm2_ceiling_gbs"] if r["raw"] else None
                prov = "Recorded run: commit `%s`; flags `%s`; calibrated HBM2 ceiling %s." % (
                    commit or "UNKNOWN", flags or "UNKNOWN", ("%.1f GB/s" % ceil) if ceil else "UNKNOWN")
                if not any(l.startswith("Recorded run:") for l in lines[i + 1:i + 4]):
                    out.extend(["", prov])
    return "\n".join(out), filled, missing


def format_text(r):
    o = []
    if r["raw"]:
        x = r["raw"]
        o.append("decode M=1      %.2f tok/s  %.2f ms/tok%s" % (x["tps"], x["ms_per_tok"],
                 "  eff_HBM2 %.1f GB/s (%.1f%% of %.1f)" % (x["eff_hbm2_gbs"], x["hbm2_pct"], x["hbm2_ceiling_gbs"]) if x["eff_hbm2_gbs"] else ""))
    for n, p in sorted(r["prefill"].items()):
        o.append("prefill pp%-5d %.2f tok/s (M=%d)" % (n, p["tps"], p["m"]))
    for s in r["spec"]:
        o.append("spec tg%d %-16s %.2f tok/s%s%s" % (s["tokens"], s["name"].strip(), s["tps"],
                 "  %.2fx" % s["speedup"] if s["speedup"] else "", "  lossless %d/%d" % tuple(s["lossless"]) if s["lossless"] else ""))
    if r["parity"]:
        o.append("parity          %d/%d %s" % (r["parity"]["match"], r["parity"]["total"], r["parity"]["verdict"] or ""))
    if r["telem"]:
        t = r["telem"]
        o.append("telemetry       avg %.1f W  peak %.1f W  TMem %.1f C  %s" % (t["rolling_avg_w"], t["peak_w"], t["max_tmem_c"], t["verdict"]))
    if r["insitu"]:
        x = r["insitu"]
        o.append("in-situ profile base %.2f ms - stages %.2f ms = %.2f ms unattributed (%.1f%%)" % (
            x["base_ms"], x["sum_ms"], x["unattributed_ms"], x["unattributed_pct"] or 0.0))
    if r["gpu0"]:
        o.append("GPU0 kernel sum %.2f ms/tok%s" % (r["gpu0"]["sum_ms"], "  (wall %.2f ms -> %.2f ms outside GPU0 kernels)" % (
            r["raw"]["ms_per_tok"], r["raw"]["ms_per_tok"] - r["gpu0"]["sum_ms"]) if r["raw"] else ""))
    if r["kstamp"]:
        for g, d in r["kstamp"]["gpus"].items():
            o.append("gpu%s stamps    %d tok  wall %.1f us  kernels %.1f us  host_gap %.1f us (%.1f%%, p99 %.1f)  idle %.1f us" % (
                g, d["tokens"], d["wall_us"]["mean"], d["kernel_sum_us"]["mean"], d["host_gap_us"]["mean"],
                d["host_gap_pct"] or 0.0, d["host_gap_us"]["p99"], d["idle_us"]["mean"]))
    for w in r["warnings"]:
        o.append("WARN " + w)
    return "\n".join(o) if o else "no recognised lines"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="bench log files, or - for stdin")
    ap.add_argument("--rtc-mhz", type=float, default=100.0)
    ap.add_argument("--json", action="store_true", help="print the parsed scorecard as JSON")
    ap.add_argument("--readme", help="README to fill")
    ap.add_argument("--out", help="write the filled README here")
    ap.add_argument("--in-place", action="store_true", help="overwrite --readme")
    ap.add_argument("--commit"); ap.add_argument("--flags")
    ap.add_argument("--strict", action="store_true", help="exit 1 if any README cell stays unfilled")
    a = ap.parse_args(argv)
    lines, docs = [], []
    for p in a.logs:
        if p == "-":
            text = sys.stdin.read()
        else:
            with open(p, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        if text.lstrip().startswith("{"):
            try:
                doc = json.loads(text)
            except ValueError:
                doc = None
            if isinstance(doc, dict) and doc.get("schema") == BENCH_SCHEMA:
                docs.append(doc)
                continue
        lines.extend(text.splitlines())
    r = parse(lines, a.rtc_mhz)
    for doc in docs:
        r = from_bench_json(doc, r)
    print(json.dumps(r, indent=2, sort_keys=True) if a.json else format_text(r))
    rc = 0
    if a.readme:
        if not (a.out or a.in_place):
            ap.error("--readme needs --out or --in-place")
        with open(a.readme, encoding="utf-8") as fh:
            text = fh.read()
        new, filled, missing = fill_readme(text, r, a.commit, a.flags)
        with open(a.readme if a.in_place else a.out, "w", encoding="utf-8") as fh:
            fh.write(new)
        print("README: filled %d cell(s)%s" % (len(filled), "; missing: " + ", ".join(missing) if missing else ""), file=sys.stderr)
        if missing and a.strict:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())

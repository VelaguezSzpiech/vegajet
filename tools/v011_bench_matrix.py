#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Reproducible v0.1.1 comparison matrix from VegaJet bench logs (stdlib only).

  v011_bench_matrix.py MANIFEST [--json OUT.json] [--md OUT.md]
  v011_bench_matrix.py --self-test

The manifest (JSON, schema "vegajet.v011_matrix_manifest.v1") names every run and how it was made; the tool parses the
engine's own log lines (the formats tools/scorecard.py and core/bench_json read, plus the v0.1.1 [SPEC-STATS] line)
and writes one matrix with the provenance next to every number:

  {"schema": "vegajet.v011_matrix_manifest.v1",
   "model": {"name": "<file name only>", "bytes": N, "fingerprint": "sha256-16M:<hex>"},   # see --fingerprint
   "parity_reference": {"golden": "...", "oracle": "..."},
   "runs": [{"label": "w4a8 super-on", "log": "<path>", "evidence": "live" | "host-replay",
             "quant": "w4a8" | "w4a4", "super": "on" | "off", "commit": "<sha>", "flags": "<flags>",
             "prompts": {"code": "<id>", "prose": "<id>"}}]}

Reported per run (null = not measured, never estimated):
  raw        tg128 / tg512 raw M=1 tok/s
  spec       per mode and prompt kind (code / prose separately): tok/s, speedup, mean accepted drafts per cycle,
             acceptance %, accepted output tokens per wall-clock second, draft depth, suffix hit rate, unique experts,
             draft / verify ms, lossless match against the 129-token release oracle (match/total, first_diff)
  prefill    pp512 / pp2048 tok/s, pp2048/pp512, first token; W4A4 runs also first-token agreement with the W4A8 run
             of the same super state and the W4A4 KLD ([KLD dot8 prefill])
  parity     the 32-token golden test ([PARITY code vs Tier-1 llama.cpp] n/32) kept apart from the 129-token oracle
  super      on-vs-off deltas between runs that differ only in "super" (same quant, same evidence)

Evidence kinds are never mixed: "live" (a bench run on the model) and "host-replay" (CPU replay output) go to separate
sections and no delta is computed across them. Model paths are never recorded; only the manifest's file name, size and
fingerprint. Exit codes: 0 ok, 2 unusable input.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

SCHEMA_IN = "vegajet.v011_matrix_manifest.v1"
SCHEMA_OUT = "vegajet.v011_matrix.v1"
NUM = r"(-?\d+(?:\.\d+)?)"

RE_RAW = re.compile(r"\[BENCH tg(128|512) raw M=1\]\s+" + NUM + r" tok/s")
RE_PP = re.compile(r"\[BENCH pp(\d+)\s+M=\s*(\d+)\s*\]\s+" + NUM + r" tok/s(?:.*first_tok=(-?\d+))?")
RE_SPEC = re.compile(r"\[BENCH tg(\d+) ([^\]]+?)\s*\]\s+" + NUM + r" tok/s(.*)$")
RE_STATS = re.compile(r"\[SPEC-STATS ([^\]]+)\] cycles=(\d+) acc/cyc=" + NUM + " tok/cyc=" + NUM + " depth=" + NUM +
                      r" sfx_hit=" + NUM + r"% uniq_exp=(\S+) draft_ms=" + NUM + " verify_ms=" + NUM + " acc_tok_s=" + NUM)
RE_GOLDEN = re.compile(r"\[PARITY code\s+vs Tier-1 llama\.cpp\]\s+(\d+)/(\d+) exact greedy tokens(?: -> (\w+))?")
RE_KLD = re.compile(r"\[KLD dot8 prefill\]\s+" + NUM)
RE_C2 = re.compile(r"\[PREFILL-C2\] (.*)$")


def fingerprint(path):
    """sha256 over the first and last 16 MiB plus the size (fast, stable for multi-GiB model files)."""
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with p.open("rb") as f:
        h.update(f.read(16 << 20))
        if size > (32 << 20):
            f.seek(size - (16 << 20))
            h.update(f.read(16 << 20))
    return {"name": p.name, "bytes": size, "fingerprint": "sha256-16M:" + h.hexdigest()}


def kind_of(label, rest):
    s = (label + " " + rest).lower()
    if "prose" in s:
        return "prose"
    if "code" in s or "hybrid" in s or "suffix" in s:
        return "code"
    return "unknown"


def mode_key(tokens, label):
    base = re.sub(r"\b(code|prose)\b", "", label.lower())
    base = re.sub(r"[^a-z0-9=+]+", "_", base).strip("_").replace("=", "").replace("+", "_")
    return "tg%s_%s" % (tokens, base)


def parse_log(text):
    out = {"raw": {"tg128_tok_s": None, "tg512_tok_s": None}, "prefill": {}, "spec": {}, "stats": {},
           "golden32": None, "kld_w4a4": None, "c2": None}
    last_spec = None  # the engine prints [SPEC-STATS] right after its [BENCH tg...] line
    for line in text.splitlines():
        m = RE_RAW.search(line)
        if m:
            out["raw"]["tg%s_tok_s" % m.group(1)] = float(m.group(2))
            last_spec = None
            continue
        m = RE_PP.search(line)
        if m:
            out["prefill"]["pp" + m.group(1)] = {"tok_s": float(m.group(3)), "m": int(m.group(2)),
                                                 "first_tok": int(m.group(4)) if m.group(4) is not None else None}
            last_spec = None
            continue
        m = RE_STATS.search(line)
        if m:
            uq = m.group(7)
            st = out["stats"][m.group(1).strip()] = {
                "cycles": int(m.group(2)), "acc_per_cycle": float(m.group(3)), "tok_per_cycle": float(m.group(4)),
                "depth": float(m.group(5)), "suffix_hit_pct": float(m.group(6)),
                "unique_experts": None if uq == "n/a" else float(uq), "draft_ms": float(m.group(8)),
                "verify_ms": float(m.group(9)), "acc_tok_s": float(m.group(10))}
            if last_spec is not None and last_spec["stats"] is None:
                last_spec["stats"] = st
            continue
        m = RE_SPEC.search(line)
        if m and "raw M=1" not in m.group(2):
            tokens, label, tps, rest = m.group(1), m.group(2).strip(), float(m.group(3)), m.group(4)
            e = {"label": label, "tokens": int(tokens), "kind": kind_of(label, rest), "tok_s": tps,
                 "speedup": None, "acceptance_pct": None, "oracle129": None, "stats": None}
            sp = re.search(r"\(" + NUM + r"x vs raw", rest)
            if sp:
                e["speedup"] = float(sp.group(1))
            ll = re.search(r"lossless=(\d+)/(\d+)(?: \(first_diff=(-?\d+)\))?", rest)
            if ll:
                e["oracle129"] = {"match": int(ll.group(1)), "total": int(ll.group(2)),
                                  "first_diff": int(ll.group(3)) if ll.group(3) is not None else None}
            ac = re.search(r"acc=(\d+)/(\d+) \(" + NUM + "%", rest)
            if ac:
                e["acceptance_pct"] = float(ac.group(3))
            out["spec"]["%s_%s" % (mode_key(tokens, label), e["kind"])] = e
            last_spec = e
            continue
        if line.startswith("[BENCH") or line.startswith("[PARITY"):
            last_spec = None
        m = RE_GOLDEN.search(line)
        if m:
            out["golden32"] = {"match": int(m.group(1)), "total": int(m.group(2)), "verdict": m.group(3)}
            continue
        m = RE_KLD.search(line)
        if m:
            out["kld_w4a4"] = float(m.group(1))
            continue
        m = RE_C2.search(line)
        if m:
            out["c2"] = m.group(1)
    pp = out["prefill"]
    if "pp512" in pp and "pp2048" in pp and pp["pp512"]["tok_s"] > 0:
        out["prefill"]["pp2048_over_pp512"] = round(pp["pp2048"]["tok_s"] / pp["pp512"]["tok_s"], 4)
    return out


def build(manifest, base_dir):
    if manifest.get("schema") != SCHEMA_IN:
        raise ValueError("manifest schema must be " + SCHEMA_IN)
    runs = manifest.get("runs") or []
    if not runs:
        raise ValueError("manifest has no runs")
    res = {"schema": SCHEMA_OUT, "model": manifest.get("model"), "parity_reference": manifest.get("parity_reference"),
           "sections": {}}
    for r in runs:
        ev = r.get("evidence")
        if ev not in ("live", "host-replay"):
            raise ValueError("run %r: evidence must be 'live' or 'host-replay'" % r.get("label"))
        path = Path(r["log"])
        if not path.is_absolute():
            path = base_dir / path
        parsed = parse_log(path.read_text(errors="replace"))
        row = {k: r.get(k) for k in ("label", "quant", "super", "commit", "flags", "prompts")}
        row["log"] = path.name  # never a directory
        row.update(parsed)
        res["sections"].setdefault(ev, {"runs": [], "super_deltas": [], "w4a4_vs_w4a8": []})["runs"].append(row)
    for ev, sec in res["sections"].items():
        rows = sec["runs"]
        for a in rows:
            if a.get("quant") != "w4a4":
                continue
            for b in rows:
                if b.get("quant") == "w4a8" and b.get("super") == a.get("super"):
                    agree = {}
                    for pp in ("pp512", "pp2048"):
                        fa = (a["prefill"].get(pp) or {}).get("first_tok")
                        fb = (b["prefill"].get(pp) or {}).get("first_tok")
                        agree[pp] = None if fa is None or fb is None else fa == fb
                    sec["w4a4_vs_w4a8"].append({"w4a4": a["label"], "w4a8": b["label"], "first_token_agreement": agree,
                                                "kld_w4a4": a.get("kld_w4a4")})
        for a in rows:
            if a.get("super") != "on":
                continue
            for b in rows:
                if b.get("super") == "off" and b.get("quant") == a.get("quant"):
                    d = {"on": a["label"], "off": b["label"], "quant": a.get("quant")}
                    for k in ("tg128_tok_s", "tg512_tok_s"):
                        x, y = a["raw"].get(k), b["raw"].get(k)
                        d["raw_" + k + "_on_over_off"] = round(x / y, 4) if x and y else None
                    for pp in ("pp512", "pp2048"):
                        x, y = (a["prefill"].get(pp) or {}).get("tok_s"), (b["prefill"].get(pp) or {}).get("tok_s")
                        d[pp + "_on_over_off"] = round(x / y, 4) if x and y else None
                    g_on, g_off = a.get("golden32"), b.get("golden32")
                    d["golden32"] = {"on": g_on, "off": g_off}
                    sec["super_deltas"].append(d)
    return res


def fmt(v, nd=2):
    return "-" if v is None else (("%." + str(nd) + "f") % v if isinstance(v, float) else str(v))


def to_markdown(res):
    out = ["# v0.1.1 comparison matrix", ""]
    mdl = res.get("model") or {}
    out.append("Model: `%s` (%s bytes, %s)" % (mdl.get("name", "?"), mdl.get("bytes", "?"), mdl.get("fingerprint", "no fingerprint")))
    pr = res.get("parity_reference") or {}
    out.append("Parity references: golden = %s; oracle = %s" % (pr.get("golden", "-"), pr.get("oracle", "-")))
    for ev in ("live", "host-replay"):
        sec = res["sections"].get(ev)
        if not sec:
            continue
        out += ["", "## Evidence: %s" % ev, "", "### Raw decode and parity", "",
                "| run | quant | super | tg128 raw | tg512 raw | golden 32 | commit |", "|---|---|---|---|---|---|---|"]
        for r in sec["runs"]:
            g = r.get("golden32")
            out.append("| %s | %s | %s | %s | %s | %s | %s |" % (r["label"], r.get("quant"), r.get("super"), fmt(r["raw"]["tg128_tok_s"]),
                       fmt(r["raw"]["tg512_tok_s"]), "-" if not g else "%d/%d" % (g["match"], g["total"]), r.get("commit") or "-"))
        out += ["", "### Speculative (code and prose separately; oracle = 129-token release oracle)", "",
                "| run | mode | kind | tok/s | speedup | acc/cyc | acc % | acc out tok/s | depth | sfx hit % | uniq exp | draft ms | verify ms | oracle |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sec["runs"]:
            for key in sorted(r["spec"]):
                e = r["spec"][key]
                st = e.get("stats") or {}
                o = e.get("oracle129")
                out.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                    r["label"], e["label"], e["kind"], fmt(e["tok_s"]), fmt(e["speedup"]), fmt(st.get("acc_per_cycle"), 3),
                    fmt(e["acceptance_pct"], 1), fmt(st.get("acc_tok_s")), fmt(st.get("depth")), fmt(st.get("suffix_hit_pct"), 1),
                    fmt(st.get("unique_experts"), 1), fmt(st.get("draft_ms"), 3), fmt(st.get("verify_ms"), 3),
                    "-" if not o else "%d/%d (first_diff=%s)" % (o["match"], o["total"], o["first_diff"])))
        out += ["", "### Prefill", "", "| run | quant | super | pp512 | pp2048 | pp2048/pp512 | first tok 512 | first tok 2048 | W4A4 KLD |",
                "|---|---|---|---|---|---|---|---|---|"]
        for r in sec["runs"]:
            p = r["prefill"]
            out.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                r["label"], r.get("quant"), r.get("super"), fmt((p.get("pp512") or {}).get("tok_s")), fmt((p.get("pp2048") or {}).get("tok_s")),
                fmt(p.get("pp2048_over_pp512"), 3), fmt((p.get("pp512") or {}).get("first_tok")), fmt((p.get("pp2048") or {}).get("first_tok")),
                fmt(r.get("kld_w4a4"), 4)))
        for c in sec["w4a4_vs_w4a8"]:
            out.append("- W4A4 `%s` vs W4A8 `%s`: first-token agreement %s, KLD %s" % (c["w4a4"], c["w4a8"], c["first_token_agreement"], fmt(c["kld_w4a4"], 4)))
        if sec["super_deltas"]:
            out += ["", "### Super-kernels on vs off (same quant)", ""]
            for d in sec["super_deltas"]:
                out.append("- %s: `%s` / `%s`: tg128 %s, tg512 %s, pp512 %s, pp2048 %s (on/off); golden on %s, off %s" % (
                    d["quant"], d["on"], d["off"], fmt(d["raw_tg128_tok_s_on_over_off"], 3), fmt(d["raw_tg512_tok_s_on_over_off"], 3),
                    fmt(d["pp512_on_over_off"], 3), fmt(d["pp2048_on_over_off"], 3),
                    d["golden32"]["on"] and "%d/%d" % (d["golden32"]["on"]["match"], d["golden32"]["on"]["total"]),
                    d["golden32"]["off"] and "%d/%d" % (d["golden32"]["off"]["match"], d["golden32"]["off"]["total"])))
    return "\n".join(out) + "\n"


SAMPLE_W4A8_ON = """\
[BENCH tg128 raw M=1] 31.20 tok/s (32.05 ms/tok) | eff_HBM2=800.0 GB/s (80.0% of 1000.0 GB/s calibrated) | real_read=1.000 GB/tok | disp/tok=120
[BENCH tg512 raw M=1] 30.10 tok/s (33.22 ms/tok) | disp/tok=120
[PARITY code  vs Tier-1 llama.cpp] 32/32 exact greedy tokens -> PASS
[BENCH tg128 MTP K=4] 48.00 tok/s (1.55x vs raw 31.00) | prose lossless=129/129 (first_diff=-1) | acc=300/400 (75.0%)
[SPEC-STATS MTP K=4 prose] cycles=40 acc/cyc=2.500 tok/cyc=3.500 depth=4.00 sfx_hit=0.0% uniq_exp=n/a draft_ms=2.100 verify_ms=9.000 acc_tok_s=47.80
[BENCH tg128 Hybrid ] 70.00 tok/s (2.24x vs raw) | code lossless=129/129 (first_diff=-1) | sfx_hits=20 acc=500/560 (89.3%)
[SPEC-STATS Hybrid K=4 code] cycles=30 acc/cyc=3.200 tok/cyc=4.200 depth=6.00 sfx_hit=66.7% uniq_exp=31.5 draft_ms=0.700 verify_ms=11.000 acc_tok_s=69.50
[BENCH tg128 DFlash2 K=7 prose] 40.00 tok/s (1.29x vs raw 31.00) | prose lossless=129/129 (first_diff=-1) | acc=200/700 (28.6%)
[SPEC-STATS DFlash2 K=7 prose] cycles=60 acc/cyc=1.100 tok/cyc=2.100 depth=7.00 sfx_hit=0.0% uniq_exp=n/a draft_ms=4.000 verify_ms=12.000 acc_tok_s=39.90
[BENCH tg128 DFlash2 K=7 code ] 55.00 tok/s (1.76x vs raw 31.20) | code lossless=129/129 (first_diff=-1) | acc=400/700 (57.1%)
[SPEC-STATS DFlash2 K=7 code] cycles=40 acc/cyc=2.200 tok/cyc=3.200 depth=7.00 sfx_hit=0.0% uniq_exp=n/a draft_ms=4.000 verify_ms=12.000 acc_tok_s=54.80
[BENCH pp512  M=1024  ] 1400.00 tok/s (WY_GDN=ON, 0 SOLVE_TRI, first_tok=271)
[BENCH pp2048 M=1024  ] 1350.00 tok/s (WY_GDN=ON, 0 SOLVE_TRI, first_tok=88)
"""


def self_test():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "w4a8_on.log").write_text(SAMPLE_W4A8_ON)
        (d / "w4a8_off.log").write_text(SAMPLE_W4A8_ON.replace("31.20 tok/s", "25.00 tok/s").replace("1400.00", "1300.00"))
        (d / "w4a4_on.log").write_text(SAMPLE_W4A8_ON.replace("first_tok=88", "first_tok=89").replace("1400.00", "1550.00")
                                       + "[KLD dot8 prefill] 0.0312\n")
        (d / "replay.log").write_text("[SPEC-STATS DFlash2 K=7 code] cycles=12 acc/cyc=3.700 tok/cyc=4.700 depth=7.00 sfx_hit=0.0% "
                                      "uniq_exp=n/a draft_ms=0.000 verify_ms=0.000 acc_tok_s=0.00\n")
        (d / "model.bin").write_bytes(b"x" * 1000)
        man = {"schema": SCHEMA_IN, "model": fingerprint(d / "model.bin"),
               "parity_reference": {"golden": "kCodeRef32", "oracle": "raw greedy 129"},
               "runs": [{"label": "w4a8 on", "log": "w4a8_on.log", "evidence": "live", "quant": "w4a8", "super": "on", "commit": "abc"},
                        {"label": "w4a8 off", "log": "w4a8_off.log", "evidence": "live", "quant": "w4a8", "super": "off", "commit": "abc"},
                        {"label": "w4a4 on", "log": "w4a4_on.log", "evidence": "live", "quant": "w4a4", "super": "on", "commit": "abc"},
                        {"label": "replay", "log": "replay.log", "evidence": "host-replay"}]}
        res = build(man, d)
        live = res["sections"]["live"]["runs"]
        a = live[0]
        assert a["raw"] == {"tg128_tok_s": 31.2, "tg512_tok_s": 30.1}, a["raw"]
        assert a["golden32"]["match"] == 32 and a["golden32"]["total"] == 32
        kinds = {(e["label"], e["kind"]) for e in a["spec"].values()}
        assert ("DFlash2 K=7 prose", "prose") in kinds and ("DFlash2 K=7 code", "code") in kinds, kinds
        df_code = [e for e in a["spec"].values() if e["label"] == "DFlash2 K=7 code"][0]
        assert df_code["stats"]["acc_per_cycle"] == 2.2 and df_code["stats"]["acc_tok_s"] == 54.8, df_code
        assert df_code["oracle129"] == {"match": 129, "total": 129, "first_diff": -1}
        hyb = [e for e in a["spec"].values() if e["label"] == "Hybrid"][0]
        assert hyb["kind"] == "code" and hyb["stats"]["unique_experts"] == 31.5, hyb
        assert a["prefill"]["pp2048_over_pp512"] == round(1350 / 1400, 4)
        cmp_ = res["sections"]["live"]["w4a4_vs_w4a8"]
        assert len(cmp_) == 1 and cmp_[0]["first_token_agreement"] == {"pp512": True, "pp2048": False} and cmp_[0]["kld_w4a4"] == 0.0312, cmp_
        sd = res["sections"]["live"]["super_deltas"]
        assert len(sd) == 1 and sd[0]["raw_tg128_tok_s_on_over_off"] == round(31.2 / 25.0, 4), sd
        assert "host-replay" in res["sections"] and not res["sections"]["host-replay"]["super_deltas"]
        assert res["model"]["name"] == "model.bin" and str(d) not in json.dumps(res)
        md = to_markdown(res)
        assert "## Evidence: live" in md and "## Evidence: host-replay" in md and "129/129" in md
        # a run without an evidence kind is refused
        bad = dict(man, runs=[dict(man["runs"][0], evidence="guess")])
        try:
            build(bad, d)
            raise AssertionError("evidence kind not enforced")
        except ValueError:
            pass
    print("v011_bench_matrix self-test: PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", nargs="?", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--md", type=Path)
    ap.add_argument("--fingerprint", type=Path, help="print the model identity block for a model file and exit")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if a.fingerprint:
        print(json.dumps(fingerprint(a.fingerprint), indent=2))
        return 0
    if not a.manifest:
        ap.print_usage()
        return 2
    try:
        man = json.loads(a.manifest.read_text())
        res = build(man, a.manifest.resolve().parent)
    except (OSError, ValueError, KeyError) as e:
        print("error: %s" % e, file=sys.stderr)
        return 2
    js = json.dumps(res, indent=2, sort_keys=True)
    md = to_markdown(res)
    if a.json:
        a.json.write_text(js + "\n")
    if a.md:
        a.md.write_text(md)
    if not a.json and not a.md:
        sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())

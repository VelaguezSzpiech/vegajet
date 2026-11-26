#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Summarize one warm-engine variant (stdin = its log lines) as a single line for scripts/fast_bench.sh."""
import json
import re
import sys

bucket = sys.argv[1] if len(sys.argv) > 1 else "TOTAL"
tps, par, ms, ev = {}, [], [], {}
for line in sys.stdin:
    if "WATCHDOG" in line or "FATAL" in line:
        print("ABORT:", line.strip())
        sys.exit(3)
    m = re.match(r"\[BENCH (.+?)\]\s+([0-9.]+) tok/s(.*)", line)
    if m:
        tps[m.group(1).strip()] = m.group(2)
        p = re.search(r"(lossless=\S+|vs_base=\S+|rows_agree=\S+)", m.group(3))
        if p:
            par.append(p.group(1))
    if re.match(r"\s+(GPU\d+:|GDN sub|MoE sub)", line):
        m2 = re.search(r"(?:^|[\s|])" + re.escape(bucket) + r"=([0-9.]+)", line)
        if m2:
            ms.append(m2.group(1))
    if line.startswith('{"event":"variant"') or line.startswith('{"event":"error"'):
        ev = json.loads(line)
t = " ".join(f"{k}={v}" for k, v in tps.items())
print(f"{t} | {bucket}={'/'.join(ms) or 'n/a'} ms | parity {','.join(par) or ev.get('parity', '?')} | "
      f"ECC {ev.get('ecc_ue', '?')}/{ev.get('ecc_de', '?')} TMem {ev.get('tmem_c', '?')} {ev.get('error', '')}")

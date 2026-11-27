#!/usr/bin/env python3
"""Join W14 measurements into one raw-decode debugging table.

Inputs: runs/W14.log (SUPER_PROFILE=2 critical-path buckets, BYTES breakdown, raw tok/s), the 20 Hz sysfs samples
(idle + load CSV), the calibrated per-card HBM bandwidth (~/.config/vegajet/hw_profile.json), and runs/W14ar.log.
Usage: raw_diag_table.py W14.log idle.csv load.csv [W14ar.log]
"""
import csv, json, os, re, statistics, sys

log = open(sys.argv[1], errors='replace').read().splitlines()
idle_csv, load_csv = sys.argv[2], sys.argv[3]
ar_log = sys.argv[4] if len(sys.argv) > 4 else None
num = r'([0-9]+(?:\.[0-9]+)?)'

# --- variants: split the log at each '"event":"variant"' line ---
variants, cur = [], []
for line in log:
    cur.append(line)
    if line.startswith('{"event":"variant"'):
        variants.append(cur); cur = []


def find_variant(key):
    for v in variants:
        if key in v[-1]:
            return v
    return None


def profile_blocks(v):
    """Return list of per-GPU dicts in print order (WG0 block first, critical block second for PROFILE=2)."""
    blocks, g = [], None
    for line in v:
        m = re.match(r'\s+GPU(\d+): (.*)', line)
        if m and 'TOTAL' in line:
            g = {k: float(x) for k, x in re.findall(r'([A-Za-z0-9+()_]+)=' + num, m.group(2))}
            g['gpu'] = int(m.group(1)); blocks.append(g); continue
        if g is None:
            continue
        m = re.match(r'\s+(GDN sub|MoE sub.*?):(.*)', line)
        if m:
            pre = 'gdn.' if m.group(1).startswith('GDN') else 'moe.'
            for part in m.group(2).split('|'):
                sub = 'gqa.' if 'GQA sub' in part else pre
                for k, x in re.findall(r'([A-Za-z0-9+_]+)=' + num, part):
                    g[sub + k] = float(x)
    return blocks


prof2 = find_variant('VEGAJET_SUPER_PROFILE=2')
blocks = profile_blocks(prof2) if prof2 else []
n = len(blocks) // 2 if len(blocks) >= 6 else len(blocks)
crit = blocks[n:] if len(blocks) >= 6 else blocks          # critical-path block (max over WGs)
wg0 = blocks[:n] if len(blocks) >= 6 else []

tps = {}
for v in variants:
    for line in v:
        m = re.match(r'\[BENCH tg(\d+) raw M=1\] ' + num + r' tok/s', line)
        if m:
            tps.setdefault(v[-1][24:110], []).append((int(m.group(1)), float(m.group(2))))

bytes_ = {}
for line in log:
    m = re.match(r'\[BYTES gpu(\d+) MB/tok\] (.*)', line)
    if m:
        bytes_[int(m.group(1))] = {k: float(x) for k, x in re.findall(r'([a-z_]+)=' + num, m.group(2))}

prof = json.load(open(os.path.expanduser('~/.config/vegajet/hw_profile.json')))
peak = [g['bw_hbm_gbs'] for g in prof['gpus']]

# --- stage groups: (label, bucket keys (ms), byte keys (MB)) ---
groups = [
    ('GDN weights (QKV+gab, out-proj)', ['gdn.QKV', 'gdn.Out'], ['gdn_w']),
    ('GDN conv + recurrence (state r/w)', ['gdn.ConvRec'], ['gdn_state']),
    ('GQA weights (QKV, out-proj)', ['gqa.QKV', 'gqa.Out'], ['gqa_w']),
    ('MoE routed Gate+Up', ['moe.GU_DOT8', 'moe.GU_IQ2S'], ['routed_gu']),
    ('MoE router + shared + routed Down', ['moe.Rtr+ShGU', 'moe.MoEDn+PP'], ['router', 'shared', 'routed_dn']),
    ('LM head', ['lm'], ['lm_head']),
]
nonbw = [
    ('AllReduce: GDN layers', ['gdn.AR']), ('AllReduce: GQA layers', ['gqa.AR']), ('AllReduce: MoE', ['moe.AR']),
    ('AllReduce: final', ['final_ar']), ('Top-8 select', ['moe.Top8']), ('GQA prep + flash + merge', ['gqa.Prep', 'gqa.Flash', 'gqa.Mrg']),
    ('Embedding + RMS', ['emb+rms']),
]

out = []
P = lambda *a: out.append(' '.join(str(x) for x in a))
P('## Raw M=1 decode: measured breakdown (W14)\n')
for k, v in tps.items():
    P(f'- `{k.strip()}`: ' + ', '.join(f'tg{t} {x:.2f} tok/s' for t, x in v))
if crit:
    P('\n### Bandwidth stages (critical path, ms/token; achieved GB/s = modelled bytes / stage time)\n')
    P('| Stage | ' + ' | '.join(f'GPU{c["gpu"]} ms | GPU{c["gpu"]} GB/s (% of {peak[c["gpu"]]:.0f})' for c in crit) + ' |')
    P('|---|' + '---|---|' * len(crit))
    tot_bw = [0.0] * len(crit)
    tot_by = [0.0] * len(crit)
    for label, ks, bk in groups:
        cells = []
        for i, c in enumerate(crit):
            ms = sum(c.get(k, 0.0) for k in ks)
            mb = sum(bytes_.get(c['gpu'], {}).get(b, 0.0) for b in bk)
            tot_bw[i] += ms; tot_by[i] += mb
            gbs = mb / ms if ms > 0 else 0.0   # MB/ms == GB/s
            cells += [f'{ms:.3f}', f'{gbs:.0f} ({100*gbs/peak[c["gpu"]]:.0f}%)' if mb else 'n/a']
        P(f'| {label} | ' + ' | '.join(cells) + ' |')
    P('| **Sum of bandwidth stages** | ' + ' | '.join(
        f'**{tot_bw[i]:.3f}** | **{tot_by[i]/tot_bw[i]:.0f} ({100*tot_by[i]/tot_bw[i]/peak[c["gpu"]]:.0f}%)**'
        for i, c in enumerate(crit)) + ' |')
    P('\n### Non-bandwidth time (critical path, ms/token)\n')
    P('| Stage | ' + ' | '.join(f'GPU{c["gpu"]}' for c in crit) + ' |')
    P('|---|' + '---|' * len(crit))
    tot_nb = [0.0] * len(crit)
    for label, ks in nonbw:
        vals = [sum(c.get(k, 0.0) for k in ks) for c in crit]
        for i, x in enumerate(vals): tot_nb[i] += x
        P(f'| {label} | ' + ' | '.join(f'{x:.3f}' for x in vals) + ' |')
    P('| **Sum non-bandwidth** | ' + ' | '.join(f'**{x:.3f}**' for x in tot_nb) + ' |')
    P('| Profile TOTAL | ' + ' | '.join(f'{c.get("TOTAL",0):.3f}' for c in crit) + ' |')
    P('\n### If every bandwidth stage ran at the calibrated peak\n')
    for i, c in enumerate(crit):
        ideal = tot_by[i] / peak[c['gpu']]
        P(f'- GPU{c["gpu"]}: bandwidth stages {tot_bw[i]:.3f} ms -> {ideal:.3f} ms at {peak[c["gpu"]]:.0f} GB/s; '
          f'+ non-bandwidth {tot_nb[i]:.3f} ms = {ideal + tot_nb[i]:.3f} ms/token -> {1000/(ideal + tot_nb[i]):.0f} tok/s '
          f'(and {1000/ideal:.0f} tok/s with zero non-bandwidth time)')


def summarize(path, label):
    rows = list(csv.DictReader(open(path)))
    by = {}
    for r in rows:
        by.setdefault(r['bdf'], []).append(r)
    P(f'\n### {label} (sysfs, {len(rows)//max(1,len(by))} samples/card)\n')
    P('| Card | sclk MHz median (min..max) | % samples at 1725 MHz | mclk MHz | power W median (max) | gpu_busy % median | temps edge/junc/mem max |')
    P('|---|---|---|---|---|---|---|')
    for bdf, rs in sorted(by.items()):
        f = lambda k: [float(r[k]) for r in rs if float(r[k]) >= 0]
        s, m, p, b = f('sclk_mhz'), f('mclk_mhz'), f('power_w'), f('gpu_busy')
        top = 100.0 * sum(1 for x in s if x >= 1725) / max(1, len(s))
        P(f'| {bdf} | {statistics.median(s):.0f} ({min(s):.0f}..{max(s):.0f}) | {top:.0f}% | {statistics.median(m):.0f} | '
          f'{statistics.median(p) if p else -1:.1f} ({max(p) if p else -1:.1f}) | {statistics.median(b) if b else -1:.0f} | '
          f'{max(f("edge_c")):.0f}/{max(f("junc_c")):.0f}/{max(f("mem_c")):.0f} |')


summarize(idle_csv, 'Idle (before load)')
# Load window: only samples where any card is busy
rows = list(csv.DictReader(open(load_csv)))
busy_t = sorted({r['t'] for r in rows if float(r['gpu_busy']) > 50})
if busy_t:
    lo, hi = float(busy_t[0]), float(busy_t[-1])
    tmp = load_csv + '.busy.csv'
    with open(tmp, 'w') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader()
        for r in rows:
            if lo <= float(r['t']) <= hi: w.writerow(r)
    summarize(tmp, f'Under raw decode load ({hi-lo:.1f} s busy window)')
if ar_log and os.path.exists(ar_log):
    P('\n### AllReduce microbench (`--bench-ar`)\n')
    for line in open(ar_log, errors='replace'):
        if re.search(r'us/AR|AR |verify|bench-ar', line):
            P('    ' + line.rstrip()[:200])
print('\n'.join(out))

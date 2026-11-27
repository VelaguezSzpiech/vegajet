#!/usr/bin/env python3
"""Deep decode telemetry: per-workgroup stage skew, per-GPU AllReduce wait, Hybrid verify breakdown (CPU-only parser).

Inputs
  --dump FILE      VEGAJET_PROFILE_DUMP binary (VEGAJET_SUPER_PROFILE=2), repeatable (one per profiled token)
  --log FILE       warm-engine log with the matching runs (BENCH, SUPER-PROFILE, VERIFY-PROFILE, spec_cycle_ms, BYTES)
  --isa FILE       scripts/isa_audit.py --json output (optional, embedded as-is)
  --idle/--load    scripts/sysfs_sampler.py CSVs (optional)
  --json OUT       machine-readable result; markdown tables go to stdout

Dump layout (models/ornith/ornith_gpu.cpp, VEGAJET_PROFILE_DUMP): u32 magic 'VJPD', N, wgs, n_stamps; per GPU:
u64 stage stamps[n_stamps], u64 critical ends[96][17], u64 per-WG ends[96][17][256], u64 WG0 AR wait [96][2];
then 48 layer-type bytes (1 = GDN). Row 2*il = mixer kernel of layer il, 2*il+1 = MoE kernel. Ticks: 25 MHz.
"""
import argparse, csv, json, re, statistics, struct, sys

TICK_MS = 40e-6
SLOT = {0: 'GDN.AR', 1: 'GDN.QKV', 2: 'GDN.ConvRec', 3: 'GDN.Out',
        4: 'GQA.AR', 5: 'GQA.QKV', 6: 'GQA.Prep', 7: 'GQA.Flash', 8: 'GQA.Mrg', 9: 'GQA.Out',
        10: 'MoE.AR', 11: 'MoE.Rtr+ShGU', 12: 'MoE.Top8', 13: 'MoE.GU_IQ2S', 15: 'MoE.GU_DOT8', 14: 'MoE.Down+PP'}
CHAIN = {'gdn': [0, 1, 2, 3], 'gqa': [4, 5, 6, 7, 8, 9], 'moe': [10, 11, 12, 13, 15, 14]}
# MoE Q3 _x sub-stamps: (slot, label, parent stage start slot)
MOE_SUB = [(1, 'MoE.s1.router_done', 10), (2, 'MoE.s1.shGU_done', 10),
           (3, 'MoE.s4.lds_in', 15), (4, 'MoE.s4.down_rows', 15), (5, 'MoE.s4.peer_stores', 15)]

ap = argparse.ArgumentParser()
ap.add_argument('--dump', action='append', default=[])
ap.add_argument('--log')
ap.add_argument('--isa')
ap.add_argument('--idle')
ap.add_argument('--load')
ap.add_argument('--json')
a = ap.parse_args() if __name__ == '__main__' else None


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = (len(xs) - 1) * p / 100.0
    f = int(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def read_dump(path):
    b = open(path, 'rb').read()
    magic, N, wgs, ns = struct.unpack_from('<4I', b, 0)
    assert magic == 0x564a5044, 'bad magic'
    off = 16
    gpus = []
    for g in range(N):
        tv = struct.unpack_from('<%dQ' % ns, b, off); off += 8 * ns
        ends = struct.unpack_from('<%dQ' % (96 * 17), b, off); off += 8 * 96 * 17
        wg = struct.unpack_from('<%dQ' % (96 * 17 * 256), b, off); off += 8 * 96 * 17 * 256
        arw = struct.unpack_from('<%dQ' % (96 * 2), b, off); off += 8 * 96 * 2
        gpus.append(dict(tv=tv, ends=ends, wg=wg, arw=arw))
    types = list(b[off:off + 48])
    return dict(N=N, wgs=wgs, gpus=gpus, is_gdn=[t == 1 for t in types])


def stage_stats(d):
    """Per GPU, per stage: per-WG finish offset from the stage start (= previous stage's critical end), summed over
    layers -> min/median/p95/max across WGs, WG0, slowest WG id histogram, tail = max - median."""
    out = []
    W = d['wgs']
    for g, G in enumerate(d['gpus']):
        ends, wg = G['ends'], G['wg']
        acc = {}   # label -> per-WG summed offsets (ticks)
        crit = {}  # label -> summed critical duration
        slow = {}  # label -> {wg: count of layers where it was slowest}
        nlay = {}
        for row in range(96):
            il, moe = row // 2, row & 1
            kind = 'moe' if moe else ('gdn' if d['is_gdn'][il] else 'gqa')
            E = ends[row * 17:(row + 1) * 17]
            if not E[16]:
                continue
            prev = E[16]
            prev_slot = 16
            for s in CHAIN[kind]:
                if not E[s]:
                    continue
                per = [wg[(row * 17 + s) * 256 + w] for w in range(W)]
                lab = SLOT[s]
                # WGs can finish (e.g. skip WG0's AR) before WG0's entry stamp: clamp to 0 (= no wait past stage start)
                offs = [max(0, t - prev) for t in per]
                if min(per) > 0:
                    v = acc.setdefault(lab, [0] * W)
                    for w in range(W):
                        v[w] += offs[w]
                    sw = max(range(W), key=lambda w: per[w])
                    slow.setdefault(lab, {}).setdefault(sw, 0)
                    slow[lab][sw] += 1
                    nlay[lab] = nlay.get(lab, 0) + 1
                crit[lab] = crit.get(lab, 0) + (E[s] - prev)
                prev = E[s]; prev_slot = s
            if moe and E[10] and E[15]:
                for s, lab, ref in MOE_SUB:
                    per = [wg[(row * 17 + s) * 256 + w] for w in range(W)]
                    refv = E[ref]
                    offs = [p - refv for p in per if p and p >= refv]
                    if offs:
                        v = acc.setdefault(lab, [])
                        v.append(offs)
        rows = []
        for lab, v in acc.items():
            if lab.startswith('MoE.s'):
                # sub-stamps: only the WGs doing that sub-task stamp; stats over (layer, WG) samples, summed per layer
                per_layer = v
                rows.append(dict(gpu=g, stage=lab, layers=len(per_layer), wgs_stamping=round(statistics.mean(len(x) for x in per_layer)),
                                 min_ms=sum(min(x) for x in per_layer) * TICK_MS,
                                 median_ms=sum(statistics.median(x) for x in per_layer) * TICK_MS,
                                 p95_ms=sum(pct(x, 95) for x in per_layer) * TICK_MS,
                                 max_ms=sum(max(x) for x in per_layer) * TICK_MS))
                continue
            med = statistics.median(v)
            sw_hist = sorted(slow.get(lab, {}).items(), key=lambda kv: -kv[1])[:3]
            rows.append(dict(gpu=g, stage=lab, layers=nlay.get(lab, 0), critical_ms=crit.get(lab, 0) * TICK_MS,
                             min_ms=min(v) * TICK_MS, median_ms=med * TICK_MS, p95_ms=pct(v, 95) * TICK_MS, max_ms=max(v) * TICK_MS,
                             wg0_ms=v[0] * TICK_MS, tail_ms=(max(v) - med) * TICK_MS,
                             slowest_wg_top3=[dict(wg=w, layers=c) for w, c in sw_hist]))
        out.extend(rows)
    return out


def ar_stats(d):
    """WG0 spin window per GPU at every in-kernel AllReduce. wait = peers' flags seen - spin start.
    Category by what was reduced: odd row (MoE kernel, Stage 0) = after the layer's mixer (GDN/GQA);
    even row il >= 1 (mixer kernel, Stage 0) = after the previous layer's MoE."""
    N = d['N']
    cats = {}
    per_ar = []
    for row in range(96):
        il, moe = row // 2, row & 1
        if moe:
            cat = 'after GDN mixer' if d['is_gdn'][il] else 'after GQA mixer'
        else:
            if il == 0:
                continue
            cat = 'after MoE'
        waits = []
        for G in d['gpus']:
            s, e = G['arw'][row * 2], G['arw'][row * 2 + 1]
            waits.append((e - s) * TICK_MS if s and e >= s else None)
        if any(w is None for w in waits):
            continue
        last = min(range(N), key=lambda g: waits[g])  # last to arrive waits least
        floor = waits[last]
        per_ar.append(dict(row=row, layer=il, cat=cat, wait_ms=waits, last_arriver=last, skew_ms=[w - floor for w in waits]))
        c = cats.setdefault(cat, dict(count=0, wait=[0.0] * N, skew=[0.0] * N, floor=0.0, last=[0] * N))
        c['count'] += 1
        c['floor'] += floor
        for g in range(N):
            c['wait'][g] += waits[g]; c['skew'][g] += waits[g] - floor
        c['last'][last] += 1
    return cats, per_ar


def stage_wall(d):
    """Whole-kernel wall per GPU from the host-side stamp kernels: emb, per-layer mixer/MoE, final AR, LM head."""
    res = []
    for g, G in enumerate(d['gpus']):
        tv = G['tv']
        p = 1
        r = dict(gpu=g, emb=(tv[1] - tv[0]) * TICK_MS, gdn=0.0, gqa=0.0, moe=0.0)
        for il in range(48):
            mix = (tv[p + 1] - tv[p]) * TICK_MS; mo = (tv[p + 2] - tv[p + 1]) * TICK_MS
            r['gdn' if d['is_gdn'][il] else 'gqa'] += mix; r['moe'] += mo; p += 2
        r['final_ar'] = (tv[p + 1] - tv[p]) * TICK_MS; r['lm'] = (tv[p + 2] - tv[p + 1]) * TICK_MS
        r['total'] = (tv[p + 2] - tv[0]) * TICK_MS
        res.append(r)
    return res


def parse_log(path):
    L = open(path, errors='replace').read().splitlines()
    variants, cur = [], []
    for line in L:
        cur.append(line)
        if line.startswith('{"event":"variant"'):
            variants.append(cur); cur = []
    out = []
    for v in variants:
        lab = re.search(r'"label":"([^"]*)"', v[-1]).group(1)
        r = dict(label=lab, bench={}, verify_profile=[], spec_cycle=[], spec_lines=[], bytes={})
        for line in v[-1:]:
            for k in ('ecc_ue', 'ecc_de', 'tmem_c'):
                m = re.search(r'"%s":([0-9.]+)' % k, line)
                if m: r[k] = float(m.group(1))
        vp = None
        for line in v:
            m = re.match(r'\[BENCH (tg\d+) (raw M=1|Hybrid)\s*\] ([0-9.]+) tok/s(.*)', line)
            if m:
                r['bench'][f'{m.group(1)} {m.group(2).split()[0]}'] = float(m.group(3))
                mm = re.search(r'(\d+)/(\d+) \(first_diff', m.group(4))
                if mm: r['bench'][f'{m.group(1)} {m.group(2).split()[0]} parity'] = f'{mm.group(1)}/{mm.group(2)}'
                mm = re.search(r'acc=(\d+)/(\d+)', m.group(4))
                if mm: r['bench'][f'{m.group(1)} acc'] = [int(mm.group(1)), int(mm.group(2))]
            m = re.match(r'\[VERIFY-PROFILE M=(\d+) TP=\d+ pos=(\d+) fast=(\d) n3=(\d)\]', line)
            if m:
                vp = dict(M=int(m.group(1)), pos=int(m.group(2)), fast=int(m.group(3)), stages={}); r['verify_profile'].append(vp); continue
            if vp is not None:
                m = re.match(r'  (.+?)\s+((?:GPU\d=\s*[0-9.]+\s*)+)$', line)
                if m:
                    vp['stages'][m.group(1).strip()] = [float(x) for x in re.findall(r'GPU\d=\s*([0-9.]+)', m.group(2))]
                    continue
                vp = None
            if line.startswith('[VERIFY-UNION]') and r['verify_profile']:
                r['verify_profile'][-1]['union'] = []
            elif r['verify_profile'] and 'union' in r['verify_profile'][-1] and re.match(r'^ \d+:\d+/', line):
                r['verify_profile'][-1]['union'] += [list(map(int, x.split('/'))) for x in re.findall(r'\d+:(\d+/\d+/\d+)', line)]
            m = re.match(r'\[spec_cycle_ms verifier=(\S+) source=(\S+) cycles=(\d+)\] (.*)', line)
            if m:
                r['spec_cycle'].append(dict(verifier=m.group(1), source=m.group(2), cycles=int(m.group(3)),
                                            **{k.replace('-', '_'): float(x) for k, x in re.findall(r'([a-z-]+)=([0-9.]+)', m.group(4))}))
            if line.startswith('[SPEC') or line.startswith('[CASCADE') or 'mean_accept' in line or 'acc/step' in line:
                r['spec_lines'].append(line[:400])
            m = re.match(r'\[BYTES gpu(\d+) MB/tok\] (.*)', line)
            if m:
                r['bytes'][int(m.group(1))] = {k: float(x) for k, x in re.findall(r'([a-z_]+)=([0-9.]+)', m.group(2))}
        out.append(r)
    return out


def sysfs(path):
    rows = list(csv.DictReader(open(path)))
    busy = [r for r in rows if float(r['gpu_busy']) > 50]
    if busy and path == a.load:
        lo, hi = min(float(r['t']) for r in busy), max(float(r['t']) for r in busy)
        rows = [r for r in rows if lo <= float(r['t']) <= hi]
    by = {}
    for r in rows:
        by.setdefault(r['bdf'], []).append(r)
    res = {}
    for bdf, rs in sorted(by.items()):
        f = lambda k: [float(r[k]) for r in rs if float(r[k]) >= 0]
        s = f('sclk_mhz')
        res[bdf] = dict(samples=len(rs), sclk_median=statistics.median(s), sclk_top_pct=100.0 * sum(x >= 1725 for x in s) / max(1, len(s)),
                        mclk_median=statistics.median(f('mclk_mhz')), power_median=statistics.median(f('power_w')),
                        power_max=max(f('power_w')), busy_median=statistics.median(f('gpu_busy')),
                        edge_max=max(f('edge_c')), junc_max=max(f('junc_c')), mem_max=max(f('mem_c')))
    return res


def main():
    R = dict(dumps=[], log=None, isa=None, sysfs={})
    P = print
    for path in a.dump:
        d = read_dump(path)
        st = stage_stats(d)
        cats, per_ar = ar_stats(d)
        R['dumps'].append(dict(path=path, N=d['N'], wgs=d['wgs'], stage_wall=stage_wall(d), stages=st, ar_by_cat=cats, ar_per_layer=per_ar))
    if a.log:
        R['log'] = parse_log(a.log)
    if a.isa:
        R['isa'] = json.load(open(a.isa))
    for k in ('idle', 'load'):
        if getattr(a, k):
            R['sysfs'][k] = sysfs(getattr(a, k))

    # ---------------- markdown ----------------
    for i, D in enumerate(R['dumps']):
        N = D['N']
        P(f'\n#### Dump {i}: `{D["path"].split("/")[-1]}` (TP={N}, {D["wgs"]} WGs, one profiled token)\n')
        P('| GPU | emb+rms | GDN kernels (36) | GQA kernels (12) | MoE kernels (48) | final AR | LM head | TOTAL |')
        P('|---|---|---|---|---|---|---|---|')
        for r in D['stage_wall']:
            P(f'| {r["gpu"]} | {r["emb"]:.3f} | {r["gdn"]:.3f} | {r["gqa"]:.3f} | {r["moe"]:.3f} | {r["final_ar"]:.3f} | {r["lm"]:.3f} | {r["total"]:.3f} |')
        P('\nPer-stage finish offset across workgroups (ms/token summed over layers; offset = WG stage end - previous stage critical end):\n')
        P('| GPU | stage | layers | critical | min | median | p95 | max | WG0 | tail (max-median) | slowest WG (layers) |')
        P('|---|---|---|---|---|---|---|---|---|---|---|')
        for r in D['stages']:
            if 'critical_ms' in r:
                sw = ', '.join(f'{x["wg"]}({x["layers"]})' for x in r['slowest_wg_top3'])
                P(f'| {r["gpu"]} | {r["stage"]} | {r["layers"]} | {r["critical_ms"]:.3f} | {r["min_ms"]:.3f} | {r["median_ms"]:.3f} | {r["p95_ms"]:.3f} | '
                  f'{r["max_ms"]:.3f} | {r["wg0_ms"]:.3f} | {r["tail_ms"]:.3f} | {sw} |')
        P('\nMoE Q3 sub-stamps (offset from Stage-0 AR end for s1.*, from GU end for s4.*; only the WGs doing that sub-task stamp):\n')
        P('| GPU | sub-stage | layers | WGs stamping | min | median | p95 | max |')
        P('|---|---|---|---|---|---|---|---|')
        for r in D['stages']:
            if 'critical_ms' not in r:
                P(f'| {r["gpu"]} | {r["stage"]} | {r["layers"]} | {r["wgs_stamping"]} | {r["min_ms"]:.3f} | {r["median_ms"]:.3f} | {r["p95_ms"]:.3f} | {r["max_ms"]:.3f} |')
        P('\nAllReduce WG0 spin window per GPU (ms/token summed; skew = wait - wait of the last-arriving GPU):\n')
        P('| AllReduce | count | ' + ' | '.join(f'GPU{g} wait' for g in range(N)) + ' | ' + ' | '.join(f'GPU{g} skew' for g in range(N)) +
          ' | latency floor | last arriver (GPU0/1/2) |')
        P('|---|---|' + '---|' * (2 * N + 2))
        for cat, c in D['ar_by_cat'].items():
            P(f'| {cat} | {c["count"]} | ' + ' | '.join(f'{x:.3f}' for x in c['wait']) + ' | ' + ' | '.join(f'{x:.3f}' for x in c['skew']) +
              f' | {c["floor"]:.3f} | {"/".join(str(x) for x in c["last"])} |')

    if R['log']:
        P('\n#### Runs in the log\n')
        for r in R['log']:
            P(f'- `{r["label"]}`: ' + ', '.join(f'{k} {v}' for k, v in r['bench'].items()) +
              f' | ECC UE/DE {r.get("ecc_ue", "?")}/{r.get("ecc_de", "?")} TMem {r.get("tmem_c", "?")}')
            for vp in r['verify_profile']:
                P(f'\n  VERIFY-PROFILE M={vp["M"]} pos={vp["pos"]} fast={vp["fast"]} (ms, one instrumented verify step):\n')
                P('  | stage | ' + ' | '.join(f'GPU{g}' for g in range(len(next(iter(vp['stages'].values()))))) + ' |')
                P('  |---|' + '---|' * len(next(iter(vp['stages'].values()))))
                for k, v in vp['stages'].items():
                    P(f'  | {k} | ' + ' | '.join(f'{x:.3f}' for x in v) + ' |')
            for sc in r['spec_cycle']:
                P(f'  - spec_cycle_ms {sc}')
            for s in r['spec_lines']:
                P(f'  - `{s}`')

    if a.json:
        json.dump(R, open(a.json, 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()

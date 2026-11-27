#!/usr/bin/env python3
"""Canonical VegaJet telemetry runner -> `vegajet.telemetry.v1` JSON + compact Markdown summary (runs/<label>_telemetry.*).

One command, decode-only, read-only on the host (no hardware/OS mutation; nothing needs root; tools that are missing or
need privileges are recorded as skipped with the reason). Phases:
  1. system   : PCIe chain per GPU (GPU / on-card switch / root port speed+width, root complex = Threadripper die),
                lspci -vv (MaxPayload/MaxReadReq/ACS when readable), NUMA, kernel cmdline + amdgpu params, CPU governor
                and clocks, numactl --hardware, power caps.
  2. isa      : scripts/isa_audit.py over every built .hsaco.
  3. live     : 20 Hz sampler: sysfs clocks/power/busy/temps + gpu_metrics v1.0 (VR temps, UMC/GFX activity, throttle
                bits) per card; host CPU utilisation, context switches/s, interrupts/s; RAS ECC before/after.
  4. engine   : resident warm engine (bench/warm.sh); raw M=1 with SUPER_PROFILE=2 + per-WG dump, raw unprofiled,
                Hybrid with VERIFY_PROFILE=1 + BYTES breakdown, Hybrid unprofiled. Abort on UE/DE > 0, new CE, HBM >= 65 C,
                watchdog/fault, engine exit.
  5. pmc      : rocprof only if the engine links the HSA runtime (VegaJet is KFD-direct -> skipped with reason).
  6. analysis : per-WG stage stats + AR skew (w16_telemetry), Hybrid verify fit over M, byte floors, dispatch gaps,
                automated top-10 rankings for Raw M=1 (ms/token) and Hybrid (ms/cycle and ms/token).
Usage: full_telemetry.py --label W19 [--skip-engine] [--hz 20]
"""
import argparse, csv, datetime, glob, json, os, re, shutil, statistics as st, struct, subprocess, sys, threading, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
import w16_telemetry as W  # noqa: E402  (parsers validated on W17)

ap = argparse.ArgumentParser()
ap.add_argument('--label', required=True)
ap.add_argument('--build', default=os.path.join(os.path.dirname(REPO), 'tp-x16-build'))
ap.add_argument('--bench-dir', default=os.environ.get('VJ_BENCH_DIR', ''), help='dir with warm.sh + warm.fifo (env VJ_BENCH_DIR)')
ap.add_argument('--objdump', default=os.environ.get('VJ_OBJDUMP', 'llvm-objdump'), help='LLVM objdump with gfx906 support (env VJ_OBJDUMP)')
ap.add_argument('--out', default=os.path.join(REPO, 'runs'))
ap.add_argument('--hz', type=float, default=20.0)
ap.add_argument('--skip-engine', action='store_true', help='static + idle sampling only (no GPU work)')
ap.add_argument('--env', default='VEGAJET_TP3_EP=1', help='env prefix for every engine variant')
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
LATCH = os.environ.get('VJ_TRIPPED_LATCH', '')  # optional site safety latch file; refuse to run while it exists
GPU_IDS = ('0x66a1',)


def sh(cmd, timeout=20):
    try:
        p = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except (OSError, subprocess.TimeoutExpired) as e:
        return -1, '', str(e)


def rd(p, default=''):
    try:
        with open(p) as f:
            return f.read().strip()
    except OSError:
        return default


def gpus():
    out = []
    for d in sorted(glob.glob('/sys/bus/pci/devices/*')):
        if rd(d + '/vendor') == '0x1002' and rd(d + '/device') in GPU_IDS:
            out.append(os.path.basename(d))
    return out


# ---------------------------------------------------------------- 1. system snapshot
def lspci_fields(bdf):
    rc, out, err = sh(['lspci', '-vv', '-s', bdf])
    f = {}
    if 'access denied' in out:
        f['capabilities'] = 'access denied (needs root; skipped)'
    for key, rx in (('LnkCap', r'LnkCap:\s*(.*)'), ('LnkSta', r'LnkSta:\s*(.*)'), ('DevCtl_MaxPayload', r'MaxPayload (\d+ bytes), MaxReadReq'),
                    ('MaxReadReq', r'MaxReadReq (\d+ bytes)'), ('ACSCtl', r'ACSCtl:\s*(.*)')):
        m = re.search(rx, out)
        if m:
            f[key] = m.group(1).strip()
    return f


def link(dev):
    d = '/sys/bus/pci/devices/' + dev
    return dict(bdf=dev, cur_speed=rd(d + '/current_link_speed'), cur_width=rd(d + '/current_link_width'),
                max_speed=rd(d + '/max_link_speed'), max_width=rd(d + '/max_link_width'), lspci=lspci_fields(dev))


def system_snapshot():
    S = dict(gpus=[])
    for bdf in gpus():
        path = os.path.realpath('/sys/bus/pci/devices/' + bdf).replace('/sys/devices/', '')
        chain = path.split('/')
        d = '/sys/bus/pci/devices/' + bdf
        hw = (glob.glob(d + '/hwmon/hwmon*') or [''])[0]
        S['gpus'].append(dict(bdf=bdf, root_complex=chain[0], die=('die0' if chain[0] == 'pci0000:00' else 'die1' if chain[0] == 'pci0000:40' else chain[0]),
                              chain=[link(x) for x in chain[1:]], numa_node=rd(d + '/numa_node'),
                              pp_dpm_pcie=rd(d + '/pp_dpm_pcie'), power_cap_w=float(rd(hw + '/power1_cap', '0') or 0) / 1e6,
                              vbios=rd(d + '/vbios_version')))
    cmd = rd('/proc/cmdline')
    S['cmdline'] = cmd
    S['cmdline_flags'] = {k: (k in cmd) for k in ('iommu=pt', 'amd_iommu=off', 'pci=realloc', 'amdgpu.runpm=0', 'amdgpu.ppfeaturemask')}
    S['amdgpu_params'] = {os.path.basename(p): rd(p) for p in glob.glob('/sys/module/amdgpu/parameters/*')
                          if os.path.basename(p) in ('ppfeaturemask', 'vm_fragment_size', 'runpm', 'noretry', 'ras_enable', 'lockup_timeout', 'sched_policy')}
    govs = {rd(p) for p in glob.glob('/sys/devices/system/cpu/cpu*/cpufreq/scaling_governor')}
    freqs = [int(rd(p, '0') or 0) / 1000 for p in glob.glob('/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq')]
    S['cpu'] = dict(governors=sorted(govs), cur_mhz_min=min(freqs) if freqs else None, cur_mhz_median=st.median(freqs) if freqs else None,
                    cur_mhz_max=max(freqs) if freqs else None, boost=rd('/sys/devices/system/cpu/cpufreq/boost'))
    rc, out, _ = sh(['lscpu'])
    S['lscpu'] = {k.strip(): v.strip() for k, v in (l.split(':', 1) for l in out.splitlines() if ':' in l)
                  if k.strip() in ('Model name', 'Socket(s)', 'Core(s) per socket', 'Thread(s) per core', 'NUMA node(s)', 'L3 cache',
                                   'NUMA node0 CPU(s)', 'NUMA node1 CPU(s)', 'CPU max MHz')}
    rc, out, err = sh(['numactl', '--hardware'])
    S['numactl_hardware'] = out.strip() if rc == 0 else f'skipped: {err.strip() or "numactl missing"}'
    return S


# ---------------------------------------------------------------- 3. live sampler
GM_NAMES = ('ts edge hotspot hbm vrgfx vrsoc vrmem gfx_act umc_act mm_act power energy avg_gfx avg_soc avg_uclk av0 ad0 av1 ad1 '
            'cur_gfx cur_soc cur_uclk cv0 cd0 cv1 cd1 throttle fan pcie_w pcie_s').split()


def gpu_metrics(bdf):
    try:
        b = open(f'/sys/bus/pci/devices/{bdf}/gpu_metrics', 'rb').read()
    except OSError:
        return None
    size, frev, crev = struct.unpack_from('<HBB', b, 0)
    if (frev, crev) != (1, 0) or len(b) < 76:
        return dict(format=f'{frev}.{crev}', note='unsupported layout (only v1.0 decoded)')
    return dict(zip(GM_NAMES, struct.unpack_from('<Q6H3HHI7H7HIHBB', b, 8)))


def active_mhz(p):
    for line in rd(p).splitlines():
        if '*' in line:
            try:
                return float(line.split(':', 1)[1].lower().split('mhz')[0])
            except ValueError:
                return -1.0
    return -1.0


def ras(bdf):
    t = rd(f'/sys/bus/pci/devices/{bdf}/ras/umc_err_count')
    m = dict(re.findall(r'(ue|ce|de):\s*(\d+)', t))
    return {k: int(v) for k, v in m.items()}


class Sampler(threading.Thread):
    def __init__(self, path, hz):
        super().__init__(daemon=True)
        self.path, self.period, self.stop_ev = path, 1.0 / hz, threading.Event()
        self.cards = gpus()

    @staticmethod
    def proc_stat():
        cpu = ctxt = intr = 0
        for line in open('/proc/stat'):
            if line.startswith('cpu '):
                v = list(map(int, line.split()[1:]))
                cpu = (sum(v), v[3] + v[4])  # total, idle+iowait
            elif line.startswith('ctxt'):
                ctxt = int(line.split()[1])
            elif line.startswith('intr'):
                intr = int(line.split()[1])
        return cpu, ctxt, intr

    def run(self):
        cols = ['t', 'bdf', 'sclk_mhz', 'mclk_mhz', 'power_w', 'gpu_busy', 'mem_busy', 'edge_c', 'junc_c', 'mem_c',
                'gm_vrgfx_c', 'gm_vrsoc_c', 'gm_vrmem_c', 'gm_gfx_act', 'gm_umc_act', 'gm_cur_gfx', 'gm_throttle',
                'host_cpu_util', 'host_ctxt_s', 'host_intr_s']
        t0 = time.monotonic(); nxt = t0; prev = self.proc_stat(); tprev = t0
        with open(self.path, 'w', newline='') as f:
            w = csv.writer(f); w.writerow(cols)
            while not self.stop_ev.is_set():
                now = time.monotonic()
                cur = self.proc_stat(); dt = max(now - tprev, 1e-6)
                (tt, ti), (pt, pi) = cur[0], prev[0]
                util = 100.0 * (1 - (ti - pi) / max(tt - pt, 1)); cs = (cur[1] - prev[1]) / dt; it = (cur[2] - prev[2]) / dt
                prev, tprev = cur, now
                for bdf in self.cards:
                    d = '/sys/bus/pci/devices/' + bdf; hw = (glob.glob(d + '/hwmon/hwmon*') or [''])[0]
                    num = lambda p, s=1.0: (float(rd(p, '-1') or -1) / s)
                    gm = gpu_metrics(bdf) or {}
                    w.writerow([f'{now - t0:.3f}', bdf, active_mhz(d + '/pp_dpm_sclk'), active_mhz(d + '/pp_dpm_mclk'),
                                num(hw + '/power1_input', 1e6), num(d + '/gpu_busy_percent'), num(d + '/mem_busy_percent'),
                                num(hw + '/temp1_input', 1e3), num(hw + '/temp2_input', 1e3), num(hw + '/temp3_input', 1e3),
                                gm.get('vrgfx', -1), gm.get('vrsoc', -1), gm.get('vrmem', -1), gm.get('gfx_act', -1), gm.get('umc_act', -1),
                                gm.get('cur_gfx', -1), gm.get('throttle', -1), f'{util:.1f}', f'{cs:.0f}', f'{it:.0f}'])
                f.flush()
                nxt += self.period
                self.stop_ev.wait(max(0.0, nxt - time.monotonic()))


def summarize_samples(path):
    rows = list(csv.DictReader(open(path)))
    busy = [r for r in rows if float(r['gpu_busy']) >= 90]
    res = dict(samples=len(rows), busy_samples=len(busy), cards={})
    by = {}
    for r in busy:
        by.setdefault(r['bdf'], []).append(r)
    for bdf, rs in sorted(by.items()):
        f = lambda k: [float(r[k]) for r in rs if float(r[k]) >= 0]
        s = f('sclk_mhz')
        thr = {}
        for r in rs:
            thr[r['gm_throttle']] = thr.get(r['gm_throttle'], 0) + 1
        med = lambda k: st.median(f(k)) if f(k) else None
        mx = lambda k: max(f(k)) if f(k) else None
        res['cards'][bdf] = dict(n=len(rs), sclk_median=st.median(s), pct_at_1725=round(100 * sum(x >= 1725 for x in s) / len(s), 1),
                                 pct_le_1606=round(100 * sum(x <= 1606 for x in s) / len(s), 1), mclk_median=med('mclk_mhz'),
                                 power_median=med('power_w'), power_max=mx('power_w'), mem_busy_median=med('mem_busy'),
                                 umc_activity_median=med('gm_umc_act'), gfx_activity_median=med('gm_gfx_act'),
                                 edge_max=mx('edge_c'), junc_max=mx('junc_c'), hbm_max=mx('mem_c'),
                                 vrgfx_max=mx('gm_vrgfx_c'), vrsoc_max=mx('gm_vrsoc_c'), vrmem_max=mx('gm_vrmem_c'),
                                 throttle_bits_hist=thr)
    hb = [r for r in busy]
    if hb:
        g = lambda k: [float(r[k]) for r in hb]
        res['host'] = dict(cpu_util_median=st.median(g('host_cpu_util')), ctxt_per_s_median=st.median(g('host_ctxt_s')),
                           intr_per_s_median=st.median(g('host_intr_s')), intr_per_s_max=max(g('host_intr_s')))
    return res


# ---------------------------------------------------------------- 4. engine
class Engine:
    def __init__(self, label):
        self.bench = a.bench_dir; self.label = label
        self.log = os.path.join(self.bench, 'runs', f'{label}.log'); self.fifo = os.path.join(self.bench, 'warm.fifo')
        self.proc = None

    def alive(self):
        return sh(['pgrep', '-x', 'vegajet'])[0] == 0

    def start(self):
        self.proc = subprocess.Popen(['./warm.sh', self.label], cwd=self.bench, stdout=open(self.log + '.warm.out', 'w'), stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 900:
            txt = rd(self.log)
            if '"event":"ready"' in txt:
                return True
            if 'end rc' in txt or self.proc.poll() is not None:
                return False
            time.sleep(2)
        return False

    def send(self, line):
        # O_NONBLOCK: fails with ENXIO when no engine has the FIFO open (never leaves a blocked writer behind; W17 lesson)
        fd = os.open(self.fifo, os.O_WRONLY | os.O_NONBLOCK)
        try:
            os.write(fd, (line + '\n').encode())
        finally:
            os.close(fd)

    def run(self, env, mode, timeout=1200):
        n0 = len(rd(self.log).splitlines())
        self.send(f'run {env} --bench-mode {mode}')
        t0 = time.time()
        while time.time() - t0 < timeout:
            lines = rd(self.log).splitlines()[n0:]
            txt = '\n'.join(lines)
            if re.search(r'"event":"(variant|error)"|WATCHDOG|FATAL|end rc', txt):
                return lines
            if not self.alive():
                return lines + ['[full_telemetry] engine exited']
            time.sleep(1)
        return lines + ['[full_telemetry] timeout']

    def quit(self):
        if self.alive():
            try:
                self.send('quit')
            except OSError:
                pass
        for _ in range(60):
            if not self.alive():
                return
            time.sleep(1)


def safety(ecc0):
    cur = {b: ras(b) for b in gpus()}
    bad = []
    for b, v in cur.items():
        e0 = ecc0.get(b, {})
        if v.get('ue', 0) > e0.get('ue', 0) or v.get('de', 0) > e0.get('de', 0):
            bad.append(f'{b} UE/DE increased')
        if v.get('ce', 0) > e0.get('ce', 0):
            bad.append(f'{b} CE +{v["ce"] - e0.get("ce", 0)}')
        hbm = float(rd(((glob.glob(f'/sys/bus/pci/devices/{b}/hwmon/hwmon*') or [''])[0]) + '/temp3_input', '0') or 0) / 1e3
        if hbm >= 65:
            bad.append(f'{b} HBM {hbm:.0f} C')
    if LATCH and os.path.exists(LATCH):
        bad.append('TRIPPED latch present')
    return bad


# ---------------------------------------------------------------- 6. analysis
def peaks():
    try:
        return [g['bw_hbm_gbs'] for g in json.load(open(os.path.expanduser('~/.config/vegajet/hw_profile.json')))['gpus']]
    except (OSError, KeyError, ValueError):
        return None


def fit(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
    return my - b * mx, b


def raw_ranking(D, unprof_ms, bytes_, pk):
    """Disjoint Raw M=1 ledger (ms/token); unprofiled-equivalent = profiled x scale."""
    N = D['N']; isg = D['is_gdn']
    S = lambda g: {x['stage']: x for x in D['stages'] if x['gpu'] == g}
    tot = st.mean(w['total'] for w in D['stage_wall']); scale = unprof_ms / tot if unprof_ms else 1.0
    bar = st.mean((S(g)['MoE.Rtr+ShGU']['critical_ms'] - S(g)['MoE.s1.router_done']['max_ms']) / 48 for g in range(N)
                  if 'MoE.s1.router_done' in S(g)) if any('MoE.s1.router_done' in S(g) for g in range(N)) else None
    ngdn = sum(isg); ngqa = 48 - ngdn
    nbar = (3 * ngdn + 5 * ngqa + 4 * 48) - 1  # layer 0 has no Stage-0 AR barrier
    led = {}
    for g in range(N):
        s = S(g); w = D['stage_wall'][g]; c = lambda k: s[k]['critical_ms'] if k in s else 0.0; t = lambda k: s[k].get('tail_ms', 0.0) if k in s else 0.0
        ar = D['ar_by_cat']
        wait = sum(v['wait'][g] for v in ar.values()); floor = sum(v['floor'] for v in ar.values())
        meanskew = st.mean(sum(v['skew'][h] for v in ar.values()) for h in range(N))
        b = bytes_.get(g) if bytes_ else None
        bb = bar or 0.0
        items = {
            'Full-grid barriers (%d/token x %.2f us)' % (nbar, bb * 1000): nbar * bb,
            'AllReduce reduce+RMS+quant on WG0 (net of wait, barrier)': c('GDN.AR') + c('GQA.AR') + c('MoE.AR') - wait - (ngdn - 1 + ngqa + 48) * bb,
            'Kernel-boundary gaps (wall - in-kernel critical span; incl. stamp kernels)': (w['gdn'] + w['gqa'] + w['moe']) - sum(x['critical_ms'] for x in s.values() if 'critical_ms' in x),
            'GQA prep+flash+merge (net of barriers)': c('GQA.Prep') + c('GQA.Flash') + c('GQA.Mrg') - 3 * ngqa * bb,
            'Top-8 softmax (every WG)': c('MoE.Top8'),
            'TP shard imbalance (mean AR skew)': meanskew,
            'Non-barrier WG tails (Down+PP, GDN.Out, GQA.Out)': t('MoE.Down+PP') + t('GDN.Out') + t('GQA.Out'),
            'AllReduce latency floor': floor,
            'MoE Stage-1 shGU WGs idle (router finishes later)': (s['MoE.s1.router_done']['max_ms'] - s['MoE.s1.shGU_done']['max_ms']) if 'MoE.s1.router_done' in s else 0.0,
        }
        if b and pk:
            p = pk[g]
            items.update({
                'BW gap: MoE router+shared+Down': c('MoE.Rtr+ShGU') + c('MoE.Down+PP') - 2 * 48 * bb - t('MoE.Down+PP') - (b['router'] + b['shared'] + b['routed_dn']) / p,
                'BW gap: GDN state (conv+recurrence)': c('GDN.ConvRec') - ngdn * bb - b['gdn_state'] / p,
                'BW gap: GDN weights': c('GDN.QKV') + c('GDN.Out') - ngdn * bb - t('GDN.Out') - b['gdn_w'] / p,
                'BW gap: routed Gate+Up': c('MoE.GU_DOT8') - 48 * bb - b['routed_gu'] / p,
                'BW gap: GQA weights': c('GQA.QKV') + c('GQA.Out') - ngqa * bb - t('GQA.Out') - b['gqa_w'] / p,
                'BW gap: LM head': w['lm'] - b['lm_head'] / p,
            })
        for k, v in items.items():
            led.setdefault(k, []).append(v)
    rows = [dict(item=k, profiled_ms_mean=st.mean(v), profiled_ms_max_gpu=max(v), ms_per_token=st.mean(v) * scale) for k, v in led.items()]
    rows.sort(key=lambda r: -r['ms_per_token'])
    return dict(scale=scale, barrier_us=(bar or 0) * 1000, barriers_per_token=nbar, items=rows[:10], all_items=rows)


def hybrid_ranking(spec_runs, bytes_, pk):
    vp = [v for r in spec_runs for v in r['verify_profile']]
    cyc = [c for r in spec_runs for c in r['spec_cycle']]
    if not vp:
        return None
    Ms = [v['M'] for v in vp]
    vmax = max(vp, key=lambda v: v['M'])
    N = len(next(iter(vmax['stages'].values())))
    crit = min(range(N), key=lambda g: vmax['stages'].get('AllReduce+norm+quant', [0] * N)[g])  # least AR wait = critical
    # tokens/cycle of the hybrid run that used this M most
    tpc = None
    for r in spec_runs:
        for l in r['spec_lines']:
            m = re.search(r'SPEC-STATS Hybrid.*tok/cyc=([0-9.]+)', l)
            if m:
                tpc = float(m.group(1))
    rows = []
    dense = 0.0
    fits = {}
    for stg, vals in vmax['stages'].items():
        if stg == 'TOTAL':
            continue
        aa, bb = fit(Ms, [v['stages'][stg][crit] for v in vp]) if len(set(Ms)) >= 2 else (vals[crit], 0.0)
        fits[stg] = dict(ms=vals, a=aa, b=bb, per_row_at_M=bb * vmax['M'])
        if 'GEMM' in stg and 'expert sort' not in stg:
            dense += bb * vmax['M']
    ar = vmax['stages'].get('AllReduce+norm+quant')
    if ar:
        mn = min(ar)
        rows.append(dict(item='TP imbalance at AllReduce (mean skew)', ms_per_cycle=sum(x - mn for x in ar) / N))
    rows.append(dict(item=f'Dense verify GEMM per-row compute (sum of b*M over GEMM stages, M={vmax["M"]})', ms_per_cycle=dense))
    if bytes_ and pk:
        b = bytes_.get(crit); p = pk[crit]
        U = st.mean(u[0] for u in vmax.get('union', [])) if vmax.get('union') else None
        def gap(stg, mb):
            if stg in vmax['stages'] and mb:
                rows.append(dict(item=f'BW gap: {stg}', ms_per_cycle=vmax['stages'][stg][crit] - mb / p, mb=mb, achieved_gbs=mb / vmax['stages'][stg][crit]))
        gap('routed Down', b['routed_dn'] * U / 8 if U else None)
        gap('routed Gate+Up', b['routed_gu'] * U / 8 if U else None)
        gap('vocab argmax', b['lm_head'])
    for stg in ('GDN recurrence (WY)', 'top-8 + shared SwiGLU', 'expert sort + shared Down GEMM', 'GDN conv', 'attention'):
        if stg in vmax['stages']:
            rows.append(dict(item=f'Stage: {stg}', ms_per_cycle=vmax['stages'][stg][crit]))
    if ar:
        rows.append(dict(item='AllReduce on critical GPU (floor + work)', ms_per_cycle=ar[crit]))
    for r in rows:
        r['ms_per_token'] = r['ms_per_cycle'] / tpc if tpc else None
    rows.sort(key=lambda r: -r['ms_per_cycle'])
    hyb = [c for c in cyc if c['cycles'] >= 10]
    gaps = [dict(source=c['source'], cycles=c['cycles'], total=c['total'], parts=c['draft'] + c['verify'] + c['catch_up'] + c['recompute_on_reject'],
                 host_gap_ms=c['total'] - (c['draft'] + c['verify'] + c['catch_up'] + c['recompute_on_reject'])) for c in cyc]
    wall_vs_gpu = None
    for r in spec_runs:
        for l in r['spec_lines']:
            m = re.search(r'verify:.*M=%d ([0-9.]+) ms' % vmax['M'], l)
            if m:
                wall_vs_gpu = dict(M=vmax['M'], verify_wall_ms=float(m.group(1)), gpu_stamp_span_ms=max(vmax['stages']['TOTAL']),
                                   dispatch_gap_ms=float(m.group(1)) - max(vmax['stages']['TOTAL']))
    return dict(critical_gpu=crit, tokens_per_cycle=tpc, M_points=Ms, stage_fits=fits, items=rows[:10], cycle_parts=gaps,
                verify_wall_vs_gpu_span=wall_vs_gpu)


# ---------------------------------------------------------------- main
def main():
    label = a.label
    stamp = datetime.datetime.now().isoformat(timespec='seconds')
    T = {'schema': 'vegajet.telemetry.v1', 'label': label, 'started': stamp, 'host': os.uname().nodename,
         'git': sh(['git', '-C', REPO, 'rev-parse', '--short', 'HEAD'])[1].strip(), 'skipped': {}}
    if LATCH and os.path.exists(LATCH):
        sys.exit('TRIPPED latch present - not running')
    if not a.skip_engine and not os.path.isfile(os.path.join(a.bench_dir, 'warm.sh')):
        sys.exit('--bench-dir (or VJ_BENCH_DIR) must point at a directory with warm.sh / warm.fifo')
    if sh(['pgrep', '-x', 'vegajet'])[0] == 0 and not a.skip_engine:
        sys.exit('GPU busy (vegajet running) - not running')
    print('[1/6] system snapshot'); T['system'] = system_snapshot()
    print('[2/6] ISA audit')
    isa_json = os.path.join(a.out, f'.{label}_isa.json')
    rc, out, err = sh(['python3', os.path.join(REPO, 'scripts/isa_audit.py'), '--objdump', a.objdump, '--json', isa_json] +
                      sorted(glob.glob(os.path.join(a.build, 'hsaco', '*.hsaco'))), timeout=600)
    T['isa'] = json.load(open(isa_json)) if rc == 0 else None
    if rc != 0:
        T['skipped']['isa'] = err[-300:]
    else:
        os.remove(isa_json)
    ecc0 = {b: ras(b) for b in gpus()}
    T['ecc_before'] = ecc0
    samples = os.path.join(a.out, f'{label}_samples.csv')
    smp = Sampler(samples, a.hz); smp.start()
    time.sleep(3)  # idle baseline
    T['engine'] = {'variants': []}
    log_lines = []
    abort = None
    if a.skip_engine:
        T['skipped']['engine'] = '--skip-engine'
    else:
        print('[4/6] engine (warm)')
        E = Engine(label)
        if not E.start():
            abort = 'engine failed to become ready'
        else:
            dump = os.path.join(a.bench_dir, 'runs', f'{label}_wg.bin')
            plan = [('raw_profiled', 'raw', f'{a.env} VEGAJET_SUPER_PROFILE=2 VEGAJET_PROFILE_DUMP={dump}'),
                    ('raw', 'raw', a.env),
                    ('hybrid_profiled', 'spec', f'{a.env} VEGAJET_VERIFY_PROFILE=1 VEGAJET_BYTES_BREAKDOWN=1'),
                    ('hybrid', 'spec', a.env)]
            for name, mode, env in plan:
                print(f'      {name}: {env} --bench-mode {mode}')
                lines = E.run(env, mode)
                log_lines += lines
                bad = safety(ecc0)
                fatal = [l for l in lines if re.search(r'WATCHDOG|FATAL|engine exited|timeout', l)]
                T['engine']['variants'].append(dict(name=name, mode=mode, env=env, safety=bad, fatal=fatal[:5]))
                if bad or fatal:
                    abort = f'{name}: ' + '; '.join(bad + fatal[:2]); break
            E.quit()
            if os.path.exists(dump):
                T['engine']['wg_dump'] = dump
    smp.stop_ev.set(); smp.join(5)
    T['ecc_after'] = {b: ras(b) for b in gpus()}
    T['abort'] = abort
    print('[5/6] PMC')
    rc, out, _ = sh(['ldd', os.path.join(a.build, 'vegajet')])
    if not re.search(r'libhsa-runtime|libamdhip', out):
        T['skipped']['pmc'] = ('engine is KFD-direct (no HSA/HIP runtime linked); rocprof/rocprofv2/rocprofv3 intercept HSA '
                               'dispatches only and cannot observe VegaJet kernels')
    else:
        T['skipped']['pmc'] = 'not implemented for HSA builds'
    T['pmc'] = None
    print('[6/6] analysis')
    T['sysfs'] = summarize_samples(samples)
    eng_log = os.path.join(a.out, f'{label}_engine.log')
    open(eng_log, 'w').write('\n'.join(log_lines))
    runs = W.parse_log(eng_log) if log_lines else []
    T['engine']['runs'] = runs
    bytes_ = next((r['bytes'] for r in runs if r['bytes']), None)
    bytes_ = {int(k): v for k, v in bytes_.items()} if bytes_ else None
    pk = peaks()
    T['engine']['bytes_mb_per_token'] = bytes_; T['engine']['hbm_peak_gbs'] = pk
    raw_unprof = next((r['bench'].get('tg128 raw') for r in runs if 'PROFILE' not in r['label'] and 'raw' in r['label']), None)
    hyb = next((r['bench'] for r in runs if 'PROFILE' not in r['label'] and 'spec' in r['label']), None)
    T['headline'] = dict(raw_tg128_tps=raw_unprof, hybrid=hyb)
    T['rankings'] = {}
    if T['engine'].get('wg_dump'):
        D = W.read_dump(T['engine']['wg_dump'])
        dd = dict(N=D['N'], wgs=D['wgs'], is_gdn=D['is_gdn'], stage_wall=W.stage_wall(D), stages=W.stage_stats(D))
        dd['ar_by_cat'], dd['ar_per_layer'] = W.ar_stats(D)
        T['engine']['raw_profile'] = dd
        T['rankings']['raw_m1'] = raw_ranking(dd, 1000.0 / raw_unprof if raw_unprof else None, bytes_, pk)
    spec_runs = [r for r in runs if r['verify_profile']]
    if spec_runs:
        T['rankings']['hybrid'] = hybrid_ranking(spec_runs + [r for r in runs if 'spec' in r['label'] and not r['verify_profile']], bytes_, pk)
    T['finished'] = datetime.datetime.now().isoformat(timespec='seconds')
    jp = os.path.join(a.out, f'{label}_telemetry.json')
    json.dump(T, open(jp, 'w'), indent=1, default=str)
    md = summary_md(T)
    mp = os.path.join(a.out, f'{label}_telemetry.md'); open(mp, 'w').write(md)
    print(md); print(f'\n-> {jp}\n-> {mp}')
    sys.exit(2 if abort else 0)


def summary_md(T):
    L = [f'# Telemetry {T["label"]} ({T["started"]}, git {T["git"]}, schema {T["schema"]})', '']
    if T.get('abort'):
        L += [f'**ABORTED:** {T["abort"]}', '']
    h = T.get('headline', {})
    hy = h.get('hybrid') or {}
    L += [f'- Raw M=1 tg128: **{h.get("raw_tg128_tps")} tok/s**; Hybrid tg128 **{hy.get("tg128 Hybrid")}** ({hy.get("tg128 Hybrid parity")}), '
          f'tg512 **{hy.get("tg512 Hybrid")}**',
          f'- ECC before {T["ecc_before"]} -> after {T["ecc_after"]}', f'- Skipped: {T["skipped"]}', '']
    L += ['## System', '', '| GPU | die | chain (speed/width) | NUMA | cap W |', '|---|---|---|---|---|']
    for g in T['system']['gpus']:
        ch = ' -> '.join(f'{c["bdf"]} {c["cur_speed"].replace(" PCIe", "")} x{c["cur_width"]}' for c in g['chain'])
        L.append(f'| {g["bdf"]} | {g["die"]} | {ch} | {g["numa_node"]} | {g["power_cap_w"]:.0f} |')
    c = T['system']['cpu']
    L += ['', f'CPU {T["system"]["lscpu"].get("Model name")} governors {c["governors"]} MHz {c["cur_mhz_min"]}/{c["cur_mhz_median"]}/{c["cur_mhz_max"]}; '
          f'cmdline flags {T["system"]["cmdline_flags"]}', '']
    s = T.get('sysfs', {})
    if s.get('cards'):
        L += ['## Under load (busy samples)', '', '| card | sclk med | % <=1606 | mclk | W med/max | UMC act | junc/HBM/VRgfx/VRmem max | throttle bits |', '|---|---|---|---|---|---|---|---|']
        for b, v in s['cards'].items():
            L.append(f'| {b} | {v["sclk_median"]:.0f} | {v["pct_le_1606"]} | {v["mclk_median"]} | {v["power_median"]}/{v["power_max"]} | {v["umc_activity_median"]} | '
                     f'{v["junc_max"]}/{v["hbm_max"]}/{v["vrgfx_max"]}/{v["vrmem_max"]} | {v["throttle_bits_hist"]} |')
        if s.get('host'):
            L += ['', f'Host: CPU util median {s["host"]["cpu_util_median"]:.1f}%, ctxt/s {s["host"]["ctxt_per_s_median"]:.0f}, '
                  f'IRQ/s {s["host"]["intr_per_s_median"]:.0f} (max {s["host"]["intr_per_s_max"]:.0f})', '']
    R = T.get('rankings', {})
    if R.get('raw_m1'):
        r = R['raw_m1']
        L += ['## Top-10 Raw M=1 (ms/token, unprofiled-equivalent)', '', f'scale {r["scale"]:.3f}, barrier {r["barrier_us"]:.2f} us x {r["barriers_per_token"]}', '',
              '| # | item | ms/token |', '|---|---|---|']
        L += [f'| {i+1} | {x["item"]} | {x["ms_per_token"]:.3f} |' for i, x in enumerate(r['items'])]
    if R.get('hybrid'):
        r = R['hybrid']
        L += ['', f'## Top-10 Hybrid (critical GPU{r["critical_gpu"]}, {r["tokens_per_cycle"]} tok/cycle)', '', '| # | item | ms/cycle | ms/token |', '|---|---|---|---|']
        L += [f'| {i+1} | {x["item"]} | {x["ms_per_cycle"]:.3f} | {x["ms_per_token"]:.3f} |' if x['ms_per_token'] is not None else
              f'| {i+1} | {x["item"]} | {x["ms_per_cycle"]:.3f} | n/a |' for i, x in enumerate(r['items'])]
        L += ['', f'Verify wall vs GPU stamp span: {r["verify_wall_vs_gpu_span"]}; cycle host gaps: ' +
              ', '.join(f'{g["source"]}x{g["cycles"]}: {g["host_gap_ms"]:.3f} ms' for g in r['cycle_parts'])]
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    main()

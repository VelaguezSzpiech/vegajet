#!/usr/bin/env python3
"""Read-only high-rate sysfs sampler for the MI50 cards (no writes, no GPU access).

Usage: sysfs_sampler.py OUT.csv DURATION_S [HZ]
Columns per sample and card: active sclk/mclk (MHz, the '*' DPM level), power1_average (W), gpu_busy_percent,
mem_busy_percent (if present), edge/junction/mem temperature (C).
"""
import glob, os, sys, time

out, dur = sys.argv[1], float(sys.argv[2])
hz = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0

cards = []
for d in sorted(glob.glob('/sys/bus/pci/devices/*')):
    try:
        if open(d + '/vendor').read().strip() != '0x1002' or open(d + '/device').read().strip() != '0x66a1':
            continue
    except OSError:
        continue
    hw = sorted(glob.glob(d + '/hwmon/hwmon*'))[0]
    cards.append((os.path.basename(d), d, hw))


def rd(p):
    try:
        with open(p) as f:
            return f.read()
    except OSError:
        return ''


def active(p):
    for line in rd(p).splitlines():
        if '*' in line:
            v = line.split(':', 1)[1].strip().lower().split('mhz')[0].strip()
            try:
                return float(v)
            except ValueError:
                return -1.0
    return -1.0


def num(p, scale=1.0):
    try:
        return float(rd(p).strip()) / scale
    except ValueError:
        return -1.0


with open(out, 'w') as f:
    f.write('t,bdf,sclk_mhz,mclk_mhz,power_w,gpu_busy,mem_busy,edge_c,junc_c,mem_c\n')
    t0 = time.monotonic()
    period = 1.0 / hz
    nxt = t0
    while True:
        now = time.monotonic()
        if now - t0 > dur:
            break
        for bdf, d, hw in cards:
            f.write('%.3f,%s,%.0f,%.0f,%.1f,%.0f,%.0f,%.1f,%.1f,%.1f\n' % (
                now - t0, bdf, active(d + '/pp_dpm_sclk'), active(d + '/pp_dpm_mclk'),
                num(hw + '/power1_input', 1e6), num(d + '/gpu_busy_percent'), num(d + '/mem_busy_percent'),
                num(hw + '/temp1_input', 1e3), num(hw + '/temp2_input', 1e3), num(hw + '/temp3_input', 1e3)))
        f.flush()
        nxt += period
        time.sleep(max(0.0, nxt - time.monotonic()))

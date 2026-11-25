#!/usr/bin/env python3
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Compare VegaJet BENCH logs; repeated observations use medians, never invented metrics."""
import argparse
import math
from pathlib import Path
import re
import statistics
import sys

NUMBER = r'(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'


def parse(text):
    values = {}
    def add(name, value, unit):
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError('invalid metric')
        values.setdefault((name, unit), []).append(value)
    for line in text.splitlines():
        match = re.search(r'\[BENCH (pp512|pp2048|tg128 raw|tg128 MTP)[^\]]*\]\s*(' + NUMBER + r')\s+tok/s', line)
        if match:
            name = {'tg128 raw': 'raw M=1', 'tg128 MTP': 'MTP'}.get(match[1], match[1])
            add(name, match[2], 'tok/s')
            latency = re.search(r'\((' + NUMBER + r') ms/tok\)', line)
            if latency:
                add(name, latency[1], 'ms/tok')
            acceptance = re.search(r'acc=(\d+)/(\d+)', line)
            if acceptance and int(acceptance[2]):
                add(name + ' acceptance', 100 * int(acceptance[1]) / int(acceptance[2]), '%')
        # Explicit normalized bucket syntax. Units are never guessed or converted from ticks.
        bucket = re.search(r'\[s_memrealtime\(\)\s+([A-Za-z0-9_.-]+)\]\s*(' + NUMBER + r')\s+(ticks|ns|us|ms)\b', line)
        if bucket:
            add('bucket ' + bucket[1], bucket[2], bucket[3])
        timeline = re.search(r'gpu(\d+) AR#(\d+)/layer: pack (' + NUMBER + r') \| post flag (' + NUMBER + r') \| wait for peer (' + NUMBER + r').*?\| reduce (' + NUMBER + r') \| norm\+quant (' + NUMBER + r') \| work until next AR (' + NUMBER + r')', line)
        if timeline:
            for label, value in zip(('pack', 'post', 'peer-wait', 'reduce', 'norm-quant', 'next-work'), timeline.groups()[2:]):
                add('bucket gpu' + timeline[1] + ' AR' + timeline[2] + ' ' + label, value, 'us/token')
        layer_bucket = re.search(r'^\s*(?:\(|->\s*)([A-Z][0-9][a-z]?[0-9]?)[):].*?\| (36|48)L=(' + NUMBER + r') ms \(best (' + NUMBER + r')\)', line)
        if layer_bucket:
            add('bucket ' + layer_bucket[1] + ' ' + layer_bucket[2] + 'L mean', layer_bucket[3], 'ms')
            add('bucket ' + layer_bucket[1] + ' ' + layer_bucket[2] + 'L best', layer_bucket[4], 'ms')
    return values


def scorecard(before, after):
    mandatory = {('raw M=1', 'tok/s'), ('pp512', 'tok/s'), ('pp2048', 'tok/s'), ('MTP', 'tok/s')}
    lines = ['| Metric | Unit | Before median (n) | After median (n) | Delta | Change |',
             '|---|---|---:|---:|---:|---|']
    for key in sorted(set(before) | set(after) | mandatory):
        a, b = before.get(key, []), after.get(key, [])
        av, bv = (statistics.median(a) if a else None), (statistics.median(b) if b else None)
        af = f'{av:.6g} ({len(a)})' if a else 'missing'
        bf = f'{bv:.6g} ({len(b)})' if b else 'missing'
        delta, change = 'n/a', 'not comparable'
        if av is not None and bv is not None and av != 0:
            percent = 100 * (bv - av) / av
            delta = f'{percent:+.2f}%'
            if key[1] == '%':
                change = 'descriptive only'
            else:
                beneficial = percent > 0 if key[1] == 'tok/s' else percent < 0
                change = 'unchanged' if percent == 0 else ('better' if beneficial else 'worse')
        lines.append(f'| {key[0]} | {key[1]} | {af} | {bf} | {delta} | {change} |')
    lines += ['', 'Deltas are descriptive, not significance tests. Different units are not comparable.',
              'Bucket syntax: `[s_memrealtime() name] value ticks|ns|us|ms`; tick frequency is not assumed.']
    return '\n'.join(lines)


def self_test():
    text = '[BENCH tg128 raw M=1] 50.00 tok/s (20.00 ms/tok)\n[BENCH pp512 M=256] 100 tok/s\n[BENCH tg128 MTP K=4] 60 tok/s acc=3/4\n[s_memrealtime() attention] 20 ticks\n'
    parsed = parse(text)
    assert parsed[('raw M=1', 'tok/s')] == [50]
    assert parsed[('MTP acceptance', '%')] == [75]
    assert parsed[('bucket attention', 'ticks')] == [20]
    output = scorecard(parsed, parse(text.replace('50.00', '55.00').replace('20 ticks', '10 ticks')))
    assert '+10.00%' in output and '-50.00%' in output and 'missing' in output
    assert 'n/a' in scorecard({('MTP', 'tok/s'): [0]}, {('MTP', 'tok/s'): [2]})
    assert parse('unrelated 99 tok/s') == {}
    assert len(parse(text + text)[('raw M=1', 'tok/s')]) == 2
    timeline = parse('gpu0 AR#1/layer: pack 10 | post flag 2 | wait for peer 7 (2 of 96 ARs waited >3 us) | reduce 3 | norm+quant 4 | work until next AR 9')
    assert len(timeline) == 6 and timeline[('bucket gpu0 AR1 peer-wait', 'us/token')] == [7]
    cold = parse('  (F0) baseline | 48L=2.5 ms (best 2.0) | 100.0 real_GB/s/GPU')
    assert cold[('bucket F0 48L mean', 'ms')] == [2.5]
    print('bench_diff self-test: PASS')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', nargs='?', type=Path)
    parser.add_argument('after', nargs='?', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.before or not args.after:
        parser.error('two log files required')
    try:
        before, after = parse(args.before.read_text()), parse(args.after.read_text())
        if not before or not after:
            raise ValueError('one log contains no recognized metrics')
        print(scorecard(before, after))
    except (OSError, ValueError):
        print('FAIL: log unavailable or no valid comparable metrics', file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()

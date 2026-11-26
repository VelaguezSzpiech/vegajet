#!/usr/bin/env python3
"""Offline GDN M=1 weight+state streaming lower bound; never opens a GPU.

Uses logical Q4_A8 traffic (0.5 B nibbles + 0.125 B FP16 affine scales/element)
and FP32 state read+write. Excludes padding, caches, activations, conv, launch/AR,
and execution inefficiency. All terms scale with whole K groups, as in the kernel.
"""
import argparse
import itertools


def compositions(total, ranks):
    for cuts in itertools.combinations(range(1, total), ranks - 1):
        edges = (0,) + cuts + (total,)
        yield tuple(b - a for a, b in zip(edges, edges[1:]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bw', required=True, help='measured GB/s, in logical rank order')
    p.add_argument('--nk', type=int, default=16)
    p.add_argument('--nv', type=int, default=64)
    p.add_argument('--head-dim', type=int, default=128)
    p.add_argument('--d', type=int, default=3072)
    p.add_argument('--layers', type=int, default=36)
    p.add_argument('--limit', type=int, default=8)
    a = p.parse_args()
    bw = tuple(float(x) for x in a.bw.split(','))
    if not bw or any(not (0 < x < float('inf')) for x in bw) or not 0 < len(bw) <= a.nk:
        p.error('need positive finite bandwidth and at least one K group per rank')
    if a.nk <= 0 or a.nv <= 0 or a.nv % a.nk or min(a.head_dim, a.d, a.layers, a.limit) <= 0:
        p.error('need integral V/K and positive dimensions, layers and limit')
    per = a.nv // a.nk
    qkv = ((2 + 2 * per) * a.head_dim + 2 * per) * a.d * 5 // 8
    out = a.d * per * a.head_dim * 5 // 8
    state = per * a.head_dim ** 2 * 8
    unit = qkv + out + state
    candidates = sorted(compositions(a.nk, len(bw)),
                        key=lambda cs: (max(c / b for c, b in zip(cs, bw)), cs))
    print(f'# nk={a.nk} nv={a.nv} dim={a.head_dim} D={a.d} layers={a.layers}; BW={bw} GB/s')
    print(f'# bytes/K-group/layer: QKV+gate+alpha+beta={qkv}, output={out}, state_R/W={state}; total={unit}')
    print('V split | K split | bytes/rank/layer | us/rank/layer | critical ms/token (lower bound)')
    for cs in candidates[:a.limit]:
        us = tuple(c * unit / b / 1000 for c, b in zip(cs, bw))
        print(','.join(str(c * per) for c in cs), '|', ','.join(map(str, cs)), '|',
              ','.join(str(c * unit) for c in cs), '|', ','.join(f'{t:.3f}' for t in us), '|',
              f'{max(us) * a.layers / 1000:.6f}')


if __name__ == '__main__':
    main()

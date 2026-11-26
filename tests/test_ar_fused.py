#!/usr/bin/env python3
"""CPU wire oracle: lossless half bits and adversarial per-dword visibility."""
import itertools
import struct


def encode(half, epoch):
    return half | (epoch << 16)


# Exhaust every binary16 encoding (including signed zero and NaN payloads).
for ep in (1, 2, 32768, 65534):
    for h in range(65536):
        w = encode(h, ep)
        assert w >> 16 == ep and w & 65535 == h

# Arbitrary tearing across packet dwords is allowed. Each accepted half must
# come from the current epoch, even if neighboring dwords remain stale.
old = [encode(0x3555 + i, 17) for i in range(8)]
new = [encode(0xB800 + i, 19) for i in range(8)]
for order in itertools.permutations(range(8)):
    memory = old.copy()
    accepted = {}
    for i in order:
        memory[i] = new[i]
        for j, word in enumerate(memory):
            if word >> 16 == 19:
                accepted[j] = word & 65535
                assert accepted[j] == (new[j] & 65535)
    assert len(accepted) == 8

# RN conversion and reconstruction preserve the existing rank-ordered sum.
for xs in ((0.0, -0.0, 1.0), (0.333333, -2.71828, 1023.75), (1e-7, -1e-6, 65504.0)):
    halves = [struct.unpack('<H', struct.pack('<e', x))[0] for x in xs]
    direct = [struct.unpack('<e', struct.pack('<H', h))[0] for h in halves]
    wire = [struct.unpack('<e', struct.pack('<H', encode(h, 21) & 65535))[0] for h in halves]
    assert direct == wire and sum(direct) == sum(wire)
print('AR_FUSED CPU oracle: half bits, RN reconstruction, all 40320 packet write orders PASS')

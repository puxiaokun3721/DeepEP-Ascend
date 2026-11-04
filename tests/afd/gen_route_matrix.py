"""Distinguishable BF16-finite data for FUNCTION-021 increment 1 (CPU prep).

Scenarios (per REV6 f06-cpu-oracle-review): 'balanced' (both Fs get real routes;
each expert exactly once per token, K=E), 'hotspot' (K=2, forced hotspot slot on
every `hotspot_every`-th token; ratio verifiable), 'zero_f' (K=E/2, routes drawn
only from F0's experts so the F1 member has an explicit zero route).
A rows encode (token,expert,slot) position-wise via hash mixing mapped into
BF16-exact finite values (multiples of 1/16 in [0,16)), so no accidental
NaN/Inf and any route swap changes visible payload. Self-check prints per-F and
per-expert route histograms against expected values.
"""
import hashlib

import numpy as np

from golden import golden

WEIGHTS = (-2.0, -0.5, 0.0, 0.25, 1.5)
SCALE = 0.0625  # 1/16; (0..255)*1/16 = [0,15.9375] are BF16-exact and finite


def _v(t, e, s, j):
    # Full-tuple hash avoids the mod-256 class collapse of XOR mixing.
    d = hashlib.blake2b(b'%d,%d,%d,%d' % (t, e, s, j), digest_size=1).digest()
    return d[0] * SCALE


def encode_row(token, expert, slot, hidden_bytes):
    """BF16 row (hidden_bytes//2 finite values) derived position-wise from (t,e,s)."""
    n = hidden_bytes // 2
    vals = np.array([_v(token, expert, slot, j) for j in range(n)], dtype=np.float32)
    bits = vals.astype('<f4', copy=False).view('<u4')
    assert np.all((bits & 0xffff) == 0), 'grid must be BF16-exact'
    return (bits >> 16).astype('<u2').tobytes()


def weight_bits(token, slot):
    return np.float32(WEIGHTS[(token + slot) % len(WEIGHTS)]).view(np.uint32)


def generate_route_matrix(batch=33, experts=8, mode='balanced', hotspot=0, hotspot_every=2):
    rng = np.random.RandomState(20261026)
    half = experts // 2
    if mode == 'zero_f':
        k = half
        topk = np.stack([rng.permutation(half) for _ in range(batch)])  # only F0 experts
    elif mode == 'hotspot':
        k = 2
        rows = []
        for t in range(batch):
            rest = [e for e in range(experts) if e != hotspot]
            if t % hotspot_every == 0:
                row = [hotspot, int(rng.choice(rest))]
            else:
                row = list(rng.choice(rest, 2, replace=False))
            rows.append(row)
        topk = np.stack(rows)
    else:  # balanced
        k = experts
        topk = np.stack([rng.permutation(experts) for _ in range(batch)])
    mask = np.ones(batch, dtype=bool)
    return dict(topk=topk.astype('<i8'), mask=mask, batch=batch, k=k, experts=experts,
                mode=mode, hotspot=hotspot if mode == 'hotspot' else None)


def route_histogram(topk, experts, mask, num_ranks=2):
    _, routes = golden(topk, experts, num_ranks, mask)
    counts = [sum(routes[e].values()) for e in range(experts)]
    half = experts // num_ranks
    f_totals = [sum(counts[i * half:(i + 1) * half]) for i in range(num_ranks)]
    return counts, f_totals


def self_check():
    batch, experts = 33, 8
    # (a) balanced: each expert exactly once per token; both Fs non-empty
    m = generate_route_matrix(batch, experts, 'balanced')
    counts, f_totals = route_histogram(m['topk'], experts, m['mask'])
    assert counts == [batch] * experts, counts
    assert all(x > 0 for x in f_totals), f_totals
    # (b) hotspot: forced ratio verifiable
    mh = generate_route_matrix(batch, experts, 'hotspot', hotspot=0, hotspot_every=2)
    ch, fh = route_histogram(mh['topk'], experts, mh['mask'])
    expect_hot = (batch + 1) // 2  # tokens 0,2,...,32 -> 17
    assert ch[0] == expect_hot, (ch[0], expect_hot)
    assert all(0 < x <= batch for x in ch), ch
    assert all(x > 0 for x in fh), fh
    # (c) zero_f: explicit zero-route F member
    mz = generate_route_matrix(batch, experts, 'zero_f')
    cz, fz = route_histogram(mz['topk'], experts, mz['mask'])
    assert cz[experts // 2:] == [0] * (experts - experts // 2), cz
    assert sum(cz[:experts // 2]) == batch * (experts // 2), cz
    assert fz[1] == 0 and fz[0] > 0, fz
    # (d) row uniqueness + BF16 finiteness across sampled triples
    hb = 256
    seen = set()
    for t in (0, 1, 2, 5, 6):
        for e in (0, 3, 7):
            for s in (0, 1, m['k'] - 1):
                row = encode_row(t, e, s, hb)
                assert row not in seen, (t, e, s)
                seen.add(row)
                words = np.frombuffer(row, dtype='<u2').astype('<u4')
                vals = (words << 16).view('<f4')
                assert np.isfinite(vals).all(), 'non-finite BF16 produced'
                assert np.allclose(vals * 16, np.round(vals * 16)), 'not BF16-exact grid'
    # (e) route swap on a present token is visible to golden
    swapped = m['topk'].copy()
    t = 2
    swapped[t, 0], swapped[t, 1] = swapped[t, 1], swapped[t, 0]
    _, r0 = golden(m['topk'], experts, 2, m['mask'])
    _, r1 = golden(swapped, experts, 2, m['mask'])
    diff = [i for i in range(experts) if r0[i] != r1[i]]
    assert diff, 'route swap invisible'
    print('SELF_CHECK_PASS', dict(
        balanced_counts=counts, balanced_f=f_totals,
        hotspot_counts=ch, hotspot_expect=expect_hot, hotspot_f=fh,
        zero_f_counts=cz, zero_f_f=fz, swap_experts=diff[:4]))


if __name__ == '__main__':
    self_check()

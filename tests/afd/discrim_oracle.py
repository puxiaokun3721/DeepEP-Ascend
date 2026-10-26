"""Independent CPU oracle v4 (AUDIT025 d1-rounding-failure-scope fix).

Changes vs v2/v3 (reviewer-identified defects):
1. expected_rows now stages the F-side BF16 rounding: y0/y1 = bf16_rne(x*S+C),
   then each term is multiplied in FP32, added per k in FP32, and the final sum
   is rounded to BF16 (CAST_RINT, ties-to-even). The old model applied the
   transform and weights in FP32 without the intermediate BF16 round.
2. bits() now takes the HIGH half-word of float32 (v2 used a raw uint16 view,
   which produced interleaved [B,2H] half-words); perturb_ulp inverts via <<16.
3. check() rejects non-finite values in the compared region before comparing.
4. make_inputs uses a four-quadrant residue encoding over the hidden dimension:
   quarter 0 carries t%241, quarter 1 carries t//241, quarter 2 carries source,
   quarter 3 carries mb. Rows are collision-free across (source, mb, token) for
   t < 241*241; all values remain integers in [base, base+mod) (BF16-exact).
"""
import numpy as np

NF = 2
F_RANKS = 2
EXPERTS = NF * F_RANKS
S = np.array([1.0, 2.0, 0.5, 1.0], dtype=np.float32)
C = np.array([0.0, 16.0, 32.0, 48.0], dtype=np.float32)
PAIRS = [(0, 2), (1, 3), (0, 3), (1, 2)]
WVEC = [(0.25, 0.125), (0.5, 0.25), (0.75, 0.375), (0.375, 0.75)]
MAX_ULP = 0


def bf16_rne(a):
    a = np.asarray(a, dtype=np.float32)
    bits = a.view(np.uint32).copy()
    exp_mask = np.uint32(0x7F800000)
    isnan = (bits & exp_mask) == exp_mask
    add = ((bits >> np.uint32(16)) & np.uint32(1)) + np.uint32(0x7FFF)
    rounded = (bits + add) & np.uint32(0xFFFF0000)
    return np.where(isnan, bits & np.uint32(0xFFFF0000), rounded).astype(np.uint32).view(np.float32)


def bits(a):
    return (np.asarray(a, dtype=np.float32).view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def _ord16(bits16):
    b = np.asarray(bits16, dtype=np.uint16).astype(np.int64)
    return np.where(b >= 0x8000, 0x8000 - (b & 0x7FFF), 0x8000 + b)


def check(actual_bits, expected_bits, mask, max_ulp=MAX_ULP):
    a = np.asarray(actual_bits, dtype=np.uint16)
    e = np.asarray(expected_bits, dtype=np.uint16)
    if a.shape != e.shape:
        return False, ('shape', None, None, None)
    act = np.asarray(mask, dtype=bool)
    if not act.any():
        return False, ('empty_selection', None, None, None)
    fa = (a[act].astype(np.uint32) << np.uint32(16)).view(np.float32)
    fe = (e[act].astype(np.uint32) << np.uint32(16)).view(np.float32)
    if not (np.isfinite(fa).all() and np.isfinite(fe).all()):
        return False, ('nonfinite', None, None, None)
    d = np.abs(_ord16(a[act]) - _ord16(e[act]))
    bad = np.argwhere(d > max_ulp)
    if bad.size:
        pos = bad[0]
        return False, ('el', int(pos[0]), int(pos[1]), int(d[tuple(pos)]))
    return True, None


def perturb_ulp(a, row, col, steps=1):
    u = np.array(np.asarray(a, np.float32).view(np.uint32), copy=True)
    u[row, col] = np.uint32(((u[row, col] >> np.uint32(16)) + np.uint32(steps)) << np.uint32(16))
    return u.view(np.float32)


def make_inputs(capacity, hidden, mb, source, base=8, mod=241):
    h = np.arange(hidden, dtype=np.int64)
    t = np.arange(capacity, dtype=np.int64)[:, None]
    q = np.minimum(h // max(1, hidden // 4), 3)
    r0 = t % mod
    r1 = t // mod
    r2 = np.int64(source % mod)
    r3 = np.int64(mb % mod)
    vals = np.where(q == 0, (r0 * 131 + h * 7) % mod,
           np.where(q == 1, (r1 * 131 + h * 7) % mod,
           np.where(q == 2, (r2 * 89 + h * 7) % mod,
                            (r3 * 97 + h * 7) % mod)))
    return (vals + base).astype(np.int32)


def route_table(capacity, mb):
    return [PAIRS[(t + 3 * mb) % len(PAIRS)] for t in range(capacity)]


def weight_table(capacity, mb):
    return [WVEC[(t + mb) % len(WVEC)] for t in range(capacity)]


def expected_rows(x, routes, weights):
    xf = np.asarray(x).astype(np.float32)
    out = np.empty_like(xf)
    for t in range(xf.shape[0]):
        g0, g1 = routes[t]
        w0 = np.float32(weights[t][0])
        w1 = np.float32(weights[t][1])
        y0 = bf16_rne(xf[t] * S[g0] + C[g0])
        y1 = bf16_rne(xf[t] * S[g1] + C[g1])
        out[t] = w0 * y0 + w1 * y1
    return bf16_rne(out)


def expected_rows_variant(x, routes, weights, *, wrong_expert=False, swap_w=False,
                          drop=False, dup=False):
    xf = np.asarray(x).astype(np.float32)
    out = np.empty_like(xf)
    for t in range(xf.shape[0]):
        g0, g1 = routes[t]
        if wrong_expert:
            g0 = (g0 + 1) % EXPERTS
        w0, w1 = np.float32(weights[t][0]), np.float32(weights[t][1])
        if swap_w:
            w0, w1 = w1, w0
        y0 = bf16_rne(xf[t] * S[g0] + C[g0])
        y1 = bf16_rne(xf[t] * S[g1] + C[g1])
        if drop:
            out[t] = w0 * y0
        elif dup:
            out[t] = w0 * y0 + w1 * y1 + w1 * y1
        else:
            out[t] = w0 * y0 + w1 * y1
    return bf16_rne(out)


def selftest():
    capacity, hidden, mb = 64, 128, 1
    results = []

    def expect(name, cond):
        results.append((name, bool(cond)))
        print(('PASS ' if cond else 'FAIL ') + name)

    xs = [make_inputs(capacity, hidden, m, src) for m in range(3) for src in range(2)]
    rows = [x[r].tobytes() for x in xs for r in range(capacity)]
    expect('inputs_cross_source_mb_distinct', len(set(rows)) == len(rows) == 384)

    xbig = make_inputs(32768, hidden, 0, 0)
    rows_big = set(xbig[r].tobytes() for r in range(32768))
    expect('inputs_32768_rows_distinct', len(rows_big) == 32768)

    expect('inputs_bf16_exact', np.array_equal(bf16_rne(np.stack(xs).astype(np.float32)),
                                              np.stack(xs).astype(np.float32)))

    x = make_inputs(capacity, hidden, 0, 0)
    rt, wt = route_table(capacity, 0), weight_table(capacity, 0)
    e = expected_rows(x, rt, wt)
    mask = np.ones(capacity, dtype=bool)
    ok, why = check(bits(e), bits(e), mask)
    expect('clean_row_passes_strict', ok)

    one = perturb_ulp(e, 5, 17, 1)
    ok, _ = check(bits(one), bits(e), mask)
    expect('one_ulp_fails_strict', not ok)
    ok, _ = check(bits(one), bits(e), mask, max_ulp=1)
    expect('one_ulp_passes_with_tolerance', ok)
    two = perturb_ulp(e, 5, 17, 2)
    ok, _ = check(bits(two), bits(e), mask, max_ulp=1)
    expect('two_ulp_fails', not ok)

    inf = e.copy()
    inf[1, 2] = np.float32(np.inf)
    ok, why = check(bits(inf), bits(e), mask)
    expect('nonfinite_rejected', (not ok) and why and why[0] == 'nonfinite')

    empty = np.zeros(capacity, dtype=bool)
    ok, why = check(bits(e), bits(e), empty)
    expect('empty_selection_rejected', (not ok) and why and why[0] == 'empty_selection')

    expect('progress_bit_parity_R1', [(((i + 1) * 1) - 1) % 2 for i in range(3)] == [0, 1, 0])
    expect('progress_bit_parity_R8_64',
           [(((i + 1) * 8) - 1) % 2 for i in range(3)] == [1, 1, 1]
           and [(((i + 1) * 64) - 1) % 2 for i in range(3)] == [1, 1, 1])

    variants = {
        'wrong_expert_fails': expected_rows_variant(x, rt, wt, wrong_expert=True),
        'swapped_weights_fails': expected_rows_variant(x, rt, wt, swap_w=True),
        'drop_contribution_fails': expected_rows_variant(x, rt, wt, drop=True),
        'dup_contribution_fails': expected_rows_variant(x, rt, wt, dup=True),
    }
    for name, v in variants.items():
        ok, _ = check(bits(v), bits(e), mask)
        expect(name, not ok)
    sw = e.copy()
    sw[[2, 5]] = sw[[5, 2]]
    ok, _ = check(bits(sw), bits(e), mask)
    expect('swapped_tokens_fails', not ok)

    xt1 = make_inputs(capacity, hidden, 1, 0)
    other = expected_rows(xt1, route_table(capacity, 1), weight_table(capacity, 1))
    ok, _ = check(bits(other), bits(e), mask)
    expect('transaction_swap_fails', not ok)
    xsrc = make_inputs(capacity, hidden, 0, 1)
    esrc = expected_rows(xsrc, rt, wt)
    ok, _ = check(bits(esrc), bits(e), mask)
    expect('source_swap_fails', not ok)

    failed = [n for n, ok in results if not ok]
    print('ORACLE_SELFTEST', 'PASS' if not failed else 'FAIL',
          'checks', len(results), 'failed', len(failed))
    return 0 if not failed else 1


if __name__ == '__main__':
    raise SystemExit(selftest())

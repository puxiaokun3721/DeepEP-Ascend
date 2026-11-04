"""FUNCTION-021 increment 2 CPU additions: signed/non-unit/zero weights (REV6 revised).

Fixes per f06-cpu-oracle-review: hand-computed exact expectations (12.75), an
inexact sample that actually forces BF16 output rounding (payload [1,2,4] with
weights [0.1,-0.3,0.7]), float64 kept in the error path, thresholds fixed before
execution. CPU model samples only; not combineRecv device precision.
"""
import unittest

import numpy as np

from test_combine_recv_cpu import bf16_rn, reduce_routes

EXACT = (-2.0, -0.5, 0.0, 0.25, 1.5)   # BF16 exact, signed, non-unit
INEXACT = (0.1, -0.3, 0.7)


def max_abs_delta(actual, expected):
    """Keep float64 in the comparison; never downcast the reference."""
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(expected, dtype=np.float64)
    return float(np.max(np.abs(a - b))) if a.size else 0.0


class WeightVectorTests(unittest.TestCase):
    def test_exact_hand_computed_12_75(self):
        payload = np.zeros((1, 5, 16), dtype=np.float32)
        for i, v in enumerate((2.0, 3.0, 5.0, 7.0, 11.0)):
            payload[0, i] = v
        w = np.array([EXACT], dtype=np.float32)
        result = reduce_routes(payload, w, [True])
        # 2*-2 + 3*-0.5 + 5*0 + 7*0.25 + 11*1.5 = -4 -1.5 + 0 + 1.75 + 16.5 = 12.75
        np.testing.assert_array_equal(result, np.full((1, 16), np.float32(12.75)))

    def test_exact_independent_sample(self):
        payload = np.zeros((1, 2, 4), dtype=np.float32)
        payload[0, 0] = 4.0
        payload[0, 1] = 8.0
        w = np.array([[0.25, -0.5]], dtype=np.float32)   # 1 - 4 = -3
        result = reduce_routes(payload, w, [True])
        np.testing.assert_array_equal(result, np.full((1, 4), np.float32(-3.0)))

    def test_zero_weight_absorbs_finite_but_not_nan(self):
        payload = np.full((1, 2, 8), 3.0, dtype=np.float32)
        w = np.array([[0.0, 1.5]], dtype=np.float32)
        result = reduce_routes(payload, w, [True])
        np.testing.assert_array_equal(result, np.full((1, 8), np.float32(4.5)))
        with np.errstate(invalid='ignore'):
            nan_row = np.full((1, 1, 8), np.nan, dtype=np.float32)
            out = reduce_routes(nan_row, np.zeros((1, 1), dtype=np.float32), [True])
        self.assertTrue(np.isnan(out).all())

    def test_inexact_forces_bf16_rounding_with_fixed_threshold(self):
        payload = np.zeros((1, 3, 8), dtype=np.float32)
        payload[0, 0] = 1.0
        payload[0, 1] = 2.0
        payload[0, 2] = 4.0
        w = np.array([INEXACT], dtype=np.float32)
        result = reduce_routes(payload, w, [True])
        # Real-valued sum: 1*0.1 + 2*(-0.3) + 4*0.7 = 0.1 - 0.6 + 2.8 = 2.3 (float64 ref)
        ref64 = np.full((1, 8), np.float64(1.0) * np.float64(np.float32(0.1))
                        + np.float64(2.0) * np.float64(np.float32(-0.3))
                        + np.float64(4.0) * np.float64(np.float32(0.7)))
        # BF16 near 2.3 rounds to 2.296875, so the delta must be nonzero but bounded.
        bound = float(np.max(np.abs(ref64))) * 2.0 ** -8 + 1e-3   # fixed <= 0.009
        delta = max_abs_delta(result, ref64)
        np.testing.assert_array_equal(result, np.full((1, 8), np.float32(2.296875)))
        self.assertGreater(delta, 0.0, 'sample did not exercise BF16 output rounding')
        self.assertLessEqual(delta, bound)
        print('INEXACT_REPORT', dict(weights=list(INEXACT), max_abs_delta=delta, bound=bound))

    def test_signed_cancellation_exact(self):
        payload = np.zeros((1, 3, 4), dtype=np.float32)
        payload[0, 0] = 2.0
        payload[0, 1] = -3.0
        payload[0, 2] = 5.0
        w = np.array([[1.5, 2.0, -0.5]], dtype=np.float32)  # 3 + (-6) + (-2.5) = -5.5 exact
        result = reduce_routes(payload, w, [True])
        np.testing.assert_array_equal(result, np.full((1, 4), np.float32(-5.5)))


if __name__ == '__main__':
    unittest.main(verbosity=2)

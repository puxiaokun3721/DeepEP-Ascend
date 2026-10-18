"""Fixture/oracle properties only; these do not execute the C++ or NPU kernel."""
import unittest

import numpy as np

from recv_reference import align, pack_active, reference
from reuse_fixture import REUSE_COUNTS, reuse_fixture


class ReuseFixtureTests(unittest.TestCase):
    def test_unsorted_entries_and_source_fanout(self):
        for fp8 in (False, True):
            for count in REUSE_COUNTS:
                with self.subTest(fp8=fp8, count=count):
                    fixture = reuse_fixture(count, fp8, hidden=128)
                    stride = align(2*count, 4)
                    x, sf, metadata = reference(**fixture, capacity=2*stride+4,
                                                token_capacity=count)
                    pack_active(2, 2, fixture['segments'], [[0], [1]])
                    # Explicit destination oracle independent of Segment traversal.
                    for lane in range(2):
                        for token in range(count):
                            row0 = lane*count + count-1-token
                            row1 = stride + lane*count + token
                            np.testing.assert_array_equal(x[row0], x[row1])
                            np.testing.assert_array_equal(sf[row0], sf[row1])
                            np.testing.assert_array_equal(metadata[row0], [lane, token, 0, lane])
                            np.testing.assert_array_equal(metadata[row1], [lane, token, 1, lane])
                    self.assertFalse(x[2*count:stride].any())
                    self.assertFalse(x[stride+2*count:].any())

    def test_segment_permutation_does_not_change_results(self):
        fixture = reuse_fixture(17, True, hidden=128)
        expected = reference(**fixture, capacity=80, token_capacity=17)
        segments = fixture['segments']
        # Preserve entry ownership while reversing each entry's Segment order.
        permuted = [segments[1], segments[0], segments[3], segments[2]]
        actual = reference(**dict(fixture, segments=permuted), capacity=80, token_capacity=17)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main(verbosity=2)

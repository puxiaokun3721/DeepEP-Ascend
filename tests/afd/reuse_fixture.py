"""Adversarial byte fixture shared by CPU oracle checks and actual NPU tests."""
import numpy as np

from recv_reference import Segment, align, make_record


REUSE_COUNTS = (1, 3, 4, 5, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129, 257, 513)


def reuse_fixture(count, fp8, hidden=7168):
    """Entry-major, deliberately non-monotonic destinations; two local routes/token.

    Each lane has distinct byte payloads but identical token numbers. Expert 0
    visits records backwards and expert 1 forwards. This catches record/token
    confusion, lane-blind dedup, and sorted-destination assumptions.
    """
    if count <= 0 or hidden <= 0 or hidden % 128:
        raise ValueError('positive count and hidden divisible by 128 required')
    hidden_bytes = hidden if fp8 else 2*hidden
    sf_bytes = hidden//32 if fp8 else 0
    rng = np.random.default_rng(20261016 + count)
    records = []
    for lane in range(2):
        lane_records = []
        for token in range(count):
            x = rng.integers(0, 256, hidden_bytes, dtype=np.uint8).tobytes()
            sf = rng.integers(0, 256, sf_bytes, dtype=np.uint8).tobytes()
            lane_records.append(make_record(x, sf, [0, 1], token))
        records.append(lane_records)
    routes = [[[2*i for i in reversed(range(count))],
               [2*i+1 for i in range(count)]] for _ in range(2)]
    expert_stride = align(2*count, 4)
    segments = [Segment(0, 0, 1, count, count, expert_stride),
                Segment(0, 0, 0, 0, count, 0),
                Segment(1, 1, 0, 0, count, count),
                Segment(1, 1, 1, count, count, expert_stride + count)]
    return dict(records=records, routes=routes, segments=segments,
                hidden_bytes=hidden_bytes, sf_bytes=sf_bytes, k=2, num_experts=2)

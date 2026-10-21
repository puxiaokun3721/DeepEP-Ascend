"""Independent byte oracle and scheduler-output fixtures; no scheduler or NPU emulation.

ABI source: docs/design/deepep_afd_operators/common_contract.md in AFD Plugin.
Only fixture construction shares this ABI with C++; the output oracle uses Python
row descriptions and does not import the device implementation.
"""
from dataclasses import dataclass
import struct

import numpy as np

ACTIVE = struct.Struct('<10I3Q')
ROW = struct.Struct('<8I')
STATE = struct.Struct('<8I')
CONTEXT_BYTES = 224
UINT32_MAX = (1 << 32) - 1
NOP, WORK, CONTROL, ABORT = range(4)
EMPTY, SELECTED, INPUT_READY, COMPUTE_READY, COMMITTED, ABORTED = range(6)


def align(n, alignment=32):
    return (n + alignment - 1) // alignment * alignment


def wire_layout(hidden_bytes, sf_bytes, k):
    """Wire offsets derived from the contract, independent of sender golden."""
    idx = hidden_bytes + align(sf_bytes)
    weights = idx + align(8 * k)
    token = weights + 4 * k
    return dict(sf=hidden_bytes, idx=idx, weights=weights, token=token,
                valid=token + 4, metadata=align(token + 8 - hidden_bytes),
                stride=align(token + 8, 512))


@dataclass(frozen=True)
class Segment:
    entry: int
    lane: int
    expert: int
    route_begin: int
    count: int
    dst: int

    def pack(self):
        return ROW.pack(self.entry, self.lane, self.expert, self.route_begin,
                        self.count, self.dst, 0, 0)


def validate_segment_intervals(segments, *, num_experts, alignment, capacity):
    """Piece-local interval oracle: arbitrary input order, no token/hidden cache.

    Independent Python sorting checks the predicates used by device preflight;
    reference() below separately checks individual route/output row uniqueness.
    This is not Ascend C execution or a performance measurement.
    """
    if any(s.count <= 0 or s.route_begin < 0 or s.dst < 0 or s.lane < 0 or
           not 0 <= s.expert < num_experts for s in segments):
        raise ValueError('segment')
    source_order = sorted(segments, key=lambda s: (s.lane, s.expert, s.route_begin))
    for previous, current in zip(source_order, source_order[1:]):
        if (previous.lane == current.lane and previous.expert == current.expert and
                current.route_begin < previous.route_begin + previous.count):
            raise ValueError('duplicate input interval')
    destination_order = sorted(segments, key=lambda s: (s.expert, s.dst))
    next_segment, physical = 0, 0
    for expert in range(num_experts):
        count = 0
        while (next_segment < len(destination_order) and
               destination_order[next_segment].expert == expert):
            segment = destination_order[next_segment]
            if segment.dst != physical + count:
                raise ValueError('dense expert interval')
            count += segment.count
            next_segment += 1
        physical += align(count, alignment)
        if physical > capacity:
            raise ValueError('capacity')
    return physical


def pack_active(lanes, experts, segments, entry_lanes, *, kind=WORK,
                valid=1, phase=0, layer=0, alignment=4, grants=None, finals=None):
    """Materialize an externally selected Active fixture, never select a task.

    entry_lanes specifies explicit member lists. Segment order is arbitrary;
    each entry's segments must be contiguous in the table by ABI contract.
    """
    if grants is None:
        grants = [0] * len(entry_lanes)
    if finals is None:
        finals = [int(kind == CONTROL)] * len(entry_lanes)
    entries_offset, members_offset, segments_offset = 64, 64 + 32 * lanes, 64 + 64 * lanes
    storage = bytearray(segments_offset + 32 * lanes * experts)
    members, entries = [], []
    cursor = 0
    for i, lane_list in enumerate(entry_lanes):
        selected = [s for s in segments if s.entry == i]
        entries.append(ROW.pack(i, len(members), len(lane_list), cursor,
                                len(selected), sum(s.count for s in selected), finals[i], grants[i]))
        for lane in lane_list:
            members.append(ROW.pack(lane, 0, 0, len(members) * experts, experts, 0, 0, 0))
        cursor += len(selected)
    expanded = sum(s.count for s in segments)
    padded = sum(align(sum(s.count for s in segments if s.expert == e), alignment)
                 for e in range(experts))
    ACTIVE.pack_into(storage, 0, valid, kind, phase, layer, len(entries), len(members),
                     len(segments), expanded, padded, 0, entries_offset, members_offset, segments_offset)
    for offset, rows in ((entries_offset, entries), (members_offset, members),
                         (segments_offset, [s.pack() for s in segments])):
        storage[offset:offset + 32 * len(rows)] = b''.join(rows)
    state = bytearray(32 + align(4 * lanes))
    STATE.pack_into(state, 0, SELECTED if valid else EMPTY, 0, len(entries), 0, 0, 0, 0, 0)
    for i, grant in enumerate(grants):
        struct.pack_into('<I', state, 32 + 4 * i, grant)
    return storage, state


def make_record(x, sf, topk, token, *, k=None, mask=None):
    k = len(topk) if k is None else k
    pos = wire_layout(len(x), len(sf), k)
    # Poison the 512B tail: neither oracle nor kernel may use it as data.
    result = bytearray([0xD7]) * pos['stride']
    result[:len(x)] = x
    result[len(x):len(x) + pos['metadata']] = bytes(pos['metadata'])
    result[pos['sf']:pos['sf'] + len(sf)] = sf
    struct.pack_into(f'<{k}q', result, pos['idx'], *topk)
    struct.pack_into('<II', result, pos['token'], token,
                     (1 << k) - 1 if mask is None else mask)
    return bytes(result)


def reference(records, routes, segments, *, hidden_bytes, sf_bytes, k,
              num_experts, rank=0, alignment=4, capacity=128, token_capacity=None):
    """Expand selected routes to exact bytes; invalid inputs raise ValueError.

    records[lane] is an ordered list of raw wire records. routes[lane][expert]
    lists record*K+slot references. Segment.route_begin indexes the lane-global
    concatenated expert route table, NOT records or original source tokens.
    """
    if hidden_bytes <= 0 or sf_bytes < 0 or not 1 <= k <= 32:
        raise ValueError('layout')
    if num_experts <= 0 or len(records) != len(routes):
        raise ValueError('lane/expert dimensions')
    for segment in segments:
        if (not 0 <= segment.expert < num_experts or
                not 0 <= segment.lane < len(records) or segment.entry < 0 or
                segment.count <= 0 or segment.route_begin < 0 or segment.dst < 0):
            raise ValueError('segment')
    if any(len(lane_routes) != num_experts for lane_routes in routes):
        raise ValueError('expert dimensions')
    pos = wire_layout(hidden_bytes, sf_bytes, k)
    token_capacity = capacity if token_capacity is None else token_capacity
    nrows = sum(align(sum(s.count for s in segments if s.expert == e), alignment)
                for e in range(num_experts))
    if nrows > capacity:
        raise ValueError('capacity')
    x = np.zeros((nrows, hidden_bytes), dtype=np.uint8)
    sf = np.zeros((nrows, sf_bytes), dtype=np.uint8)
    metadata = np.full((nrows, 4), -1, dtype=np.int32)
    occupied, used_routes = set(), set()
    expert_base = 0
    for expert in range(num_experts):
        group = [s for s in segments if s.expert == expert]
        count = sum(s.count for s in group)
        for segment in group:
            if segment.count <= 0 or segment.lane >= len(records) or segment.entry < 0:
                raise ValueError('segment')
            local_routes = routes[segment.lane]
            prefix = sum(len(r) for r in local_routes[:expert])
            start = segment.route_begin - prefix
            if start < 0 or start + segment.count > len(local_routes[expert]):
                raise ValueError('route interval')
            if segment.dst < expert_base or segment.dst + segment.count > expert_base + count:
                raise ValueError('dense expert interval')
            for i, ref in enumerate(local_routes[expert][start:start + segment.count]):
                route_key = (segment.lane, expert, start + i)
                dst = segment.dst + i
                if route_key in used_routes or dst in occupied or not 0 <= ref < UINT32_MAX:
                    raise ValueError('duplicate or invalid route')
                used_routes.add(route_key)
                occupied.add(dst)
                record, slot = divmod(ref, k)
                if record >= len(records[segment.lane]):
                    raise ValueError('record')
                raw = records[segment.lane][record]
                if len(raw) != pos['stride']:
                    raise ValueError('record stride')
                token, mask = struct.unpack_from('<II', raw, pos['token'])
                expected_expert = rank * num_experts + expert
                actual_expert = struct.unpack_from('<q', raw, pos['idx'] + 8 * slot)[0]
                if token >= token_capacity or not mask & (1 << slot) or mask >> k or actual_expert != expected_expert:
                    raise ValueError('record metadata')
                x[dst] = np.frombuffer(raw[:hidden_bytes], dtype=np.uint8)
                sf[dst] = np.frombuffer(raw[pos['sf']:pos['sf'] + sf_bytes], dtype=np.uint8)
                metadata[dst] = segment.lane, token, slot, segment.entry
        expert_base += align(count, alignment)
    return x, sf, metadata


def sample_fixture(fp8=False, *, hidden=128):
    """Two source lanes, repeated token across experts, shuffled record order."""
    k, experts = 2, 2
    hidden_bytes, sf_bytes = hidden * (1 if fp8 else 2), (4 if fp8 else 0)
    records = []
    for lane in range(2):
        lane_records = []
        for token in (2, 0, 1):
            x = bytes((j + token * 37 + lane * 91) % 256 for j in range(hidden_bytes))
            sf = bytes((j * 71 + token * 23 + lane * 13) % 256 for j in range(sf_bytes))
            lane_records.append(make_record(x, sf, [0, 1], token))
        records.append(lane_records)
    routes = [[[4, 0, 2], [3, 5, 1]], [[2, 4, 0], [1, 3, 5]]]
    segments = [Segment(0, 0, 0, 0, 2, 0), Segment(0, 1, 0, 1, 1, 2),
                Segment(0, 1, 1, 3, 2, 4), Segment(0, 0, 1, 5, 1, 6)]
    return dict(records=records, routes=routes, segments=segments,
                hidden_bytes=hidden_bytes, sf_bytes=sf_bytes, k=k, num_experts=experts)


def bit_errors(actual, expected):
    """Return mismatched byte count (zero required), preserving NaN/sign payloads."""
    a = np.ascontiguousarray(actual).view(np.uint8).ravel()
    b = np.ascontiguousarray(expected).view(np.uint8).ravel()
    if a.shape != b.shape:
        raise AssertionError(f'byte shape {a.shape} != {b.shape}')
    return int(np.count_nonzero(a != b))

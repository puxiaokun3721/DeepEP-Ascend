"""Independent dispatchSend byte oracle. No device implementation is imported."""
from collections import Counter
import struct
import numpy as np


def golden(topk, num_experts, num_ranks, mask=None):
    topk = np.asarray(topk, dtype=np.int64)
    if topk.ndim != 2 or not 0 < topk.shape[1] <= min(32, num_experts):
        raise ValueError("K")
    if num_ranks <= 0 or num_experts <= 0 or num_experts > 512 or num_experts % num_ranks:
        raise ValueError("experts/ranks")
    mask = np.ones(len(topk), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    records = [set() for _ in range(num_ranks)]
    routes = [Counter() for _ in range(num_experts)]
    for token, row in enumerate(topk):
        if not mask[token]:
            continue
        if any(e < 0 or e >= num_experts for e in row) or len(set(row)) != len(row):
            raise ValueError("BAD_ROUTE")
        for slot, expert in enumerate(row):
            records[int(expert) // (num_experts // num_ranks)].add(token)
            routes[int(expert)][token, slot] += 1
    return records, routes


def layout(hidden_bytes, sf_bytes, k):
    align = lambda n, a: (n + a - 1) // a * a
    sf = hidden_bytes
    idx = sf + align(sf_bytes, 32)
    weights = idx + align(8 * k, 32)
    token = weights + 4 * k
    valid = token + 4
    return dict(sf=sf, idx=idx, weights=weights, token=token, valid=valid,
                metadata=align(valid + 4 - hidden_bytes, 32), stride=align(valid + 4, 512))


def verify_snapshot(payload, route_refs, counts, prefixes, rank, x_bytes, sf_bytes,
                    topk, weight_bits, num_experts, num_ranks, mask=None):
    """Validate an actual F snapshot; records may be in arbitrary atomic order.

    payload is trimmed to Inbox.num_recv_records * record_stride. Counts and
    prefixes are local-expert counts and cumulative end offsets, route_refs packed.
    x_bytes/sf_bytes are uint8 rows (SF must first be gathered by original strides).
    """
    expected_records, expected_routes = golden(topk, num_experts, num_ranks, mask)
    k = topk.shape[1]
    pos = layout(x_bytes.shape[1], sf_bytes.shape[1], k)
    assert len(payload) % pos['stride'] == 0, 'record stride'
    seen, records = set(), []
    for offset in range(0, len(payload), pos['stride']):
        raw = payload[offset:offset + pos['stride']]
        token = struct.unpack_from('<I', raw, pos['token'])[0]
        assert token in expected_records[rank] and token not in seen, 'record identity'
        seen.add(token)
        records.append(token)
        assert raw[:x_bytes.shape[1]] == x_bytes[token].tobytes(), 'hidden bits'
        assert raw[pos['sf']:pos['sf'] + sf_bytes.shape[1]] == sf_bytes[token].tobytes(), 'SF bits'
        assert raw[pos['idx']:pos['idx'] + 8*k] == topk[token].astype('<i8').tobytes(), 'topk bits'
        assert raw[pos['weights']:pos['weights'] + 4*k] == weight_bits[token].astype('<u4').tobytes(), 'weight bits'
        assert struct.unpack_from('<I', raw, pos['valid'])[0] == (1 << k) - 1, 'valid mask'
        occupied = set(range(pos['sf'], pos['sf'] + sf_bytes.shape[1]))
        occupied.update(range(pos['idx'], pos['idx'] + 8*k))
        occupied.update(range(pos['weights'], pos['valid'] + 4))
        assert all(raw[j] == 0 for j in range(pos['sf'], pos['sf'] + pos['metadata']) if j not in occupied), 'metadata padding'
    assert seen == expected_records[rank], 'missing record'
    r = num_experts // num_ranks
    assert len(counts) == len(prefixes) == r
    cursor = 0
    for expert in range(r):
        count = int(counts[expert])
        assert 0 <= count <= 0xffffffff, 'count domain'
        assert prefixes[expert] == cursor + count, 'prefix'
        actual = Counter()
        for ref in route_refs[cursor:cursor + count]:
            assert 0 <= int(ref) <= 0xffffffff, 'route reference domain'
            record, slot = divmod(int(ref), k)
            assert 0 <= record < len(records), 'record index'
            actual[records[record], slot] += 1
        assert actual == expected_routes[rank*r + expert], 'route multiset'
        cursor += count
    assert cursor == len(route_refs), 'route length'
    return {'hidden_bit_errors': 0, 'sf_bit_errors': 0, 'metadata_bit_errors': 0,
            'route_errors': 0, 'padding_errors': 0}


class InputLane:
    """Executable protocol SPECIFICATION, not a model of DMA visibility."""
    def __init__(self, num_ranks):
        self.next_bit = 0
        self.last_bit = 1
        self.has_published = False
        self.consumed = [1] * num_ranks
        self.ready = [1] * num_ranks
        self.ledger = [False, False]

    def publish(self, round_idx):
        if self.has_published and any(v != self.last_bit for v in self.consumed):
            raise BlockingIOError('ACK')
        if self.ledger[round_idx % 2]:
            raise BlockingIOError('Ledger')
        self.ledger[round_idx % 2] = True
        self.ready[:] = [self.next_bit] * len(self.ready)
        self.last_bit = self.next_bit
        self.next_bit ^= 1
        self.has_published = True
        return self.last_bit

    def ack(self, rank):
        self.consumed[rank] = self.last_bit

    def recv_release(self, round_idx):
        self.ledger[round_idx % 2] = False

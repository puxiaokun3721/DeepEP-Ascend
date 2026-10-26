"""Host-side fixtures for the actual dispatchRecv kernel, not a scheduler.

The caller must synchronize real dispatchSend before snapshot_active(). This
module never polls Ready, ACKs inputs, selects queued tasks or recycles slots.
"""
import struct

import numpy as np
import torch

from recv_reference import (ACTIVE, INPUT_READY, Segment, bit_errors, pack_active,
                            reference, wire_layout)

SENTINEL_BYTE = 0xA5
SENTINEL_METADATA = 0x123456


def put_bytes(tensor, raw):
    data = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy())
    if tensor.numel() != len(raw):
        raise ValueError(f'fixture storage {len(raw)} != tensor {tensor.numel()}')
    tensor.copy_(data.to(tensor.device))


def raw_bytes(tensor):
    return tensor.detach().view(torch.uint8).cpu().numpy().tobytes()


def directory(owner):
    raw = raw_bytes(owner.context)
    # ABI 2 adds channel_stride and reserved_pad0 before the uint64 offsets.
    fields = struct.unpack_from('<18I8Q', raw)
    lanes, peers, addresses = fields[18:21]
    return dict(raw=raw, ranks=fields[1], num_lanes=fields[3], experts=fields[4],
                k=fields[5], hidden_bytes=fields[6], sf_packs=fields[7],
                lanes=[struct.unpack_from('<8I', raw, lanes + 32*i) for i in range(fields[3])],
                addresses=[struct.unpack_from('<8Q', raw, addresses + 64*i)
                           for i in range(fields[3]*fields[1])])


def install_window(owner, fixture, rank=0):
    """Write explicit local F window, route table and Inbox fixture."""
    d = directory(owner)
    payload = bytearray([0xC3]) * owner.payload.numel()
    controls = bytearray(owner.control_buffer.numel())
    for lane, records in enumerate(fixture['records']):
        records_offset, refs_offset, counts_offset, prefix_offset, inbox, ready, _, _ = d['addresses'][lane*d['ranks'] + rank]
        packed = b''.join(records)
        payload[records_offset:records_offset + len(packed)] = packed
        routes = fixture['routes'][lane]
        counts = np.array([len(refs) for refs in routes], dtype='<u4')
        prefixes = counts.cumsum(dtype=np.uint32)
        refs = np.array([ref for expert in routes for ref in expert], dtype='<u4')
        for offset, data in ((refs_offset, refs), (counts_offset, counts), (prefix_offset, prefixes)):
            payload[offset:offset + data.nbytes] = data.tobytes()
        struct.pack_into('<16I', controls, inbox, 0, 0, 0, 1, len(records), 0, len(records), *([0]*9))
        struct.pack_into('<I', controls, ready, 0)
    put_bytes(owner.payload, payload)
    put_bytes(owner.control_buffer, controls)
    return payload, controls


def install_active(owner, segments, entry_lanes, **kwargs):
    d = directory(owner)
    active, state = pack_active(d['num_lanes'], d['experts']//d['ranks'], segments, entry_lanes, **kwargs)
    put_bytes(owner.active, active)
    put_bytes(owner.execution_state, state)
    return active, state


def invoke(owner, *, recv_x=None, recv_sf=None, recv_src_metadata=None):
    owner.dispatchRecv(owner.payload, owner.context, owner.active, owner.execution_state,
        owner.recv_workspace, owner.recv_x if recv_x is None else recv_x,
        owner.recv_sf if recv_sf is None else recv_sf,
        owner.recv_src_metadata if recv_src_metadata is None else recv_src_metadata)
    torch.npu.synchronize()


def poison_outputs(owner):
    owner.recv_x.view(torch.uint8).fill_(SENTINEL_BYTE)
    if owner.recv_sf is not None:
        owner.recv_sf.fill_(-23131)  # 0xA5A5, including noncontiguous layouts.
    owner.recv_src_metadata.fill_(SENTINEL_METADATA)


def assert_outputs(owner, expected, *, stage=INPUT_READY, entries=1, grants=None):
    x, sf, metadata = expected
    rows = len(x)
    actual_x = owner.recv_x.view(torch.uint8).cpu().numpy()
    actual_metadata = owner.recv_src_metadata.cpu().numpy()
    errors = {'recv_x_byte_errors': bit_errors(actual_x[:rows], x),
              'metadata_byte_errors': bit_errors(actual_metadata[:rows], metadata),
              'recv_sf_byte_errors': 0}
    assert not actual_x[rows:].size or np.all(actual_x[rows:] == SENTINEL_BYTE), 'unselected hidden tail modified'
    assert np.all(actual_metadata[rows:] == SENTINEL_METADATA), 'unselected metadata tail modified'
    if owner.recv_sf is not None:
        actual_sf = owner.recv_sf.cpu().numpy()
        errors['recv_sf_byte_errors'] = bit_errors(actual_sf[:rows], sf.copy().view(np.int16))
        assert np.all(actual_sf[rows:] == -23131), 'unselected SF tail modified'
    state_bytes = raw_bytes(owner.execution_state)
    state = struct.unpack_from('<8I', state_bytes)
    assert state[:3] == (stage, 0, entries), f'execution state {state}'
    expected_grants = [0]*entries if grants is None else grants
    assert list(struct.unpack_from(f'<{entries}I', state_bytes, 32)) == expected_grants, 'grant snapshot changed'
    assert all(error == 0 for error in errors.values()), errors
    return errors


def snapshot_active(owner, rank=0, alignment=4):
    """Fixture full current stable input; snapshot after true send completion.

    This is intentionally an offline test fixture, not ffnScheduler. It takes no
    scheduling decision, does not poll DP readiness, and never sends Consumed.
    """
    d = directory(owner)
    payload, controls = raw_bytes(owner.payload), raw_bytes(owner.control_buffer)
    local_experts = d['experts']//d['ranks']
    pos = wire_layout(d['hidden_bytes'], 2*d['sf_packs'], d['k'])
    records, routes = [], []
    for lane in range(d['num_lanes']):
        address = d['addresses'][lane*d['ranks'] + rank]
        record_offset, refs_offset, counts_offset, _, inbox, _, _, _ = address
        count = struct.unpack_from('<16I', controls, inbox)[6]
        records.append([payload[record_offset + i*pos['stride']:record_offset + (i+1)*pos['stride']]
                        for i in range(count)])
        counts = struct.unpack_from(f'<{local_experts}I', payload, counts_offset)
        lane_routes, cursor = [], 0
        for n in counts:
            lane_routes.append(list(struct.unpack_from(f'<{n}I', payload, refs_offset + 4*cursor)))
            cursor += n
        routes.append(lane_routes)
    # ABI entry-owned segments are contiguous; one explicit entry includes all
    # test lanes in the same DP. Destination rows still group by expert.
    segments, row = [], 0
    for expert in range(local_experts):
        start = row
        for lane in range(d['num_lanes']):
            count = len(routes[lane][expert])
            if count:
                begin = sum(len(v) for v in routes[lane][:expert])
                segments.append(Segment(0, lane, expert, begin, count, row))
                row += count
        row = start + ((row-start + alignment-1)//alignment)*alignment
    fixture = dict(records=records, routes=routes, segments=segments,
                   hidden_bytes=d['hidden_bytes'], sf_bytes=2*d['sf_packs'],
                   k=d['k'], num_experts=local_experts)
    return fixture, reference(**fixture, rank=rank, alignment=alignment,
                              capacity=max(row, 1), token_capacity=max(v[5] for v in d['lanes']))

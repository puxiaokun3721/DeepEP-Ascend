"""Actual single-card ffnScheduler fixtures with synthetic peer metadata.

Use an external watchdog and an independently approved device/run binding:
  python tests/afd/run_scheduler_a5.py --device VISIBLE_DEVICE --output RUN/scheduler.json \
      --run-id RUN_ID --deepep-sha FULL_SHA --afd-sha FULL_SHA
Only ffnScheduler is launched. No communication, completion, graph or performance
acceptance is implied. Detailed case JSON files accompany the output manifest.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess
import sys
import traceback

import torch
import torch_npu  # Registers the actual NPU backend.
import deep_ep
import deep_ep._C as deep_ep_extension

from recv_npu_support import directory, put_bytes, raw_bytes


ROOT = Path(__file__).resolve().parents[2]
AFD_CONTRACT_SHA = '5ac3177c9a2fcade47051f251ba4b19a5954bbb4'
NUM_RANKS = 1
NUM_SOURCE_RANKS = 2
NUM_EXPERTS = 2
NUM_LANES = 2
NUM_TOPK = 2
CAPACITY = 8
ALIGNMENT = 4
MEMORY_BUDGET = 64 * 1024**2
TABLE_RECORD_BYTES = 32
CONTEXT_PREFIX = struct.Struct('<16I8Q')
ACTIVE_HEADER = struct.Struct('<10I3Q')
WORDS = struct.Struct('<8I')
INBOX = struct.Struct('<16I')
COUNTS = struct.Struct('<2I')
RETURN_DIRECTORY = struct.Struct('<3Q2I')
RETURN_LANE = struct.Struct('<4Q')
EXPECTED_TABLE_OFFSETS = (64, 128, 192)
MAX_REPORT_BYTES = 9000
SENTINEL_PAYLOAD = 0xC3
SENTINEL_OUTPUT = -7
NOP, WORK, CONTROL, ABORT = range(4)
EMPTY, SELECTED, ABORTED = 0, 1, 5
BAD_LAYER = 3
SCOPE = 'single-card actual ffnScheduler kernel; synthetic peer metadata; no communication/graph'


@dataclass(frozen=True)
class Lane:
    counts: tuple[int, int] = (0, 0)
    tokens: int = 0
    records: int = 0
    published: bool = False
    credit: int = 0
    layer: int = 0


def no_selection(*, bad_layer=False):
    header = [0] * 10 + list(EXPECTED_TABLE_OFFSETS)
    execution = [0] * 8
    if bad_layer:
        header[1] = ABORT
        execution[:2] = [ABORTED, BAD_LAYER]
    return dict(active_header=header, entries=[], members=[], segments=[],
                execution_header=execution, execution_grants=[],
                completion_header=[0] * 8, rows=[0, 0], prefix=[0, 0])


def selection(kind, entry, segments, expanded, padded, rows, prefix):
    return dict(
        active_header=[1, kind, 0, 0, 1, 2, len(segments), expanded, padded, 0,
                       *EXPECTED_TABLE_OFFSETS],
        entries=[list(entry)],
        members=[[0, 0, 0, 0, 2, 0, 0, 0], [1, 0, 0, 2, 2, 0, 0, 0]],
        segments=[[*segment, 0, 0] for segment in segments],
        execution_header=[SELECTED, 0, 1, 0, 0, 0, 0, 0],
        execution_grants=[entry[7]], completion_header=[1, 0, 1, 2, 4, 0, 0, 0],
        rows=list(rows), prefix=list(prefix))


def cases():
    """Literal contract goldens; no CPU scheduler substitutes for the kernel."""
    normal = (Lane((3, 1), 3, 3, True), Lane((0, 2), 2, 2, True))
    work = selection(WORK, (0, 0, 2, 0, 3, 6, 1, 0),
                     [(0, 0, 0, 0, 3, 0), (0, 0, 1, 3, 1, 4),
                      (0, 1, 1, 0, 2, 5)], 6, 8, [3, 3], [4, 8])
    specifications = [
        dict(name='S0-no-publish', lanes=(Lane(), Lane()), expected=[no_selection()]),
        dict(name='S1-normal', lanes=normal, expected=[work]),
        dict(name='S2-padding-budget', lanes=normal, q=4, qa=4, expected=[
            selection(WORK, (0, 0, 2, 0, 1, 3, 0, 0),
                      [(0, 0, 0, 0, 3, 0)], 3, 4, [3, 0], [4, 4])]),
        dict(name='S3-zero-member-arrives', lanes=(Lane((2, 0), 2, 2, True), Lane()),
             update='publish-zero-member', expected=[no_selection(),
             selection(WORK, (0, 0, 2, 0, 1, 2, 1, 0),
                       [(0, 0, 0, 0, 2, 0)], 2, 4, [2, 0], [4, 4])]),
    ]
    for precheck in (False, True):
        specifications.append(dict(
            name=f'S4-zero-credit-precheck-{int(precheck)}', precheck=precheck,
            lanes=(Lane(published=True), Lane(published=True, credit=1)),
            update='return-credit', expected=[no_selection(),
            selection(CONTROL, (0, 0, 2, 0, 0, 0, 1, int(precheck)),
                      [], 0, 0, [0, 0], [0, 0])]))
    for precheck in (False, True):
        specifications.append(dict(
            name=f'S5-work-credit-precheck-{int(precheck)}', precheck=precheck,
            lanes=(normal[0], Lane((0, 2), 2, 2, True, credit=1)),
            expected=[no_selection() if precheck else work]))
    specifications.append(dict(name='S6-bad-layer',
                               lanes=(Lane(published=True, layer=99), Lane()),
                               expected=[no_selection(bad_layer=True)]))
    return specifications


def pack_lane(payload, controls, address, credit_offset, lane):
    if not (0 <= lane.records <= lane.tokens <= CAPACITY and
            all(count >= 0 for count in lane.counts) and
            sum(lane.counts) <= lane.tokens * NUM_TOPK):
        raise ValueError(f'invalid fixture lane: {lane}')
    COUNTS.pack_into(payload, address[2], *lane.counts)
    COUNTS.pack_into(payload, address[3], lane.counts[0], sum(lane.counts))
    INBOX.pack_into(controls, address[4], 0, lane.layer, 0, 1,
                    lane.tokens, 0, lane.records, *([0] * 9))
    struct.pack_into('<I', controls, address[5], 0 if lane.published else 1)
    struct.pack_into('<I', controls, credit_offset, lane.credit)


def install_fixture(owner, lanes):
    d = directory(owner)
    context = d['raw']
    fields = CONTEXT_PREFIX.unpack_from(context)
    if (len(context), fields[20], fields[2], d['ranks'], d['num_lanes'], d['experts'], d['k']) != (
            640, 640, NUM_SOURCE_RANKS, NUM_RANKS, NUM_LANES, NUM_EXPERTS, NUM_TOPK):
        raise AssertionError('unexpected complete context size or fixture family')
    return_directory = RETURN_DIRECTORY.unpack_from(context, fields[23])
    return_lanes_offset = return_directory[1]
    if (fields[23], return_lanes_offset) != (480, 576):
        raise AssertionError('unexpected return-directory layout for the fixed family')
    credit_offsets = [RETURN_LANE.unpack_from(
        context, return_lanes_offset + RETURN_LANE.size * lane)[2]
        for lane in range(NUM_LANES)]
    payload = bytearray([SENTINEL_PAYLOAD]) * owner.payload.numel()
    # Preserve prepare_ffn's unpublished Ready=1 state instead of clearing control.
    controls = bytearray(raw_bytes(owner.control_buffer))
    for lane_idx, lane in enumerate(lanes):
        address = d['addresses'][lane_idx * NUM_RANKS]
        if struct.unpack_from('<I', controls, address[5])[0] != 1:
            raise AssertionError('prepare_ffn did not initialize unpublished Ready')
        pack_lane(payload, controls, address, credit_offsets[lane_idx], lane)
    put_bytes(owner.payload, payload)
    put_bytes(owner.control_buffer, controls)
    torch.npu.synchronize()
    return d, credit_offsets


def input_snapshot(owner):
    return dict(context=raw_bytes(owner.context), payload=raw_bytes(owner.payload),
                control=raw_bytes(owner.control_buffer))


def input_comparison(before, after):
    return {name: dict(unchanged=before[name] == after[name],
                       before_sha256=hashlib.sha256(before[name]).hexdigest(),
                       after_sha256=hashlib.sha256(after[name]).hexdigest(),
                       bytes=len(before[name])) for name in before}


def update_waiting_fixture(owner, update, d, credit_offsets):
    before = input_snapshot(owner)
    payload, controls = bytearray(before['payload']), bytearray(before['control'])
    address = d['addresses'][NUM_RANKS]  # Second member, local F rank zero.
    allowed = dict(context=[], payload=[], control=[])
    if update == 'publish-zero-member':
        COUNTS.pack_into(payload, address[2], 0, 0)
        INBOX.pack_into(controls, address[4], 0, 0, 0, 1, 0, 0, 0, *([0] * 9))
        struct.pack_into('<I', controls, address[5], 0)
        allowed['payload'] = [(address[2], address[2] + COUNTS.size)]
        allowed['control'] = [(address[4], address[4] + INBOX.size),
                              (address[5], address[5] + 4)]
        put_bytes(owner.payload, payload)
    elif update == 'return-credit':
        struct.pack_into('<I', controls, credit_offsets[1], 0)
        allowed['control'] = [(credit_offsets[1], credit_offsets[1] + 4)]
    else:
        raise ValueError(f'unknown fixture update: {update}')
    put_bytes(owner.control_buffer, controls)
    torch.npu.synchronize()
    after = input_snapshot(owner)
    changes = {}
    for name in before:
        if len(before[name]) != len(after[name]):
            raise AssertionError(f'{name} size changed during fixture update')
        offsets = [i for i, (old, new) in enumerate(zip(before[name], after[name]))
                   if old != new]
        if any(not any(start <= i < end for start, end in allowed[name]) for i in offsets):
            raise AssertionError(f'fixture update modified forbidden {name} bytes')
        changes[name] = dict(allowed_ranges=allowed[name], changed_offsets=offsets)
    if after['payload'] != payload or after['control'] != controls:
        raise AssertionError('fixture upload differs from its complete CPU buffer')
    return changes


def read_outputs(owner):
    active = raw_bytes(owner.active)
    header = list(ACTIVE_HEADER.unpack_from(active))
    state = raw_bytes(owner.execution_state)
    actual = dict(active_header=header, execution_header=list(WORDS.unpack_from(state)),
                  completion_header=list(WORDS.unpack_from(raw_bytes(owner.completion))),
                  rows=owner.num_unaligned_recv_tokens_per_expert.cpu().tolist(),
                  prefix=owner.psum_num_recv_tokens_per_expert.cpu().tolist())
    counts = header[4:7]
    if any(n > bound for n, bound in zip(counts, (NUM_LANES, NUM_LANES, NUM_LANES * NUM_EXPERTS))):
        actual['decode_error'] = 'Active declares more records than the fixed allocation'
        return actual
    for name, count, offset in zip(('entries', 'members', 'segments'), counts, header[10:13]):
        if offset > len(active) or count * TABLE_RECORD_BYTES > len(active) - offset:
            actual['decode_error'] = f'{name} range exceeds Active allocation'
            return actual
        actual[name] = [list(WORDS.unpack_from(active, offset + TABLE_RECORD_BYTES * i))
                        for i in range(count)]
    actual['execution_grants'] = list(struct.unpack_from(f'<{counts[0]}I', state, WORDS.size))
    return actual


def run_case(specification, result):
    q, qa = specification.get('q', 8), specification.get('qa', 12)
    precheck = specification.get('precheck', False)
    result['parameters'] = dict(q=q, qa=qa, alignment=ALIGNMENT, precheck=precheck,
                               lanes=[vars(lane) for lane in specification['lanes']])
    owner = deep_ep.AFDBuffer.create_local_receiver(
        NUM_RANKS, 0, [[0, 0, 0, 0, 0, CAPACITY], [1, 0, 0, 1, 0, CAPACITY]],
        [0], 128, 0, NUM_TOPK, NUM_EXPERTS, CAPACITY, False, 1, MEMORY_BUDGET)
    owner.prepare_recv(q, qa, ALIGNMENT, False)
    owner.prepare_ffn(precheck)
    torch.npu.synchronize()
    d, credit_offsets = install_fixture(owner, specification['lanes'])
    result['context_bytes'] = len(d['raw'])
    for step_idx, expected in enumerate(specification['expected']):
        step = dict(call=step_idx + 1, status='RUNNING', expected=expected)
        result['calls'].append(step)
        if step_idx:
            # The previous exact NOP check passed; retain its owner and Pending state.
            step['fixture_update'] = update_waiting_fixture(
                owner, specification['update'], d, credit_offsets)
        owner.num_unaligned_recv_tokens_per_expert.fill_(SENTINEL_OUTPUT)
        owner.psum_num_recv_tokens_per_expert.fill_(SENTINEL_OUTPUT)
        torch.npu.synchronize()
        before = input_snapshot(owner)
        owner.ffnScheduler(owner.context, owner.control_buffer, owner.scheduler_workspace,
                           owner.active, owner.execution_state, owner.completion,
                           owner.psum_num_recv_tokens_per_expert,
                           owner.num_unaligned_recv_tokens_per_expert)
        torch.npu.synchronize()
        step['actual'] = read_outputs(owner)
        step['inputs'] = input_comparison(before, input_snapshot(owner))
        step['mismatches'] = [key for key in expected if step['actual'].get(key) != expected[key]]
        if step['mismatches'] or any(not item['unchanged'] for item in step['inputs'].values()):
            step['status'] = 'FAIL'
            raise AssertionError(f"{specification['name']} call {step_idx + 1}: {step['mismatches']}; "
                                 f"inputs unchanged={all(v['unchanged'] for v in step['inputs'].values())}")
        step['status'] = 'PASS'
    # No finally cleanup after errors/timeouts and no invocation after ARMED success.
    torch.npu.synchronize()
    owner.destroy()
    result['status'] = 'PASS'


def save_json(path, content, *, exclusive=False):
    encoded = (json.dumps(content, separators=(',', ':'), ensure_ascii=True) + '\n').encode('utf-8')
    if len(encoded) > MAX_REPORT_BYTES:
        raise ValueError(f'{path.name}: report is {len(encoded)} bytes, limit {MAX_REPORT_BYTES}')
    with path.open('xb' if exclusive else 'wb') as handle:
        handle.write(encoded)
    return hashlib.sha256(encoded).hexdigest()


def full_sha(value):
    if re.fullmatch('[0-9a-f]{40}', value) is None:
        raise argparse.ArgumentTypeError('expected a full lowercase 40-character Git SHA')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=int, required=True, help='approved visible NPU device number')
    parser.add_argument('--output', type=Path, required=True, help='new output manifest, never overwritten')
    parser.add_argument('--run-id', required=True, help='run identifier from the execution REQUEST')
    parser.add_argument('--deepep-sha', type=full_sha, required=True)
    parser.add_argument('--afd-sha', type=full_sha, required=True, help='bound AFD contract revision')
    args = parser.parse_args()
    specifications = cases()
    paths = {spec['name']: args.output.with_name(f"{args.output.stem}-{spec['name']}.json")
             for spec in specifications}
    if args.output.exists() or any(path.exists() for path in paths.values()):
        raise FileExistsError('output manifest or case file already exists; use a new run path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(status='RUNNING', run_id=args.run_id, scope=SCOPE,
                  deepep_sha=args.deepep_sha, afd_contract_sha=args.afd_sha,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  device_requested=args.device,
                  deep_ep_module=str(Path(deep_ep.__file__).resolve()),
                  extension_path=str(Path(deep_ep_extension.__file__).resolve()),
                  required_cases=[spec['name'] for spec in specifications], cases=[])
    save_json(args.output, result, exclusive=True)
    current_case = None
    try:
        if args.afd_sha != AFD_CONTRACT_SHA:
            raise RuntimeError('AFD contract differs from the revision used for these goldens')
        with Path(result['extension_path']).open('rb') as extension:
            result['extension_sha256'] = hashlib.file_digest(extension, 'sha256').hexdigest()
        result['extension_commit_binding'] = 'not inferred; execution REQUEST must bind built artifacts and JIT'
        result['source_head'] = subprocess.check_output(
            ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        dirty = subprocess.check_output(
            ['git', '-C', str(ROOT), 'status', '--porcelain', '--untracked-files=no'], text=True)
        if result['source_head'] != args.deepep_sha or dirty:
            raise RuntimeError('source HEAD differs from bound SHA or tracked checkout is dirty')
        torch.npu.set_device(args.device)
        result['device_actual'] = torch.npu.current_device()
        if result['device_actual'] != args.device:
            raise RuntimeError('actual NPU device differs from requested visible number')
        for specification in specifications:
            current_case = dict(case=specification['name'], status='RUNNING', calls=[])
            run_case(specification, current_case)
            if (len(current_case['calls']) != len(specification['expected']) or
                    any(call['status'] != 'PASS' for call in current_case['calls'])):
                raise AssertionError('case did not pass every planned invocation')
            path = paths[specification['name']]
            digest = save_json(path, current_case, exclusive=True)
            result['cases'].append(dict(case=specification['name'], status='PASS',
                                        calls=len(current_case['calls']), file=path.name, sha256=digest))
            current_case = None
            save_json(args.output, result)
        if [case['case'] for case in result['cases']] != result['required_cases']:
            raise AssertionError('not all planned cases completed')
        result['status'] = 'PASS'
        save_json(args.output, result)
        print(json.dumps(result, separators=(',', ':')))
    except Exception as error:
        result['status'] = 'FAIL'
        result['error'] = f'{type(error).__name__}: {error}'[:1000]
        if current_case is not None:
            current_case['status'] = 'FAIL'
            current_case['error'] = result['error']
            if current_case['calls'] and current_case['calls'][-1]['status'] == 'RUNNING':
                current_case['calls'][-1]['status'] = 'FAIL'
            result['failed_case'] = current_case['case']
        # Persist failure before any optional detail that can exceed its size limit.
        # Do not rewrite this manifest after attempting the failure detail.
        try:
            save_json(args.output, result)
        except Exception as report_error:
            print(f'Could not persist FAIL manifest: {report_error}', file=sys.stderr)
        if current_case is not None:
            path = paths[current_case['case']]
            try:
                if not path.exists():
                    save_json(path, current_case, exclusive=True)
            except Exception as report_error:
                print(f'Could not persist failure detail: {report_error}', file=sys.stderr)
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()

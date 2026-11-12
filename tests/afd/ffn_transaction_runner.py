"""Diagnostic CAM transaction driver. Host snapshots make this an eager bring-up."""
import json
import struct
import time

import torch
import torch_npu


HEADER = struct.Struct("<8I")
INBOX = struct.Struct("<16I")
ACTIVE = struct.Struct("<10I3Q")
MAGIC = 0x41465431
IDLE, SELECTED, SLICE, COMPLETE, ABORTED = range(5)


def align32(size):
    return (size + 31) // 32 * 32


def snapshot(tensor):
    return tensor.cpu().numpy().tobytes()


def read_transaction(owner, lanes, experts, layers):
    raw = snapshot(owner.active)
    active = ACTIVE.unpack_from(raw)
    assert active[9] == MAGIC, f"unexpected Active mode {active[9]:#x}"
    tx_offset = 64 + 64 * lanes + 32 * lanes * experts
    state = HEADER.unpack_from(raw, tx_offset)
    entries = [HEADER.unpack_from(raw, tx_offset + 32 + i * 32) for i in range(state[4])]
    group_offset = tx_offset + 32 + 64 * lanes + align32(4 * lanes * experts)
    groups = struct.unpack_from(f"<{layers * experts}I", raw, group_offset)
    return active, state, entries, groups


def selected_rounds(owner, entries, lane_specs):
    workspace = snapshot(owner.scheduler_workspace)
    rounds = []
    for pending_idx, _, _, layer_slot, _, _, _, _ in entries:
        mb, layer, comm_round, total = INBOX.unpack_from(workspace, 64 + 96 * pending_idx)[:4]
        _, _, dp, _, phase, _ = lane_specs[pending_idx]
        rounds.append(dict(dp=dp, mb=mb, layer=layer, layer_slot=layer_slot,
                           round=comm_round, rounds=total, phase=phase))
    return rounds


def missing_members(owner, observations, lane_specs):
    controls = snapshot(owner.control_buffer)
    pending = snapshot(owner.scheduler_workspace)
    groups = {}
    for lane, obs in enumerate(observations):
        classification = obs[0]
        if classification not in (1, 2):
            continue
        _, _, dp, member, phase, _ = lane_specs[lane]
        inbox = (INBOX.unpack_from(controls, 128 * lane) if classification == 1 else
                 INBOX.unpack_from(pending, 64 + 96 * lane))
        groups.setdefault((dp, phase, *inbox[:4]), set()).add(member)
    partial = []
    for key, arrived in groups.items():
        dp, phase, mb, layer, comm_round, rounds = key
        expected = {member for _, _, lane_dp, member, lane_phase, _ in lane_specs
                    if (dp, phase) == (lane_dp, lane_phase)}
        if expected != arrived:
            partial.append(dict(dp=dp, mb=mb, layer=layer, round=comm_round, rounds=rounds,
                                phase=phase, missing=sorted(expected - arrived)))
    return partial


def run_transaction_ffn(owner, cfg, *, rank, lane_specs, layer_ids, expected_rounds,
                        timeout, recv, send, grouped_gemm, wait_first_cohort):
    lanes = cfg["num_source_lanes"]
    experts = cfg["num_experts"] // cfg["num_ranks"]
    alignment = cfg["recv_alignment"]
    hidden = cfg["num_hidden"]
    identity = None
    if grouped_gemm:
        # Small diagnostic weights. Real serving supplies W13/activation/W2
        # and a backend-specific layout via the same layer/expert row groups.
        identity = torch.eye(hidden, dtype=torch.bfloat16, device=owner.context.device)
    deadline = time.monotonic() + timeout
    finished = 0
    last_poll = None

    def emit(event, **fields):
        print("FFN_TX " + json.dumps({"event": event, "f_rank": rank, **fields}), flush=True)

    while finished < expected_rounds:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"FFN transaction finished {finished}/{expected_rounds} rounds")
        start = time.monotonic_ns()
        with torch.profiler.record_function("afd.ffn_poll"):
            observations = owner.ffnPoll().cpu().tolist()
        fingerprint = tuple(tuple(obs) for obs in observations)
        if fingerprint != last_poll:
            emit("poll", ready=[i for i, obs in enumerate(observations) if obs[0] == 1],
                 pending=[i for i, obs in enumerate(observations) if obs[0] == 2],
                 missing_members=missing_members(owner, observations, lane_specs),
                 host_wait_ns=time.monotonic_ns() - start)
            last_poll = fingerprint
        if wait_first_cohort and finished == 0 and any(obs[0] != 1 for obs in observations):
            continue
        with torch.profiler.record_function("afd.ffn_scheduler"):
            owner.ffnScheduler(owner.context, owner.control_buffer, owner.scheduler_workspace,
                               owner.active, owner.execution_state, owner.completion,
                               owner.psum_num_recv_tokens_per_expert,
                               owner.num_unaligned_recv_tokens_per_expert)
        active, state, entries, _ = read_transaction(owner, lanes, experts, len(layer_ids))
        if state[0] == IDLE:
            continue
        if state[0] != SELECTED:
            raise RuntimeError(f"FFN scheduler state={state} exec={owner.execution_state.cpu().tolist()[:8]}")
        txn = state[1]
        rounds = selected_rounds(owner, entries, lane_specs)
        emit("scheduler", txn=txn, phase=active[2], entries=rounds, result="selected")
        seen_groups = set()
        retired = set()
        while state[0] == SELECTED:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"FFN transaction {txn} did not finish")
            with torch.profiler.record_function("afd.dispatch_recv"):
                recv()
            active, state, _, counts = read_transaction(owner, lanes, experts, len(layer_ids))
            if state[0] != SLICE or state[1] != txn or HEADER.unpack_from(snapshot(owner.execution_state))[0] != 2:
                raise RuntimeError(f"dispatchRecv failed to prepare txn={txn}: state={state}")
            groups = []
            offset = 0
            for group in range(state[2], state[3]):
                count = counts[group]
                if not count:
                    continue
                layer = layer_ids[group // experts]
                expert = group % experts
                padded = (count + alignment - 1) // alignment * alignment
                assert (layer, expert) not in seen_groups
                seen_groups.add((layer, expert))
                groups.append(dict(layer=layer, expert=expert, rows=count, padded=padded, row_begin=offset))
                offset += padded
            assert sum(g["rows"] for g in groups) == active[7] and offset == active[8]
            emit("dispatch_recv", txn=txn, slice=state[7], groups=groups, padded=offset)
            if groups:
                with torch.profiler.record_function("afd.ffn_grouped_gemm" if grouped_gemm else "afd.ffn_fixture"):
                    if grouped_gemm:
                        group_counts = torch.tensor([g["padded"] for g in groups], dtype=torch.int64,
                                                    device=owner.context.device)
                        weights = torch.stack([identity * (g["layer"] + 1) for g in groups])
                        result = torch_npu.npu_grouped_matmul(
                            x=[owner.recv_x[:offset]], weight=[weights], split_item=2,
                            group_list_type=1, group_type=0, group_list=group_counts)[0]
                        owner.combine_x[:offset].copy_(result)
                    else:
                        for group in groups:
                            start = group["row_begin"]
                            end = start + group["padded"]
                            owner.combine_x[start:end].copy_(owner.recv_x[start:end] * (group["layer"] + 1))
                emit("gmm" if grouped_gemm else "ffn_fixture", txn=txn, slice=state[7])
            owner.computeReady()
            slice_index = state[7]
            with torch.profiler.record_function("afd.combine_send"):
                send()
            _, state, entries, _ = read_transaction(owner, lanes, experts, len(layer_ids))
            if state[0] not in (SELECTED, COMPLETE) or state[7] != slice_index + 1:
                raise RuntimeError(f"combineSend did not commit txn={txn}, slice={slice_index}: {state}")
            emit("combine_send", txn=txn, slice=slice_index, committed=True)
            newly_retired = [i for i, entry in enumerate(entries) if entry[6] and i not in retired]
            for i in newly_retired:
                emit("done_consumed", txn=txn, slice=slice_index, entry=rounds[i])
                retired.add(i)
                finished += 1
        assert state[0] == COMPLETE and len(retired) == len(rounds)
        emit("transaction_end", txn=txn, completed_rounds=len(retired))
    assert finished == expected_rounds
    owner.ffnScheduler(owner.context, owner.control_buffer, owner.scheduler_workspace,
                       owner.active, owner.execution_state, owner.completion,
                       owner.psum_num_recv_tokens_per_expert, owner.num_unaligned_recv_tokens_per_expert)
    assert read_transaction(owner, lanes, experts, len(layer_ids))[1][0] == IDLE

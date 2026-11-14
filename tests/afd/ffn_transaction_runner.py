"""Diagnostic CAM transaction driver. Host snapshots make this an eager bring-up."""
import json
import struct
import time

import torch
import torch_npu


HEADER = struct.Struct("<8I")
INBOX = struct.Struct("<16I")
POLL_TIMING = struct.Struct("<IIQQ10I")
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


def selected_rounds(owner, entries, lane_specs, experts):
    workspace = snapshot(owner.scheduler_workspace)
    active = snapshot(owner.active)
    lanes = len(lane_specs)
    members_offset = 64 + 64 * lanes + 32 * lanes * experts + 32 + 32 * lanes
    rounds = []
    for pending_idx, member_begin, member_count, layer_slot, _, _, _, _ in entries:
        mb, layer, comm_round, total, length, _, _, mode = INBOX.unpack_from(
            workspace, 64 + 96 * pending_idx)[:8]
        _, _, dp, _, phase, _ = lane_specs[pending_idx]
        members = []
        for member_offset in range(member_begin, member_begin + member_count):
            lane, input_bit, return_bit, *_ = HEADER.unpack_from(active, members_offset + 32 * member_offset)
            inbox = INBOX.unpack_from(workspace, 64 + 96 * lane)
            members.append(dict(lane=lane, local_total_rounds=inbox[3], round_tokens=inbox[4],
                                input_bit=input_bit, return_bit=return_bit))
        rounds.append(dict(dp=dp, mb=mb, layer=layer, layer_slot=layer_slot,
                           round=comm_round, rounds=total, phase=phase, members=members,
                           representative_tokens=length, local_rounds=bool(mode)))
    return rounds


def missing_members(owner, observations, lane_specs, local_rounds):
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
        mb, layer, comm_round, total = inbox[:4]
        if local_rounds and comm_round > 0:
            continue  # Later-round membership belongs to the device continuation table.
        groups.setdefault((dp, phase, mb, layer, comm_round, None if local_rounds else total), set()).add(member)
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
                        timeout, recv, send, grouped_gemm, wait_first_cohort,
                        local_comm_rounds=False, poll_cycles=False, wait_tail_overlap=False):
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
    timestamps = (torch.zeros(lanes * POLL_TIMING.size, dtype=torch.uint8, device=owner.context.device)
                  if poll_cycles else None)
    latest_cycle = {}
    observed_tail_overlap = False
    completed_rounds = set()

    def emit(event, **fields):
        print("FFN_TX " + json.dumps({"event": event, "f_rank": rank, **fields}), flush=True)

    def poll():
        if timestamps is None:
            return owner.ffnPoll().cpu().tolist()
        observations = owner.ffnPollTimestamp(timestamps).cpu().tolist()
        # Observe twice without scheduler adoption. A repeated Ready must keep its cycle.
        first_snapshot = snapshot(timestamps)
        repeated = owner.ffnPollTimestamp(timestamps).cpu().tolist()
        second_snapshot = snapshot(timestamps)
        for lane, (first, second) in enumerate(zip(observations, repeated)):
            offset = lane * POLL_TIMING.size
            old = POLL_TIMING.unpack_from(first_snapshot, offset)
            new = POLL_TIMING.unpack_from(second_snapshot, offset)
            if second[0] == 1 and (not new[0] or new[1] != second[1] or not new[3]):
                raise RuntimeError(f"Poll did not timestamp fresh Ready for lane {lane}: {second}, {new}")
            if first[0] == second[0] == 1 and first == second and old != new:
                raise RuntimeError(f"Poll rewrote first cycle for unchanged lane {lane}")
            if new[0] and latest_cycle.get(lane) != new:
                previous = latest_cycle.get(lane)
                if previous is not None and new[3] <= previous[3]:
                    raise RuntimeError(f"Poll cycles did not advance for lane {lane}: {previous}, {new}")
                latest_cycle[lane] = new
                emit("poll_first_ready", lane=lane, bit=new[1], previous_adopt_tick=new[2],
                     system_cycles=new[3], mb=new[4], layer=new[5], round=new[6],
                     num_comm_rounds=new[7], round_tokens=new[8], local_rounds=bool(new[11]))
        return repeated

    while finished < expected_rounds:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"FFN transaction finished {finished}/{expected_rounds} rounds")
        start = time.monotonic_ns()
        with torch.profiler.record_function("afd.ffn_poll"):
            observations = poll()
        fingerprint = tuple(tuple(obs) for obs in observations)
        if fingerprint != last_poll:
            emit("poll", ready=[i for i, obs in enumerate(observations) if obs[0] == 1],
                 pending=[i for i, obs in enumerate(observations) if obs[0] == 2],
                 missing_members=missing_members(owner, observations, lane_specs, local_comm_rounds),
                 host_wait_ns=time.monotonic_ns() - start)
            last_poll = fingerprint
        if wait_first_cohort and finished == 0 and any(obs[0] != 1 for obs in observations):
            continue
        if wait_tail_overlap and finished > 0 and not observed_tail_overlap:
            long_ready = latest_cycle.get(0)
            short_ready = latest_cycle.get(1)
            if not (long_ready and short_ready and observations[0][0] == observations[1][0] == 1 and
                    (long_ready[4], long_ready[6]) == (0, 1) and
                    (short_ready[4], short_ready[6]) == (1, 0)):
                continue
            emit("slot_overlap", old_mb=0, old_tail_lane=0, new_mb=1, reused_lane=1,
                 old_tail_cycles=long_ready[3], new_mb_cycles=short_ready[3])
            observed_tail_overlap = True
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
        rounds = selected_rounds(owner, entries, lane_specs, experts)
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
            if poll_cycles and state[0] == SELECTED:
                poll()
            newly_retired = [i for i, entry in enumerate(entries) if entry[6] and i not in retired]
            for i in newly_retired:
                entry = rounds[i]
                key = (entry["dp"], entry["phase"], entry["mb"], entry["layer"], entry["round"])
                if key in completed_rounds:
                    raise RuntimeError(f"duplicate round retirement: {entry}")
                completed_rounds.add(key)
                if wait_tail_overlap:
                    expected = [0, 1] if entry["round"] == 0 else [0]
                    if sorted(member["lane"] for member in entry["members"]) != expected:
                        raise RuntimeError(f"unexpected member roster: {entry}")
                emit("done_consumed", txn=txn, slice=slice_index, entry=entry)
                retired.add(i)
                finished += 1
        assert state[0] == COMPLETE and len(retired) == len(rounds)
        emit("transaction_end", txn=txn, completed_rounds=len(retired))
    assert finished == expected_rounds
    assert not wait_tail_overlap or observed_tail_overlap
    owner.ffnScheduler(owner.context, owner.control_buffer, owner.scheduler_workspace,
                       owner.active, owner.execution_state, owner.completion,
                       owner.psum_num_recv_tokens_per_expert, owner.num_unaligned_recv_tokens_per_expert)
    assert read_transaction(owner, lanes, experts, len(layer_ids))[1][0] == IDLE

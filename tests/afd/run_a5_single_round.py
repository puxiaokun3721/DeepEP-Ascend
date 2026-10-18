"""Actual two-rank, single-round A5 dispatchSend smoke; never run on CPU.

Run under an external watchdog, e.g. timeout 300s torchrun --standalone
--nproc_per_node=2 tests/afd/run_a5_single_round.py --batch 33 [--fp8].
This validates payload after sender completion, NOT asynchronous visibility,
ACK/reuse, multi-round flow, receiver concurrency or graph replay.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import struct
import numpy as np
import torch
import torch.distributed as dist
import torch_npu  # Registers the NPU backend.
import deep_ep
from deep_ep.utils.envs import get_hccl_group_name
from gen_data import generate
from golden import layout, verify_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=int,default=33)
    parser.add_argument('--fp8',action='store_true')
    parser.add_argument('--all-masked',action='store_true')
    parser.add_argument('--transposed-sf',action='store_true')
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    if not 0 <= args.batch <= 32768:
        parser.error('single round requires 0 <= batch <= 32768')
    if args.transposed_sf and not args.fp8:
        parser.error('--transposed-sf requires --fp8')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.npu.set_device(local_rank)
    dist.init_process_group('hccl',timeout=timedelta(seconds=180))
    rank = dist.get_rank()
    if dist.get_world_size() != 2:
        raise ValueError('exactly two ranks required: A=0, F=1')
    group = dist.distributed_c10d._get_default_group()
    dist.barrier()
    group_name = get_hccl_group_name(group,rank)
    hidden, k, experts = 128, 8, 8
    capacity = max(1,args.batch)
    data = generate(batch=capacity,hidden=hidden,k=k,experts=experts,ranks=1,fp8=args.fp8)
    if args.all_masked:
        data['mask'][:] = False
    num_sf_packs = data['sf_bytes'].shape[1]//2
    owner = deep_ep.AFDBuffer(rank,group_name,[0],[1],[[0,0,0,0,0,capacity]],[0],
        hidden,num_sf_packs,k,experts,capacity,args.fp8,1,60,30,
        8*1024**3,256*1024**2)
    # Finish initialization on both owners before any incoming writes.
    torch.npu.synchronize()
    dist.barrier()
    if rank == 0:
        x = torch.from_numpy(data['x_bytes'].copy()).view(torch.float8_e4m3fn if args.fp8 else torch.bfloat16).npu()
        topk = torch.from_numpy(data['topk'].copy()).npu()
        weights = torch.from_numpy(data['weight_bits'].copy().view(np.float32)).npu()
        mask = torch.from_numpy(data['mask'].copy()).npu()
        sf = None
        if args.fp8:
            sf = torch.from_numpy(data['sf_bytes'].copy().view(np.int16)).npu()
            if args.transposed_sf:
                sf = sf.T.contiguous().T
        task = np.frombuffer(struct.pack('<16I',1,0,0,0,0,args.batch,1,*([0]*9)),dtype=np.uint8).copy()
        task_info = torch.from_numpy(task).npu()
        owner.dispatchSend(x,sf,topk,weights,mask,task_info,owner.context,
                           owner.control_buffer,owner.round_ledger,owner.workspace,owner.send_status)
        torch.npu.synchronize()
        status = owner.send_status.cpu().tolist()
        assert status[:3] == [2,0,0], f'not PUBLISHED: {status}'
        # No F ACK was sent. This checks final publication can complete without ACK.
        assert struct.unpack_from('<I',owner.control_buffer.cpu().numpy().tobytes())[0] == 1
    dist.barrier()
    if rank == 1:
        context = owner.context.cpu().numpy().tobytes()
        addresses_offset = struct.unpack_from('<Q',context,80)[0]
        records,refs_offset,count_offset,prefix_offset,inbox_offset,ready_offset,_,_ = struct.unpack_from('<8Q',context,addresses_offset)
        controls = owner.control_buffer.cpu().numpy().tobytes()
        inbox = struct.unpack_from('<16I',controls,inbox_offset)
        assert struct.unpack_from('<I',controls,ready_offset)[0] == 0, 'InputReady first bit'
        assert inbox[:6] == (0,0,0,1,args.batch,0), f'Inbox {inbox}'
        payload = owner.payload.cpu().numpy().tobytes()
        counts = np.frombuffer(payload,dtype='<u4',count=experts,offset=count_offset).copy()
        prefixes = np.frombuffer(payload,dtype='<u4',count=experts,offset=prefix_offset).copy()
        refs = np.frombuffer(payload,dtype='<u4',count=int(prefixes[-1]),offset=refs_offset).copy()
        pos = layout(data['x_bytes'].shape[1],data['sf_bytes'].shape[1],k)
        record_bytes = payload[records:records+inbox[6]*pos['stride']]
        result = verify_snapshot(record_bytes,refs,counts,prefixes,0,
            data['x_bytes'][:args.batch],data['sf_bytes'][:args.batch],data['topk'][:args.batch],
            data['weight_bits'][:args.batch],experts,1,data['mask'][:args.batch])
        if args.output:
            args.output.mkdir(parents=True,exist_ok=True)
            np.savez(args.output/'inputs.npz',**{key:value[:args.batch] for key,value in data.items()})
            np.savez(args.output/'snapshot.npz',payload=np.frombuffer(record_bytes,np.uint8),
                route_refs=refs,counts=counts,prefixes=prefixes,rank=0,num_experts=experts,num_ranks=1)
        print(json.dumps({'status':'PASS','scope':'A5 single round after send completion',
                          'batch':args.batch,'fp8':args.fp8,**result}),flush=True)
    # Sender has drained its transport; snapshot complete. No second transaction or
    # simulated ACK is launched, and no unresolved device task survives destruction.
    dist.barrier()
    torch.npu.synchronize()
    owner.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

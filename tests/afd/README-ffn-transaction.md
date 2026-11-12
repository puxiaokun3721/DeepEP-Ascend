# CAM FFN transaction bring-up

The CAM base is UMDK `cam_feature@9179865e` (PRs #3860 and #3861). The owner and F-side scheduler base is DeepEP-Ascend `19fe6b7a`. Build the transaction changes in both repositories. The existing CAM DispatchSend/CombineRecv and A-side wire protocol are reused.

## Offline trace

From the DeepEP root:

```bash
python -B tests/afd/ffn_transaction_sim.py --trace /tmp/afd-ffn-txn.jsonl
python -B -m unittest discover -s tests/afd -p 'test_*cpu*.py'
```

The checked-in `ffn_transaction_trace.jsonl` is a deterministic CPU trace. Transaction 1 selects two D communication rounds once. Slice 0 retires DP0 (including its zero-route member), a new Ready appears on those lanes, and slice 1 retires DP1 without another selection. The ready P round is selected next. Different layer IDs with the same expert number use distinct groups. The tests also cover 80/40 groups under Q=100, per-group alignment, exact capacity, rejection below the configured group bound or backend cap, transport failure before commit, and independent selections on different F ranks.

## Device integration example

In the CANN/torch_npu build environment, build and install the run package and wheel from the patched UMDK tree:

```bash
bash build/cam/build.sh comm_operator -c ascend950 -a 'cam_deepep_dispatch_send;cam_deepep_dispatch_recv;cam_deepep_combine_send;cam_deepep_combine_recv'
```

Build the paired DeepEP extension using its normal submodule and CANN environment setup:

```bash
python -m pip install --no-build-isolation .
```

Run the small BF16 cohort on an available eight-card A5 group after confirming the matching builds are loaded:

```bash
timeout 930s torchrun --standalone --nproc_per_node=8 tests/afd/run_pipeline_cam.py \
  --ffn-transaction --transaction-cohort --transaction-layers 2 \
  --batch 16 --window 16 --hidden 128 --topk 2 --experts 8 \
  --q 64 --transaction-alignment 8 --backend-max-rows 64 --transaction-gmm
```

The cohort is a diagnostic waiting policy for one round and one slot only. For multiple rounds or slots, omit it and keep the existing A-side round-step requirements. With `--transaction-gmm`, a small BF16 grouped GEMM applies layer-specific diagonal weights for correctness diagnosis. The default transaction diagnostic uses a scaled copy. Neither replaces the serving model's W13/activation/W2 or measures its performance. `--profile-trace` reuses the existing Torch NPU profiler. Structured `FFN_TX` lines carry F rank, transaction ID, phase, DP/MB/layer/round, slice index, groups and per-round completion. Host snapshots make this eager instrumentation, and the reported polling duration includes the snapshot copy.

## Capacity and shared state

`prepare_recv(q, q_alloc, alignment, ffn_transaction=True, backend_max_rows=...)` activates the new mode. Q is a hard effective-row cap. The owner requires Q to cover the largest configured same-phase sum of lane capacities and Q_alloc to cover that bound after alignment. Q_alloc must fit the explicitly declared backend padded-row limit. Production also needs the target backend to accept the registered group count, alignment, weight and activation layouts, and its own workspace. The 64-row limit above is a diagnostic configuration, not a hardware capability assertion.

`ffnPoll()` reports `[classification, observed_bit, expected_bit, phase]` per lane (0 absent, 1 fresh, 2 pending, -1 error). Scheduler freezes a transaction in an Active trailer. DR derives each slice from the committed group cursor, and CS alone advances it after transport completion. The old per-expert Q list is for legacy mode. In transaction mode the diagnostic derives layer/expert counts and physical row boundaries from the trailer's group table and current slice interval. Serving must feed the matching device layout to its backend, with W2 writing the same row order to `combine_x`.

A5 checks still needed include CANN/Bisheng and ACLNN/pybind build and load, mutation and same-stream ordering of the extended Active arena, independent F selection and early lane reuse under transport, and grouped GEMM support on the intended SKU and shapes. Graph replay and performance need separate verification.

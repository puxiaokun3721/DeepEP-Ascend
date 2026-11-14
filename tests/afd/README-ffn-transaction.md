# CAM FFN transaction bring-up

Build and load the paired CAM operators and DeepEP owner/JIT from the candidate revisions. The default mode uses a common DP round count. The optional local-round mode uses the same dispatch and combine entry points with a per-source round count.

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

On four available A5 cards, verify two A ranks in the same DP, two F ranks, and one physical slot per A. A0 sends 8,193 rows and A1 sends 8,192 rows per microbatch. Each uses an 8,192-row window.

```bash
timeout 930s torchrun --standalone --nproc_per_node=4 tests/afd/run_pipeline_cam.py \
  --ffn-transaction --local-comm-rounds --poll-cycles --reuse-short-slot --microbatches 2 \
  --batch 8193 --short-second 8192 --window 8192 --hidden 7168 --topk 2 --experts 4 \
  --q 16384 --transaction-alignment 32 --backend-max-rows 16384 --weights mix \
  --timeout-seconds 120 --run-tag local-rounds-h7168
```

The runner checks both F ranks see the old long-member tail and the new short-member Ready together, both first rounds freeze both members, and each tail freezes only A0. Each A checks its own output, all its acknowledgments and the empty terminal Ledger. The F ranks report final Credit after all ranks finish. The diagnostics use BF16 scaled-copy compute and unequal combine weights. `--transaction-gmm` uses diagnostic grouped GEMM. Structured `FFN_TX` lines separate raw device `system_cycles` from host poll and snapshot duration.

For the common-round regression, remove `--local-comm-rounds`, `--reuse-short-slot` and `--microbatches 2` while keeping the unequal input lengths. The short member must publish an empty tail. For the zero-row member, retain `--local-comm-rounds --poll-cycles`, remove `--reuse-short-slot --microbatches 2`, and set `--short-second 0`. The zero-row member participates in round 0 and completes its own acknowledgment path.

## Capacity and shared state

`prepare_recv(q, q_alloc, alignment, ffn_transaction=True, backend_max_rows=...)` activates FFN transactions. To opt into local rounds, call `enable_local_rounds()` on every rank before preparation. Q is a hard effective-row cap. The owner requires Q to cover the largest configured same-phase sum of lane capacities and Q_alloc to cover that bound after alignment. Q_alloc must fit the declared backend padded-row limit. The device case above declares a 16,384-row diagnostic configuration. Production also needs the target backend to support the actual groups, alignment, weights, activations and workspace.

`ffnPoll()` reports `[classification, observed_bit, expected_bit, phase]` per lane (0 absent, 1 fresh, 2 pending, -1 error). `ffnPollTimestamp()` returns the same report and optionally records the first device-system-cycle observation in a caller-owned, zero-initialized 64-byte-per-lane sidecar. A publication can be adopted by Scheduler between Poll scans. Match the sidecar to the publication identity before using its timing. Scheduler freezes a transaction in an Active trailer. DR derives each slice from the committed group cursor, and CS advances it after transport completion. The diagnostic derives layer/expert counts and physical row boundaries from the trailer's group table and current slice interval. Serving must feed the matching device layout to its backend, with W2 writing the same row order to `combine_x`.

A5 checks still needed include CANN/Bisheng and ACLNN/pybind build and load, mutation and same-stream ordering of the extended Active arena, independent F selection and early lane reuse under transport, and grouped GEMM support on the intended SKU and shapes. Graph replay and performance need separate verification.

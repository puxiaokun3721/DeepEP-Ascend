# AFD operator test artifacts

## Current multi-slot/multi-round A5 bring-up (2026-10-21, not executed here)

`run_pipeline_a5.py` accepts `--microbatches 1|2|3` (default 1) and
`--round-step` (default off). Defaults preserve the prior single-slot multi-round
case. Multiple local slots with multiple rounds require `--round-step`; both
argument validation and device admission reject the unsafe legacy combination.
Every A rank owns separate task/input/output tensors per microbatch, submits
sends serially on one send stream, and receives on one independent recv stream.
In step mode, start is called once per slot, then every slot receives one call in
each fixed fair sweep, including slots waiting for final ACK. Status/progress
copies are queued on the recv stream; the harness never waits for the live send
stream or performs an A-wide device synchronize before return progress finishes.
Only then does it join the last recv work with the send Epub event.

This is a functional bring-up driver with host observations and identity experts,
not a graph or performance benchmark. `--timeout-seconds` (default 600) bounds
host progress and kernel/domain timeouts; `--max-sweeps` (default 100000) bounds
sweep submission. Use an external watchdog for runtime failures that prevent a
host API returning. No Ready, Done, Consumed or credit is fabricated or reset.
F completion counts `num_comm_rounds * microbatches` DP entries, including zero
routes and exhausted members. Only this run's process group may be stopped on
failure; never kill an unrelated NPU workload.

**Preconditions before any command below:**

1. Use the approved existing Docker/container image and isolated test directory.
   Record host driver/firmware, CANN, compiler, torch/torch_npu, HCOMM/URMA and
   repository revisions before compiling. Do not install, upgrade, downgrade,
   edit host/container configuration or change drivers. If a prerequisite or
   version is incompatible, stop and ask the environment owner.
2. Inspect devices with the site's `mup-smi` tool (or its documented `npu-smi`
   command if that is the actual installed name). Select only explicitly free,
   authorized devices and map them into the container. Do not disturb active
   jobs, reset a device, or use commands targeting another process.
3. Build the matching modified DeepEP extension and make the pinned DeepJIT
   dependency available inside that approved container. Resolve unknown SDK/API
   compatibility before the communication test. Cold JIT compilation during an
   active transaction can exceed the timeout; record it separately, and do not
   claim an operator protocol failure solely from cold-compile timing.
4. Run from the DeepEP repository inside the container; the container must expose
   the requested 2 or 4 devices. The watchdog below targets only the new test job.

```bash
# Legacy P: one physical slot, multiple communication rounds (unchanged default).
timeout 900s torchrun --standalone --nproc_per_node=4 tests/afd/run_pipeline_a5.py
# Legacy D: M=2/3, one round each remains supported.
timeout 900s torchrun --standalone --nproc_per_node=4 tests/afd/run_pipeline_a5.py --batch 13 --window 16 --microbatches 3
# New fair round-step: two owners, M=2/3, hotspot F plus zero-route F.
timeout 900s torchrun --standalone --nproc_per_node=4 tests/afd/run_pipeline_a5.py --microbatches 2 --round-step
timeout 900s torchrun --standalone --nproc_per_node=4 tests/afd/run_pipeline_a5.py --microbatches 3 --round-step
# Short second owner, then all-zero second owner; both keep zero-length tail rounds.
timeout 900s torchrun --standalone --nproc_per_node=4 tests/afd/run_pipeline_a5.py --microbatches 3 --round-step --short-second 3
timeout 900s torchrun --standalone --nproc_per_node=4 tests/afd/run_pipeline_a5.py --microbatches 3 --round-step --short-second 0
# All-masked CONTROL and optional credit-precheck policy.
timeout 900s torchrun --standalone --nproc_per_node=4 tests/afd/run_pipeline_a5.py --microbatches 3 --round-step --all-masked
timeout 900s torchrun --standalone --nproc_per_node=4 tests/afd/run_pipeline_a5.py --microbatches 3 --round-step --precheck
# Small topology alternative: one A, one F, two slots.
timeout 900s torchrun --standalone --nproc_per_node=2 tests/afd/run_pipeline_a5.py --microbatches 2 --round-step
```

Expected success is a final JSON `status=PASS` after every per-slot BF16 output,
masked zero and untouched allocation tail is checked, all real final ACKs have
closed, send statuses succeeded, and every F has committed the expected entries.
The script prints `microbatches`, `round_step` and rank0 sweep count. On failure,
retain command, versions, full log, per-slot progress/error state and local patch;
do not change the environment or push to GitHub automatically. A sweep count is
not a latency/bandwidth result. Graph replay, visibility, synchronization resource
isolation and measured performance remain separate A5 gates.

Local verification of this new entry is **AST parsing only**: no torch import,
C++ compilation or NPU execution was performed here.

## Historical remaining-operator CPU record (2026-10-16)

Current suite: **51 CPU tests passed**, **19 Python AST checks passed**.
The 18 new cases cover scheduler state/selection, combineSend route/Completion
contracts and combineRecv weighted arithmetic/protocol models. These are not
C++ compilation or device execution evidence.

`run_pipeline_a5.py` now launches all five actual operators with identity experts;
it does not inject ACK/credit. It is prepared but **not run**. See
`docs/afd/remaining_operators.md` for commands and platform gates.

## Capacity clarification regression (2026-10-16)

`test_capacity_cpu.py` adds six independent specification/oracle tests: two same-DP
owners each C=32768 (DP=65536), each full B=262144 over eight common rounds,
heterogeneous windows/lengths and zero-length exhausted members, mask holes,
TP nonowner B=0, independent Decode slots, hot-F 524288 routes over sixteen
Q=32768 pieces with fixed padding budget, byte expansion retaining full-B source
token indices, and the single-owner workspace regression. Registered slots are
all counted in F memory even if not selected in one DP round. These are CPU
models plus actual-header arithmetic checks, NOT a scheduler implementation or
proof of device round/credit liveness.

```powershell
python -m unittest discover -s tests/afd -p 'test_*cpu.py' -v
```

Result on the development host: **33 CPU tests passed** (27 existing + 6 new).
All 15 `tests/afd/*.py` files parsed with Python AST. The ABI parser now ignores
C++ comments; it still reads and verifies actual field layouts.

Actual target host check, **not executed here**:

```bash
python tests/afd/run_capacity_a5.py --device 0
```

This invokes the compiled `AFDBuffer.create_local_receiver` constructor for
single 32k, same-DP 2x32k and heterogeneous 32k/16k lanes, checks bounded receive
allocation and doubled input reservation, and requires per-lane32769 rejection.
It does not execute a communication kernel. The real HCCL constructor, sender
8-round execution, live ACK/credit closure, scheduler aggregation, graph replay
and A5 kernel output/performance remain separate hardware gates. Do not invoke
multi-round send without a real ACK consumer and mistake a hang for capacity
failure. The downstream operators now have source implementations and the real
five-operator harness above; hardware validation remains outstanding.

This is the CPU contract/oracle portion of the approved test plan, not a completed A5 test harness. No torch or pytest dependency is required; NumPy is required. Tests do not import or execute the Ascend kernel.

```powershell
python -B -m unittest discover -s tests/afd -p test_cpu.py -v
```

On Linux, `bash tests/afd/run.sh` runs the same CPU suite. `gen_data.py OUTPUT.npz [--fp8] [--batch N]` creates deterministic raw-byte inputs. BF16/FP8 payloads and FP32 weights are represented as bytes/bits so NaN payloads are not canonicalized. `verify_dump.py INPUTS.npz SNAPSHOT.npz` checks a **provided real F snapshot**. It does not collect a snapshot or launch an operator. Its module docstring gives the required keys.

The independent oracle checks distinct target-rank records and every logical expert route. Physical record order may differ. `src_token_idx` is round-local: pass the matching input slice for each communication round. F `prefixes[e]` is the cumulative end of local expert e, not its start. Snapshot payload must be trimmed to `Inbox.num_recv_records * record_stride`; route_refs must be trimmed to the final prefix. Counts/prefix arrays have exactly E/N entries. Metadata padding is zero; unused record-stride holes have no zero requirement.

Three tests inspect real source rather than a substitute model: C++ wire declarations are parsed and compared with fixed ABI sizes/offsets; actual `make_workspace` arithmetic is evaluated against a separately formulated budget; actual TokenLayout expressions are evaluated against the independent byte layout. This restricted evaluator is **not a C++ compiler**: ABI packing/alignment, target compiler semantics and device API overloads remain unverified. Tests fail on grammar they cannot interpret.

Other tests validate the CPU oracle with corruption injections and the abstract protocol specification. They do not claim device branch coverage. Boundary arrays reach B=3*32768+1 without allocating full hidden payloads. Protocol tests cover zero-route F participation, delayed final ACK, 0→1→0, independent Decode slots and delayed Ledger reuse.

## A5 execution gap and required procedure

A two-process, single-round real kernel entry is provided below, and the five-operator multi-round harness is documented above. Neither has been run in this CPU-only workspace. Do **not** use a host `tensor.cpu()` poll loop on the same stream as a persistent send kernel: that can deadlock and does not test stream concurrency. The round-step driver observes progress on the independent recv stream.

1. On the actual Ascend950 stack, initialize DeepJIT dependencies, build the DeepEP host extension, and compile the new JIT dispatchSend path. Record compiler, CANN, HCOMM/URMA and driver versions.
2. Validate the actual ffnScheduler/dispatchRecv/combineSend chain in `run_pipeline_a5.py`. Include every F, including zero-route F, and inspect its real owner/control mapping rather than guessing remote addresses.
3. Validate combineRecv on a genuinely concurrent stream. For multiple slots and rounds use its round-step driver; only real return consumption can release Ledger and credit. Do not make dispatchSend clear Ledger itself.
4. Use AFDBuffer.dispatchSend with its exact public out arguments. Assert final send_status PUBLISHED can precede final Consumed. Wait for both consumers before owner destruction. Snapshot checker inputs must be sliced by Ledger.token_begin/num_round_tokens.
5. Run plan L0-01..07 first, then L1 and L2. Delay ACK/one jetty/Ledger independently, force SQ wrap, test empty members and cross-transaction persistent state. Use an external all-process watchdog for faults.
6. Only after functional verification measure maximum-rank elapsed time with device events plus cross-rank completion. Report actual transmitted bytes separately from reserved memory and workspace; no performance result is available now.

Not covered by CPU execution: API compilation, host extension validation branches, real SIMT dedup/histogram, source tensor DMA/SF stride, remote offset mapping, queue wrap/CQ completion, URMA→UBMEM visibility, control cache behavior, peer errors/domain exit, simultaneous send/recv residency, memory guards on hardware, graph replay, or performance. None of L0/L1/L2 groups is marked NPU-passed.

## Actual single-round A5 entry (provided, not executed here)

After building the modified DeepEP extension, run from this repository under an external watchdog:

```bash
timeout 300s torchrun --standalone --nproc_per_node=2 tests/afd/run_a5_single_round.py --batch 33 --output /tmp/afd-smoke
timeout 300s torchrun --standalone --nproc_per_node=2 tests/afd/run_a5_single_round.py --batch 33 --fp8 --transposed-sf
timeout 300s torchrun --standalone --nproc_per_node=2 tests/afd/run_a5_single_round.py --batch 0
timeout 300s torchrun --standalone --nproc_per_node=2 tests/afd/run_a5_single_round.py --batch 33 --all-masked
```

The script constructs the actual AFDBuffer on A=0/F=1, calls the actual JIT dispatchSend, checks PUBLISHED while final Consumed is still its initial value, then reads the actual F record/route/Inbox/Ready after sender completion and compares against the independent golden. It includes deterministic mixed masks, rank dedup (all 8 experts on F), raw BF16/FP8/SF bits and explicit optional SF transpose. Errors propagate; it does not fall back to CPU or silently skip. `--output` stores a snapshot compatible with verify_dump.py.

This is deliberately a single invocation. No ACK is fabricated, no slot is reused and no Ledger is released. All work is synchronized before explicit resource destruction. It proves neither the live Ready visibility ordering nor ACK, replay, multi-round, multiple members or concurrent progress. It was only AST checked in the current CPU environment; toolchain/API execution remains unverified.

The single-round snapshot smoke intentionally does not drive reverse-channel ACKs. The implemented five-operator path obtains A destinations through the version-1 return-directory extension and its source-peer table; it must be exercised through the pipeline harness above.
# Native-logic restoration regression matrix

`run_reuse_a5.py` invokes the actual local-F `dispatchRecv` kernel. It requires
an A5 SDK/NPU environment and does not implement a scheduler. Run:

```bash
timeout 1800s python tests/afd/run_reuse_a5.py --cores 8 --hidden 7168
```

Default coverage is 16 route counts (1, 3, 4, 5, 15, 16, 17, 31, 32, 33, 63,
64, 65, 129, 257, 513) x BF16 / FP8 row SF / FP8 column SF x cores 1 and 8:
96 valid invocations and 96 late-invalid-route invocations. Each case checks
byte-exact hidden/SF/metadata, padding, untouched output tail, untouched input
and control, and no partial outputs on invalid input. Entry-owned Segments are
not sorted by destination; long Segments cross core boundaries, and records
fan out to two experts. H=7168 uses 224 bytes SF per FP8 token. This is a
correctness suite, not a performance benchmark or proof of pipeline overlap.

For actual two-rank `dispatchSend` -> offline fixture -> `dispatchRecv`, exercise
batching across the 32-request drain threshold and four-token metadata tails:

```bash
for batch in 1 3 4 5 7 8 9 31 32 33 63 64 65 255 256 257; do
  timeout 300s torchrun --standalone --nproc_per_node=2 tests/afd/run_send_recv_a5.py --batch "$batch" --piece-size 17 --hidden 7168 --all-active
  timeout 300s torchrun --standalone --nproc_per_node=2 tests/afd/run_send_recv_a5.py --batch "$batch" --piece-size 17 --hidden 7168 --all-active --fp8
  timeout 300s torchrun --standalone --nproc_per_node=2 tests/afd/run_send_recv_a5.py --batch "$batch" --piece-size 17 --hidden 7168 --all-active --fp8 --transposed-sf
done
timeout 300s torchrun --standalone --nproc_per_node=2 tests/afd/run_send_recv_a5.py --batch 65 --hidden 7168 --fp8 --poison-masked-topk
timeout 300s torchrun --standalone --nproc_per_node=2 tests/afd/run_send_recv_a5.py --batch 65 --hidden 7168 --fp8 --all-masked --poison-masked-topk
timeout 300s torchrun --standalone --nproc_per_node=2 tests/afd/run_send_recv_a5.py --batch 257 --hidden 7168 --fp8 --all-active --cores 16
```

The all-active matrix makes token counts equal the per-destination record SQE
counts; default masks otherwise remove every third token. The two masked commands
exercise masked tokens with invalid topk sentinels inside metadata batches and
all-masked no-route publication. This harness synchronizes A
before constructing F Active; it does not validate async visibility, ACK reuse,
online scheduler behavior, or graph replay. The existing single-round sender
entry also uses exactly two ranks (A=0/F=1); multi-F execution is provided by the
four-rank pipeline harness above but still requires hardware validation. These short single-call cases do not exercise actual
SQ wrap; a long-running/cross-wrap device test remains pending. No C++ compilation or hardware results are
implied by the CPU fixture/oracle tests (`test_reuse_cpu.py`).
The final 16-core run covers more cores than the nine rank/expert prefix columns
in this topology; it requires a device exposing at least 16 vector cores.
